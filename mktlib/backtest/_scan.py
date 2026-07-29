"""The entry/exit chain as a single forward pass over plain Python lists.

Deliberately free of polars and of ``mktlib.backtest`` types. Everything arrives
as lists of ``float`` / ``bool`` and leaves as lists of indices. That boundary is
the point: it is the contract a compiled accelerator would implement, and it is
what lets the kernel be tested without constructing a backtest.

Why one forward pass rather than something cleverer: the realized chain only ever
asks for the exit bar at entries it actually reaches, and those queries partition
the timeline — so a single scan touches each bar exactly once. **O(n)**, no
auxiliary structure. A doubling max/min table over all candidates was prototyped
and rejected on measurement; it answers a strictly harder question that nothing
here consumes, at 5.7x the cost even with its build amortized away.

Four things keep the loop near the floor of what CPython can do, each measured
rather than assumed:

* **Flat stretches are skipped with** ``list.index``, which searches in C. While
  flat there is nothing to evaluate, so the whole gap costs one call.
* **The entry array is not read per bar.** An entry signal only matters on a bar
  where an exit would otherwise fire — that is the entire content of the "entry
  wins" rule — so it is consulted once per firing bar.
* **Strictness folds into the latched threshold**, once per trade, because
  ``v > t`` is exactly ``v >= nextafter(t, +inf)`` for floats.
* **The take-profit-and-stop-loss shape gets its own loop.** Two legs reading one
  column, one upward and one downward, is what essentially every eligible tree
  compiles to, and sharing the single per-bar read halves its work.

Together those took the kernel from 2.7x slower than an equivalent hand-rolled
loop to parity with it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import nextafter
from typing import Literal

#: Which side of its level a leg watches.
Direction = Literal["above", "below"]

_INF = float("inf")
_NEG_INF = float("-inf")


@dataclass(frozen=True, slots=True)
class ScanLeg:
    """One exit level pinned at the entry bar.

    ``value`` is read on the bar being tested. ``threshold`` is read **once**, on
    the bar the position opened, and held for the trade — which is what makes
    this a fixed barrier rather than a trailing one.

    Nulls must arrive as ``NaN``, not as ``None`` and not as an infinity. Every
    comparison against ``NaN`` is false in both directions and at both strictness
    settings, so a null bar simply never fires and the loop needs no extra
    branch. ``+inf >= +inf`` is true, so an infinity would tie against a real one.
    """

    value: list[float]
    threshold: list[float]
    direction: Direction
    strict: bool = True


@dataclass(frozen=True, slots=True)
class ScanResult:
    """Realized entries, plus where each of them closed.

    ``exits[i]`` is the bar that closed the trade opened at ``entries[i]``, or
    ``None`` for a position still open when the frame ended. ``winners[i]`` is
    the index into ``legs`` of the leg that fired, or ``-1`` when the close came
    from the fixed predicate or a forced session flatten.
    """

    realized: list[bool]
    entries: list[int] = field(default_factory=list)
    exits: list[int | None] = field(default_factory=list)
    winners: list[int] = field(default_factory=list)


def _next_entry(entry: list[bool], start: int) -> int:
    """First entry signal at or after *start*, or -1."""
    try:
        return entry.index(True, start)
    except ValueError:
        return -1


def _latch(leg: ScanLeg, bar: int) -> float:
    """The leg's level for a trade opened on *bar*, with strictness folded in."""
    level = leg.threshold[bar]
    if not leg.strict:
        return level
    return nextafter(level, _INF if leg.direction == "above" else _NEG_INF)


def scan_realized(  # noqa: C901, PLR0912, PLR0915
    *,
    entry: list[bool],
    legs: tuple[ScanLeg, ...],
    fixed: list[bool] | None = None,
    session_last: list[bool] | None = None,
    n: int,
) -> ScanResult:
    """Resolve which entry signals open a position, and where each one closes.

    Mirrors the general resolver bar for bar, including the two rules that are
    easy to get wrong:

    * **An entry signal wins over an exit on a shared bar.** The position stays
      open, the exit does not happen, and the trade keeps its original anchor.
    * **Under ``flatten_eod`` the session-last bar closes regardless**, and an
      entry signal on such a bar opens nothing — it is deferred, and the
      deferred signal is already present on the next session's first bar.
    """
    realized = [False] * n
    entries: list[int] = []
    exits: list[int | None] = []
    winners: list[int] = []

    n_legs = len(legs)
    forced_any = session_last is not None
    fixed_any = fixed is not None
    plain = not forced_any and not fixed_any

    pair = (
        plain
        and n_legs == 2
        and legs[0].value is legs[1].value
        and legs[0].direction != legs[1].direction
    )
    solo = plain and n_legs == 1
    values = legs[0].value if n_legs else []
    up_first = bool(n_legs) and legs[0].direction == "above"

    leg_values = [leg.value for leg in legs]
    leg_above = [leg.direction == "above" for leg in legs]
    span = range(n_legs)

    cursor = 0
    while cursor < n:
        candidate = _next_entry(entry, cursor)
        if candidate < 0:
            break
        if forced_any and session_last[candidate]:  # type: ignore[index]
            # Opens nothing: deferred to the next session, whose own signal the
            # engine has already materialized.
            cursor = candidate + 1
            continue

        realized[candidate] = True
        entries.append(candidate)
        latched = [_latch(leg, candidate) for leg in legs]

        closed: int | None = None
        winner = -1
        bar = candidate + 1

        if pair:
            # Two loops rather than one with a per-bar direction test. Legs are
            # tried in DECLARATION order, matching the general path below — a
            # bar that tags both levels must not resolve differently just
            # because it took the specialized route.
            t0, t1 = latched[0], latched[1]
            if up_first:
                while bar < n:
                    x = values[bar]
                    if x >= t0:
                        hit = 0
                    elif x <= t1:
                        hit = 1
                    else:
                        bar += 1
                        continue
                    if entry[bar]:
                        bar += 1
                        continue
                    closed = bar
                    winner = hit
                    break
            else:
                while bar < n:
                    x = values[bar]
                    if x <= t0:
                        hit = 0
                    elif x >= t1:
                        hit = 1
                    else:
                        bar += 1
                        continue
                    if entry[bar]:
                        bar += 1
                        continue
                    closed = bar
                    winner = hit
                    break
        elif solo:
            level = latched[0]
            while bar < n:
                x = values[bar]
                if (x >= level) if up_first else (x <= level):
                    if entry[bar]:
                        bar += 1
                        continue
                    closed = bar
                    winner = 0
                    break
                bar += 1
        else:
            while bar < n:
                forced = forced_any and session_last[bar]  # type: ignore[index]
                hit = -1
                for i in span:
                    value = leg_values[i][bar]
                    level = latched[i]
                    if (value >= level) if leg_above[i] else (value <= level):
                        hit = i
                        break
                if hit < 0 and not forced and not (fixed_any and fixed[bar]):  # type: ignore[index]
                    bar += 1
                    continue
                if entry[bar] and not forced:
                    bar += 1
                    continue
                closed = bar
                winner = hit
                break

        exits.append(closed)
        winners.append(winner)
        if closed is None:
            break  # open at the end of the frame; nothing after it can realize
        cursor = closed + 1

    return ScanResult(
        realized=realized, entries=entries, exits=exits, winners=winners
    )
