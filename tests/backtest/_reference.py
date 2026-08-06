"""An independent, bar-by-bar reference backtester.

This exists to be *obviously* right rather than fast. It is a plain Python loop
over Python lists, written from the documented semantics, and it is the oracle
the vectorized engine is checked against. Its value depends entirely on being an
independent implementation, so it must not import ``mktlib.backtest`` — that is
enforced by an AST lint in ``test_reference_independence.py``, not by discipline.

Read the fill ladder in :func:`run_reference` against the prose in
``docs/api/backtest.rst`` and ``mktlib/backtest/_bracket.py``; if they disagree,
one of them is wrong and that is the point of this file.

Semantics implemented
---------------------

*Position.* ``entry`` wins over ``exit`` on a shared bar, matching the engine's
``when(entry).then(1).when(exit).then(0)`` chain. A consequence worth stating:
an entry signal landing on the bar an exit would have fired keeps the position
open, so the exit does not happen and the trade continues **on its original
anchor**. A "realized" entry is therefore one that fires while flat.

*Fills.* An entry fills at the next bar's open. An exit fills at the next bar's
open too, unless something owns the bar it fired on — a bracket level or a limit
price — in which case it fills inside that bar.

*Bracket anchoring.* ``RefBracket.anchor`` mirrors ``Bracket.anchor``.
``"position"`` latches each leg once, on the entry that opened the position.
``"signal"`` also moves it on every later entry signal that fires while the
position is still held — armed on the bar after the signal, so the level the
signal bar itself is checked against is still the old one. See
:func:`_block_levels`.

*Returns.* One value per bar: ``close/open - 1`` on an entry fill bar,
``close/prev_close - 1`` while held, ``fill/prev_close - 1`` on an exit bar, and
``0.0`` when flat. Costs are subtracted on the bars that pay a fill.

Out of scope, and asserted rather than silently mis-modelled: ``flatten_eod``,
dual-side runs, and portfolio weights. Multi-instrument is reachable by looping
this per partition.
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: The reference is O(bars) in Python. It is an oracle, not an engine.
MAX_BARS = 25_000


@dataclass(frozen=True, slots=True)
class RefLeg:
    """An exit level pinned at the entry bar: ``anchor[source] * mult``."""

    source: str
    mult: float
    direction: str  # "above" | "below"
    strict: bool = True


@dataclass(frozen=True, slots=True)
class RefBracket:
    """Protective levels checked against each bar's range while held.

    ``tp`` / ``sl`` are either a fraction of the entry fill price (``float``) or
    the name of a per-bar column of absolute levels (``str``), latched at the
    entry signal bar — mirroring ``Bracket``.

    ``anchor`` mirrors ``Bracket.anchor``. Under ``"position"`` each leg is
    latched once and holds for the life of the trade. Under ``"signal"`` an
    entry signal that fires while the position is still open moves it — see
    :func:`_block_levels` for the arming convention.
    """

    tp: float | str | None = None
    sl: float | str | None = None
    both_touch: str = "stop_first"
    anchor: str = "position"


@dataclass(frozen=True, slots=True)
class RefConfig:
    side: int = 1
    legs: tuple[RefLeg, ...] = ()
    bracket: RefBracket | None = None
    limit_exit: bool = False
    cost_bps: float = 0.0
    #: Anchor at every raw entry signal instead of every realized one. Models
    #: the pre-fix behaviour so the defect itself can be regression-tested.
    legacy_anchoring: bool = False


@dataclass(frozen=True, slots=True)
class RefBars:
    open: list[float]
    high: list[float]
    low: list[float]
    close: list[float]
    entry: list[bool]
    exit: list[bool]
    #: Raw columns available for anchoring and for absolute bracket levels.
    cols: dict[str, list[float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = len(self.close)
        assert n <= MAX_BARS, f"reference is capped at {MAX_BARS} bars, got {n}"
        for name in ("open", "high", "low", "entry", "exit"):
            assert len(getattr(self, name)) == n, f"{name} length mismatch"


@dataclass(frozen=True, slots=True)
class RefTrade:
    entry_bar: int
    exit_bar: int
    side: int
    pnl: float
    bars_held: int


@dataclass(frozen=True, slots=True)
class RefResult:
    trades: list[RefTrade]
    returns: list[float]


def _crosses(value: float, level: float, direction: str, strict: bool) -> bool:
    if direction == "above":
        return value > level if strict else value >= level
    return value < level if strict else value <= level


def run_reference(bars: RefBars, cfg: RefConfig) -> RefResult:  # noqa: C901, PLR0912, PLR0915
    """Walk the bars once. No expressions, no vectorization, no cleverness."""
    n = len(bars.close)
    rets = [0.0] * n
    trades: list[RefTrade] = []
    cost = cfg.cost_bps / 1e4

    # --- pass 1: position and anchors, one bar at a time --------------------
    pos = [0] * n
    anchors: list[dict[str, float] | None] = [None] * n
    held = 0
    anchor: dict[str, float] | None = None
    sources = {leg.source for leg in cfg.legs}

    for t in range(n):
        fired = bars.exit[t]
        if held == 1 and anchor is not None:
            for leg in cfg.legs:
                level = anchor[leg.source] * leg.mult
                if _crosses(bars.close[t], level, leg.direction, leg.strict):
                    fired = True

        if bars.entry[t]:
            if held == 0:
                anchor = {s: bars.cols[s][t] for s in sources}
            elif cfg.legacy_anchoring:
                # The defect: re-latch even though this signal opens nothing.
                anchor = {s: bars.cols[s][t] for s in sources}
            held = 1
        elif fired:
            held = 0
        pos[t] = held
        anchors[t] = anchor

    # --- pass 2: fills, returns and trades ---------------------------------
    # Blocks are runs of pos == 1 at signal level. The entry fills on the bar
    # after the block opens; the exit fills on the bar after it closes, unless
    # a bracket or limit owns a bar inside it.
    t = 0
    while t < n:
        if pos[t] != 1:
            t += 1
            continue
        start = t
        while t + 1 < n and pos[t + 1] == 1:
            t += 1
        end = t  # last signal bar of the block
        t += 1

        fill_bar = start + 1
        if fill_bar >= n:
            break
        entry_price = bars.open[fill_bar]

        block_anchor = anchors[start] or {}
        exit_bar, exit_price = _resolve_exit(
            bars, cfg, fill_bar, end, entry_price, block_anchor, n
        )
        if exit_bar is None or exit_price is None:
            # The frame ends with the position still open. `returns` is
            # mark-to-market, so the held bars still accrue; `trades` records
            # closed trades only, so nothing is appended.
            last = min(end + 1, n - 1)
            for bar in range(fill_bar, last + 1):
                raw = (
                    bars.close[bar] / bars.open[bar] - 1.0
                    if bar == fill_bar
                    else bars.close[bar] / bars.close[bar - 1] - 1.0
                )
                charged = cost if bar == fill_bar else 0.0
                rets[bar] = cfg.side * raw - charged
            continue

        gross = cfg.side * (exit_price / entry_price - 1.0)
        pnl = gross - 2.0 * cost if cost else gross
        trades.append(
            RefTrade(
                entry_bar=start,
                exit_bar=exit_bar,
                side=cfg.side,
                pnl=pnl,
                bars_held=exit_bar - fill_bar,
            )
        )

        for bar in range(fill_bar, exit_bar + 1):
            if bar == fill_bar and bar == exit_bar:
                raw = exit_price / bars.open[bar] - 1.0
                charged = 2.0 * cost
            elif bar == fill_bar:
                raw = bars.close[bar] / bars.open[bar] - 1.0
                charged = cost
            elif bar == exit_bar:
                raw = exit_price / bars.close[bar - 1] - 1.0
                charged = cost
            else:
                raw = bars.close[bar] / bars.close[bar - 1] - 1.0
                charged = 0.0
            rets[bar] = cfg.side * raw - charged

    return RefResult(trades=trades, returns=rets)


def _resolve_exit(  # noqa: C901, PLR0912
    bars: RefBars,
    cfg: RefConfig,
    fill_bar: int,
    end: int,
    entry_price: float,
    anchor: dict[str, float],
    n: int,
) -> tuple[int | None, float | None]:
    """Where and at what price this position closes.

    Priority, highest first: a bracket level tagged inside a bar, a limit price
    on the bar its condition fired, then the next bar's open.
    """
    brk = cfg.bracket
    if brk is not None:
        tp_levels = _block_levels(bars, cfg, brk.tp, "tp", fill_bar, end, entry_price, n)
        sl_levels = _block_levels(bars, cfg, brk.sl, "sl", fill_bar, end, entry_price, n)
        # Live from the entry fill bar through `end + 1`: the position is
        # held whenever the previous bar's signal position was 1.
        for bar in range(fill_bar, min(end + 1, n - 1) + 1):
            tp_level = tp_levels.get(bar)
            sl_level = sl_levels.get(bar)
            tp_hit = tp_level is not None and _tagged(bars, bar, tp_level, cfg.side, "tp")
            sl_hit = sl_level is not None and _tagged(bars, bar, sl_level, cfg.side, "sl")
            if not (tp_hit or sl_hit):
                continue
            stop_first = brk.both_touch == "stop_first"
            take_sl = sl_hit and (stop_first or not tp_hit)
            if take_sl:
                assert sl_level is not None
                price = (
                    min(bars.open[bar], sl_level)
                    if cfg.side == 1
                    else max(bars.open[bar], sl_level)
                )
            else:
                assert tp_level is not None
                price = (
                    max(bars.open[bar], tp_level)
                    if cfg.side == 1
                    else min(bars.open[bar], tp_level)
                )
            return (bar, price)

    if cfg.limit_exit:
        for bar in range(fill_bar, min(end + 1, n - 1) + 1):
            if bars.exit[bar]:
                return (bar, bars.cols["_limit_price"][bar])

    # The position is live through bar ``end + 1`` — ``end`` is the last SIGNAL
    # bar at which it was open, and the exit signal lands on ``end + 1``. A
    # next-open exit therefore fills one bar later still.
    if end + 2 >= n:
        return (None, None)
    return (end + 2, bars.open[end + 2])


def _block_levels(
    bars: RefBars,
    cfg: RefConfig,
    spec: float | str | None,
    leg: str,
    fill_bar: int,
    end: int,
    entry_price: float,
    n: int,
) -> dict[int, float]:
    """The level in force on each live bar of one block, for one leg.

    Under ``anchor="position"`` this is one latched value repeated — the level
    the position was opened with.

    Under ``anchor="signal"`` an entry signal that fires while the position is
    still open moves it. The signal on bar ``k`` is observed at that bar's
    *close*, so the OLD level still governs bar ``k`` itself and the new one is
    in force from ``k + 1``. A bar that tags a leg therefore closes the position
    before any re-anchor on that same bar could take effect.

    What the new level is, per spec kind:

    ``float``
        ``open[k + 1]`` scaled by the side-appropriate multiplier — the price a
        fresh entry on that signal would have filled at, which is the engine's
        own entry-fill model and not a second price convention.
    ``str``
        the spec column re-read on bar ``k``, the same bar the position-opening
        signal reads it on.

    A re-signal on the frame's last bar has no ``k + 1`` to land on and is
    dropped.
    """
    if spec is None:
        return {}
    assert cfg.bracket is not None
    re_anchors = cfg.bracket.anchor == "signal"
    current = _latch(bars, spec, cfg.side, entry_price, fill_bar, leg)
    levels: dict[int, float] = {}
    for bar in range(fill_bar, min(end + 1, n - 1) + 1):
        levels[bar] = current
        # Every bar in this range is one the position is held on, so an entry
        # signal here is by definition a re-signal: it opens nothing.
        if not re_anchors or not bars.entry[bar] or bar + 1 >= n:
            continue
        current = (
            bars.cols[spec][bar]
            if isinstance(spec, str)
            else _pct_level(spec, side=cfg.side, anchor_price=bars.open[bar + 1], leg=leg)
        )
    return levels


def _latch(
    bars: RefBars,
    spec: float | str,
    side: int,
    entry_price: float,
    fill_bar: int,
    leg: str,
) -> float:
    """The level the position opens with."""
    if isinstance(spec, str):
        # Absolute levels are latched on the entry SIGNAL bar.
        return bars.cols[spec][fill_bar - 1]
    return _pct_level(spec, side=side, anchor_price=entry_price, leg=leg)


def _pct_level(spec: float, *, side: int, anchor_price: float, leg: str) -> float:
    """A fractional leg measured off *anchor_price*."""
    favourable = leg == "tp"
    up = favourable == (side == 1)
    return anchor_price * (1.0 + spec if up else 1.0 - spec)


def _tagged(bars: RefBars, bar: int, level: float, side: int, leg: str) -> bool:
    """Does this bar's range reach the level? Long TP is a high, long SL a low."""
    if leg == "tp":
        return bars.high[bar] >= level if side == 1 else bars.low[bar] <= level
    return bars.low[bar] <= level if side == 1 else bars.high[bar] >= level
