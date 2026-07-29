"""The re-arm position recurrence, against a sequential reference.

:func:`mktlib.backtest._bracket.held_expr` claims two things that are easy to
state and easy to get quietly wrong:

1. A bracket touch closes the position, and the **next** entry signal opens a
   new one — "first trigger in the block wins" plus re-arm, both falling out of
   a single forward-fill rather than a window over block ids.
2. The touch can be left **ungated** — not masked by "were we holding?" — because
   forcing ``0`` on an already-flat bar is a no-op. This is the load-bearing
   simplification: gating would require the position state that is being
   computed, which is the circularity the whole design avoids.

Both are pinned here against a plain-Python loop that applies the *gated*
semantics one bar at a time. The loop is the definition; the polars expression
has to match it, including on inputs where a touch fires while flat, where an
entry and a touch land on the same bar, and where a touch lands on the very
first or very last row.

Randomized frames come from an LCG rather than ``random`` so the fixtures are
reproducible without depending on CPython's PRNG stream.
"""

from __future__ import annotations

import polars as pl
import pytest

from mktlib.backtest._bracket import held_expr

_ENTRY, _EXIT, _TOUCH = "_entry", "_exit", "_touch"


def _reference(
    entry: list[bool], exit_: list[bool], touch: list[bool]
) -> list[int]:
    """Sequential, gated definition. One bar at a time, no cleverness.

    A touch only closes a position that is actually open — the ``held == 1``
    test is exactly the gate the vectorized form drops.
    """
    out: list[int] = []
    held = 0
    for e, x, t in zip(entry, exit_, touch, strict=True):
        if e:
            held = 1
        elif x or (t and held == 1):
            held = 0
        out.append(held)
    return out


def _vectorized(
    entry: list[bool], exit_: list[bool], touch: list[bool]
) -> list[int]:
    frame = pl.DataFrame(
        {
            _ENTRY: pl.Series(entry, dtype=pl.Boolean),
            _EXIT: pl.Series(exit_, dtype=pl.Boolean),
            _TOUCH: pl.Series(touch, dtype=pl.Boolean),
        }
    )
    return (
        frame.select(
            held_expr(
                entry_col=_ENTRY, exit_col=_EXIT, touch=pl.col(_TOUCH)
            ).alias("h")
        )["h"]
        .cast(pl.Int64)
        .to_list()
    )


def _check(entry: list[bool], exit_: list[bool], touch: list[bool]) -> None:
    assert _vectorized(entry, exit_, touch) == _reference(entry, exit_, touch)


# (label, entry, exit, touch)
CASES: list[tuple[str, list[bool], list[bool], list[bool]]] = [
    (
        "touch closes, next entry re-arms",
        [True, False, False, True, False, False],
        [False] * 6,
        [False, True, False, False, True, False],
    ),
    (
        "touch while flat is a no-op",
        [False, False, True, False, False, False],
        [False] * 6,
        [True, True, False, False, False, True],
    ),
    (
        "entry and touch on the same bar — entry wins",
        [False, True, True, False, False],
        [False] * 5,
        [False, False, True, False, False],
    ),
    (
        "touch on the first row",
        [False, True, False, False],
        [False] * 4,
        [True, False, False, False],
    ),
    (
        "touch on the last row",
        [True, False, False, False],
        [False] * 4,
        [False, False, False, True],
    ),
    (
        "signal exit and touch on the same bar",
        [True, False, False, False],
        [False, False, True, False],
        [False, False, True, False],
    ),
    (
        "repeated touches inside one position — only the first can matter",
        [True, False, False, False, False],
        [False] * 5,
        [False, True, True, True, True],
    ),
    (
        "no events at all",
        [False] * 5,
        [False] * 5,
        [False] * 5,
    ),
    (
        "entry on every bar",
        [True] * 5,
        [False] * 5,
        [True, False, True, False, True],
    ),
    (
        "single row, entry only",
        [True],
        [False],
        [False],
    ),
    (
        "single row, touch only",
        [False],
        [False],
        [True],
    ),
]


@pytest.mark.parametrize("label,entry,exit_,touch", CASES, ids=[c[0] for c in CASES])
def test_matches_sequential_reference(
    label: str, entry: list[bool], exit_: list[bool], touch: list[bool]
) -> None:
    _check(entry, exit_, touch)


def _lcg(seed: int, n: int) -> list[int]:
    """Numerical Recipes LCG — reproducible without depending on CPython."""
    out: list[int] = []
    state = seed
    for _ in range(n):
        state = (1_664_525 * state + 1_013_904_223) % (2**32)
        out.append(state)
    return out


@pytest.mark.parametrize("seed", [1, 7, 42, 12345, 999_983])
@pytest.mark.parametrize(
    "entry_rate,touch_rate", [(0.05, 0.10), (0.50, 0.50), (0.02, 0.90), (0.90, 0.02)]
)
def test_matches_reference_on_randomized_frames(
    seed: int, entry_rate: float, touch_rate: float
) -> None:
    """Densities span the realistic range and both degenerate ends.

    ``entry_rate=0.50`` is the persistent-entry shape that the engine guard
    rejects; the recurrence must still be *defined* there, because the guard
    reads the state this produces in order to decide whether to raise.
    """
    n = 4_000
    raw = _lcg(seed, n * 3)
    scale = float(2**32)
    entry = [raw[i] / scale < entry_rate for i in range(n)]
    exit_ = [raw[n + i] / scale < 0.05 for i in range(n)]
    touch = [raw[2 * n + i] / scale < touch_rate for i in range(n)]

    got = _vectorized(entry, exit_, touch)
    assert got == _reference(entry, exit_, touch)
    # Non-vacuity: the fixture must actually exercise both states.
    assert 0 < sum(got) < n


def test_gating_the_touch_would_change_nothing() -> None:
    """State the load-bearing claim directly, not just via the reference.

    The vectorized form drops the ``held == 1`` gate on the touch. Re-running
    the reference with the gate removed must give the same answer — if it ever
    does not, the simplification in ``held_expr`` is unsound.
    """

    def ungated_reference(
        entry: list[bool], exit_: list[bool], touch: list[bool]
    ) -> list[int]:
        out: list[int] = []
        held = 0
        for e, x, t in zip(entry, exit_, touch, strict=True):
            if e:
                held = 1
            elif x or t:
                held = 0
            out.append(held)
        return out

    n = 4_000
    raw = _lcg(2_026, n * 3)
    scale = float(2**32)
    entry = [raw[i] / scale < 0.20 for i in range(n)]
    exit_ = [raw[n + i] / scale < 0.05 for i in range(n)]
    touch = [raw[2 * n + i] / scale < 0.40 for i in range(n)]

    assert ungated_reference(entry, exit_, touch) == _reference(entry, exit_, touch)


def test_first_touch_in_a_position_is_the_one_that_closes_it() -> None:
    """The property the block-id machinery used to provide explicitly."""
    entry = [True] + [False] * 9
    exit_ = [False] * 10
    touch = [False, False, True, False, True, False, False, False, False, False]

    held = _vectorized(entry, exit_, touch)
    # Held from the entry bar through the first touch; flat afterwards, and the
    # second touch changes nothing because there is nothing left to close.
    assert held == [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
