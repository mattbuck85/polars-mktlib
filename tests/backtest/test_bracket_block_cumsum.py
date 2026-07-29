"""The window-free per-block trigger count in :func:`_apply_bracket`.

``_apply_bracket`` needs, for every bar, how many bracket triggers have fired so
far *within the current position block* — that is what makes "first trigger in
the block wins" expressible. The obvious spelling is
``trig.cast(UInt32).cum_sum().over(BLOCK)``. The engine instead computes a global
``cum_sum`` and subtracts a forward-filled per-block base, which is materially
cheaper but only valid because ``BLOCK_COLUMN`` is contiguous and monotonically
non-decreasing (it is itself a ``cum_sum`` of a non-negative cast).

These tests pin that identity directly, across the boundary cases that a
whole-backtest fixture will not reliably produce. The golden Parquet baselines
run on 60 daily / 75 intraday bars — far too few blocks to be trusted to
exercise a single-row block, a block that runs to the end of the frame, or a
frame whose very first row triggers. This file is the durable guard for the
rewrite; the baselines are not.
"""

from __future__ import annotations

import polars as pl
import pytest
from polars.testing import assert_series_equal

from mktlib.backtest._bracket import BLOCK_COLUMN

_TRIG = "trig"


def _windowed(df: pl.DataFrame) -> pl.Series:
    """The formulation being replaced. The reference, by definition."""
    return df.select(
        pl.col(_TRIG).cast(pl.UInt32).cum_sum().over(BLOCK_COLUMN).alias("n")
    )["n"]


def _window_free(df: pl.DataFrame) -> pl.Series:
    """The formulation in ``_apply_bracket``, spelled identically."""
    out = df.with_columns(
        pl.col(_TRIG).cast(pl.UInt32).cum_sum().alias("_cum"),
    )
    out = out.with_columns(
        pl.when(pl.col(BLOCK_COLUMN) != pl.col(BLOCK_COLUMN).shift(1))
        .then(pl.col("_cum").shift(1).fill_null(0))
        .otherwise(None)
        .forward_fill()
        .fill_null(0)
        .alias("_base"),
    )
    return out.select((pl.col("_cum") - pl.col("_base")).alias("n"))["n"]


def _frame(blocks: list[int], trigs: list[bool]) -> pl.DataFrame:
    assert len(blocks) == len(trigs)
    return pl.DataFrame(
        {
            BLOCK_COLUMN: pl.Series(blocks, dtype=pl.UInt32),
            _TRIG: pl.Series(trigs, dtype=pl.Boolean),
        }
    )


def _entry_blocks(entries: list[bool]) -> list[int]:
    """Build a block id column the way the engine does: cum_sum of entry bars."""
    total = 0
    out: list[int] = []
    for e in entries:
        total += int(e)
        out.append(total)
    return out


# Each case is (label, entry-bar mask, trigger mask). Blocks are derived from the
# entry mask exactly as `_apply_bracket` derives BLOCK_COLUMN, so no case can
# accidentally describe a block layout the engine cannot produce.
CASES: list[tuple[str, list[bool], list[bool]]] = [
    (
        "no entries at all — BLOCK is constant 0",
        [False] * 12,
        [False, True, False, True] * 3,
    ),
    (
        "all triggers false",
        [True, False, False, True, False, False],
        [False] * 6,
    ),
    (
        "every trigger true",
        [True, False, False, True, False, False],
        [True] * 6,
    ),
    (
        "single-row blocks — a new block on every row",
        [True] * 8,
        [True, False, True, True, False, False, True, True],
    ),
    (
        "zero-trigger block between two triggering blocks",
        [True, False, True, False, False, True, False],
        [True, True, False, False, False, True, True],
    ),
    (
        "block runs to the end of the frame and triggers on the last row",
        [True, False, False, True, False, False],
        [False, False, False, False, False, True],
    ),
    (
        "first row of the frame is both an entry and a trigger",
        [True, False, True, False],
        [True, False, False, True],
    ),
    (
        "trigger before the first entry (flat rows lead the frame)",
        [False, False, True, False, True, False],
        [True, True, True, False, False, True],
    ),
    (
        "long run of a single block",
        [True] + [False] * 40,
        [False] * 20 + [True] + [True] * 20,
    ),
]


@pytest.mark.parametrize("label,entries,trigs", CASES, ids=[c[0] for c in CASES])
def test_matches_windowed(label: str, entries: list[bool], trigs: list[bool]) -> None:
    df = _frame(_entry_blocks(entries), trigs)
    assert_series_equal(_window_free(df), _windowed(df))


def test_matches_windowed_at_scale() -> None:
    """~50k rows over thousands of blocks — the shape the rewrite runs at.

    Deterministic, and built through the same cum_sum-of-entries construction so
    the block column is contiguous and non-decreasing by construction rather
    than by assertion.
    """
    n = 50_000
    entries = [(i * 7919) % 23 == 0 for i in range(n)]
    trigs = [(i * 104_729) % 5 == 0 for i in range(n)]
    df = _frame(_entry_blocks(entries), trigs)

    n_blocks = df[BLOCK_COLUMN].n_unique()
    assert n_blocks > 2_000, f"fixture is not block-dense enough: {n_blocks}"

    assert_series_equal(_window_free(df), _windowed(df))


def test_no_underflow_at_scale() -> None:
    """The subtraction is UInt32; a base above the running total would wrap.

    ``_base`` is an earlier value of a non-decreasing series, so it can never
    exceed ``_cum`` — but the failure mode of getting that wrong is a silent
    wrap to ~4e9 rather than a negative number, so it is asserted rather than
    reasoned about.
    """
    n = 50_000
    entries = [(i * 7919) % 23 == 0 for i in range(n)]
    trigs = [(i * 104_729) % 5 == 0 for i in range(n)]
    df = _frame(_entry_blocks(entries), trigs)

    counts = _window_free(df)
    assert counts.dtype == pl.UInt32
    # A wrap would land near 2**32; a correct count cannot exceed the row count.
    assert bool((counts <= n).all())


def test_count_resets_on_every_block_boundary() -> None:
    """Independent of the windowed reference: the property that is actually
    wanted. On the first row of a block the count is that row's own trigger."""
    n = 5_000
    entries = [(i * 31) % 17 == 0 for i in range(n)]
    trigs = [(i * 13) % 3 == 0 for i in range(n)]
    df = _frame(_entry_blocks(entries), trigs).with_columns(
        _window_free(_frame(_entry_blocks(entries), trigs)).alias("n"),
    )
    firsts = df.filter(
        pl.col(BLOCK_COLUMN) != pl.col(BLOCK_COLUMN).shift(1)
    )
    assert firsts.height > 100
    assert firsts.select(
        (pl.col("n") == pl.col(_TRIG).cast(pl.UInt32)).all()
    ).item()
