"""A generated corpus of exit trees, run down both resolvers.

The enumerated gate in `test_anchor_equivalence.py` covers the shapes anyone
actually writes. This one exists for the shapes nobody thought of, and for the
day someone widens `plan_exit`.

Nothing here declares which trees are eligible. The corpus is built blind and
partitioned **at runtime by `plan_arrays`** — the same call production
dispatches on — so a widening automatically pulls new shapes into the compared
partition. A test that had to be edited to keep covering things would not.
"""

from __future__ import annotations

import itertools

import polars as pl
import pytest

from mktlib.backtest import (
    Col,
    Condition,
    Crossunder,
    EntryRef,
    Limit,
    Pct,
    ValueGT,
    ValueGTE,
    ValueLT,
    ValueLTE,
)
from mktlib.backtest._anchor import (
    _realized_entries_windowed,
    collect_entry_refs,
    plan_arrays,
    plan_exit,
    realized_entries,
)

_N = 140

_OPS = (ValueGT, ValueGTE, ValueLT, ValueLTE)

#: Threshold shapes. The first three are pinned at entry; the last two move with
#: the bar and must therefore land in the fallback partition.
_THRESHOLDS = (
    ("pct_up", Pct(EntryRef("close"), 4.0)),
    ("pct_down", Pct(EntryRef("close"), -4.0)),
    ("two_snapshots", EntryRef("close") - EntryRef("atr") * 2),
    ("moving_col", EntryRef("close") - Col("atr") * 2),
    ("bare_col", Col("slow")),
)


def _frame(seed: int, entry_every: int) -> pl.DataFrame:
    state = seed
    close: list[float] = []
    px = 100.0
    for _ in range(_N):
        state = (1_664_525 * state + 1_013_904_223) % (2**32)
        px *= 1.0 + ((state / 2**32) - 0.5) * 0.05
        close.append(px)
    return pl.DataFrame(
        {"close": close, "_entry": [i % entry_every == 0 for i in range(_N)]}
    ).with_columns(
        (pl.col("close") * 0.03).alias("atr"),
        pl.col("close").rolling_mean(3).fill_null(strategy="backward").alias("fast"),
        pl.col("close").rolling_mean(10).fill_null(strategy="backward").alias("slow"),
        (pl.int_range(pl.len()) % 23 == 0).alias("_session_last"),
    )


FRAMES = (_frame(7, 11), _frame(29, 4))


def _corpus() -> list[tuple[str, Condition]]:
    """Bounded combinator sweep, depth <= 2. Built blind to eligibility."""
    out: list[tuple[str, Condition]] = []
    singles: list[tuple[str, Condition]] = []

    for op, (label, threshold), flip in itertools.product(
        _OPS, _THRESHOLDS, (False, True)
    ):
        name = f"{op.__name__}-{label}{'-flipped' if flip else ''}"
        cond = op(threshold, "close") if flip else op("close", threshold)  # type: ignore[arg-type]
        singles.append((name, cond))

    out.extend(singles)
    # OR of two anchored terms, OR with a fixed term, wrapped in Limit, and an
    # AND that must never be dispatched to the scan.
    for name, cond in singles[::3]:
        for other_name, other in singles[1::7]:
            out.append((f"{name}|{other_name}", cond | other))
        out.append((f"{name}|crossunder", cond | Crossunder("fast", "slow")))
        out.append((f"{name}&crossunder", cond & Crossunder("fast", "slow")))
        out.append((f"limit({name})", Limit(cond)))
        out.append((f"~{name}", ~cond))
    return out


CORPUS = _corpus()


@pytest.mark.parametrize("name,cond", CORPUS, ids=[n for n, _ in CORPUS])
@pytest.mark.parametrize("frame_index", range(len(FRAMES)))
def test_generated_tree_resolves_identically(
    name: str, cond: Condition, frame_index: int, scan_backend: str
) -> None:
    frame = FRAMES[frame_index]
    arrays = plan_arrays(frame, plan_exit(cond))
    if arrays is None:
        return  # fallback partition: nothing to compare, by construction

    fast = realized_entries(
        frame,
        entry_col="_entry",
        exit_cond=cond,
        snapshot_cols=collect_entry_refs(cond),
        session_last_col=None,
    )
    windowed = _realized_entries_windowed(
        frame,
        entry_col="_entry",
        exit_cond=cond,
        snapshot_cols=collect_entry_refs(cond),
        session_last_col=None,
    )
    assert fast.equals(windowed), f"{name} / backend={scan_backend}"


def test_the_corpus_reaches_both_partitions() -> None:
    """Neither partition may be empty, or this file proves nothing.

    Measured against the live predicate, so it keeps its meaning after a
    widening rather than silently degrading into a one-sided test.
    """
    frame = FRAMES[0]
    eligible = [
        name
        for name, cond in CORPUS
        if plan_arrays(frame, plan_exit(cond)) is not None
    ]
    fallback = [
        name
        for name, cond in CORPUS
        if plan_arrays(frame, plan_exit(cond)) is None
    ]
    assert len(eligible) > 20, f"only {len(eligible)} trees reach the scan"
    assert len(fallback) > 20, f"only {len(fallback)} trees reach the fallback"


def test_moving_thresholds_never_reach_the_scan() -> None:
    """The distinction the whole eligibility predicate exists to draw.

    A *moving* threshold — one whose level re-reads the current bar, like
    ``EntryRef("close") - Col("atr") * 2`` — disqualifies the whole tree, in any
    position, because a monotone scan cannot chase a barrier that moves.
    """
    frame = FRAMES[0]
    checked = 0
    for name, cond in CORPUS:
        if "moving_col" in name:
            assert plan_arrays(frame, plan_exit(cond)) is None, name
            checked += 1
    assert checked > 10, "corpus stopped generating moving-threshold trees"


def test_a_plain_column_term_rides_along_as_the_fixed_predicate() -> None:
    """`ValueGT("close", Col("slow"))` is not a moving threshold.

    It mentions no ``EntryRef`` at all, so it is an ordinary exit condition that
    travels as the plan's ``fixed`` part. OR-ing one alongside an anchored leg
    must therefore stay eligible — the distinction is "does the *threshold* move
    with the bar", not "does the term read a column".
    """
    frame = FRAMES[0]
    anchored = ValueGT("close", Pct(EntryRef("close"), 4.0))
    fixed_term = ValueGT("close", Col("slow"))

    assert plan_arrays(frame, plan_exit(fixed_term)) is None, (
        "on its own there is no anchored leg to resolve"
    )
    plan = plan_exit(anchored | fixed_term)
    assert plan.eligible
    assert plan.fixed is not None
    assert plan_arrays(frame, plan) is not None
