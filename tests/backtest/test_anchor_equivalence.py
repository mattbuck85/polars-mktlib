"""The two resolvers must agree, on every tree the fast one claims.

`realized_entries` dispatches: exit trees `plan_exit` recognizes resolve in a
single forward pass, everything else falls back to evaluating the user's own
expression over a window. That is only safe if the two never disagree.

The corpus is partitioned **at runtime by `plan_arrays`** — the same call
production dispatches on — rather than by a hand-maintained list. So when
eligibility is widened later, the newly-eligible shapes are pulled into the gated
partition automatically and cannot silently change results. The conditions
themselves are imported from `test_anchor_plan` for the same reason: two copies
of that list would drift.

Unmarked and in the default job. A gate behind a `slow` marker is not a gate.
"""

from __future__ import annotations

import polars as pl
import pytest

from mktlib.backtest._anchor import (
    _realized_entries_windowed,
    collect_entry_refs,
    plan_arrays,
    plan_exit,
    realized_entries,
)
from tests.backtest.test_anchor_plan import _ELIGIBLE, _INELIGIBLE

_N = 160


def _frame(
    *,
    seed: int,
    entry_every: int,
    nulls: bool = False,
    no_entries: bool = False,
    all_entries: bool = False,
) -> pl.DataFrame:
    """A small frame carrying every column the corpus can reference."""
    state = seed
    close: list[float] = []
    px = 100.0
    for _ in range(_N):
        state = (1_664_525 * state + 1_013_904_223) % (2**32)
        px *= 1.0 + ((state / 2**32) - 0.5) * 0.06
        close.append(px)

    if no_entries:
        entry = [False] * _N
    elif all_entries:
        entry = [True] * _N
    else:
        entry = [i % entry_every == 0 for i in range(_N)]

    frame = pl.DataFrame({"close": close, "_entry": entry})
    frame = frame.with_columns(
        (pl.col("close") * 0.02).alias("atr"),
        (pl.col("close") * 1_000.0).alias("volume"),
        pl.col("close").rolling_mean(3).fill_null(strategy="backward").alias("fast"),
        pl.col("close").rolling_mean(12).fill_null(strategy="backward").alias("slow"),
        (pl.int_range(pl.len()) % 17 == 0).alias("_session_last"),
    )
    if nulls:
        frame = frame.with_columns(
            pl.when(pl.int_range(pl.len()) % 11 == 0)
            .then(None)
            .otherwise(pl.col("close"))
            .alias("close")
        )
    return frame


FRAMES: dict[str, pl.DataFrame] = {
    "ordinary": _frame(seed=7, entry_every=13),
    "dense entries": _frame(seed=11, entry_every=3),
    "sparse entries": _frame(seed=23, entry_every=57),
    "null-laden close": _frame(seed=31, entry_every=13, nulls=True),
    "no entries": _frame(seed=5, entry_every=13, no_entries=True),
    "entry on every bar": _frame(seed=5, entry_every=1, all_entries=True),
}

CASES = [
    (label, index)
    for label in FRAMES
    for index in range(len(_ELIGIBLE))
]


@pytest.mark.parametrize(
    "frame_label,cond_index", CASES, ids=[f"{f}-{i}" for f, i in CASES]
)
@pytest.mark.parametrize("session", [None, "_session_last"])
def test_fast_and_windowed_agree(
    frame_label: str, cond_index: int, session: str | None, scan_backend: str
) -> None:
    frame = FRAMES[frame_label]
    cond = _ELIGIBLE[cond_index]

    if plan_arrays(frame, plan_exit(cond)) is None:
        pytest.skip("dispatch would not take the fast path for this pairing")

    # Through the real dispatcher, so the backend under test is the one
    # production would pick — not a hand-picked inner function.
    fast = realized_entries(
        frame,
        entry_col="_entry",
        exit_cond=cond,
        snapshot_cols=collect_entry_refs(cond),
        session_last_col=session,
    )
    windowed = _realized_entries_windowed(
        frame,
        entry_col="_entry",
        exit_cond=cond,
        snapshot_cols=collect_entry_refs(cond),
        session_last_col=session,
    )
    assert fast.equals(windowed), (
        f"{frame_label} / condition {cond_index} / session={session} / "
        f"backend={scan_backend}: fast realized {fast.sum()} entries, "
        f"windowed realized {windowed.sum()}"
    )


@pytest.mark.parametrize("cond_index", range(len(_INELIGIBLE)))
def test_ineligible_trees_are_not_dispatched_to_the_fast_path(
    cond_index: int,
) -> None:
    """The fallback must actually be reachable, or the gate above proves little."""
    cond = _INELIGIBLE[cond_index]
    assert plan_arrays(FRAMES["ordinary"], plan_exit(cond)) is None


def test_both_partitions_are_non_empty() -> None:
    """A planner that accepted everything would make the gate vacuous.

    Asserted against the live predicate rather than the declared lists, so it
    keeps meaning something if eligibility is widened.
    """
    frame = FRAMES["ordinary"]
    eligible = sum(
        plan_arrays(frame, plan_exit(c)) is not None for c in _ELIGIBLE
    )
    ineligible = sum(
        plan_arrays(frame, plan_exit(c)) is None for c in _INELIGIBLE
    )
    assert eligible > 0, "no condition reaches the fast path"
    assert ineligible > 0, "no condition reaches the fallback"


def test_the_gate_would_catch_a_divergent_fast_path(scan_backend: str) -> None:
    """Non-vacuity: perturb the scan and the comparison must fail.

    A gate that cannot fail is decoration. Shifting the latched level by one
    percent is a plausible-looking bug — the levels are still monotone, the
    trades still look sane — and it must not survive.

    Both backends are perturbed, by different means, because "the gate can
    fail" has to be true of whichever implementation is actually running. The
    Python kernel is monkeypatched at ``_latch``; compiled code cannot be, so
    the accelerator exports a test-only perturbed kernel for exactly this, and
    asserts on its own side that the perturbation still moves a trade.
    """
    from mktlib.backtest import _anchor, _scan

    frame = FRAMES["ordinary"]
    cond = _ELIGIBLE[1]  # take-profit OR stop-loss
    assert plan_arrays(frame, plan_exit(cond)) is not None

    def resolve() -> pl.Series:
        return realized_entries(
            frame,
            entry_col="_entry",
            exit_cond=cond,
            snapshot_cols=collect_entry_refs(cond),
            session_last_col=None,
        )

    if scan_backend == "python":
        original = _scan._latch
        _scan._latch = lambda leg, bar: original(leg, bar) * 1.01  # type: ignore[assignment]
        try:
            perturbed = resolve()
        finally:
            _scan._latch = original  # type: ignore[assignment]
    else:
        from mktlib_scan import _mktlib_scan

        class _PerturbedModule:
            scan_realized = staticmethod(_mktlib_scan._perturbed_scan_realized)

        original_loader = _anchor.native_module
        _anchor.native_module = lambda: _PerturbedModule  # type: ignore[assignment,return-value]
        try:
            perturbed = resolve()
        finally:
            _anchor.native_module = original_loader  # type: ignore[assignment]

    windowed = _realized_entries_windowed(
        frame,
        entry_col="_entry",
        exit_cond=cond,
        snapshot_cols=collect_entry_refs(cond),
        session_last_col=None,
    )
    assert not perturbed.equals(windowed), (
        f"a shifted level slipped past the gate on the {scan_backend} backend"
    )


def test_limit_wrapped_entry_ref_creates_its_snapshot() -> None:
    """Regression: this combination used to raise ColumnNotFoundError.

    `Limit` and `EntryRef` are both documented public API, but the walker that
    decides which snapshot columns to create had no `Limit` case — so wrapping
    an anchored exit in one produced no `_entry_*` column and blew up when the
    condition resolved. No test combined them; the equivalence gate above found
    it, because the fast path unwraps `Limit` and the fallback did not.
    """
    from dataclasses import dataclass

    from mktlib.backtest import Col, Crossover, EntryRef, Limit, Pct, ValueGTE, run

    from tests.backtest.test_golden_baseline import _daily_frame

    @dataclass(frozen=True, slots=True)
    class LimitAnchoredTarget:
        def entry(self) -> Crossover:
            return Crossover("fast", "slow")

        def exit(self) -> Limit:
            return Limit(ValueGTE(Col("close"), Pct(EntryRef("close"), 0.25)))

    result = run(_daily_frame(), LimitAnchoredTarget())
    assert result.trades.height > 0
    # The snapshot is created (it used to be absent, which is what crashed) and
    # is surfaced on `signals`, as it is for any EntryRef exit.
    assert "_entry_close" in result.signals.columns
    # ...and it holds for the life of each position, like every other anchor.
    blocks = result.signals.with_columns(
        ((pl.col("_position") == 1) & (pl.col("_position").shift(1) != 1))
        .cum_sum()
        .alias("_block")
    ).filter(pl.col("_position") == 1)
    per_block = blocks.group_by("_block").agg(
        pl.col("_entry_close").n_unique().alias("anchors")
    )
    assert per_block.select((pl.col("anchors") == 1).all()).item()
