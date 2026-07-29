"""`plan_exit` — which exit conditions can be resolved per candidate entry.

The fast resolver needs each `EntryRef` term to be a level **pinned at the entry
bar**: constant for a given candidate, whatever the bar under test does. That is
the only property it needs, and it is narrower than "mentions EntryRef" and
wider than "one EntryRef times a constant".

The distinction that matters, and which two adjacent lines of the public docs
happen to straddle:

    EntryRef("close") - EntryRef("atr") * 2   pinned   — a fixed stop
    EntryRef("close") - Col("atr") * 2        moving   — a trailing stop

Both are legitimate strategies and both must produce correct results; only the
first can be resolved by a monotone search. Misclassifying in the permissive
direction is a correctness bug, so every ambiguous shape must land as
ineligible — these tests assert the eligible set is exactly the enumerated
shapes.
"""

from __future__ import annotations

import polars as pl
import pytest

from mktlib.backtest import (
    Col,
    Condition,
    Crossunder,
    Custom,
    EntryRef,
    Limit,
    Pct,
    ValueGT,
    ValueGTE,
    ValueLT,
    ValueLTE,
)
from mktlib.backtest._anchor import plan_exit

TP = ValueGT("close", Pct(EntryRef("close"), 5.0))
SL = ValueLT("close", Pct(EntryRef("close"), -3.0))


# --- eligible ---------------------------------------------------------------


def test_single_anchored_leg() -> None:
    plan = plan_exit(TP)
    assert plan.eligible
    assert len(plan.legs) == 1
    assert plan.legs[0].direction == "above"
    assert plan.legs[0].strict is True
    assert plan.fixed is None


def test_take_profit_or_stop_loss() -> None:
    """The canonical idiom, and the entire measured blast radius."""
    plan = plan_exit(TP | SL)
    assert plan.eligible
    assert {leg.direction for leg in plan.legs} == {"above", "below"}


def test_arithmetic_over_two_snapshots_is_pinned() -> None:
    """`EntryRef('close') - EntryRef('atr') * 2` is a per-candidate constant.

    Every leaf is snapshotted at the entry bar, so the barrier does not move —
    which is all the search requires. Arbitrary arithmetic among snapshots is
    fine; it is a *plain column* anywhere in the threshold that disqualifies.
    """
    plan = plan_exit(ValueLT("close", EntryRef("close") - EntryRef("atr") * 2))
    assert plan.eligible
    assert len(plan.legs) == 1
    assert plan.legs[0].direction == "below"


def test_non_strict_comparisons_keep_their_strictness() -> None:
    for cond, strict in (
        (ValueGTE("close", EntryRef("close")), False),
        (ValueGT("close", EntryRef("close")), True),
        (ValueLTE("close", EntryRef("close")), False),
        (ValueLT("close", EntryRef("close")), True),
    ):
        plan = plan_exit(cond)
        assert plan.eligible
        assert plan.legs[0].strict is strict


def test_reversed_orientation_flips_direction() -> None:
    """`anchor > close` is the same crossing as `close < anchor`."""
    plan = plan_exit(ValueGT(EntryRef("close"), "close"))
    assert plan.eligible
    assert plan.legs[0].direction == "below"


def test_anchored_or_fixed_term() -> None:
    """A non-anchored term rides along as a fixed predicate."""
    plan = plan_exit(TP | Crossunder("fast", "slow"))
    assert plan.eligible
    assert len(plan.legs) == 1
    assert plan.fixed is not None


def test_limit_wrapper_is_unwrapped() -> None:
    plan = plan_exit(Limit(TP))
    assert plan.eligible
    assert len(plan.legs) == 1


# --- ineligible -------------------------------------------------------------


def test_moving_threshold_is_refused() -> None:
    """The docs' own ATR example — a trailing stop, not a pinned one."""
    plan = plan_exit(ValueLT("close", EntryRef("close") - Col("atr") * 2))
    assert not plan.eligible
    assert plan.reason is not None
    assert "moving barrier" in plan.reason


def test_and_with_an_anchored_term_is_refused() -> None:
    """Under AND the predicate stops being 'the running extreme reaches L'."""
    plan = plan_exit(TP & Crossunder("fast", "slow"))
    assert not plan.eligible
    assert plan.reason is not None
    assert "EntryRef" in plan.reason


def test_negated_anchored_term_is_refused() -> None:
    plan = plan_exit(~TP)
    assert not plan.eligible


def test_custom_reading_a_snapshot_is_refused() -> None:
    """`Custom` bypasses the typed AST, so the roots are inspected instead."""
    plan = plan_exit(Custom(pl.col("close") > pl.col("_entry_close")))
    assert not plan.eligible
    assert plan.reason is not None
    assert "Custom" in plan.reason


def test_custom_not_reading_a_snapshot_rides_along_as_fixed() -> None:
    plan = plan_exit(TP | Custom(pl.col("volume") > 1_000))
    assert plan.eligible
    assert plan.fixed is not None


def test_condition_without_any_entry_ref_is_not_planned() -> None:
    """Nothing to resolve — the ordinary engine path already handles it."""
    plan = plan_exit(Crossunder("fast", "slow"))
    assert not plan.eligible


# --- the eligible set is exactly what is enumerated --------------------------

_ELIGIBLE: list[Condition] = [
    TP,
    TP | SL,
    ValueLT("close", EntryRef("close") - EntryRef("atr") * 2),
    ValueGT(EntryRef("close"), "close"),
    TP | Crossunder("fast", "slow"),
    Limit(TP),
]

_INELIGIBLE: list[Condition] = [
    ValueLT("close", EntryRef("close") - Col("atr") * 2),
    TP & Crossunder("fast", "slow"),
    ~TP,
    Custom(pl.col("close") > pl.col("_entry_close")),
    Crossunder("fast", "slow"),
    ValueGT(EntryRef("close"), EntryRef("atr")),
]


@pytest.mark.parametrize("cond", _ELIGIBLE, ids=range(len(_ELIGIBLE)))
def test_enumerated_eligible(cond: Condition) -> None:
    assert plan_exit(cond).eligible


@pytest.mark.parametrize("cond", _INELIGIBLE, ids=range(len(_INELIGIBLE)))
def test_enumerated_ineligible(cond: Condition) -> None:
    plan = plan_exit(cond)
    assert not plan.eligible
    assert plan.reason is not None


def test_threshold_reads_the_raw_column_not_the_snapshot() -> None:
    """The threshold is evaluated AT the candidate bar, so it must read `close`.

    Reading `_entry_close` here would reintroduce the forward-fill this whole
    module exists to avoid.
    """
    leg = plan_exit(TP).legs[0]
    assert "close" in leg.threshold.meta.root_names()
    assert not any(
        name.startswith("_entry_") for name in leg.threshold.meta.root_names()
    )
