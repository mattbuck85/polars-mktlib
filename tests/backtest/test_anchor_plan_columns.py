"""`_plan_columns` alone decides whether the fast path applies.

`plan_arrays` is two jobs wearing one name: decide eligibility, then materialize.
Only the second is expensive, and a resolver that walks polars Series instead of
Python lists wants to skip it — but eligibility is also the predicate production
dispatch and all three test corpora partition on, so if a second materializer
could disagree about *which* trees it accepts, dispatch and the corpus would
drift apart. That is the exact failure `plan_arrays`' own docstring promises is
impossible.

Splitting the decision into `_plan_columns` makes that structural rather than
hopeful: every refusal lives there, conversion cannot reach a different verdict,
and these tests fail if that stops being true.

The corpus is imported from `test_anchor_plan` rather than restated, for the same
reason the other gates import it — two copies would drift.
"""

from __future__ import annotations

import polars as pl
import pytest

from mktlib.backtest import Condition
from mktlib.backtest._anchor import _plan_columns, plan_arrays, plan_exit
from tests.backtest.test_anchor_plan import _ELIGIBLE, _INELIGIBLE

_N = 120


def _frame(*, ints: bool = False) -> pl.DataFrame:
    state = 7
    close: list[float] = []
    px = 100.0
    for _ in range(_N):
        state = (1_664_525 * state + 1_013_904_223) % (2**32)
        px *= 1.0 + ((state / 2**32) - 0.5) * 0.04
        close.append(px)
    frame = pl.DataFrame({"close": close})
    frame = frame.with_columns(
        (pl.col("close") * 0.02).alias("atr"),
        pl.col("close").rolling_mean(3).fill_null(strategy="backward").alias("fast"),
        pl.col("close").rolling_mean(10).fill_null(strategy="backward").alias("slow"),
        (pl.int_range(pl.len()) % 19 == 0).alias("_session_last"),
    )
    if ints:
        # The >2**53 refusal path: a non-float value column must be rejected by
        # _plan_columns, not noticed later during conversion.
        frame = frame.with_columns(pl.col("close").cast(pl.Int64))
    return frame


CORPUS: list[Condition] = [*_ELIGIBLE, *_INELIGIBLE]


@pytest.mark.parametrize("index", range(len(CORPUS)), ids=range(len(CORPUS)))
@pytest.mark.parametrize("ints", [False, True], ids=["float", "int-close"])
def test_plan_columns_decides_eligibility_alone(index: int, ints: bool) -> None:
    """`plan_arrays` is None exactly when `_plan_columns` is.

    Parameterized over an int-typed `close` as well, because the dtype refusal is
    the one rejection that happens *after* the frame is evaluated — the place a
    second materializer is most likely to diverge.
    """
    frame = _frame(ints=ints)
    cond = CORPUS[index]
    plan = plan_exit(cond)

    columns = _plan_columns(frame, plan)
    arrays = plan_arrays(frame, plan)

    assert (columns is None) == (arrays is None), (
        f"condition {index} (ints={ints}): _plan_columns "
        f"{'refused' if columns is None else 'accepted'} but plan_arrays "
        f"{'refused' if arrays is None else 'accepted'}"
    )


def test_both_verdicts_occur_in_the_corpus() -> None:
    """Otherwise the agreement above is vacuous."""
    frame = _frame()
    verdicts = [_plan_columns(frame, plan_exit(c)) is None for c in CORPUS]
    assert any(verdicts), "nothing in the corpus is refused"
    assert not all(verdicts), "nothing in the corpus is accepted"


def test_the_dtype_refusal_is_reachable() -> None:
    """An int value column must be refused, not silently cast.

    Integers above 2**53 do not survive the cast to float, so resolving one would
    put a level quietly off by one.
    """
    plan = plan_exit(_ELIGIBLE[0])
    assert _plan_columns(_frame(), plan) is not None
    assert _plan_columns(_frame(ints=True), plan) is None


def test_legs_sharing_a_value_column_share_an_id_and_a_list() -> None:
    """The pair-loop optimization survives both representations.

    `_scan.py`'s specialized loop keys on `legs[0].value is legs[1].value` to
    halve its per-bar reads. `plan_arrays` preserves that by handing both legs one
    list object; `PlannedColumns.value_ids` states the same fact without relying
    on object identity, which is what a resolver reading Series needs.
    """
    frame = _frame()
    # take-profit OR stop-loss: both legs written against `close`
    plan = plan_exit(_ELIGIBLE[1])
    assert len(plan.legs) == 2

    columns = _plan_columns(frame, plan)
    assert columns is not None
    assert columns.value_ids[0] == columns.value_ids[1], "both legs read close"
    assert len(columns.values) == 1, "the shared column is materialized once"

    arrays = plan_arrays(frame, plan)
    assert arrays is not None
    legs, _ = arrays
    assert legs[0].value is legs[1].value, "identity feeds the pair loop"


def test_distinct_value_columns_get_distinct_ids() -> None:
    """The sharing must be real, not unconditional."""
    from mktlib.backtest import EntryRef, Pct, ValueGT, ValueLT

    cond = ValueGT("close", Pct(EntryRef("close"), 5.0)) | ValueLT(
        "fast", Pct(EntryRef("close"), -3.0)
    )
    columns = _plan_columns(_frame(), plan_exit(cond))
    assert columns is not None
    assert columns.value_ids[0] != columns.value_ids[1]
    assert len(columns.values) == 2

    arrays = plan_arrays(_frame(), plan_exit(cond))
    assert arrays is not None
    legs, _ = arrays
    assert legs[0].value is not legs[1].value
