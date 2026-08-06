"""Tests for protective bracket exits (:class:`mktlib.backtest.Bracket`).

The invariants under test, in order of importance:

1. ``bracket=None`` is an exact no-op, and a bracket whose levels are never
   reached leaves the backtest byte-identical (also pinned against frozen
   Parquet baselines in ``test_golden_baseline.py``).
2. The trigger/fill decision table matches a conventional event-driven
   OHLC broker exactly, on both sides, including gaps.
3. A bracket exit closes the position **on the bar that tagged the level**,
   suppresses the signal exit that would have followed, and zeroes every
   bar after it in that block.
4. Same-bar both-touch resolves per policy, and the default deliberately
   diverges from live submission order.
"""

from __future__ import annotations

import datetime
from collections.abc import Sequence
from dataclasses import FrozenInstanceError, dataclass
from typing import Any

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from mktlib.backtest import (
    Bracket,
    Col,
    Condition,
    Cost,
    Crossover,
    Crossunder,
    Limit,
    Lit,
    TradeSide,
    ValueGTE,
    run,
)
from mktlib.backtest._bracket import (
    BLOCK_COLUMN,
    RESIGNAL_COLUMN,
    SL_LEVEL_COLUMN,
    TAKE_PROFIT,
    TP_LEVEL_COLUMN,
    level_expr,
)
from mktlib.backtest._engine import _apply_bracket
from mktlib.scheduling import get_calendar

BPS = 1e-4

# The fixture is deliberately flat — every bar opens and closes at 100 — so
# that each test's expectations come only from the bars it perturbs.
_FLAT = 100.0
_N = 8
#: Crossover at bar 2 → position from bar 2 → entry fills at ``open[3]``.
_ENTRY_BAR = 3
#: Crossunder at bar 6 → signal exit at bar 6 → would fill at ``open[7]``.
_ENTRY_FILL = _FLAT


# ---------------------------------------------------------------------------
# Fixtures (module-local, matching tests/backtest conventions)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CrossStrategy:
    def entry(self) -> Crossover:
        return Crossover("fast", "slow")

    def exit(self) -> Crossunder:
        return Crossunder("fast", "slow")


@dataclass(frozen=True, slots=True)
class CrossEntryLimitExit:
    def entry(self) -> Crossover:
        return Crossover("fast", "slow")

    def exit(self) -> Condition:
        return Limit(ValueGTE(Col("high"), Col("close")))


def _frame(
    *,
    highs: dict[int, float] | None = None,
    lows: dict[int, float] | None = None,
    opens: dict[int, float] | None = None,
    extra: dict[str, list[float]] | None = None,
) -> pl.DataFrame:
    """Flat 8-bar frame with the named bars perturbed."""
    open_ = [_FLAT] * _N
    high = [_FLAT] * _N
    low = [_FLAT] * _N
    for idx, value in (opens or {}).items():
        open_[idx] = value
    for idx, value in (highs or {}).items():
        high[idx] = value
    for idx, value in (lows or {}).items():
        low[idx] = value
    data: dict[str, object] = {
        "date": pl.date_range(pl.date(2024, 1, 1), pl.date(2024, 1, 8), eager=True),
        "open": open_,
        "high": high,
        "low": low,
        "close": [_FLAT] * _N,
        "fast": [1.0, 1.0, 3.0, 4.0, 4.0, 4.0, 1.0, 1.0],
        "slow": [2.0] * _N,
    }
    data.update(extra or {})
    return pl.DataFrame(data)


@pytest.fixture
def flat() -> pl.DataFrame:
    return _frame()


def _only_trade(result: object) -> dict[str, Any]:
    trades = result.trades  # type: ignore[attr-defined]
    assert trades.height == 1, f"expected exactly one trade, got {trades.height}"
    return trades.to_dicts()[0]


def _bracket(leg: str, value: float | str) -> Bracket:
    """Single-leg bracket, built without splatting an untyped mapping."""
    if leg == "take_profit":
        return Bracket(take_profit=value)
    return Bracket(stop_loss=value)


# ---------------------------------------------------------------------------
# Construction / validation
# ---------------------------------------------------------------------------


def test_bracket_is_frozen() -> None:
    bracket = Bracket(take_profit=0.02)
    with pytest.raises(FrozenInstanceError):
        bracket.take_profit = 0.03  # type: ignore[misc]


def test_empty_bracket_rejected() -> None:
    with pytest.raises(ValueError, match="at least one of take_profit"):
        Bracket()


@pytest.mark.parametrize("leg", ["take_profit", "stop_loss"])
@pytest.mark.parametrize("value", [0.0, -0.01])
def test_non_positive_fraction_rejected(leg: str, value: float) -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        _bracket(leg, value)


@pytest.mark.parametrize("leg", ["take_profit", "stop_loss"])
@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_non_finite_fraction_rejected(leg: str, value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        _bracket(leg, value)


@pytest.mark.parametrize("value", [True, None.__class__, object()])
def test_non_numeric_non_column_rejected(value: object) -> None:
    with pytest.raises(TypeError, match="fraction, a column name or None"):
        Bracket(take_profit=value)  # type: ignore[arg-type]


def test_empty_column_name_rejected() -> None:
    with pytest.raises(ValueError, match="non-empty column name"):
        Bracket(stop_loss="")


def test_unknown_both_touch_policy_rejected() -> None:
    with pytest.raises(ValueError, match="both_touch must be one of"):
        Bracket(take_profit=0.02, both_touch="whichever")  # type: ignore[arg-type]


def test_unknown_anchor_policy_rejected() -> None:
    with pytest.raises(ValueError, match="anchor must be one of"):
        Bracket(take_profit=0.02, anchor="entry")  # type: ignore[arg-type]


def test_anchor_defaults_to_position() -> None:
    assert Bracket(take_profit=0.02).anchor == "position"


@pytest.mark.parametrize("anchor", ["position", "signal"])
def test_both_anchor_values_are_accepted(anchor: str) -> None:
    """The guard must not over-refuse: both documented policies construct.

    A membership check written from a rejection case is exactly the shape
    that bans a legitimate value, so pin the accepting half explicitly.
    """
    bracket = Bracket(take_profit=0.02, stop_loss=0.01, anchor=anchor)  # type: ignore[arg-type]
    assert bracket.anchor == anchor


def test_anchor_participates_in_equality_and_hash() -> None:
    """Two brackets differing only in *anchor* are distinct keys.

    ``Bracket`` is primitives-only precisely so a consumer can hash it into
    a cache key. *anchor* changes what the backtest computes with everything
    else held equal, so it must not collide with the default.
    """
    default = Bracket(take_profit=0.02, stop_loss=0.01)
    resignal = Bracket(take_profit=0.02, stop_loss=0.01, anchor="signal")

    assert default != resignal
    assert hash(default) != hash(resignal)
    assert len({default, resignal}) == 2


def test_positional_construction_is_unchanged_by_anchor() -> None:
    """*anchor* is last on the dataclass, so the three existing positions hold."""
    assert Bracket(0.02, 0.01, "take_profit_first") == Bracket(
        take_profit=0.02, stop_loss=0.01, both_touch="take_profit_first"
    )


def test_anchor_is_frozen() -> None:
    bracket = Bracket(take_profit=0.02)
    with pytest.raises(FrozenInstanceError):
        bracket.anchor = "signal"  # type: ignore[misc]


def test_level_columns_lists_only_string_specs() -> None:
    assert Bracket(take_profit="tp", stop_loss=0.01).level_columns == ("tp",)
    assert Bracket(take_profit=0.02, stop_loss=0.01).level_columns == ()


# ---------------------------------------------------------------------------
# The decision table — one test per row, plus the gap variants
# ---------------------------------------------------------------------------


def test_long_take_profit_triggers_on_high_and_fills_at_level() -> None:
    """long TP: ``high >= tp`` → ``max(open, tp)``. Level = 100 * 1.02 = 102."""
    result = run(_frame(highs={4: 103.0}), CrossStrategy(), bracket=Bracket(take_profit=0.02))

    trade = _only_trade(result)
    assert trade["pnl"] == pytest.approx(102.0 / _ENTRY_FILL - 1)
    assert trade["exit_date"] == datetime.date(2024, 1, 5)
    rets = result.returns["return"].to_list()
    assert rets[4] == pytest.approx((102.0 - _FLAT) / _FLAT)


def test_long_take_profit_gap_up_fills_at_the_open_not_the_level() -> None:
    """A favourable gap fills better than the resting limit: ``max(open, tp)``."""
    result = run(
        _frame(opens={4: 105.0}, highs={4: 105.0}),
        CrossStrategy(),
        bracket=Bracket(take_profit=0.02),
    )
    assert _only_trade(result)["pnl"] == pytest.approx(105.0 / _ENTRY_FILL - 1)


def test_long_stop_loss_triggers_on_low_and_fills_at_level() -> None:
    """long SL: ``low <= sl`` → ``min(open, sl)``. Level = 100 * 0.99 = 99."""
    result = run(_frame(lows={4: 98.0}), CrossStrategy(), bracket=Bracket(stop_loss=0.01))

    trade = _only_trade(result)
    assert trade["pnl"] == pytest.approx(99.0 / _ENTRY_FILL - 1)
    assert result.returns["return"].to_list()[4] == pytest.approx(-0.01)


def test_long_stop_loss_gap_down_fills_at_the_open_not_the_level() -> None:
    """An adverse gap fills worse than the stop: ``min(open, sl)`` = the open."""
    result = run(
        _frame(opens={4: 95.0}, lows={4: 94.0}),
        CrossStrategy(),
        bracket=Bracket(stop_loss=0.01),
    )
    assert _only_trade(result)["pnl"] == pytest.approx(95.0 / _ENTRY_FILL - 1)


def test_short_take_profit_triggers_on_low_and_fills_at_level() -> None:
    """short TP: ``low <= tp`` → ``min(open, tp)``. Level = 100 * 0.98 = 98."""
    result = run(
        _frame(lows={4: 97.0}),
        CrossStrategy(),
        trade_side=TradeSide.SHORT,
        bracket=Bracket(take_profit=0.02),
    )

    trade = _only_trade(result)
    assert trade["side"] == -1
    assert trade["pnl"] == pytest.approx(-1 * (98.0 / _ENTRY_FILL - 1))
    assert result.returns["return"].to_list()[4] == pytest.approx(0.02)


def test_short_stop_loss_triggers_on_high_and_fills_at_level() -> None:
    """short SL: ``high >= sl`` → ``max(open, sl)``. Level = 100 * 1.01 = 101."""
    result = run(
        _frame(highs={4: 102.0}),
        CrossStrategy(),
        trade_side=TradeSide.SHORT,
        bracket=Bracket(stop_loss=0.01),
    )

    trade = _only_trade(result)
    assert trade["pnl"] == pytest.approx(-1 * (101.0 / _ENTRY_FILL - 1))
    assert result.returns["return"].to_list()[4] == pytest.approx(-0.01)


def test_short_stop_loss_gap_up_fills_at_the_open() -> None:
    result = run(
        _frame(opens={4: 106.0}, highs={4: 107.0}),
        CrossStrategy(),
        trade_side=TradeSide.SHORT,
        bracket=Bracket(stop_loss=0.01),
    )
    assert _only_trade(result)["pnl"] == pytest.approx(-1 * (106.0 / _ENTRY_FILL - 1))


# ---------------------------------------------------------------------------
# The entry-fill bar is armed
# ---------------------------------------------------------------------------


def test_bracket_is_armed_on_the_entry_fill_bar() -> None:
    """A stop tagged on the bar the entry filled closes the trade immediately.

    The entry-bar return is measured against the *entry fill price*, not the
    previous close, so it is exactly the bracket loss.
    """
    result = run(
        _frame(lows={_ENTRY_BAR: 98.0}),
        CrossStrategy(),
        bracket=Bracket(stop_loss=0.01),
    )

    trade = _only_trade(result)
    assert trade["exit_date"] == datetime.date(2024, 1, 4)
    assert trade["pnl"] == pytest.approx(-0.01)
    rets = result.returns["return"].to_list()
    assert rets[_ENTRY_BAR] == pytest.approx(-0.01)
    assert rets[_ENTRY_BAR + 1 :] == [0.0] * (_N - _ENTRY_BAR - 1)


def test_gap_through_the_stop_on_the_entry_bar_fills_at_the_entry_open() -> None:
    """Levels are struck off the entry fill, so an entry-bar gap is a wick.

    ``open[3]`` is both the fill price and the level's reference, so the
    ``min(open, sl)`` clamp can only bind on a *later* bar. Here the low
    reaches far through the stop but the fill is still the level.
    """
    result = run(
        _frame(lows={_ENTRY_BAR: 80.0}),
        CrossStrategy(),
        bracket=Bracket(stop_loss=0.01),
    )
    assert _only_trade(result)["pnl"] == pytest.approx(-0.01)


# ---------------------------------------------------------------------------
# Same-bar both-touch
# ---------------------------------------------------------------------------


def _both_touch_frame() -> pl.DataFrame:
    """Bar 4 tags a 2% target (102) *and* a 1% stop (99)."""
    return _frame(highs={4: 103.0}, lows={4: 98.0})


def test_both_touch_stop_first_books_the_loss() -> None:
    result = run(
        _both_touch_frame(),
        CrossStrategy(),
        bracket=Bracket(take_profit=0.02, stop_loss=0.01, both_touch="stop_first"),
    )
    assert _only_trade(result)["pnl"] == pytest.approx(-0.01)


def test_both_touch_take_profit_first_books_the_gain() -> None:
    result = run(
        _both_touch_frame(),
        CrossStrategy(),
        bracket=Bracket(
            take_profit=0.02, stop_loss=0.01, both_touch="take_profit_first"
        ),
    )
    assert _only_trade(result)["pnl"] == pytest.approx(0.02)


def test_stop_first_is_the_default() -> None:
    assert Bracket(take_profit=0.02).both_touch == "stop_first"


def test_default_policy_diverges_from_submission_order_oco() -> None:
    """Pin the deliberate divergence so it cannot be forgotten.

    A live bracket is commonly an OCO pair whose take-profit leg is
    submitted first and filled in submission order — so the realized policy
    on a both-touch bar is ``take_profit_first``. mktlib defaults to ``stop_first`` because a
    backtest must not book the favourable resolution of an ambiguity that
    OHLC cannot resolve. The two must therefore disagree, and the live
    behaviour must remain reachable by opting in.
    """
    df = _both_touch_frame()

    default = _only_trade(
        run(df, CrossStrategy(), bracket=Bracket(take_profit=0.02, stop_loss=0.01))
    )
    live = _only_trade(
        run(
            df,
            CrossStrategy(),
            bracket=Bracket(
                take_profit=0.02, stop_loss=0.01, both_touch="take_profit_first"
            ),
        )
    )

    assert default["pnl"] < live["pnl"], "the conservative default must be the worse one"
    assert default["pnl"] == pytest.approx(-0.01)
    assert live["pnl"] == pytest.approx(0.02)


def test_both_touch_short_mirror() -> None:
    """Short: bar 4 tags TP (98) on the low and SL (101) on the high."""
    df = _frame(highs={4: 102.0}, lows={4: 97.0})

    stopped = _only_trade(
        run(
            df,
            CrossStrategy(),
            trade_side=TradeSide.SHORT,
            bracket=Bracket(take_profit=0.02, stop_loss=0.01),
        )
    )
    took = _only_trade(
        run(
            df,
            CrossStrategy(),
            trade_side=TradeSide.SHORT,
            bracket=Bracket(
                take_profit=0.02, stop_loss=0.01, both_touch="take_profit_first"
            ),
        )
    )
    assert stopped["pnl"] == pytest.approx(-0.01)
    assert took["pnl"] == pytest.approx(0.02)


# ---------------------------------------------------------------------------
# No-trigger passthrough and block truncation
# ---------------------------------------------------------------------------


def test_untriggered_bracket_is_a_passthrough(flat: pl.DataFrame) -> None:
    """Levels never reached → every artifact identical to the plain run."""
    plain = run(flat, CrossStrategy())
    bracketed = run(flat, CrossStrategy(), bracket=Bracket(take_profit=0.5, stop_loss=0.5))

    for artifact in ("returns", "trades", "signals"):
        assert_frame_equal(
            getattr(bracketed, artifact),
            getattr(plain, artifact),
            check_exact=True,
            check_dtypes=True,
            check_column_order=True,
        )


def test_bracket_does_not_change_the_signals_schema(flat: pl.DataFrame) -> None:
    """Every bracket working column is internal and dropped before return."""
    plain = run(flat, CrossStrategy())
    bracketed = run(_frame(highs={4: 103.0}), CrossStrategy(), bracket=Bracket(take_profit=0.02))

    assert bracketed.signals.columns == plain.signals.columns
    assert not [col for col in bracketed.signals.columns if col.startswith("_bracket")]
    assert bracketed.trades.columns == plain.trades.columns


def test_bracket_exit_suppresses_the_later_signal_exit() -> None:
    """One trade, closed by the bracket — not two, and not the signal price."""
    result = run(_frame(highs={4: 103.0}), CrossStrategy(), bracket=Bracket(take_profit=0.02))

    trade = _only_trade(result)
    assert trade["exit_date"] == datetime.date(2024, 1, 5)
    # The signal exit would have filled at open[7] == 100 for a pnl of 0.0.
    assert trade["pnl"] != pytest.approx(0.0)


def test_bars_after_a_bracket_exit_return_zero() -> None:
    """The stale ``_position`` must not keep accruing after the bracket fired."""
    # Bar 5 would have returned +3% had the position still been open.
    df = _frame(highs={4: 103.0}, opens={5: 103.0, 6: 103.0, 7: 103.0})
    result = run(df, CrossStrategy(), bracket=Bracket(take_profit=0.02))

    rets = result.returns["return"].to_list()
    assert rets[5:] == [0.0, 0.0, 0.0]


def test_only_the_first_trigger_in_a_block_closes_the_position() -> None:
    """A later leg tagging the same block is noise against a flat position."""
    df = _frame(highs={4: 103.0}, lows={5: 90.0})
    result = run(
        df, CrossStrategy(), bracket=Bracket(take_profit=0.02, stop_loss=0.01)
    )

    trade = _only_trade(result)
    assert trade["exit_date"] == datetime.date(2024, 1, 5)
    assert trade["pnl"] == pytest.approx(0.02)


def test_bracket_exit_does_not_re_enter_within_the_same_block() -> None:
    """Documented divergence from live: no re-entry until the block ends.

    ``fast`` stays above ``slow`` for the whole block, so live the entry
    condition would re-fire on the bar after the stop. Here it does not.
    """
    result = run(_frame(lows={4: 98.0}), CrossStrategy(), bracket=Bracket(stop_loss=0.01))

    assert result.trades.height == 1
    assert result.returns["return"].to_list()[5:] == [0.0, 0.0, 0.0]


# ---------------------------------------------------------------------------
# Column-driven absolute levels
# ---------------------------------------------------------------------------


def test_column_levels_are_latched_at_the_entry_signal_bar() -> None:
    """The level read is the entry *signal* bar's, one bar before the fill.

    Only bar 2 carries the real level (102); every other bar carries a
    decoy, so a read on the wrong bar produces a different exit price.
    """
    levels = [999.0, 999.0, 102.0, 50.0, 50.0, 50.0, 50.0, 50.0]
    df = _frame(highs={4: 103.0}, extra={"tp_col": levels})
    result = run(df, CrossStrategy(), bracket=Bracket(take_profit="tp_col"))

    assert _only_trade(result)["pnl"] == pytest.approx(102.0 / _ENTRY_FILL - 1)


def test_column_levels_mix_with_fractional_levels() -> None:
    df = _frame(lows={4: 98.0}, extra={"tp_col": [200.0] * _N})
    result = run(
        df, CrossStrategy(), bracket=Bracket(take_profit="tp_col", stop_loss=0.01)
    )
    assert _only_trade(result)["pnl"] == pytest.approx(-0.01)


def test_missing_level_column_raises() -> None:
    with pytest.raises(ValueError, match="not found in DataFrame columns"):
        run(_frame(), CrossStrategy(), bracket=Bracket(take_profit="nope"))


def test_null_level_on_an_entry_bar_raises_rather_than_carrying_a_stale_level() -> None:
    levels: list[float | None] = [102.0] * _N
    levels[2] = None
    df = _frame(highs={4: 103.0}).with_columns(pl.Series("tp_col", levels))
    with pytest.raises(ValueError, match="null on at least one entry"):
        run(df, CrossStrategy(), bracket=Bracket(take_profit="tp_col"))


# ---------------------------------------------------------------------------
# Interaction with Cost
# ---------------------------------------------------------------------------


def test_bracket_exit_pays_one_side_of_cost() -> None:
    result = run(
        _frame(highs={4: 103.0}),
        CrossStrategy(),
        bracket=Bracket(take_profit=0.02),
        cost=Cost(commission_bps=10.0),
    )

    rets = result.returns["return"].to_list()
    assert rets[_ENTRY_BAR] == pytest.approx(-10.0 * BPS)
    assert rets[4] == pytest.approx((102.0 - _FLAT) / _FLAT - 10.0 * BPS)
    assert _only_trade(result)["pnl"] == pytest.approx(0.02 - 20.0 * BPS)


def test_entry_bar_bracket_exit_pays_both_sides_in_returns_and_pnl() -> None:
    """Two fills land on one bar, so that bar is charged twice — as is pnl."""
    result = run(
        _frame(lows={_ENTRY_BAR: 98.0}),
        CrossStrategy(),
        bracket=Bracket(stop_loss=0.01),
        cost=Cost(commission_bps=10.0),
    )

    rets = result.returns["return"].to_list()
    assert rets[_ENTRY_BAR] == pytest.approx(-0.01 - 2 * 10.0 * BPS)
    assert _only_trade(result)["pnl"] == pytest.approx(-0.01 - 2 * 10.0 * BPS)


def test_zero_cost_with_a_bracket_is_a_no_op() -> None:
    df = _frame(highs={4: 103.0})
    bracket = Bracket(take_profit=0.02)
    assert_frame_equal(
        run(df, CrossStrategy(), bracket=bracket, cost=Cost()).returns,
        run(df, CrossStrategy(), bracket=bracket).returns,
        check_exact=True,
    )


# ---------------------------------------------------------------------------
# flatten_eod and multi-instrument
# ---------------------------------------------------------------------------


def _intraday_frame(
    *, highs: dict[int, float] | None = None
) -> pl.DataFrame:
    """Two XNYS sessions of four 90-minute bars (09:30 → 14:00)."""
    stamps: list[datetime.datetime] = []
    for day in (datetime.date(2024, 1, 2), datetime.date(2024, 1, 3)):
        ts = datetime.datetime(day.year, day.month, day.day, 9, 30)
        for _ in range(4):
            stamps.append(ts)
            ts += datetime.timedelta(minutes=90)
    n = len(stamps)
    high = [_FLAT] * n
    for idx, value in (highs or {}).items():
        high[idx] = value
    return pl.DataFrame({
        "date": stamps,
        "open": [_FLAT] * n,
        "high": high,
        "low": [_FLAT] * n,
        "close": [_FLAT] * n,
        # Crossover at bar 1 → entry fills at bar 2; no crossunder.
        "fast": [1.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
        "slow": [2.0] * n,
    })


def test_bracket_fires_inside_a_session_under_flatten_eod() -> None:
    result = run(
        _intraday_frame(highs={2: 103.0}),
        CrossStrategy(),
        calendar=get_calendar("XNYS"),
        flatten_eod=True,
        bracket=Bracket(take_profit=0.02),
    )
    assert _only_trade(result)["pnl"] == pytest.approx(0.02)


def test_session_flatten_wins_on_the_session_last_bar() -> None:
    """The engine flattens at the session-last bar's *open*.

    The position is therefore already closed before any intra-bar level on
    that bar could be tagged, so the bracket must not fire there.
    """
    # Bar 3 is the session-last bar; its high would tag a 2% target.
    result = run(
        _intraday_frame(highs={3: 103.0}),
        CrossStrategy(),
        calendar=get_calendar("XNYS"),
        flatten_eod=True,
        bracket=Bracket(take_profit=0.02),
    )
    assert _only_trade(result)["pnl"] == pytest.approx(0.0)


def test_bracket_applies_per_instrument() -> None:
    triggered = _frame(highs={4: 103.0}).with_columns(pl.lit("AAA").alias("symbol"))
    quiet = _frame().with_columns(pl.lit("BBB").alias("symbol"))
    result = run(
        pl.concat([triggered, quiet]),
        CrossStrategy(),
        instrument_col="symbol",
        bracket=Bracket(take_profit=0.02),
    )

    assert _only_trade(result["AAA"])["pnl"] == pytest.approx(0.02)
    assert _only_trade(result["BBB"])["pnl"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Unsupported combinations
# ---------------------------------------------------------------------------


def test_dual_strategy_with_bracket_raises(flat: pl.DataFrame) -> None:
    with pytest.raises(NotImplementedError, match="short_strategy"):
        run(
            flat,
            CrossStrategy(),
            short_strategy=CrossStrategy(),
            bracket=Bracket(take_profit=0.02),
        )


def test_multi_instrument_dual_strategy_with_bracket_raises(flat: pl.DataFrame) -> None:
    df = flat.with_columns(pl.lit("AAA").alias("symbol"))
    with pytest.raises(NotImplementedError, match="short_strategy"):
        run(
            df,
            CrossStrategy(),
            short_strategy=CrossStrategy(),
            instrument_col="symbol",
            bracket=Bracket(take_profit=0.02),
        )


def test_dual_guard_is_defended_at_the_dual_path_too(flat: pl.DataFrame) -> None:
    """The guard in ``_run_dual`` is load-bearing, not redundant.

    ``run()`` rejects the combination first, so this reaches past it to the
    private helper — the point being that a future refactor adding another
    route into the dual path must not silently acquire long-leg levels on a
    short-leg position.
    """
    from mktlib.backtest._engine import _run_dual

    with pytest.raises(NotImplementedError, match="short_strategy"):
        _run_dual(
            flat,
            CrossStrategy(),
            CrossStrategy(),
            calendar=None,
            flatten=None,
            bracket=Bracket(take_profit=0.02),
        )


def test_limit_exit_with_bracket_raises(flat: pl.DataFrame) -> None:
    with pytest.raises(NotImplementedError, match="Limit"):
        run(flat, CrossEntryLimitExit(), bracket=Bracket(take_profit=0.02))


@pytest.mark.parametrize("dropped", ["high", "low"])
def test_missing_range_column_raises(flat: pl.DataFrame, dropped: str) -> None:
    with pytest.raises(ValueError, match=f"Bracket requires \\['{dropped}'\\]"):
        run(flat.drop(dropped), CrossStrategy(), bracket=Bracket(take_profit=0.02))


# ---------------------------------------------------------------------------
# Bracket(anchor="signal") — the corner cases
#
# These fixtures drive entry and exit from plain 0/1 columns rather than a
# crossover pair. Two reasons, both structural:
#
# * A `Crossover` cannot fire on two adjacent bars by construction, so a
#   crossover-driven fixture cannot express "a re-signal on the entry fill
#   bar" or "consecutive re-signals" at all.
# * `Crossover` and `Crossunder` are exact complements, so a crossunder exit
#   closes the position on the very bar that would have made the next
#   crossover a mid-hold signal. Nothing here could ever re-anchor.
#
# Nothing in the bracket depends on how a signal was derived — it reads
# `_entry` and the position columns — so driving them directly is a faithful
# substitute and not a shortcut around anything.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SignalColumnStrategy:
    """Entry and exit each read a plain 0/1 column of the fixture."""

    def entry(self) -> Condition:
        return ValueGTE(Col("entry_sig"), Lit(0.5))

    def exit(self) -> Condition:
        return ValueGTE(Col("exit_sig"), Lit(0.5))


def _sig_frame(
    n: int = 8,
    *,
    entries: Sequence[int] = (),
    exits: Sequence[int] = (),
    opens: dict[int, float] | None = None,
    highs: dict[int, float] | None = None,
    lows: dict[int, float] | None = None,
    extra: dict[str, list[Any]] | None = None,
) -> pl.DataFrame:
    """Flat *n*-bar daily frame with entry/exit signals on the named bars."""
    open_ = [_FLAT] * n
    high = [_FLAT] * n
    low = [_FLAT] * n
    for spec, arr in ((opens, open_), (highs, high), (lows, low)):
        for idx, value in (spec or {}).items():
            arr[idx] = value
    data: dict[str, object] = {
        "date": pl.date_range(
            datetime.date(2024, 1, 1),
            datetime.date(2024, 1, 1) + datetime.timedelta(days=n - 1),
            eager=True,
        ),
        "open": open_,
        "high": high,
        "low": low,
        "close": [_FLAT] * n,
        "entry_sig": [1.0 if i in entries else 0.0 for i in range(n)],
        "exit_sig": [1.0 if i in exits else 0.0 for i in range(n)],
    }
    data.update(extra or {})
    return pl.DataFrame(data)


def _sig_intraday_frame(
    *,
    entries: Sequence[int] = (),
    exits: Sequence[int] = (),
    opens: dict[int, float] | None = None,
    highs: dict[int, float] | None = None,
    lows: dict[int, float] | None = None,
) -> pl.DataFrame:
    """Two XNYS sessions of four 90-minute bars, signal-column driven.

    Bars 0-3 are 2024-01-02 09:30/11:00/12:30/14:00, bars 4-7 the same times
    on 2024-01-03. Bars 3 and 7 are session-last.
    """
    stamps: list[datetime.datetime] = []
    for day in (datetime.date(2024, 1, 2), datetime.date(2024, 1, 3)):
        ts = datetime.datetime(day.year, day.month, day.day, 9, 30)
        for _ in range(4):
            stamps.append(ts)
            ts += datetime.timedelta(minutes=90)
    n = len(stamps)
    open_ = [_FLAT] * n
    high = [_FLAT] * n
    low = [_FLAT] * n
    for spec, arr in ((opens, open_), (highs, high), (lows, low)):
        for idx, value in (spec or {}).items():
            arr[idx] = value
    return pl.DataFrame(
        {
            "date": stamps,
            "open": open_,
            "high": high,
            "low": low,
            "close": [_FLAT] * n,
            "entry_sig": [1.0 if i in entries else 0.0 for i in range(n)],
            "exit_sig": [1.0 if i in exits else 0.0 for i in range(n)],
        }
    )


def _sig_run(df: pl.DataFrame, bracket: Bracket, **kw: Any) -> Any:
    return run(df, SignalColumnStrategy(), bracket=bracket, **kw)


# --- 1. a leg tagged on the re-signal bar closes the position first ---------


def test_a_leg_tagged_on_the_re_signal_bar_beats_the_re_anchor_float() -> None:
    """The old level governs bar ``k`` inclusive; the new one starts at ``k+1``.

    The re-signal is observed at bar 5's close, while the bracket is a resting
    order that filled *during* bar 5 — so the bracket wins. Bar 5 opens at 90,
    which is what makes the assertion discriminating: a re-latch that moved
    the level on bar 5 rather than after it would anchor to 90 and fill at
    91.8, not 102.
    """
    df = _sig_frame(
        8, entries=(2, 5), opens={5: 90.0}, lows={5: 90.0}, highs={5: 103.0}
    )
    trade = _only_trade(_sig_run(df, Bracket(take_profit=0.02, anchor="signal")))

    assert trade["exit_date"] == datetime.date(2024, 1, 6)
    assert trade["pnl"] == pytest.approx(102.0 / _ENTRY_FILL - 1)


def test_a_leg_tagged_on_the_re_signal_bar_beats_the_re_anchor_col() -> None:
    """The ``str`` mirror. Bar 5's own column value is 110, which the bar's
    high of 103 would not reach — so reading it a bar early would silently
    turn a bracket exit into no exit at all."""
    levels = [999.0] * 8
    levels[2] = 102.0
    levels[5] = 110.0
    df = _sig_frame(8, entries=(2, 5), highs={5: 103.0}, extra={"tp_col": levels})
    trade = _only_trade(_sig_run(df, Bracket(take_profit="tp_col", anchor="signal")))

    assert trade["exit_date"] == datetime.date(2024, 1, 6)
    assert trade["pnl"] == pytest.approx(102.0 / _ENTRY_FILL - 1)


# --- 2. session-last under flatten_eod --------------------------------------


def test_session_last_re_signal_opens_a_fresh_trade_rather_than_re_anchoring() -> None:
    """The deferred-entry fill bar, pinned against measured engine output.

    Under ``flatten_eod`` a session-last entry signal is deferred to the next
    session's *first* bar — the SIGNAL moves, not the fill. The recurrence
    then opens the position on that bar and the usual fill-at-next-open rule
    puts the entry fill on the bar AFTER it. So a 14:00 signal produces a
    position dated 09:30 whose P&L is measured from the 11:00 open.

    Every number below was read off the engine, not derived from the
    docstrings: CHANGELOG 0.7.0 and the older comment in ``_engine`` both
    describe the deferral of the signal and leave the reader to infer a fill
    one bar too early.

    The position that was live into 14:00 is flattened at that bar's open, so
    there is nothing left for the re-signal to re-anchor — which is why the
    two anchor policies agree here.
    """
    opens = {i: 100.0 + i for i in range(8)}
    df = _sig_intraday_frame(entries=(1, 3), opens=opens)
    kw: dict[str, Any] = {"calendar": get_calendar("XNYS"), "flatten_eod": True}
    result = _sig_run(df, Bracket(take_profit=0.5, anchor="signal"), **kw)

    signals = result.signals
    # The signal moved off 14:00 and onto the next session's 09:30 ...
    assert signals["_entry"].to_list()[3] is True
    assert signals["_position"].to_list()[3] == 0
    assert signals["_entry"].to_list()[4] is True
    assert signals["_position"].to_list()[4] == 1

    trades = result.trades.to_dicts()
    assert len(trades) == 2
    # ... the second trade is therefore dated 09:30 and priced from 11:00.
    assert trades[1]["entry_date"] == datetime.datetime(2024, 1, 3, 9, 30)
    assert trades[1]["exit_date"] == datetime.datetime(2024, 1, 3, 14, 0)
    assert trades[1]["pnl"] == pytest.approx(107.0 / 105.0 - 1)
    # The first one flattens at the session-last bar's own open.
    assert trades[0]["pnl"] == pytest.approx(103.0 / 102.0 - 1)

    held = _sig_run(df, Bracket(take_profit=0.5, anchor="position"), **kw)
    assert_frame_equal(result.trades, held.trades, check_exact=True)
    assert_frame_equal(result.returns, held.returns, check_exact=True)


def test_the_session_last_bar_is_kept_out_of_the_re_signal_mask() -> None:
    """The ``~_session_last`` guard, read straight off the column it builds.

    End-to-end the guard is unobservable on this fixture — the position is
    already flat on that bar, and ``level_expr``'s initial-latch-first
    ``coalesce`` independently resolves the one collision it could cause. So
    it is asserted where it is actually decided rather than through an output
    that cannot see it.
    """
    opens = {i: 100.0 + i for i in range(8)}
    df = _sig_intraday_frame(entries=(1, 3), opens=opens)
    engine = _sig_run(df, Bracket(take_profit=0.5, anchor="signal"),
                      calendar=get_calendar("XNYS"), flatten_eod=True)
    prepared = _pre_bracket_frame(
        engine.signals.with_columns(
            (pl.col("date").dt.time() == datetime.time(14, 0)).alias("_session_last")
        )
    )
    out = _apply_bracket(
        prepared,
        Bracket(take_profit=0.5, anchor="signal"),
        is_long=True,
        flatten_eod=True,
    )
    resignal = out[RESIGNAL_COLUMN].to_list()
    assert resignal[3] is False, "a session-last bar must not be a re-signal"
    assert resignal[7] is False


# --- 3. consecutive re-signals ----------------------------------------------


def test_the_latest_re_signal_wins() -> None:
    """Three latches in one block: the level in force is the newest one."""
    levels = [999.0] * 12
    levels[2] = 130.0
    levels[5] = 120.0
    levels[7] = 105.0
    df = _sig_frame(12, entries=(2, 5, 7), highs={9: 106.0}, extra={"tp_col": levels})
    trade = _only_trade(_sig_run(df, Bracket(take_profit="tp_col", anchor="signal")))

    assert trade["exit_date"] == datetime.date(2024, 1, 10)
    assert trade["pnl"] == pytest.approx(105.0 / _ENTRY_FILL - 1)


def test_an_intermediate_re_signal_is_in_force_until_the_next_one() -> None:
    """The same three latches, tagged between the second and the third."""
    levels = [999.0] * 12
    levels[2] = 130.0
    levels[5] = 120.0
    levels[7] = 105.0
    df = _sig_frame(12, entries=(2, 5, 7), highs={6: 121.0}, extra={"tp_col": levels})
    trade = _only_trade(_sig_run(df, Bracket(take_profit="tp_col", anchor="signal")))

    assert trade["exit_date"] == datetime.date(2024, 1, 7)
    assert trade["pnl"] == pytest.approx(120.0 / _ENTRY_FILL - 1)


# --- 4. a re-signal on the entry fill bar -----------------------------------


def test_a_re_signal_on_the_entry_fill_bar_takes_effect_one_bar_later() -> None:
    """Signal at bar 2 fills at bar 3; a second signal on bar 3 arms bar 4.

    Bar 4 opens at 110 and its high of 110 would tag the *original* 102 level
    — so the position surviving that bar is the assertion, and the fill at
    112.2 on bar 5 is where the re-anchored level shows up.
    """
    df = _sig_frame(8, entries=(2, 3), opens={4: 110.0}, highs={4: 110.0, 5: 113.0})
    moved = _only_trade(_sig_run(df, Bracket(take_profit=0.02, anchor="signal")))
    held = _only_trade(_sig_run(df, Bracket(take_profit=0.02, anchor="position")))

    assert moved["exit_date"] == datetime.date(2024, 1, 6)
    assert moved["pnl"] == pytest.approx(110.0 * 1.02 / _ENTRY_FILL - 1)
    assert held["exit_date"] == datetime.date(2024, 1, 5)
    assert held["pnl"] == pytest.approx(110.0 / _ENTRY_FILL - 1)


# --- 5. null level columns on a re-latch bar --------------------------------


def test_null_level_on_a_live_re_latch_bar_raises() -> None:
    """A null where a re-signal fires would hold the level it meant to move."""
    levels: list[float | None] = [999.0] * 8
    levels[2] = 150.0
    levels[5] = None
    df = _sig_frame(8, entries=(2, 5)).with_columns(pl.Series("tp_col", levels))
    with pytest.raises(ValueError, match="null on at least one re-anchoring"):
        _sig_run(df, Bracket(take_profit="tp_col", anchor="signal"))


def test_null_level_after_the_block_already_fired_is_not_an_error() -> None:
    """Past the first trigger the block is dead, so the null cannot be used.

    The guard is post-hoc — it runs after the levels are computed — so it has
    to be exact rather than conservative, or it would refuse runs whose stale
    level is never read by anything.
    """
    levels: list[float | None] = [999.0] * 8
    levels[2] = 102.0
    levels[5] = None
    df = _sig_frame(8, entries=(2, 5), highs={4: 103.0}).with_columns(
        pl.Series("tp_col", levels)
    )
    trade = _only_trade(_sig_run(df, Bracket(take_profit="tp_col", anchor="signal")))
    assert trade["exit_date"] == datetime.date(2024, 1, 5)
    assert trade["pnl"] == pytest.approx(0.02)


# --- 6. both-touch on a re-anchored pair ------------------------------------


@pytest.mark.parametrize(
    ("both_touch", "expected"),
    [("stop_first", 108.9), ("take_profit_first", 112.2)],
)
def test_both_touch_resolves_a_re_anchored_pair(
    both_touch: str, expected: float
) -> None:
    """Bar 5 tags both re-anchored legs; the policy still decides, unchanged.

    The re-anchor moves both legs to 112.2 / 108.9 off bar 5's open of 110.
    Under ``anchor="position"`` the same bar tags only the take-profit, so
    both policies would book 110 — which is what makes this a test of the
    interaction rather than of ``both_touch`` on its own.
    """
    df = _sig_frame(
        8, entries=(2, 4), opens={5: 110.0}, highs={5: 113.0}, lows={5: 108.0}
    )
    bracket = Bracket(
        take_profit=0.02, stop_loss=0.01, both_touch=both_touch, anchor="signal"  # type: ignore[arg-type]
    )
    assert _only_trade(_sig_run(df, bracket))["pnl"] == pytest.approx(
        expected / _ENTRY_FILL - 1
    )


def test_position_anchor_tags_only_one_leg_on_that_bar() -> None:
    """The control for the pair above: without the re-anchor there is no tie."""
    df = _sig_frame(
        8, entries=(2, 4), opens={5: 110.0}, highs={5: 113.0}, lows={5: 108.0}
    )
    for both_touch in ("stop_first", "take_profit_first"):
        bracket = Bracket(
            take_profit=0.02, stop_loss=0.01, both_touch=both_touch, anchor="position"  # type: ignore[arg-type]
        )
        assert _only_trade(_sig_run(df, bracket))["pnl"] == pytest.approx(
            110.0 / _ENTRY_FILL - 1
        )


# --- 7. a re-signal on the frame's last bar ---------------------------------


def test_a_re_signal_on_the_last_bar_of_the_frame_arms_nothing() -> None:
    """The new level takes effect one bar later, and there is no such bar.

    The paired case below is what makes this an assertion rather than an
    accident: move the identical re-signal one bar earlier and the level does
    land, and the bracket does fire.
    """
    levels = [999.0] * 8
    levels[2] = 150.0
    levels[7] = 101.0
    df = _sig_frame(8, entries=(2, 7), highs={7: 102.0}, extra={"tp_col": levels})
    result = _sig_run(df, Bracket(take_profit="tp_col", anchor="signal"))
    assert result.trades.height == 0


def test_the_same_re_signal_one_bar_earlier_does_arm() -> None:
    levels = [999.0] * 8
    levels[2] = 150.0
    levels[6] = 101.0
    df = _sig_frame(8, entries=(2, 6), highs={7: 102.0}, extra={"tp_col": levels})
    trade = _only_trade(_sig_run(df, Bracket(take_profit="tp_col", anchor="signal")))
    assert trade["pnl"] == pytest.approx(101.0 / _ENTRY_FILL - 1)


# --- 8. a re-signal in the post-fire zone -----------------------------------


def test_a_re_signal_after_the_bracket_fired_cannot_reopen_the_block() -> None:
    """The stale ``_position`` still marks bar 5 as held, but the block is dead.

    Bar 6's high would tag the level a re-anchor at bar 5 produces. The
    ``_count == 1`` gate discards it, exactly as it discards any later
    trigger — the re-anchor needs no special case for this.
    """
    df = _sig_frame(8, entries=(2, 5), highs={4: 103.0, 6: 103.0})
    result = _sig_run(df, Bracket(take_profit=0.02, anchor="signal"))

    trade = _only_trade(result)
    assert trade["exit_date"] == datetime.date(2024, 1, 5)
    assert trade["pnl"] == pytest.approx(0.02)
    assert result.returns["return"].to_list()[5:] == [0.0, 0.0, 0.0]


# --- 9. a null `_entry` ------------------------------------------------------


def test_the_re_signal_mask_is_a_non_nullable_boolean() -> None:
    """``_entry`` is null while a strategy's indicators warm up.

    ``null & (pos == 1)`` is null under Kleene logic, so the mask would be
    nullable without the ``fill_null(False)``. Today's three consumers all
    happen to be null-safe, so no output value can discriminate this — the
    column's own dtype and null count are where the guarantee lives, and a
    future consumer combining the mask with ``&``/``|`` outside a ``when`` is
    what it protects.
    """
    levels: list[float | None] = [None, None, 1.0, 0.0, None, 1.0, 0.0, 0.0]
    df = _sig_frame(8, entries=(2, 5)).with_columns(pl.Series("entry_sig", levels))
    prepared = _pre_bracket_frame(
        df.with_columns(
            (pl.col("entry_sig") >= 0.5).alias("_entry"),
            (pl.col("exit_sig") >= 0.5).alias("_exit"),
        )
    )
    assert prepared["_entry"].null_count() > 0, "fixture must produce a null _entry"

    out = _apply_bracket(
        prepared,
        Bracket(take_profit=0.02, anchor="signal"),
        is_long=True,
        flatten_eod=False,
    )
    assert out[RESIGNAL_COLUMN].dtype == pl.Boolean
    assert out[RESIGNAL_COLUMN].null_count() == 0


def test_a_null_entry_during_warm_up_does_not_change_the_result() -> None:
    """End-to-end: the nulls are inert, not merely tolerated."""
    nullable: list[float | None] = [None, None, 1.0, 0.0, None, 1.0, 0.0, 0.0]
    filled = [0.0 if v is None else v for v in nullable]
    highs = {4: 103.0}
    with_nulls = _sig_frame(8, highs=highs).with_columns(
        pl.Series("entry_sig", nullable)
    )
    without = _sig_frame(8, highs=highs).with_columns(pl.Series("entry_sig", filled))
    bracket = Bracket(take_profit=0.02, anchor="signal")

    assert_frame_equal(
        _sig_run(with_nulls, bracket).trades,
        _sig_run(without, bracket).trades,
        check_exact=True,
    )


# --- 10. flatten_eod=False across a day boundary ----------------------------


def test_without_flatten_eod_a_day_last_re_signal_anchors_to_the_next_open() -> None:
    """No session mask exists when ``flatten_eod`` is off, so the boundary is
    an ordinary bar boundary and the overnight gap carries the levels with it.

    Bar 4 opens the second day at 110, so the re-signal on bar 3 anchors
    there: 112.2, not the original 102. Bar 4's high of 110 would have tagged
    102, so surviving that bar is the observable difference.
    """
    df = _sig_intraday_frame(
        entries=(1, 3), opens={4: 110.0}, highs={4: 110.0, 5: 113.0}
    )
    moved = _only_trade(_sig_run(df, Bracket(take_profit=0.02, anchor="signal")))
    held = _only_trade(_sig_run(df, Bracket(take_profit=0.02, anchor="position")))

    assert moved["exit_date"] == datetime.datetime(2024, 1, 3, 11, 0)
    assert moved["pnl"] == pytest.approx(110.0 * 1.02 / _ENTRY_FILL - 1)
    assert held["exit_date"] == datetime.datetime(2024, 1, 3, 9, 30)
    assert held["pnl"] == pytest.approx(110.0 / _ENTRY_FILL - 1)


# ---------------------------------------------------------------------------
# Baseline-free invariants, asserted on the bracket columns directly
#
# `_bracket_tp` / `_bracket_sl` are dropped before the result is returned, so
# these call `_apply_bracket` on a hand-built frame. `_pre_bracket_frame`
# derives the position columns exactly as `_run_core` does, so no case here
# can describe a state the engine cannot produce.
# ---------------------------------------------------------------------------


def _pre_bracket_frame(df: pl.DataFrame) -> pl.DataFrame:
    """The columns ``_apply_bracket`` consumes, derived as ``_run_core`` does."""
    out = df.with_columns(
        pl.when(pl.col("_entry"))
        .then(pl.lit(1))
        .when(pl.col("_exit"))
        .then(pl.lit(0))
        .otherwise(pl.lit(None))
        .forward_fill()
        .fill_null(0)
        .alias("_position"),
    )
    out = out.with_columns(
        pl.col("_position").shift(1).fill_null(0).alias("_pos_d1"),
        pl.col("_position").shift(2).fill_null(0).alias("_pos_d2"),
    )
    return out.with_columns(
        ((pl.col("_position") == 1) & (pl.col("_pos_d1") == 0)).alias("_entry_clean"),
        ((pl.col("_position") == 0) & (pl.col("_pos_d1") == 1)).alias("_exit_clean"),
    )


def _levels_frame() -> pl.DataFrame:
    """Several position blocks, each carrying at least one mid-hold signal.

    Prices step up by one per bar. On the flat fixture used elsewhere in this
    module a re-anchored ``float`` leg would land on the same number it
    started at, and the control test below could not tell "did not move" from
    "moved to the same place".

    ``tp_col`` sits 50% above the bar and the tests pair it with a 50% stop,
    so no leg is ever tagged and the block structure comes from the signals
    alone.
    """
    n = 24
    entries = (2, 4, 7, 12, 14, 19)
    exits = (9, 17)
    prices = [100.0 + i for i in range(n)]
    df = pl.DataFrame(
        {
            "date": pl.date_range(
                datetime.date(2024, 1, 1),
                datetime.date(2024, 1, 1) + datetime.timedelta(days=n - 1),
                eager=True,
            ),
            "open": prices,
            "high": prices,
            "low": prices,
            "close": prices,
            "tp_col": [p * 1.5 for p in prices],
            "_entry": [i in entries for i in range(n)],
            "_exit": [i in exits for i in range(n)],
        }
    )
    return _pre_bracket_frame(df)


@pytest.mark.parametrize("spec", [0.5, "tp_col"])
def test_position_anchor_holds_one_level_per_block(spec: float | str) -> None:
    """One level per position, by construction — the bracket twin of the
    ``EntryRef`` anchor invariant, and baseline-free like it.

    The fixture deliberately carries mid-hold entry signals: under
    ``anchor="position"`` every one of them must be inert.
    """
    prepared = _levels_frame()
    out = _apply_bracket(
        prepared,
        Bracket(take_profit=spec, stop_loss=0.5, anchor="position"),
        is_long=True,
        flatten_eod=False,
    )
    live = out.filter(pl.col("_pos_d1") == 1)
    assert live[BLOCK_COLUMN].n_unique() > 1, "fixture must open several positions"
    per_block = live.group_by(BLOCK_COLUMN).agg(
        pl.col(TP_LEVEL_COLUMN).n_unique().alias("tp"),
        pl.col(SL_LEVEL_COLUMN).n_unique().alias("sl"),
    )
    assert per_block.select(
        ((pl.col("tp") == 1) & (pl.col("sl") == 1)).all()
    ).item(), f"a level moved mid-position: {per_block.sort(BLOCK_COLUMN).to_dicts()}"


@pytest.mark.parametrize("spec", [0.5, "tp_col"])
def test_signal_anchor_moves_a_level_within_a_block(spec: float | str) -> None:
    """The control. Without this the invariant above would also pass on a
    fixture where no signal ever fires mid-hold, and would pin nothing."""
    prepared = _levels_frame()
    out = _apply_bracket(
        prepared,
        Bracket(take_profit=spec, stop_loss=0.5, anchor="signal"),
        is_long=True,
        flatten_eod=False,
    )
    live = out.filter(pl.col("_pos_d1") == 1)
    per_block = live.group_by(BLOCK_COLUMN).agg(
        pl.col(TP_LEVEL_COLUMN).n_unique().alias("tp")
    )
    assert per_block.select((pl.col("tp") > 1).any()).item(), (
        "no level moved, so the invariant above is vacuous on this fixture"
    )


def test_position_anchor_emits_no_re_anchor_columns() -> None:
    """The default path builds neither new column, so its expression tree is
    untouched and default byte-identity is structural rather than asserted."""
    prepared = _levels_frame()
    kw: dict[str, Any] = {"is_long": True, "flatten_eod": False}
    held = _apply_bracket(prepared, Bracket(take_profit=0.02, anchor="position"), **kw)
    moved = _apply_bracket(prepared, Bracket(take_profit=0.02, anchor="signal"), **kw)

    assert RESIGNAL_COLUMN not in held.columns
    assert RESIGNAL_COLUMN in moved.columns
    assert set(moved.columns) - set(held.columns) == {
        RESIGNAL_COLUMN,
        "_bracket_anchor_fill",
    }


def test_the_opening_latch_wins_a_same_bar_collision_with_a_re_latch() -> None:
    """``coalesce`` order in ``level_expr``, pinned where it is decidable.

    The two latches can land on the same bar only under ``flatten_eod``, and
    there the caller's ``~_session_last`` mask empties the re-latch first — so
    no engine-level fixture can distinguish the two orderings. This drives the
    expression directly with both flags true on one bar, which is the state
    the ordering exists to resolve.
    """
    df = pl.DataFrame(
        {
            "spec": [10.0, 20.0, 30.0, 40.0],
            "_entry_clean": [False, False, True, False],
            "_resignal": [False, True, False, False],
        }
    )
    got = df.select(
        level_expr(
            "spec",
            leg=TAKE_PROFIT,
            is_long=True,
            entry_clean_col="_entry_clean",
            entry_fill_col="unused",
            resignal_col="_resignal",
        ).alias("level")
    )["level"].to_list()

    # Bar 2 is both the opening latch and the landing bar of bar 1's re-latch.
    assert got == [None, None, 30.0, 30.0]
