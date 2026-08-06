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
    TradeSide,
    ValueGTE,
    run,
)
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
            flatten_eod=False,
            bracket=Bracket(take_profit=0.02),
        )


def test_limit_exit_with_bracket_raises(flat: pl.DataFrame) -> None:
    with pytest.raises(NotImplementedError, match="Limit"):
        run(flat, CrossEntryLimitExit(), bracket=Bracket(take_profit=0.02))


@pytest.mark.parametrize("dropped", ["high", "low"])
def test_missing_range_column_raises(flat: pl.DataFrame, dropped: str) -> None:
    with pytest.raises(ValueError, match=f"Bracket requires \\['{dropped}'\\]"):
        run(flat.drop(dropped), CrossStrategy(), bracket=Bracket(take_profit=0.02))
