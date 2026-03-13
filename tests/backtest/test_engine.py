from __future__ import annotations

import datetime
import math
from dataclasses import dataclass

import polars as pl
import pytest

from mktlib.backtest._conditions import Col, Crossover, Crossunder, Lit, Pct, PriceIsAbove, PriceIsBelow
from mktlib.backtest._engine import run
from mktlib.backtest._types import TradeSide


@dataclass(frozen=True, slots=True)
class SimpleCrossStrategy:
    fast: str = "fast"
    slow: str = "slow"

    def entry(self) -> Crossover:
        return Crossover(self.fast, self.slow)

    def exit(self) -> Crossunder:
        return Crossunder(self.fast, self.slow)


@dataclass(frozen=True, slots=True)
class ReEntryStrategy:
    """Entry = Crossover (edge), Exit = PriceIsBelow (level, independent)."""

    exit_threshold: float = 102.0

    def entry(self) -> Crossover:
        return Crossover("fast", "slow")

    def exit(self) -> PriceIsBelow:
        return PriceIsBelow("close", self.exit_threshold)


@dataclass(frozen=True, slots=True)
class LevelEntryStrategy:
    """Entry = PriceIsAbove (fires every bar fast > slow), Exit = Crossunder."""

    def entry(self) -> PriceIsAbove:
        return PriceIsAbove("fast", "slow")

    def exit(self) -> Crossunder:
        return Crossunder("fast", "slow")


@pytest.fixture
def ohlcv() -> pl.DataFrame:
    """Synthetic data where fast crosses above slow at bar 2, below at bar 5."""
    return pl.DataFrame(
        {
            "date": pl.date_range(pl.date(2024, 1, 1), pl.date(2024, 1, 8), eager=True),
            "open": [100.0, 100.5, 101.5, 103.5, 105.5, 103.5, 99.5, 97.5],
            "close": [100.0, 101.0, 103.0, 105.0, 104.0, 100.0, 98.0, 97.0],
            # fast crosses above slow at index 2, crosses below at index 5
            "fast": [1.0, 1.0, 3.0, 4.0, 3.5, 1.0, 0.5, 0.3],
            "slow": [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
        }
    )


@pytest.fixture
def re_entry_df() -> pl.DataFrame:
    """12-bar data: Crossover at bars 2 & 7, PriceIsBelow(102) exit at 0,1,4,5,6,10,11."""
    return pl.DataFrame(
        {
            "date": pl.date_range(pl.date(2024, 1, 1), pl.date(2024, 1, 12), eager=True),
            "open": [100.0, 100.5, 102.0, 103.0, 104.0, 101.5, 99.0, 100.5, 103.5, 104.5, 102.0, 100.0],
            "close": [100.0, 101.0, 101.5, 105.0, 101.5, 100.0, 99.0, 100.0, 104.0, 104.0, 101.0, 100.0],
            # Crossover at bars 2 and 7 (fast goes from <=slow to >slow)
            "fast": [1.0, 1.0, 3.0, 4.0, 3.0, 1.0, 1.0, 3.0, 4.0, 4.0, 1.0, 1.0],
            "slow": [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
        }
    )


class TestRun:
    def test_returns_shape(self, ohlcv: pl.DataFrame) -> None:
        result = run(ohlcv, SimpleCrossStrategy())
        assert result.returns.columns == ["date", "return"]
        assert result.returns.height == ohlcv.height

    def test_signals_columns(self, ohlcv: pl.DataFrame) -> None:
        result = run(ohlcv, SimpleCrossStrategy())
        assert "_entry" in result.signals.columns
        assert "_exit" in result.signals.columns
        assert "_position" in result.signals.columns

    def test_position_tracking(self, ohlcv: pl.DataFrame) -> None:
        result = run(ohlcv, SimpleCrossStrategy())
        positions = result.signals["_position"].to_list()
        # Entry at bar 2 (fast crosses above slow), exit at bar 5
        assert positions == [0, 0, 1, 1, 1, 0, 0, 0]

    def test_returns_correctness(self, ohlcv: pl.DataFrame) -> None:
        result = run(ohlcv, SimpleCrossStrategy())
        rets = result.returns["return"].to_list()
        # Position is [0,0,1,1,1,0,0,0]
        # Delayed(shift=1): [0,0,0,1,1,1,0,0]
        # Delayed2(shift=2): [0,0,0,0,1,1,1,0]
        # Bar 3: entry bar (fill at open=103.5), ret = (105-103.5)/103.5
        # Bar 4: middle bar, ret = 104/105 - 1
        # Bar 5: middle bar, ret = 100/104 - 1
        # Bar 6: exit bar (fill at open=99.5), ret = (99.5-100)/100
        assert rets[0] == pytest.approx(0.0)
        assert rets[1] == pytest.approx(0.0)
        assert rets[2] == pytest.approx(0.0)
        assert rets[3] == pytest.approx((105 - 103.5) / 103.5, rel=1e-6)
        assert rets[4] == pytest.approx(104 / 105 - 1, rel=1e-6)
        assert rets[5] == pytest.approx(100 / 104 - 1, rel=1e-6)
        assert rets[6] == pytest.approx((99.5 - 100) / 100, rel=1e-6)
        assert rets[7] == pytest.approx(0.0)

    def test_trades_log(self, ohlcv: pl.DataFrame) -> None:
        result = run(ohlcv, SimpleCrossStrategy())
        assert result.trades.height == 1
        trade = result.trades.row(0, named=True)
        # Entry signal at bar 2 (Jan 3) → fill at bar 3's open = 103.5
        assert trade["entry_date"] == datetime.date(2024, 1, 3)
        # Exit signal at bar 5 (Jan 6) → fill at bar 6's open = 99.5
        assert trade["exit_date"] == datetime.date(2024, 1, 6)
        assert trade["pnl"] == pytest.approx(99.5 / 103.5 - 1, rel=1e-6)
        assert trade["bars_held"] == 3

    def test_no_trades(self) -> None:
        """When no crossover occurs, result should have zero trades."""
        df = pl.DataFrame(
            {
                "date": pl.date_range(pl.date(2024, 1, 1), pl.date(2024, 1, 3), eager=True),
                "open": [100.0, 100.5, 101.5],
                "close": [100.0, 101.0, 102.0],
                "fast": [1.0, 1.0, 1.0],
                "slow": [2.0, 2.0, 2.0],
            }
        )
        result = run(df, SimpleCrossStrategy())
        assert result.trades.height == 0
        assert all(r == 0.0 for r in result.returns["return"].to_list())


# ---------------------------------------------------------------------------
# Market hours tests
# ---------------------------------------------------------------------------

from mktlib.scheduling import get_calendar


@dataclass(frozen=True, slots=True)
class AlwaysInStrategy:
    """Enter on first bar, never exit — useful for testing position persistence."""

    def entry(self) -> Crossover:
        # fast > slow on every bar after first
        return Crossover("fast", "slow")

    def exit(self) -> Crossunder:
        # never triggers (fast stays above slow)
        return Crossunder("never_cross", "slow")


def _make_minute_df(
    date: datetime.date,
    hours: list[tuple[int, int]],
    *,
    base_price: float = 100.0,
) -> pl.DataFrame:
    """Build a minute-bar DataFrame for specific (hour, minute) timestamps."""
    timestamps = [
        datetime.datetime(date.year, date.month, date.day, h, m)
        for h, m in hours
    ]
    n = len(timestamps)
    prices = [base_price + i * 0.5 for i in range(n)]
    return pl.DataFrame(
        {
            "date": timestamps,
            "open": prices,
            "close": [p + 0.1 for p in prices],
            # fast crosses above slow at bar 1 (stays above)
            "fast": [1.0] + [3.0] * (n - 1),
            "slow": [2.0] * n,
            "never_cross": [5.0] * n,
        }
    )


def _make_two_session_df() -> pl.DataFrame:
    """Two trading sessions with an overnight gap.

    Day 1: 3 bars (09:30, 09:31, 15:59) — entry signal on bar 0->1
    Day 2: 3 bars (09:30, 09:31, 15:59) — gap open higher
    """
    day1 = datetime.date(2024, 1, 2)  # Tuesday
    day2 = datetime.date(2024, 1, 3)  # Wednesday
    timestamps = [
        datetime.datetime(2024, 1, 2, 9, 30),
        datetime.datetime(2024, 1, 2, 9, 31),
        datetime.datetime(2024, 1, 2, 15, 59),
        datetime.datetime(2024, 1, 3, 9, 30),
        datetime.datetime(2024, 1, 3, 9, 31),
        datetime.datetime(2024, 1, 3, 15, 59),
    ]
    return pl.DataFrame(
        {
            "date": timestamps,
            "open":  [100.0, 100.5, 101.0, 105.0, 105.5, 106.0],
            "close": [100.2, 100.8, 101.5, 105.3, 105.8, 106.5],
            # fast crosses above slow at bar 1
            "fast":  [1.0, 3.0, 3.0, 3.0, 3.0, 3.0],
            "slow":  [2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
            "never_cross": [5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
        }
    )


class TestMarketHours:
    def test_calendar_filters_non_market_bars(self) -> None:
        """With calendar, non-market bars are removed."""
        cal = get_calendar("XNYS")
        # Include bars outside market hours (08:00, 18:00)
        df = _make_minute_df(
            datetime.date(2024, 1, 2),
            [(8, 0), (9, 30), (10, 0), (18, 0)],
        )
        result = run(
            df,
            SimpleCrossStrategy(),
            calendar=cal,
        )
        result_times = result.signals["date"].to_list()
        # Only 09:30 and 10:00 should survive
        assert len(result_times) == 2
        assert all(
            datetime.time(9, 30) <= t.time() <= datetime.time(15, 59)
            for t in result_times
        )

    def test_no_calendar_unchanged(self, ohlcv: pl.DataFrame) -> None:
        """calendar=None produces identical results to current behavior."""
        result_no_cal = run(ohlcv, SimpleCrossStrategy())
        result_with_none = run(ohlcv, SimpleCrossStrategy(), calendar=None)
        assert result_no_cal.returns.equals(result_with_none.returns)
        assert result_no_cal.trades.equals(result_with_none.trades)

    def test_timezone_alignment(self) -> None:
        """Naive datetime column + tz-aware calendar works without crash."""
        cal = get_calendar("XNYS")
        # Naive timestamps (no tz) — calendar produces tz-aware index
        df = _make_minute_df(
            datetime.date(2024, 1, 2),
            [(9, 30), (10, 0), (11, 0)],
        )
        assert df["date"].dtype.time_zone is None
        # Should not raise
        result = run(df, SimpleCrossStrategy(), calendar=cal)
        assert result.returns.height > 0

    def test_flatten_eod_closes_positions_at_session_end(self) -> None:
        """flatten_eod=True forces position to 0 at session end."""
        cal = get_calendar("XNYS")
        df = _make_two_session_df()
        result = run(
            df,
            AlwaysInStrategy(),
            calendar=cal,
            flatten_eod=True,
        )
        positions = result.signals["_position"].to_list()
        # Entry signal at bar 0→1 (fast crosses above slow at bar 1)
        # With flatten_eod, position should go to 0 at 15:59 (session last)
        # Day 1: bar0=0, bar1=1, bar2(15:59)=0 (session end forces close)
        # Day 2: no fresh crossover (fast stays above slow), all positions 0
        assert positions == [0, 1, 0, 0, 0, 0]

        rets = result.returns["return"].to_list()
        # Bar 2 (session-last): entry fills at open[2]=101.0, forced exit at open[2]=101.0
        # Same-bar entry+exit → return = 0
        assert rets[2] == pytest.approx(0.0, abs=1e-12), (
            "session-last entry+exit on same bar should have 0 return"
        )
        # Day 2 bars: no position, all returns 0
        assert rets[3] == pytest.approx(0.0), "no overnight gap return with flatten_eod"
        assert rets[4] == pytest.approx(0.0)
        assert rets[5] == pytest.approx(0.0)

    def test_flatten_eod_false_captures_overnight_gap(self) -> None:
        """Without flatten_eod, position persists and overnight gap return is captured."""
        cal = get_calendar("XNYS")
        df = _make_two_session_df()
        result = run(
            df,
            AlwaysInStrategy(),
            calendar=cal,
            flatten_eod=False,
        )
        positions = result.signals["_position"].to_list()
        # Position should persist through session end
        assert positions[2] == 1, "position persists at Day 1 session end"

        # Day 2 first bar captures overnight gap
        rets = result.returns["return"].to_list()
        # Bar 3: middle bar, ret = close[3]/close[2] - 1 = 105.3/101.5 - 1
        assert rets[3] == pytest.approx(105.3 / 101.5 - 1, rel=1e-6)

    def test_flatten_eod_requires_calendar(self) -> None:
        """flatten_eod=True without calendar raises ValueError."""
        df = _make_minute_df(datetime.date(2024, 1, 2), [(9, 30), (10, 0)])
        with pytest.raises(ValueError, match="flatten_eod.*requires.*calendar"):
            run(df, SimpleCrossStrategy(), flatten_eod=True)


class TestShortSide:
    def test_short_returns_negated(self, ohlcv: pl.DataFrame) -> None:
        """Short returns are the exact negation of long returns."""
        strategy = SimpleCrossStrategy()
        long_result = run(ohlcv, strategy, trade_side=TradeSide.LONG)
        short_result = run(ohlcv, strategy, trade_side=TradeSide.SHORT)
        long_rets = long_result.returns["return"].to_list()
        short_rets = short_result.returns["return"].to_list()
        for i, (lr, sr) in enumerate(zip(long_rets, short_rets)):
            assert sr == pytest.approx(-lr, abs=1e-12), f"bar {i}: short={sr} != -long={-lr}"

    def test_short_trade_pnl_negated(self, ohlcv: pl.DataFrame) -> None:
        """Short trade PnL is the exact negation of long trade PnL."""
        strategy = SimpleCrossStrategy()
        long_result = run(ohlcv, strategy, trade_side=TradeSide.LONG)
        short_result = run(ohlcv, strategy, trade_side=TradeSide.SHORT)
        assert long_result.trades.height == short_result.trades.height
        long_pnl = long_result.trades["pnl"].to_list()
        short_pnl = short_result.trades["pnl"].to_list()
        for i, (lp, sp) in enumerate(zip(long_pnl, short_pnl)):
            assert sp == pytest.approx(-lp, abs=1e-12), f"trade {i}: short={sp} != -long={-lp}"

    def test_default_is_long(self, ohlcv: pl.DataFrame) -> None:
        """Omitting trade_side gives identical results to explicit TradeSide.LONG."""
        strategy = SimpleCrossStrategy()
        default_result = run(ohlcv, strategy)
        long_result = run(ohlcv, strategy, trade_side=TradeSide.LONG)
        assert default_result.returns.equals(long_result.returns)
        assert default_result.trades.equals(long_result.trades)

    def test_condition_trade_side_overrides_run_default(self, ohlcv: pl.DataFrame) -> None:
        """Entry condition's trade_side overrides the run() default."""

        @dataclass(frozen=True, slots=True)
        class ShortViaConditionStrategy:
            """Same signals as SimpleCrossStrategy, but entry carries SHORT."""
            fast: str = "fast"
            slow: str = "slow"

            def entry(self) -> Crossover:
                return Crossover(self.fast, self.slow, trade_side=TradeSide.SHORT)

            def exit(self) -> Crossunder:
                return Crossunder(self.fast, self.slow)

        # run() default is LONG, but entry condition says SHORT
        result = run(ohlcv, ShortViaConditionStrategy(), trade_side=TradeSide.LONG)
        # Compare with explicit SHORT at run level using the base strategy
        explicit_short = run(ohlcv, SimpleCrossStrategy(), trade_side=TradeSide.SHORT)
        # Returns should match (condition override wins)
        for i, (a, b) in enumerate(
            zip(
                result.returns["return"].to_list(),
                explicit_short.returns["return"].to_list(),
            )
        ):
            assert a == pytest.approx(b, abs=1e-12), f"bar {i}: condition-level={a} != run-level={b}"

    def test_condition_trade_side_none_falls_back_to_run(self, ohlcv: pl.DataFrame) -> None:
        """When entry condition has trade_side=None, run() default is used."""
        # SimpleCrossStrategy doesn't set trade_side on conditions
        long_result = run(ohlcv, SimpleCrossStrategy(), trade_side=TradeSide.LONG)
        short_result = run(ohlcv, SimpleCrossStrategy(), trade_side=TradeSide.SHORT)
        # They should differ (proving the run() default is used)
        long_rets = long_result.returns["return"].to_list()
        short_rets = short_result.returns["return"].to_list()
        for i, (lr, sr) in enumerate(zip(long_rets, short_rets)):
            assert sr == pytest.approx(-lr, abs=1e-12), f"bar {i}"


# ---------------------------------------------------------------------------
# Re-entry tests — two complete trade cycles
# ---------------------------------------------------------------------------


class TestReEntry:
    """Two complete trade cycles using independent entry/exit signals."""

    def test_position_two_cycles(self, re_entry_df: pl.DataFrame) -> None:
        result = run(re_entry_df, ReEntryStrategy())
        positions = result.signals["_position"].to_list()
        assert positions == [0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0]

    def test_two_trades_logged(self, re_entry_df: pl.DataFrame) -> None:
        result = run(re_entry_df, ReEntryStrategy())
        assert result.trades.height == 2

    def test_trade1_dates_and_pnl(self, re_entry_df: pl.DataFrame) -> None:
        result = run(re_entry_df, ReEntryStrategy())
        trade = result.trades.row(0, named=True)
        assert trade["entry_date"] == datetime.date(2024, 1, 3)
        assert trade["exit_date"] == datetime.date(2024, 1, 5)
        assert trade["pnl"] == pytest.approx(101.5 / 103.0 - 1, rel=1e-6)
        assert trade["bars_held"] == 2

    def test_trade2_dates_and_pnl(self, re_entry_df: pl.DataFrame) -> None:
        result = run(re_entry_df, ReEntryStrategy())
        trade = result.trades.row(1, named=True)
        assert trade["entry_date"] == datetime.date(2024, 1, 8)
        assert trade["exit_date"] == datetime.date(2024, 1, 11)
        assert trade["pnl"] == pytest.approx(100.0 / 103.5 - 1, rel=1e-6)
        assert trade["bars_held"] == 3

    def test_returns_per_bar(self, re_entry_df: pl.DataFrame) -> None:
        result = run(re_entry_df, ReEntryStrategy())
        rets = result.returns["return"].to_list()
        expected = [
            0.0,                             # bar 0: flat
            0.0,                             # bar 1: flat
            0.0,                             # bar 2: entry signal, no fill yet
            (105.0 - 103.0) / 103.0,         # bar 3: entry fill
            101.5 / 105.0 - 1,               # bar 4: middle
            (101.5 - 101.5) / 101.5,         # bar 5: exit fill
            0.0,                             # bar 6: flat
            0.0,                             # bar 7: entry signal, no fill yet
            (104.0 - 103.5) / 103.5,         # bar 8: entry fill
            104.0 / 104.0 - 1,               # bar 9: middle
            101.0 / 104.0 - 1,               # bar 10: middle
            (100.0 - 101.0) / 101.0,         # bar 11: exit fill
        ]
        for i, (actual, exp) in enumerate(zip(rets, expected)):
            assert actual == pytest.approx(exp, abs=1e-12), f"bar {i}"

    def test_compounded_returns_match_pnl(self, re_entry_df: pl.DataFrame) -> None:
        result = run(re_entry_df, ReEntryStrategy())
        rets = result.returns["return"].to_list()
        trades = result.trades
        # Trade 1: return bars 3-5
        compounded_1 = math.prod(1 + rets[i] for i in range(3, 6))
        pnl_1 = trades.row(0, named=True)["pnl"]
        assert compounded_1 == pytest.approx(1 + pnl_1, rel=1e-6)
        # Trade 2: return bars 8-11
        compounded_2 = math.prod(1 + rets[i] for i in range(8, 12))
        pnl_2 = trades.row(1, named=True)["pnl"]
        assert compounded_2 == pytest.approx(1 + pnl_2, rel=1e-6)


# ---------------------------------------------------------------------------
# Overlapping signal tests
# ---------------------------------------------------------------------------


class TestOverlappingSignals:
    """Test behavior when entry/exit signals overlap."""

    def test_redundant_entry_single_trade(self, ohlcv: pl.DataFrame) -> None:
        """PriceIsAbove fires on consecutive bars, but only 1 trade is opened."""
        result = run(ohlcv, LevelEntryStrategy())
        assert result.trades.height == 1

    def test_redundant_exit_ignored(self, ohlcv: pl.DataFrame) -> None:
        """Exit fires while flat — position stays 0, no phantom trades."""
        result = run(ohlcv, ReEntryStrategy())
        positions = result.signals["_position"].to_list()
        # After exit at bar 5, bars 6-7 have exit=True but position stays 0
        assert positions[5:] == [0, 0, 0]
        assert result.trades.height == 1

    def test_entry_wins_on_same_bar(self, re_entry_df: pl.DataFrame) -> None:
        """When both entry and exit fire on the same bar, entry wins."""
        result = run(re_entry_df, ReEntryStrategy(exit_threshold=103.0))
        positions = result.signals["_position"].to_list()
        # Bar 2: Crossover fires AND close=101.5 < 103 → entry wins
        assert positions[2] == 1


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases: boundary bars, empty data, unpaired signals."""

    def test_entry_last_bar_no_trade(self) -> None:
        """Crossover at the final bar → 0 trades (no next-bar fill)."""
        df = pl.DataFrame(
            {
                "date": pl.date_range(pl.date(2024, 1, 1), pl.date(2024, 1, 3), eager=True),
                "open": [100.0, 100.5, 101.5],
                "close": [100.0, 101.0, 102.0],
                "fast": [1.0, 1.0, 3.0],
                "slow": [2.0, 2.0, 2.0],
            }
        )
        result = run(df, SimpleCrossStrategy())
        assert result.trades.height == 0

    def test_exit_last_bar_null_pnl(self) -> None:
        """Exit at the final bar → trade logged with null pnl."""
        df = pl.DataFrame(
            {
                "date": pl.date_range(pl.date(2024, 1, 1), pl.date(2024, 1, 4), eager=True),
                "open": [100.0, 100.5, 101.5, 102.0],
                "close": [100.0, 101.0, 102.0, 103.0],
                "fast": [1.0, 3.0, 3.0, 1.0],
                "slow": [2.0, 2.0, 2.0, 2.0],
            }
        )
        result = run(df, SimpleCrossStrategy())
        assert result.trades.height == 1
        trade = result.trades.row(0, named=True)
        assert trade["pnl"] is None

    def test_empty_dataframe(self) -> None:
        """0-row DataFrame → no crash, 0 returns, 0 trades."""
        df = pl.DataFrame(
            schema={
                "date": pl.Date,
                "open": pl.Float64,
                "close": pl.Float64,
                "fast": pl.Float64,
                "slow": pl.Float64,
            }
        )
        result = run(df, SimpleCrossStrategy())
        assert result.returns.height == 0
        assert result.trades.height == 0

    def test_single_bar(self) -> None:
        """1-row DataFrame → 0 trades, return = [0.0]."""
        df = pl.DataFrame(
            {
                "date": [datetime.date(2024, 1, 1)],
                "open": [100.0],
                "close": [100.0],
                "fast": [1.0],
                "slow": [2.0],
            }
        )
        result = run(df, SimpleCrossStrategy())
        assert result.trades.height == 0
        assert result.returns["return"].to_list() == [0.0]

    def test_unpaired_entry_no_trade(self, ohlcv: pl.DataFrame) -> None:
        """Entry without exit → 0 trades."""
        df = ohlcv.with_columns(pl.lit(5.0).alias("never_cross"))
        result = run(df, AlwaysInStrategy())
        assert result.trades.height == 0


# ---------------------------------------------------------------------------
# PriceExpr unit tests
# ---------------------------------------------------------------------------


class TestInitHook:
    """Tests for the optional strategy.init() hook."""

    def test_init_adds_columns(self) -> None:
        """Strategy with init() that adds a rolling mean; entry/exit reference it."""
        df = pl.DataFrame(
            {
                "date": pl.date_range(pl.date(2024, 1, 1), pl.date(2024, 1, 8), eager=True),
                "open": [100.0, 100.5, 101.5, 103.5, 105.5, 103.5, 99.5, 97.5],
                "close": [100.0, 101.0, 103.0, 105.0, 104.0, 100.0, 98.0, 97.0],
            }
        )

        @dataclass(frozen=True, slots=True)
        class InitStrategy:
            def init(self, df: pl.DataFrame) -> pl.DataFrame:
                return df.with_columns(
                    pl.col("close").rolling_mean(2).alias("fast"),
                    pl.lit(101.5).alias("slow"),
                )

            def entry(self) -> Crossover:
                return Crossover("fast", "slow")

            def exit(self) -> Crossunder:
                return Crossunder("fast", "slow")

        result = run(df, InitStrategy())
        # fast (rolling mean of 2): [null, 100.5, 102.0, 104.0, 104.5, 102.0, 99.0, 97.5]
        # slow = 101.5 everywhere
        # Crossover at bar 2 (fast goes from 100.5 to 102.0, crossing above 101.5)
        assert "fast" in result.signals.columns
        assert "slow" in result.signals.columns
        assert result.trades.height >= 1

    def test_init_not_required(self) -> None:
        """Existing strategies without init() still work."""
        assert not hasattr(SimpleCrossStrategy(), "init")
        df = pl.DataFrame(
            {
                "date": pl.date_range(pl.date(2024, 1, 1), pl.date(2024, 1, 3), eager=True),
                "open": [100.0, 100.5, 101.5],
                "close": [100.0, 101.0, 102.0],
                "fast": [1.0, 1.0, 1.0],
                "slow": [2.0, 2.0, 2.0],
            }
        )
        result = run(df, SimpleCrossStrategy())
        assert result.returns.height == 3

    def test_init_sees_calendar_filtered_data(self) -> None:
        """When calendar is provided, init() receives already-filtered df."""
        cal = get_calendar("XNYS")
        received_heights: list[int] = []

        @dataclass(frozen=True, slots=True)
        class SpyInitStrategy:
            def init(self, df: pl.DataFrame) -> pl.DataFrame:
                received_heights.append(df.height)
                return df.with_columns(
                    pl.lit(1.0).alias("fast"),
                    pl.lit(2.0).alias("slow"),
                )

            def entry(self) -> Crossover:
                return Crossover("fast", "slow")

            def exit(self) -> Crossunder:
                return Crossunder("fast", "slow")

        # Include bars outside market hours
        df = _make_minute_df(
            datetime.date(2024, 1, 2),
            [(8, 0), (9, 30), (10, 0), (18, 0)],
        )
        original_height = df.height  # 4 bars
        run(df, SpyInitStrategy(), calendar=cal)
        # init should see only 2 market-hours bars, not 4
        assert len(received_heights) == 1
        assert received_heights[0] < original_height
        assert received_heights[0] == 2

    def test_init_with_bare_expr(self) -> None:
        """Combine init() with bare pl.Expr return from entry/exit."""
        df = pl.DataFrame(
            {
                "date": pl.date_range(pl.date(2024, 1, 1), pl.date(2024, 1, 6), eager=True),
                "open": [100.0, 100.5, 101.5, 103.5, 105.5, 103.5],
                "close": [100.0, 101.0, 103.0, 105.0, 104.0, 100.0],
            }
        )

        @dataclass(frozen=True, slots=True)
        class BareExprInitStrategy:
            def init(self, df: pl.DataFrame) -> pl.DataFrame:
                return df.with_columns(
                    pl.col("close").rolling_mean(2).alias("sma2"),
                )

            def entry(self) -> pl.Expr:
                return pl.col("close") > pl.col("sma2") + 1.0

            def exit(self) -> pl.Expr:
                return pl.col("close") < pl.col("sma2")

        result = run(df, BareExprInitStrategy())
        assert "sma2" in result.signals.columns
        assert result.returns.height == 6


class TestPriceExpr:
    def test_col_resolve(self) -> None:
        expr = Col("close").resolve()
        expected = pl.col("close")
        df = pl.DataFrame({"close": [1.0, 2.0]})
        assert df.select(expr)["close"].to_list() == df.select(expected)["close"].to_list()

    def test_lit_resolve(self) -> None:
        expr = Lit(5.0).resolve()
        df = pl.DataFrame({"x": [1.0, 2.0]})
        result = df.with_columns(expr.alias("v"))["v"].to_list()
        assert result == [5.0, 5.0]

    def test_mul(self) -> None:
        expr = (Col("close") * 0.95).resolve()
        df = pl.DataFrame({"close": [100.0, 200.0]})
        result = df.select(expr.alias("v"))["v"].to_list()
        assert result == pytest.approx([95.0, 190.0])

    def test_add_with_precedence(self) -> None:
        # Col("close") + Col("vol") * 2 should be close + (vol * 2)
        expr = (Col("close") + Col("vol") * 2).resolve()
        df = pl.DataFrame({"close": [100.0], "vol": [5.0]})
        result = df.select(expr.alias("v"))["v"].to_list()
        assert result == pytest.approx([110.0])

    def test_reverse_mul(self) -> None:
        expr = (2.0 * Col("vol")).resolve()
        df = pl.DataFrame({"vol": [5.0, 10.0]})
        result = df.select(expr.alias("v"))["v"].to_list()
        assert result == pytest.approx([10.0, 20.0])

    def test_modulo(self) -> None:
        expr = (Col("a") % Col("b")).resolve()
        df = pl.DataFrame({"a": [7.0, 10.0], "b": [3.0, 4.0]})
        result = df.select(expr.alias("v"))["v"].to_list()
        assert result == pytest.approx([1.0, 2.0])

    def test_negation(self) -> None:
        expr = (-Col("close")).resolve()
        df = pl.DataFrame({"close": [100.0, -50.0]})
        result = df.select(expr.alias("v"))["v"].to_list()
        assert result == pytest.approx([-100.0, 50.0])

    def test_sub(self) -> None:
        expr = (Col("close") - Col("vol")).resolve()
        df = pl.DataFrame({"close": [100.0], "vol": [5.0]})
        result = df.select(expr.alias("v"))["v"].to_list()
        assert result == pytest.approx([95.0])

    def test_truediv(self) -> None:
        expr = (Col("close") / 2.0).resolve()
        df = pl.DataFrame({"close": [100.0]})
        result = df.select(expr.alias("v"))["v"].to_list()
        assert result == pytest.approx([50.0])

    def test_rsub(self) -> None:
        expr = (100.0 - Col("close")).resolve()
        df = pl.DataFrame({"close": [30.0]})
        result = df.select(expr.alias("v"))["v"].to_list()
        assert result == pytest.approx([70.0])

    def test_rtruediv(self) -> None:
        expr = (1.0 / Col("close")).resolve()
        df = pl.DataFrame({"close": [4.0]})
        result = df.select(expr.alias("v"))["v"].to_list()
        assert result == pytest.approx([0.25])

    def test_rmod(self) -> None:
        expr = (10.0 % Col("b")).resolve()
        df = pl.DataFrame({"b": [3.0]})
        result = df.select(expr.alias("v"))["v"].to_list()
        assert result == pytest.approx([1.0])

    def test_radd(self) -> None:
        expr = (5.0 + Col("close")).resolve()
        df = pl.DataFrame({"close": [100.0]})
        result = df.select(expr.alias("v"))["v"].to_list()
        assert result == pytest.approx([105.0])


# ---------------------------------------------------------------------------
# Pct helper tests
# ---------------------------------------------------------------------------


class TestPct:
    def test_pct_positive(self) -> None:
        """Pct("close", 1.0) resolves to close * 1.01."""
        expr = Pct("close", 1.0).resolve()
        df = pl.DataFrame({"close": [100.0, 200.0]})
        result = df.select(expr.alias("v"))["v"].to_list()
        assert result == pytest.approx([101.0, 202.0])

    def test_pct_negative(self) -> None:
        """Pct("close", -0.5) resolves to close * 0.995."""
        expr = Pct("close", -0.5).resolve()
        df = pl.DataFrame({"close": [100.0, 200.0]})
        result = df.select(expr.alias("v"))["v"].to_list()
        assert result == pytest.approx([99.5, 199.0])

    def test_pct_with_price_expr_base(self) -> None:
        """Pct(Col("close"), 2.0) accepts PriceExpr base."""
        expr = Pct(Col("close"), 2.0).resolve()
        df = pl.DataFrame({"close": [100.0]})
        result = df.select(expr.alias("v"))["v"].to_list()
        assert result == pytest.approx([102.0])

    def test_pct_with_float_base(self) -> None:
        """Pct(100.0, -10.0) with float base resolves to literal 90.0."""
        expr = Pct(100.0, -10.0).resolve()
        df = pl.DataFrame({"x": [1]})  # dummy frame
        result = df.select(expr.alias("v"))["v"].to_list()
        assert result == pytest.approx([90.0])

    def test_pct_zero(self) -> None:
        """Pct("close", 0.0) is identity."""
        expr = Pct("close", 0.0).resolve()
        df = pl.DataFrame({"close": [100.0]})
        result = df.select(expr.alias("v"))["v"].to_list()
        assert result == pytest.approx([100.0])


# ---------------------------------------------------------------------------
# PriceExpr condition integration tests
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PctExitStrategy:
    """Entry = Crossover, Exit = PriceIsAbove with Pct expression."""

    def entry(self) -> Crossover:
        return Crossover("fast", "slow")

    def exit(self) -> PriceIsAbove:
        return PriceIsAbove("close", Pct("ref", 5))


@dataclass(frozen=True, slots=True)
class ColExprExitStrategy:
    """Entry = Crossover, Exit = PriceIsBelow with Col arithmetic."""

    def entry(self) -> Crossover:
        return Crossover("fast", "slow")

    def exit(self) -> PriceIsBelow:
        return PriceIsBelow("close", Col("sma") - Col("vol") * 2)


# ---------------------------------------------------------------------------
# Multi-symbol tests
# ---------------------------------------------------------------------------


def _make_multi_symbol_df() -> pl.DataFrame:
    """Two symbols with different crossover timing.

    AAPL: crossover at bar 2, crossunder at bar 5 (same as ohlcv fixture).
    TSLA: crossover at bar 3, crossunder at bar 6 (shifted by 1).
    """
    aapl = pl.DataFrame(
        {
            "symbol": ["AAPL"] * 8,
            "date": pl.date_range(pl.date(2024, 1, 1), pl.date(2024, 1, 8), eager=True),
            "open": [100.0, 100.5, 101.5, 103.5, 105.5, 103.5, 99.5, 97.5],
            "close": [100.0, 101.0, 103.0, 105.0, 104.0, 100.0, 98.0, 97.0],
            "fast": [1.0, 1.0, 3.0, 4.0, 3.5, 1.0, 0.5, 0.3],
            "slow": [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
        }
    )
    tsla = pl.DataFrame(
        {
            "symbol": ["TSLA"] * 8,
            "date": pl.date_range(pl.date(2024, 1, 1), pl.date(2024, 1, 8), eager=True),
            "open": [200.0, 200.5, 201.5, 203.5, 205.5, 207.5, 203.5, 199.5],
            "close": [200.0, 201.0, 201.5, 205.0, 206.0, 204.0, 200.0, 198.0],
            # Crossover at bar 3 (shifted by 1 vs AAPL), crossunder at bar 6
            "fast": [1.0, 1.0, 1.0, 3.0, 4.0, 3.5, 1.0, 0.5],
            "slow": [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
        }
    )
    return pl.concat([aapl, tsla])


class TestMultiSymbol:
    def test_getitem(self) -> None:
        """result['AAPL'] returns a BacktestResult for that symbol."""
        df = _make_multi_symbol_df()
        result = run(df, SimpleCrossStrategy(), instrument_col="symbol")
        aapl = result["AAPL"]
        assert aapl.returns.columns == ["date", "return"]
        assert aapl.trades.columns == ["entry_date", "exit_date", "pnl", "bars_held"]

    def test_symbols(self) -> None:
        """result.symbols returns ordered list of symbol keys."""
        df = _make_multi_symbol_df()
        result = run(df, SimpleCrossStrategy(), instrument_col="symbol")
        assert result.symbols == ["AAPL", "TSLA"]

    def test_len(self) -> None:
        df = _make_multi_symbol_df()
        result = run(df, SimpleCrossStrategy(), instrument_col="symbol")
        assert len(result) == 2

    def test_contains(self) -> None:
        df = _make_multi_symbol_df()
        result = run(df, SimpleCrossStrategy(), instrument_col="symbol")
        assert "AAPL" in result
        assert "SPY" not in result

    def test_items(self) -> None:
        df = _make_multi_symbol_df()
        result = run(df, SimpleCrossStrategy(), instrument_col="symbol")
        symbols = [sym for sym, _ in result.items()]
        assert symbols == ["AAPL", "TSLA"]

    def test_missing_symbol_keyerror(self) -> None:
        df = _make_multi_symbol_df()
        result = run(df, SimpleCrossStrategy(), instrument_col="symbol")
        with pytest.raises(KeyError):
            result["SPY"]

    def test_returns_schema(self) -> None:
        """Combined .returns has [symbol, date, return] columns."""
        df = _make_multi_symbol_df()
        result = run(df, SimpleCrossStrategy(), instrument_col="symbol")
        assert result.returns.columns == ["symbol", "date", "return"]

    def test_trades_schema(self) -> None:
        """Combined .trades has [symbol, entry_date, exit_date, pnl, bars_held]."""
        df = _make_multi_symbol_df()
        result = run(df, SimpleCrossStrategy(), instrument_col="symbol")
        assert result.trades.columns == ["symbol", "entry_date", "exit_date", "pnl", "bars_held"]

    def test_independent_positions(self) -> None:
        """Two symbols produce independent position sequences."""
        df = _make_multi_symbol_df()
        result = run(df, SimpleCrossStrategy(), instrument_col="symbol")
        aapl_pos = result["AAPL"].signals["_position"].to_list()
        tsla_pos = result["TSLA"].signals["_position"].to_list()
        # AAPL: entry bar 2, exit bar 5
        assert aapl_pos == [0, 0, 1, 1, 1, 0, 0, 0]
        # TSLA: entry bar 3, exit bar 6
        assert tsla_pos == [0, 0, 0, 1, 1, 1, 0, 0]

    def test_matches_individual_runs(self) -> None:
        """Multi-symbol run matches running each symbol individually."""
        df = _make_multi_symbol_df()
        combined = run(df, SimpleCrossStrategy(), instrument_col="symbol")

        for sym in ["AAPL", "TSLA"]:
            individual_df = df.filter(pl.col("symbol") == sym).drop("symbol")
            individual = run(individual_df, SimpleCrossStrategy())

            assert combined[sym].returns.equals(individual.returns), f"{sym} returns mismatch"
            assert combined[sym].trades.equals(individual.trades), f"{sym} trades mismatch"

    def test_with_init_hook(self) -> None:
        """Strategy with init() works per-symbol."""
        df = _make_multi_symbol_df().drop("fast", "slow")

        @dataclass(frozen=True, slots=True)
        class InitMultiStrategy:
            def init(self, df: pl.DataFrame) -> pl.DataFrame:
                return df.with_columns(
                    pl.col("close").rolling_mean(2).alias("fast"),
                    pl.lit(101.5).alias("slow"),
                )

            def entry(self) -> Crossover:
                return Crossover("fast", "slow")

            def exit(self) -> Crossunder:
                return Crossunder("fast", "slow")

        result = run(df, InitMultiStrategy(), instrument_col="symbol")
        assert result.symbols == ["AAPL", "TSLA"]
        # Each symbol was processed independently
        assert "fast" in result["AAPL"].signals.columns

    def test_with_flatten_eod(self) -> None:
        """flatten_eod works per-symbol with minute data."""
        cal = get_calendar("XNYS")

        def _minute_symbol(sym: str, base: float) -> pl.DataFrame:
            timestamps = [
                datetime.datetime(2024, 1, 2, 9, 30),
                datetime.datetime(2024, 1, 2, 9, 31),
                datetime.datetime(2024, 1, 2, 15, 59),
            ]
            n = len(timestamps)
            prices = [base + i * 0.5 for i in range(n)]
            return pl.DataFrame(
                {
                    "symbol": [sym] * n,
                    "date": timestamps,
                    "open": prices,
                    "close": [p + 0.1 for p in prices],
                    "fast": [1.0, 3.0, 3.0],
                    "slow": [2.0, 2.0, 2.0],
                    "never_cross": [5.0] * n,
                }
            )

        df = pl.concat([_minute_symbol("AAPL", 100.0), _minute_symbol("TSLA", 200.0)])
        result = run(
            df, AlwaysInStrategy(), calendar=cal, flatten_eod=True, instrument_col="symbol",
        )
        # Both symbols should have position forced to 0 at session end
        for sym in ["AAPL", "TSLA"]:
            pos = result[sym].signals["_position"].to_list()
            assert pos[-1] == 0, f"{sym} not flattened at EOD"

    def test_missing_column(self) -> None:
        """instrument_col='nope' raises ValueError."""
        df = _make_multi_symbol_df()
        with pytest.raises(ValueError, match="instrument_col='nope' not found"):
            run(df, SimpleCrossStrategy(), instrument_col="nope")

    def test_backward_compat(self, ohlcv: pl.DataFrame) -> None:
        """instrument_col=None (default) matches existing behavior."""
        result_default = run(ohlcv, SimpleCrossStrategy())
        result_none = run(ohlcv, SimpleCrossStrategy(), instrument_col=None)
        assert result_default.returns.equals(result_none.returns)
        assert result_default.trades.equals(result_none.trades)

    def test_empty_trades_one_symbol(self) -> None:
        """One symbol has 0 trades, results still include both symbols."""
        # AAPL has a crossover, TSLA does not (fast stays below slow)
        aapl = pl.DataFrame(
            {
                "symbol": ["AAPL"] * 6,
                "date": pl.date_range(pl.date(2024, 1, 1), pl.date(2024, 1, 6), eager=True),
                "open": [100.0, 100.5, 101.5, 103.5, 105.5, 103.5],
                "close": [100.0, 101.0, 103.0, 105.0, 104.0, 100.0],
                "fast": [1.0, 1.0, 3.0, 4.0, 3.5, 1.0],
                "slow": [2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
            }
        )
        tsla = pl.DataFrame(
            {
                "symbol": ["TSLA"] * 6,
                "date": pl.date_range(pl.date(2024, 1, 1), pl.date(2024, 1, 6), eager=True),
                "open": [200.0, 200.5, 201.5, 203.5, 205.5, 207.5],
                "close": [200.0, 201.0, 201.5, 203.0, 204.0, 205.0],
                # fast never crosses above slow
                "fast": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                "slow": [2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
            }
        )
        df = pl.concat([aapl, tsla])
        result = run(df, SimpleCrossStrategy(), instrument_col="symbol")
        # AAPL has trades, TSLA has none
        assert result["AAPL"].trades.height >= 1
        assert result["TSLA"].trades.height == 0
        # Both symbols present in combined returns
        return_symbols = result.returns["symbol"].unique().sort().to_list()
        assert return_symbols == ["AAPL", "TSLA"]


class TestPriceExprConditions:
    def test_price_is_above_with_pct(self) -> None:
        """PriceIsAbove("close", Pct("ref", 5)) fires when close > ref * 1.05."""
        df = pl.DataFrame(
            {
                "date": pl.date_range(pl.date(2024, 1, 1), pl.date(2024, 1, 8), eager=True),
                "open": [100.0, 100.5, 101.5, 103.5, 105.5, 108.0, 110.0, 112.0],
                "close": [100.0, 101.0, 103.0, 105.0, 107.0, 109.0, 111.0, 113.0],
                "fast": [1.0, 1.0, 3.0, 4.0, 4.5, 5.0, 5.5, 6.0],
                "slow": [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
                "ref": [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
            }
        )
        # Crossover at bar 2. Exit = PriceIsAbove("close", Pct("ref", 5)) = close > 105.0
        # close: 100, 101, 103, 105, 107, 109, ...
        # Exit fires at bar 4 (close=107 > 105)
        result = run(df, PctExitStrategy())
        positions = result.signals["_position"].to_list()
        assert positions[2] == 1, "entry at bar 2"
        assert positions[3] == 1, "still in at bar 3 (close=105 == 105, not >)"
        assert positions[4] == 0, "exit at bar 4 (close=107 > 105)"

    def test_price_is_below_with_col_expr(self) -> None:
        """PriceIsBelow("close", Col("sma") - Col("vol") * 2) fires correctly."""
        df = pl.DataFrame(
            {
                "date": pl.date_range(pl.date(2024, 1, 1), pl.date(2024, 1, 8), eager=True),
                "open": [100.0, 100.5, 101.5, 103.5, 105.5, 103.5, 99.5, 97.5],
                "close": [100.0, 101.0, 103.0, 105.0, 104.0, 100.0, 98.0, 97.0],
                "fast": [1.0, 1.0, 3.0, 4.0, 3.5, 3.0, 2.5, 2.1],
                "slow": [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
                # sma - vol*2 = 105 - 2*2 = 101
                "sma": [105.0, 105.0, 105.0, 105.0, 105.0, 105.0, 105.0, 105.0],
                "vol": [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
            }
        )
        # Crossover at bar 2. Exit = PriceIsBelow("close", Col("sma") - Col("vol") * 2)
        # Threshold = 105 - 4 = 101. close: 100, 101, 103, 105, 104, 100, 98, 97
        # Exit fires at bar 5 (close=100 < 101)
        result = run(df, ColExprExitStrategy())
        positions = result.signals["_position"].to_list()
        assert positions[2] == 1, "entry at bar 2"
        assert positions[4] == 1, "still in at bar 4 (close=104 > 101)"
        assert positions[5] == 0, "exit at bar 5 (close=100 < 101)"

    def test_price_expr_exit_in_engine(self) -> None:
        """Full engine run with PriceExpr exit produces correct trades."""
        df = pl.DataFrame(
            {
                "date": pl.date_range(pl.date(2024, 1, 1), pl.date(2024, 1, 8), eager=True),
                "open": [100.0, 100.5, 101.5, 103.5, 105.5, 108.0, 110.0, 112.0],
                "close": [100.0, 101.0, 103.0, 105.0, 107.0, 109.0, 111.0, 113.0],
                "fast": [1.0, 1.0, 3.0, 4.0, 4.5, 5.0, 5.5, 6.0],
                "slow": [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
                "ref": [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
            }
        )
        result = run(df, PctExitStrategy())
        assert result.trades.height == 1
        trade = result.trades.row(0, named=True)
        # Entry signal bar 2, fill at bar 3 open=103.5
        assert trade["entry_date"] == datetime.date(2024, 1, 3)
        # Exit signal bar 4, fill at bar 5 open=108.0
        assert trade["exit_date"] == datetime.date(2024, 1, 5)
        assert trade["pnl"] == pytest.approx(108.0 / 103.5 - 1, rel=1e-6)

    def test_backward_compat_str_float(self, ohlcv: pl.DataFrame) -> None:
        """PriceIsAbove("close", 100.0) still works unchanged."""
        cond = PriceIsAbove("close", 100.0)
        expr = cond.resolve()
        result = ohlcv.select(expr.alias("v"))["v"].to_list()
        # close: 100, 101, 103, 105, 104, 100, 98, 97
        assert result == [False, True, True, True, True, False, False, False]

    def test_price_expr_both_sides(self) -> None:
        """PriceIsAbove(Col("a"), Col("b") + 1) with PriceExpr on both sides."""
        df = pl.DataFrame({"a": [5.0, 10.0, 3.0], "b": [3.0, 10.0, 5.0]})
        cond = PriceIsAbove(Col("a"), Col("b") + 1)
        result = df.select(cond.resolve().alias("v"))["v"].to_list()
        # a > b+1: 5>4=True, 10>11=False, 3>6=False
        assert result == [True, False, False]
