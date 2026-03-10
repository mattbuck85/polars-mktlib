from __future__ import annotations

import datetime
import math
from dataclasses import dataclass

import polars as pl
import pytest

from mktlib.backtest._conditions import Crossover, Crossunder, PriceIsAbove, PriceIsBelow
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
