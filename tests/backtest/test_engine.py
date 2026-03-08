from __future__ import annotations

import datetime
from dataclasses import dataclass

import polars as pl
import pytest

from mktlib.backtest._conditions import Crossover, Crossunder
from mktlib.backtest._engine import run


@dataclass(frozen=True, slots=True)
class SimpleCrossStrategy:
    fast: str = "fast"
    slow: str = "slow"

    def entry(self) -> Crossover:
        return Crossover(self.fast, self.slow)

    def exit(self) -> Crossunder:
        return Crossunder(self.fast, self.slow)


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

    def test_trade_on_custom_column(self, ohlcv: pl.DataFrame) -> None:
        """Can use a different price column for return calculation."""
        df = ohlcv.with_columns(pl.col("close").alias("vwap"))
        result = run(df, SimpleCrossStrategy(), trade_on="vwap")
        assert result.returns.height == df.height


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
    def test_prefilter_removes_non_market_bars(self) -> None:
        """With prefilter_market_data=True, non-market bars are removed."""
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
            prefilter_market_data=True,
        )
        result_times = result.signals["date"].to_list()
        # Only 09:30 and 10:00 should survive
        assert len(result_times) == 2
        assert all(
            datetime.time(9, 30) <= t.time() <= datetime.time(15, 59)
            for t in result_times
        )

    def test_mask_mode_zeros_non_market_returns(self) -> None:
        """With prefilter_market_data=False, all rows present but non-market returns are 0."""
        cal = get_calendar("XNYS")
        df = _make_minute_df(
            datetime.date(2024, 1, 2),
            [(8, 0), (9, 30), (10, 0), (18, 0)],
        )
        result = run(
            df,
            SimpleCrossStrategy(),
            calendar=cal,
            prefilter_market_data=False,
        )
        assert result.signals.height == 4  # all rows kept
        rets = result.returns["return"].to_list()
        # Bar 0 (08:00) and bar 3 (18:00) are outside market hours
        assert rets[0] == 0.0
        assert rets[3] == 0.0

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
            prefilter_market_data=True,
            flatten_eod=True,
        )
        positions = result.signals["_position"].to_list()
        # Entry signal at bar 0 (fast crosses above slow at bar 1)
        # With flatten_eod, position should go to 0 at 15:59 (session last)
        # Day 1: bar0=0, bar1=1, bar2(15:59)=0 (session end forces close)
        # Day 2: needs fresh entry signal — bar3=0, bar4=1, bar5(15:59)=0
        assert positions[2] == 0, "position should be 0 at Day 1 session end"
        assert positions[5] == 0, "position should be 0 at Day 2 session end"

        # Day 2 first bar should have 0 return (no position held overnight)
        rets = result.returns["return"].to_list()
        # The overnight gap return should NOT be captured
        # Bar 3 (Day2 09:30): _pos_delayed=0 (previous bar position=0), so return=0
        assert rets[3] == pytest.approx(0.0), "no overnight gap return with flatten_eod"

    def test_flatten_eod_false_captures_overnight_gap(self) -> None:
        """Without flatten_eod, position persists and overnight gap return is captured."""
        cal = get_calendar("XNYS")
        df = _make_two_session_df()
        result = run(
            df,
            AlwaysInStrategy(),
            calendar=cal,
            prefilter_market_data=True,
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
