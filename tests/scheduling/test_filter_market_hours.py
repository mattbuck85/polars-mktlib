from __future__ import annotations

from datetime import date, datetime, time
from zoneinfo import ZoneInfo

import polars as pl
import pytest

from mktlib.scheduling import get_calendar
from mktlib.scheduling.calendar import ExchangeCalendar, ExchangeCalendarWithBreaks


@pytest.fixture
def nyse():
    return get_calendar("XNYS")


@pytest.fixture
def jpx():
    return get_calendar("XTKS")


def _make_df(
    datetimes: list[datetime], col: str = "date"
) -> pl.DataFrame:
    return pl.DataFrame({col: datetimes})


class TestFilterMarketHours:
    def test_filters_non_market_bars(self, nyse: ExchangeCalendar):
        """Pre/post market bars are removed."""
        tz = ZoneInfo("America/New_York")
        bars = [
            datetime(2024, 1, 2, 8, 0, tzinfo=tz),   # pre-market
            datetime(2024, 1, 2, 10, 0, tzinfo=tz),  # market
            datetime(2024, 1, 2, 12, 0, tzinfo=tz),  # market
            datetime(2024, 1, 2, 17, 0, tzinfo=tz),  # post-market
        ]
        df = _make_df(bars)
        result = nyse.filter_market_hours(df)
        assert result.height == 2
        assert result["date"].to_list() == [bars[1], bars[2]]

    def test_keeps_all_market_bars(self, nyse: ExchangeCalendar):
        """All bars within market hours pass through."""
        tz = ZoneInfo("America/New_York")
        bars = [
            datetime(2024, 1, 2, 9, 30, tzinfo=tz),
            datetime(2024, 1, 2, 12, 0, tzinfo=tz),
            datetime(2024, 1, 2, 15, 59, tzinfo=tz),  # last minute
        ]
        df = _make_df(bars)
        result = nyse.filter_market_hours(df)
        assert result.height == 3

    def test_early_close_respected(self, nyse: ExchangeCalendar):
        """Bars after early close time are removed."""
        tz = ZoneInfo("America/New_York")
        # 2024-11-29 is Black Friday — NYSE closes at 13:00
        bars = [
            datetime(2024, 11, 29, 10, 0, tzinfo=tz),  # market
            datetime(2024, 11, 29, 12, 59, tzinfo=tz),  # last minute
            datetime(2024, 11, 29, 13, 0, tzinfo=tz),  # at close (excluded)
            datetime(2024, 11, 29, 14, 0, tzinfo=tz),  # post close
        ]
        df = _make_df(bars)
        result = nyse.filter_market_hours(df)
        assert result.height == 2
        assert result["date"].to_list() == [bars[0], bars[1]]

    def test_break_calendar(self, jpx: ExchangeCalendarWithBreaks):
        """Lunch break bars are removed for JPX."""
        tz = ZoneInfo("Asia/Tokyo")
        bars = [
            datetime(2024, 1, 4, 9, 30, tzinfo=tz),   # morning session
            datetime(2024, 1, 4, 11, 30, tzinfo=tz),  # in break
            datetime(2024, 1, 4, 12, 0, tzinfo=tz),   # in break
            datetime(2024, 1, 4, 12, 30, tzinfo=tz),  # afternoon session
            datetime(2024, 1, 4, 14, 59, tzinfo=tz),  # last minute
        ]
        df = _make_df(bars)
        result = jpx.filter_market_hours(df)
        # Break bars should be excluded
        times = [dt.time() for dt in result["date"].to_list()]
        assert time(11, 30) not in times
        assert time(12, 0) not in times
        assert time(9, 30) in times
        assert time(12, 30) in times

    def test_naive_datetime_column(self, nyse: ExchangeCalendar):
        """Naive timestamps work with tz-aware calendar."""
        bars = [
            datetime(2024, 1, 2, 8, 0),   # pre-market
            datetime(2024, 1, 2, 10, 0),  # market
            datetime(2024, 1, 2, 15, 59),  # last minute
            datetime(2024, 1, 2, 16, 0),  # at close
        ]
        df = _make_df(bars)
        result = nyse.filter_market_hours(df)
        assert result.height == 2

    def test_empty_dataframe(self, nyse: ExchangeCalendar):
        """No crash on 0-row input."""
        df = pl.DataFrame(
            {"date": []},
            schema={"date": pl.Datetime("us", "America/New_York")},
        )
        result = nyse.filter_market_hours(df)
        assert result.height == 0
        assert result.columns == ["date"]

    def test_custom_date_column(self, nyse: ExchangeCalendar):
        """date_column='timestamp' works."""
        tz = ZoneInfo("America/New_York")
        bars = [
            datetime(2024, 1, 2, 10, 0, tzinfo=tz),
            datetime(2024, 1, 2, 17, 0, tzinfo=tz),
        ]
        df = _make_df(bars, col="timestamp")
        result = nyse.filter_market_hours(df, date_column="timestamp")
        assert result.height == 1
        assert result.columns == ["timestamp"]
