from __future__ import annotations

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import polars as pl


class TestFXCalendar:
    """FX (24/5) calendar: every weekday is a session, no holidays."""

    def test_every_weekday_is_session(self, fx):
        """All weekdays in a week are sessions."""
        days = fx.valid_days("2024-01-08", "2024-01-12")  # Mon-Fri
        assert len(days) == 5

    def test_no_holidays(self, fx):
        """Christmas and New Year's are regular sessions for FX."""
        assert fx.is_session(date(2024, 12, 25)) is True  # Christmas (Wed)
        assert fx.is_session(date(2025, 1, 1)) is True  # New Year's (Wed)

    def test_weekends_excluded(self, fx):
        assert fx.is_session(date(2024, 1, 6)) is False  # Saturday
        assert fx.is_session(date(2024, 1, 7)) is False  # Sunday

    def test_open_offset_monday_opens_sunday(self, fx):
        """Monday session opens Sunday 17:00 ET via open_offset=-1."""
        sched = fx.get_schedule(date(2024, 1, 8))  # Monday
        assert sched is not None
        assert sched.market_open.weekday() == 6  # Sunday
        assert sched.market_open.hour == 17
        assert sched.market_open.minute == 0

    def test_24h_session(self, fx):
        """Open and close are both 17:00, spanning 24 hours across days."""
        sched = fx.get_schedule(date(2024, 1, 9))  # Tuesday
        assert sched is not None
        assert sched.market_open.hour == 17
        assert sched.market_close.hour == 17
        # Open is previous day (Monday), close is Tuesday
        assert sched.market_open.day == 8
        assert sched.market_close.day == 9

    def test_is_open_on_minute_overnight(self, fx):
        """is_open_on_minute works for overnight sessions."""
        tz = ZoneInfo("America/New_York")
        # Monday 2024-01-08 session opens Sunday 17:00
        dt_open = datetime(2024, 1, 7, 17, 0, tzinfo=tz)
        assert fx.is_open_on_minute(dt_open) is True
        # Still open Monday morning
        dt_morning = datetime(2024, 1, 8, 10, 0, tzinfo=tz)
        assert fx.is_open_on_minute(dt_morning) is True

    def test_full_year_all_weekdays(self, fx):
        """Full year should have every weekday as a session."""
        days = fx.valid_days("2024-01-01", "2024-12-31")
        # Every weekday in 2024 (leap year, 366 days)
        expected = sum(
            1 for d in (date(2024, 1, 1) + timedelta(days=i) for i in range(366))
            if d.weekday() < 5
        )
        assert len(days) == expected

    def test_aliases(self):
        """FX calendar is accessible via all registered aliases."""
        from mktlib.scheduling import get_calendar

        fx1 = get_calendar("CMES")
        fx2 = get_calendar("CME-FX")
        fx3 = get_calendar("FX")
        assert fx1.name == fx2.name == fx3.name == "CMES"


class TestValidDays:
    def test_excludes_weekends(self, nyse):
        days = nyse.valid_days("2024-01-08", "2024-01-12")  # Mon-Fri, no holidays
        assert len(days) == 5
        for d in days.to_list():
            assert d.weekday() < 5

    def test_excludes_holidays(self, nyse):
        # 2024-01-01 is New Year's Day (Mon)
        days = nyse.valid_days("2024-01-01", "2024-01-05")
        dates = days.to_list()
        assert date(2024, 1, 1) not in dates
        assert date(2024, 1, 2) in dates

    def test_returns_polars_series(self, nyse):
        days = nyse.valid_days("2024-01-02", "2024-01-05")
        assert isinstance(days, pl.Series)
        assert days.dtype == pl.Date

    def test_string_and_date_args(self, nyse):
        s1 = nyse.valid_days("2024-06-01", "2024-06-30")
        s2 = nyse.valid_days(date(2024, 6, 1), date(2024, 6, 30))
        assert s1.to_list() == s2.to_list()


class TestSchedule:
    def test_columns(self, nyse):
        df = nyse.schedule("2024-01-02", "2024-01-05")
        assert set(df.columns) == {"date", "market_open", "market_close"}

    def test_open_close_times(self, nyse):
        df = nyse.schedule("2024-01-02", "2024-01-02")
        assert len(df) == 1
        row = df.row(0, named=True)
        tz = ZoneInfo("America/New_York")
        assert row["market_open"] == datetime(2024, 1, 2, 9, 30, tzinfo=tz)
        assert row["market_close"] == datetime(2024, 1, 2, 16, 0, tzinfo=tz)

    def test_early_close_black_friday(self, nyse):
        # 2024 Thanksgiving is Nov 28 (Thu), Black Friday is Nov 29
        df = nyse.schedule("2024-11-29", "2024-11-29")
        assert len(df) == 1
        row = df.row(0, named=True)
        assert row["market_close"].hour == 13
        assert row["market_close"].minute == 0

    def test_early_close_christmas_eve(self, nyse):
        # 2024-12-24 is Tuesday
        df = nyse.schedule("2024-12-24", "2024-12-24")
        assert len(df) == 1
        row = df.row(0, named=True)
        assert row["market_close"].hour == 13

    def test_no_holidays_in_schedule(self, nyse):
        df = nyse.schedule("2024-01-01", "2024-01-05")
        dates = df["date"].to_list()
        assert date(2024, 1, 1) not in dates

    def test_schedule_length_matches_valid_days(self, nyse):
        start, end = "2024-01-01", "2024-12-31"
        days = nyse.valid_days(start, end)
        df = nyse.schedule(start, end)
        assert len(days) == len(df)


class TestIsSession:
    def test_weekday_trading_day(self, nyse):
        assert nyse.is_session(date(2024, 1, 2)) is True  # Tuesday

    def test_weekend(self, nyse):
        assert nyse.is_session(date(2024, 1, 6)) is False  # Saturday
        assert nyse.is_session(date(2024, 1, 7)) is False  # Sunday

    def test_holiday(self, nyse):
        assert nyse.is_session(date(2024, 1, 1)) is False  # New Year's
        assert nyse.is_session(date(2024, 12, 25)) is False  # Christmas

    def test_string_arg(self, nyse):
        assert nyse.is_session("2024-01-02") is True


class TestGetSchedule:
    def test_trading_day(self, nyse):
        sched = nyse.get_schedule(date(2024, 1, 2))
        assert sched is not None
        assert sched.date == date(2024, 1, 2)
        assert sched.market_open.hour == 9
        assert sched.market_open.minute == 30
        assert sched.market_close.hour == 16

    def test_non_trading_day(self, nyse):
        assert nyse.get_schedule(date(2024, 1, 1)) is None

    def test_early_close(self, nyse):
        # Black Friday 2024 = Nov 29
        sched = nyse.get_schedule(date(2024, 11, 29))
        assert sched is not None
        assert sched.market_close.hour == 13
