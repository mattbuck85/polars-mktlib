from __future__ import annotations

from datetime import date, datetime, time
from zoneinfo import ZoneInfo

import pytest

from mktlib.scheduling import get_calendar

ET = ZoneInfo("America/New_York")


# --- CME RTH ---


class TestCMERTHSchedule:
    def test_open_close_times(self, cme):
        sched = cme.get_schedule(date(2024, 6, 3))  # Monday
        assert sched is not None
        assert sched.market_open.time() == time(9, 30)
        assert sched.market_close.time() == time(16, 15)

    def test_holidays_match_nyse(self, cme):
        # Spot-check shared holidays
        assert not cme.is_session(date(2024, 1, 1))   # New Year's
        assert not cme.is_session(date(2024, 1, 15))  # MLK
        assert not cme.is_session(date(2024, 12, 25)) # Christmas
        assert not cme.is_session(date(2024, 3, 29))  # Good Friday

    def test_early_close_black_friday(self, cme):
        sched = cme.get_schedule(date(2024, 11, 29))
        assert sched is not None
        assert sched.market_close.hour == 13

    def test_early_close_christmas_eve(self, cme):
        sched = cme.get_schedule(date(2024, 12, 24))
        assert sched is not None
        assert sched.market_close.hour == 13

    def test_early_close_independence_day(self, cme):
        # 2024: July 4 Thu, early close July 3 Wed
        sched = cme.get_schedule(date(2024, 7, 3))
        assert sched is not None
        assert sched.market_close.hour == 13

    @pytest.mark.parametrize("year", [2020, 2021, 2022, 2023, 2024])
    def test_trading_days_in_range(self, cme, year):
        days = cme.valid_days(date(year, 1, 1), date(year, 12, 31))
        assert 248 <= len(days) <= 253, f"Year {year}: {len(days)} trading days"


# --- CME Globex ---


class TestGlobexSchedule:
    def test_cross_midnight_open(self, globex):
        """Monday session opens Sunday 6 PM."""
        sched = globex.get_schedule(date(2024, 6, 3))  # Monday
        assert sched is not None
        assert sched.market_open == datetime(2024, 6, 2, 18, 0, tzinfo=ET)  # Sunday 6 PM
        assert sched.market_close == datetime(2024, 6, 3, 17, 0, tzinfo=ET)  # Monday 5 PM

    def test_close_time(self, globex):
        sched = globex.get_schedule(date(2024, 6, 4))  # Tuesday
        assert sched is not None
        assert sched.market_close.time() == time(17, 0)

    def test_holidays_match_nyse(self, globex):
        assert not globex.is_session(date(2024, 1, 1))
        assert not globex.is_session(date(2024, 12, 25))
        assert not globex.is_session(date(2024, 3, 29))

    def test_early_close(self, globex):
        sched = globex.get_schedule(date(2024, 11, 29))  # Black Friday
        assert sched is not None
        assert sched.market_close.hour == 13

    @pytest.mark.parametrize("year", [2020, 2021, 2022, 2023, 2024])
    def test_trading_days_in_range(self, globex, year):
        days = globex.valid_days(date(year, 1, 1), date(year, 12, 31))
        assert 248 <= len(days) <= 253, f"Year {year}: {len(days)} trading days"


class TestGlobexMinuteQueries:
    def test_is_open_sunday_evening(self, globex):
        """Sunday 6:30 PM ET is within Monday's Globex session."""
        dt = datetime(2024, 6, 2, 18, 30, tzinfo=ET)  # Sunday 6:30 PM
        assert globex.is_open_on_minute(dt) is True

    def test_is_closed_during_gap(self, globex):
        """5:00-6:00 PM ET is the daily maintenance window — market closed."""
        dt = datetime(2024, 6, 3, 17, 30, tzinfo=ET)  # Monday 5:30 PM
        assert globex.is_open_on_minute(dt) is False

    def test_is_open_afternoon(self, globex):
        """Monday 2 PM is within Monday's session (open Sun 6 PM - close Mon 5 PM)."""
        dt = datetime(2024, 6, 3, 14, 0, tzinfo=ET)
        assert globex.is_open_on_minute(dt) is True

    def test_minute_to_session_sunday_evening(self, globex):
        """Sunday 6:30 PM belongs to Monday's session."""
        dt = datetime(2024, 6, 2, 18, 30, tzinfo=ET)
        assert globex.minute_to_session(dt) == date(2024, 6, 3)

    def test_minute_to_session_during_gap(self, globex):
        dt = datetime(2024, 6, 3, 17, 30, tzinfo=ET)
        assert globex.minute_to_session(dt) is None

    def test_next_open_from_sunday_evening(self, globex):
        """From Sunday 7 PM (already past Monday session's open), next open is Tuesday session's open (Monday 6 PM)."""
        dt = datetime(2024, 6, 2, 19, 0, tzinfo=ET)  # Sunday 7 PM
        result = globex.next_open(dt)
        assert result == datetime(2024, 6, 3, 18, 0, tzinfo=ET)  # Monday 6 PM (Tue session)

    def test_next_open_before_session_open(self, globex):
        """Sunday 5 PM is before Monday session's open (Sunday 6 PM)."""
        dt = datetime(2024, 6, 2, 17, 0, tzinfo=ET)  # Sunday 5 PM
        result = globex.next_open(dt)
        assert result == datetime(2024, 6, 2, 18, 0, tzinfo=ET)  # Sunday 6 PM (Mon session)

    def test_next_open_from_gap(self, globex):
        """From Mon 5:30 PM (gap), next open is Mon 6 PM (Tue session)."""
        dt = datetime(2024, 6, 3, 17, 30, tzinfo=ET)
        result = globex.next_open(dt)
        assert result == datetime(2024, 6, 3, 18, 0, tzinfo=ET)


# --- Registry aliases ---


class TestCMERegistryAliases:
    def test_cme_alias(self):
        cal = get_calendar("CME")
        assert cal.name == "XCME"

    def test_cme_rth_alias(self):
        cal = get_calendar("CME-RTH")
        assert cal.name == "XCME"

    def test_globex_alias(self):
        cal = get_calendar("Globex")
        assert cal.name == "GLBX"

    def test_cme_globex_alias(self):
        cal = get_calendar("CME-GLOBEX")
        assert cal.name == "GLBX"
