from __future__ import annotations

from datetime import datetime, time
from zoneinfo import ZoneInfo

import pytest

from mktlib.scheduling import get_calendar


class TestJPXBreaks:
    """Lunch break tests for Tokyo Stock Exchange (JPX)."""

    @pytest.fixture
    def cal(self):
        return get_calendar("XTKS")

    def test_is_open_morning_session(self, cal):
        """Market open during morning session (before break)."""
        dt = datetime(2024, 1, 4, 10, 0, tzinfo=ZoneInfo("Asia/Tokyo"))
        assert cal.is_open_on_minute(dt) is True

    def test_is_closed_during_lunch(self, cal):
        """Market closed during lunch break (11:30-12:30)."""
        dt = datetime(2024, 1, 4, 12, 0, tzinfo=ZoneInfo("Asia/Tokyo"))
        assert cal.is_open_on_minute(dt) is False

    def test_is_closed_at_break_start(self, cal):
        """Break start (11:30) is closed — [break_start, break_end) semantics."""
        dt = datetime(2024, 1, 4, 11, 30, tzinfo=ZoneInfo("Asia/Tokyo"))
        assert cal.is_open_on_minute(dt) is False

    def test_is_open_at_break_end(self, cal):
        """Break end (12:30) is open — afternoon session resumes."""
        dt = datetime(2024, 1, 4, 12, 30, tzinfo=ZoneInfo("Asia/Tokyo"))
        assert cal.is_open_on_minute(dt) is True

    def test_is_open_afternoon_session(self, cal):
        """Market open during afternoon session (after break)."""
        dt = datetime(2024, 1, 4, 14, 0, tzinfo=ZoneInfo("Asia/Tokyo"))
        assert cal.is_open_on_minute(dt) is True

    def test_trading_index_skips_lunch(self, cal):
        """trading_index at 1m skips 11:30-12:29 for JPX."""
        idx = cal.trading_index("2024-01-04", "2024-01-04", "1m")
        times = [dt.time() for dt in idx.to_list()]

        # Should not contain any minute in [11:30, 12:30)
        lunch_start = time(11, 30)
        lunch_end = time(12, 30)
        lunch_times = [t for t in times if lunch_start <= t < lunch_end]
        assert (
            lunch_times == []
        ), f"Found lunch minutes in trading_index: {lunch_times}"

        # Should contain minutes before and after lunch
        assert time(9, 30) in times
        assert time(11, 29) in times
        assert time(12, 30) in times
        assert time(14, 59) in times

    def test_schedule_has_break_columns(self, cal):
        """schedule() DataFrame includes break_start/break_end columns."""
        sched = cal.schedule("2024-01-04", "2024-01-04")
        assert "break_start" in sched.columns
        assert "break_end" in sched.columns

        row = sched.row(0, named=True)
        assert row["break_start"] is not None
        assert row["break_end"] is not None
        assert row["break_start"].time() == time(11, 30)
        assert row["break_end"].time() == time(12, 30)

    def test_get_schedule_has_break_fields(self, cal):
        """get_schedule() returns MarketDailySchedule with break fields."""
        sched = cal.get_schedule("2024-01-04")
        assert sched is not None
        assert sched.break_start is not None
        assert sched.break_end is not None
        assert sched.break_start.time() == time(11, 30)
        assert sched.break_end.time() == time(12, 30)


class TestHKEXBreaks:
    """Lunch break tests for Hong Kong Stock Exchange (HKEX)."""

    @pytest.fixture
    def cal(self):
        return get_calendar("XHKG")

    def test_is_open_morning_session(self, cal):
        """Market open during morning session (before break)."""
        dt = datetime(2024, 1, 2, 10, 0, tzinfo=ZoneInfo("Asia/Hong_Kong"))
        assert cal.is_open_on_minute(dt) is True

    def test_is_closed_during_lunch(self, cal):
        """Market closed during lunch break (12:00-13:00)."""
        dt = datetime(2024, 1, 2, 12, 30, tzinfo=ZoneInfo("Asia/Hong_Kong"))
        assert cal.is_open_on_minute(dt) is False

    def test_is_closed_at_break_start(self, cal):
        """Break start (12:00) is closed — [break_start, break_end) semantics."""
        dt = datetime(2024, 1, 2, 12, 0, tzinfo=ZoneInfo("Asia/Hong_Kong"))
        assert cal.is_open_on_minute(dt) is False

    def test_is_open_at_break_end(self, cal):
        """Break end (13:00) is open — afternoon session resumes."""
        dt = datetime(2024, 1, 2, 13, 0, tzinfo=ZoneInfo("Asia/Hong_Kong"))
        assert cal.is_open_on_minute(dt) is True

    def test_is_open_afternoon_session(self, cal):
        """Market open during afternoon session (after break)."""
        dt = datetime(2024, 1, 2, 14, 0, tzinfo=ZoneInfo("Asia/Hong_Kong"))
        assert cal.is_open_on_minute(dt) is True

    def test_trading_index_skips_lunch(self, cal):
        """trading_index at 1m skips 12:00-12:59 for HKEX."""
        idx = cal.trading_index("2024-01-02", "2024-01-02", "1m")
        times = [dt.time() for dt in idx.to_list()]

        # Should not contain any minute in [12:00, 13:00)
        lunch_start = time(12, 0)
        lunch_end = time(13, 0)
        lunch_times = [t for t in times if lunch_start <= t < lunch_end]
        assert (
            lunch_times == []
        ), f"Found lunch minutes in trading_index: {lunch_times}"

        # Should contain minutes before and after lunch
        assert time(9, 30) in times
        assert time(11, 59) in times
        assert time(13, 0) in times
        assert time(15, 59) in times

    def test_schedule_has_break_columns(self, cal):
        """schedule() DataFrame includes break_start/break_end columns."""
        sched = cal.schedule("2024-01-02", "2024-01-02")
        assert "break_start" in sched.columns
        assert "break_end" in sched.columns

        row = sched.row(0, named=True)
        assert row["break_start"] is not None
        assert row["break_end"] is not None
        assert row["break_start"].time() == time(12, 0)
        assert row["break_end"].time() == time(13, 0)

    def test_get_schedule_has_break_fields(self, cal):
        """get_schedule() returns MarketDailySchedule with break fields."""
        sched = cal.get_schedule("2024-01-02")
        assert sched is not None
        assert sched.break_start is not None
        assert sched.break_end is not None
        assert sched.break_start.time() == time(12, 0)
        assert sched.break_end.time() == time(13, 0)


class TestNoBreakExchange:
    """Verify non-break exchanges have null break columns."""

    @pytest.fixture
    def cal(self):
        return get_calendar("XNYS")

    def test_schedule_no_break_columns(self, cal):
        """NYSE schedule does not have break columns."""
        sched = cal.schedule("2024-01-02", "2024-01-02")
        assert "break_start" not in sched.columns
        assert "break_end" not in sched.columns

    def test_get_schedule_break_fields_none(self, cal):
        """NYSE get_schedule returns None for break fields."""
        sched = cal.get_schedule("2024-01-02")
        assert sched is not None
        assert sched.break_start is None
        assert sched.break_end is None
