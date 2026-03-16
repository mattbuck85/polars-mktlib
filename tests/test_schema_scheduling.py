"""Schema validation tests for mktlib.scheduling output DataFrames."""

from __future__ import annotations

import pytest

from mktlib.scheduling import get_calendar

from tests.schemas.scheduling import validate_schedule, validate_valid_days


@pytest.mark.parametrize(
    ("exchange", "tz"),
    [
        ("XNYS", "America/New_York"),
        ("XLON", "Europe/London"),
    ],
)
class TestScheduleSchema:
    def test_schedule(self, exchange: str, tz: str):
        cal = get_calendar(exchange)
        df = cal.schedule("2024-01-01", "2024-01-31")
        validate_schedule(df, tz)

    def test_valid_days(self, exchange: str, tz: str):  # noqa: ARG002
        cal = get_calendar(exchange)
        days = cal.valid_days("2024-01-01", "2024-01-31")
        validate_valid_days(days)


class TestBreakExchangeSchema:
    def test_hkex_schedule_has_breaks(self):
        cal = get_calendar("XHKG")
        df = cal.schedule("2024-01-01", "2024-01-31")
        validate_schedule(df, "Asia/Hong_Kong", has_breaks=True)

    def test_hkex_valid_days(self):
        cal = get_calendar("XHKG")
        days = cal.valid_days("2024-01-01", "2024-01-31")
        validate_valid_days(days)
