from __future__ import annotations

from datetime import date, datetime
from zoneinfo import ZoneInfo

import polars as pl
import pytest

from mktlib.scheduling import ExchangeCalendar


# ── Session navigation ─────────────────────────────────────────────


class TestNextSession:
    def test_from_trading_day(self, nyse: ExchangeCalendar):
        # Tuesday Jan 2 → Wednesday Jan 3
        assert nyse.next_session(date(2024, 1, 2)) == date(2024, 1, 3)

    def test_skips_weekend(self, nyse: ExchangeCalendar):
        # Friday Jan 5 → Monday Jan 8
        assert nyse.next_session(date(2024, 1, 5)) == date(2024, 1, 8)

    def test_skips_holiday(self, nyse: ExchangeCalendar):
        # Dec 24, 2024 (Tue) → Dec 26 (Thu), skipping Christmas
        assert nyse.next_session(date(2024, 12, 24)) == date(2024, 12, 26)

    def test_from_weekend(self, nyse: ExchangeCalendar):
        # Saturday Jan 6 → Monday Jan 8
        assert nyse.next_session(date(2024, 1, 6)) == date(2024, 1, 8)

    def test_accepts_string(self, nyse: ExchangeCalendar):
        assert nyse.next_session("2024-01-02") == date(2024, 1, 3)


class TestPreviousSession:
    def test_from_trading_day(self, nyse: ExchangeCalendar):
        assert nyse.previous_session(date(2024, 1, 3)) == date(2024, 1, 2)

    def test_skips_weekend(self, nyse: ExchangeCalendar):
        # Monday Jan 8 → Friday Jan 5
        assert nyse.previous_session(date(2024, 1, 8)) == date(2024, 1, 5)

    def test_skips_holiday(self, nyse: ExchangeCalendar):
        # Dec 26, 2024 (Thu) → Dec 24 (Tue), skipping Christmas
        assert nyse.previous_session(date(2024, 12, 26)) == date(2024, 12, 24)

    def test_from_weekend(self, nyse: ExchangeCalendar):
        # Sunday Jan 7 → Friday Jan 5
        assert nyse.previous_session(date(2024, 1, 7)) == date(2024, 1, 5)

    def test_accepts_string(self, nyse: ExchangeCalendar):
        assert nyse.previous_session("2024-01-03") == date(2024, 1, 2)


class TestSessionOffset:
    def test_zero_returns_same(self, nyse: ExchangeCalendar):
        assert nyse.session_offset(date(2024, 1, 2), 0) == date(2024, 1, 2)

    def test_plus_one_equals_next(self, nyse: ExchangeCalendar):
        d = date(2024, 1, 2)
        assert nyse.session_offset(d, 1) == nyse.next_session(d)

    def test_minus_one_equals_previous(self, nyse: ExchangeCalendar):
        d = date(2024, 1, 3)
        assert nyse.session_offset(d, -1) == nyse.previous_session(d)

    def test_plus_five_skips_weekend(self, nyse: ExchangeCalendar):
        # Mon Jan 8 + 5 sessions → Mon Jan 15 (no holidays that week)
        # Jan 15 2024 is MLK day, so it should be Jan 16
        assert nyse.session_offset(date(2024, 1, 8), 5) == date(2024, 1, 16)

    def test_raises_on_non_session(self, nyse: ExchangeCalendar):
        with pytest.raises(ValueError, match="not a trading session"):
            nyse.session_offset(date(2024, 1, 6), 1)  # Saturday


class TestDateToSession:
    def test_none_on_session(self, nyse: ExchangeCalendar):
        assert nyse.date_to_session(date(2024, 1, 2)) == date(2024, 1, 2)

    def test_none_raises_on_non_session(self, nyse: ExchangeCalendar):
        with pytest.raises(ValueError, match="not a trading session"):
            nyse.date_to_session(date(2024, 1, 6))

    def test_next_from_weekend(self, nyse: ExchangeCalendar):
        assert nyse.date_to_session(date(2024, 1, 6), "next") == date(2024, 1, 8)

    def test_previous_from_weekend(self, nyse: ExchangeCalendar):
        assert nyse.date_to_session(date(2024, 1, 7), "previous") == date(2024, 1, 5)

    def test_invalid_direction(self, nyse: ExchangeCalendar):
        with pytest.raises(ValueError, match="Invalid direction"):
            nyse.date_to_session(date(2024, 1, 6), "up")


class TestSessionsInRange:
    def test_matches_valid_days(self, nyse: ExchangeCalendar):
        start, end = date(2024, 1, 1), date(2024, 1, 31)
        assert nyse.sessions_in_range(start, end) == len(nyse.valid_days(start, end))

    def test_full_year(self, nyse: ExchangeCalendar):
        count = nyse.sessions_in_range("2024-01-01", "2024-12-31")
        assert 250 <= count <= 253


# ── Minute-level queries ───────────────────────────────────────────

ET = ZoneInfo("America/New_York")


class TestIsOpenOnMinute:
    def test_during_trading(self, nyse: ExchangeCalendar):
        dt = datetime(2024, 1, 2, 12, 0, tzinfo=ET)
        assert nyse.is_open_on_minute(dt) is True

    def test_before_open(self, nyse: ExchangeCalendar):
        dt = datetime(2024, 1, 2, 9, 0, tzinfo=ET)
        assert nyse.is_open_on_minute(dt) is False

    def test_at_open(self, nyse: ExchangeCalendar):
        dt = datetime(2024, 1, 2, 9, 30, tzinfo=ET)
        assert nyse.is_open_on_minute(dt) is True

    def test_at_close(self, nyse: ExchangeCalendar):
        # [open, close) — close excluded
        dt = datetime(2024, 1, 2, 16, 0, tzinfo=ET)
        assert nyse.is_open_on_minute(dt) is False

    def test_one_minute_before_close(self, nyse: ExchangeCalendar):
        dt = datetime(2024, 1, 2, 15, 59, tzinfo=ET)
        assert nyse.is_open_on_minute(dt) is True

    def test_weekend(self, nyse: ExchangeCalendar):
        dt = datetime(2024, 1, 6, 12, 0, tzinfo=ET)  # Saturday
        assert nyse.is_open_on_minute(dt) is False

    def test_holiday(self, nyse: ExchangeCalendar):
        dt = datetime(2024, 1, 1, 12, 0, tzinfo=ET)  # New Year's
        assert nyse.is_open_on_minute(dt) is False

    def test_early_close_before(self, nyse: ExchangeCalendar):
        # Black Friday 2024, closes at 13:00
        dt = datetime(2024, 11, 29, 12, 0, tzinfo=ET)
        assert nyse.is_open_on_minute(dt) is True

    def test_early_close_after(self, nyse: ExchangeCalendar):
        dt = datetime(2024, 11, 29, 13, 30, tzinfo=ET)
        assert nyse.is_open_on_minute(dt) is False

    def test_naive_assumes_exchange_tz(self, nyse: ExchangeCalendar):
        dt = datetime(2024, 1, 2, 12, 0)  # naive
        assert nyse.is_open_on_minute(dt) is True

    def test_aware_different_tz(self, nyse: ExchangeCalendar):
        # 17:00 UTC = 12:00 ET
        utc = ZoneInfo("UTC")
        dt = datetime(2024, 1, 2, 17, 0, tzinfo=utc)
        assert nyse.is_open_on_minute(dt) is True


class TestNextOpen:
    def test_before_todays_open(self, nyse: ExchangeCalendar):
        dt = datetime(2024, 1, 2, 8, 0, tzinfo=ET)
        assert nyse.next_open(dt) == datetime(2024, 1, 2, 9, 30, tzinfo=ET)

    def test_during_trading(self, nyse: ExchangeCalendar):
        dt = datetime(2024, 1, 2, 12, 0, tzinfo=ET)
        assert nyse.next_open(dt) == datetime(2024, 1, 3, 9, 30, tzinfo=ET)

    def test_after_close(self, nyse: ExchangeCalendar):
        dt = datetime(2024, 1, 2, 17, 0, tzinfo=ET)
        assert nyse.next_open(dt) == datetime(2024, 1, 3, 9, 30, tzinfo=ET)

    def test_friday_after_close(self, nyse: ExchangeCalendar):
        dt = datetime(2024, 1, 5, 17, 0, tzinfo=ET)
        assert nyse.next_open(dt) == datetime(2024, 1, 8, 9, 30, tzinfo=ET)


class TestNextClose:
    def test_during_trading(self, nyse: ExchangeCalendar):
        dt = datetime(2024, 1, 2, 12, 0, tzinfo=ET)
        assert nyse.next_close(dt) == datetime(2024, 1, 2, 16, 0, tzinfo=ET)

    def test_after_close(self, nyse: ExchangeCalendar):
        dt = datetime(2024, 1, 2, 17, 0, tzinfo=ET)
        assert nyse.next_close(dt) == datetime(2024, 1, 3, 16, 0, tzinfo=ET)

    def test_early_close_day(self, nyse: ExchangeCalendar):
        # Black Friday 2024 closes at 13:00
        dt = datetime(2024, 11, 29, 10, 0, tzinfo=ET)
        assert nyse.next_close(dt) == datetime(2024, 11, 29, 13, 0, tzinfo=ET)


class TestPreviousOpen:
    def test_after_todays_open(self, nyse: ExchangeCalendar):
        dt = datetime(2024, 1, 2, 12, 0, tzinfo=ET)
        assert nyse.previous_open(dt) == datetime(2024, 1, 2, 9, 30, tzinfo=ET)

    def test_before_todays_open(self, nyse: ExchangeCalendar):
        dt = datetime(2024, 1, 8, 8, 0, tzinfo=ET)  # Monday before open
        # Previous session is Friday Jan 5
        assert nyse.previous_open(dt) == datetime(2024, 1, 5, 9, 30, tzinfo=ET)


class TestPreviousClose:
    def test_after_todays_close(self, nyse: ExchangeCalendar):
        dt = datetime(2024, 1, 2, 17, 0, tzinfo=ET)
        assert nyse.previous_close(dt) == datetime(2024, 1, 2, 16, 0, tzinfo=ET)

    def test_during_trading(self, nyse: ExchangeCalendar):
        # Before close → previous session's close
        dt = datetime(2024, 1, 3, 12, 0, tzinfo=ET)
        assert nyse.previous_close(dt) == datetime(2024, 1, 2, 16, 0, tzinfo=ET)


class TestMinuteToSession:
    def test_during_trading(self, nyse: ExchangeCalendar):
        dt = datetime(2024, 1, 2, 12, 0, tzinfo=ET)
        assert nyse.minute_to_session(dt) == date(2024, 1, 2)

    def test_outside_hours(self, nyse: ExchangeCalendar):
        dt = datetime(2024, 1, 2, 17, 0, tzinfo=ET)
        assert nyse.minute_to_session(dt) is None

    def test_weekend(self, nyse: ExchangeCalendar):
        dt = datetime(2024, 1, 6, 12, 0, tzinfo=ET)
        assert nyse.minute_to_session(dt) is None


# ── Trading index ──────────────────────────────────────────────────


class TestTradingIndex:
    def test_single_day_1m(self, nyse: ExchangeCalendar):
        idx = nyse.trading_index(date(2024, 1, 2), date(2024, 1, 2), "1m")
        # 9:30 to 16:00 = 6.5h = 390 minutes, closed="left" → 390 points
        assert len(idx) == 390

    def test_single_day_5m(self, nyse: ExchangeCalendar):
        idx = nyse.trading_index(date(2024, 1, 2), date(2024, 1, 2), "5m")
        assert len(idx) == 78  # 390 / 5

    def test_multi_day(self, nyse: ExchangeCalendar):
        idx = nyse.trading_index(date(2024, 1, 2), date(2024, 1, 3), "1m")
        assert len(idx) == 390 * 2

    def test_early_close(self, nyse: ExchangeCalendar):
        # Black Friday 2024: 9:30–13:00 = 3.5h = 210 min
        idx = nyse.trading_index(date(2024, 11, 29), date(2024, 11, 29), "1m")
        assert len(idx) == 210

    def test_closed_left_excludes_close(self, nyse: ExchangeCalendar):
        idx = nyse.trading_index(date(2024, 1, 2), date(2024, 1, 2), "1m", closed="left")
        ts_list = idx.to_list()
        close = datetime(2024, 1, 2, 16, 0, tzinfo=ET)
        assert close not in ts_list

    def test_returns_series_with_tz(self, nyse: ExchangeCalendar):
        idx = nyse.trading_index(date(2024, 1, 2), date(2024, 1, 2))
        assert isinstance(idx, pl.Series)
        assert idx.dtype == pl.Datetime("us", "America/New_York")

    def test_empty_range(self, nyse: ExchangeCalendar):
        # Weekend only
        idx = nyse.trading_index(date(2024, 1, 6), date(2024, 1, 7))
        assert len(idx) == 0
        assert idx.dtype == pl.Datetime("us", "America/New_York")

    def test_lse_different_hours(self, lse: ExchangeCalendar):
        # LSE: 8:00–16:30 = 8.5h = 510 min
        idx = lse.trading_index(date(2024, 1, 2), date(2024, 1, 2), "1m")
        assert len(idx) == 510

    def test_euronext_different_hours(self, euronext: ExchangeCalendar):
        # Euronext: 9:00–17:30 = 8.5h = 510 min
        idx = euronext.trading_index(date(2024, 1, 2), date(2024, 1, 2), "1m")
        assert len(idx) == 510
