from __future__ import annotations

from datetime import date

import pytest


class TestEuronextHolidays:
    def test_new_years_day_weekday(self, euronext):
        assert not euronext.is_session(date(2024, 1, 1))  # Monday

    def test_new_years_day_weekend_no_observance(self, euronext):
        # 2022-01-01 is Saturday — no Friday observance for Euronext
        assert euronext.is_session(date(2021, 12, 31))
        # 2023-01-01 is Sunday — no Monday observance
        assert euronext.is_session(date(2023, 1, 2))

    def test_good_friday(self, euronext):
        assert not euronext.is_session(date(2024, 3, 29))
        assert not euronext.is_session(date(2025, 4, 18))

    def test_easter_monday(self, euronext):
        assert not euronext.is_session(date(2024, 4, 1))
        assert not euronext.is_session(date(2025, 4, 21))

    def test_labour_day_weekday(self, euronext):
        # 2024: May 1 is Wednesday
        assert not euronext.is_session(date(2024, 5, 1))

    def test_labour_day_weekend_no_observance(self, euronext):
        # 2022: May 1 is Sunday — no Monday observance
        assert euronext.is_session(date(2022, 5, 2))

    def test_christmas(self, euronext):
        assert not euronext.is_session(date(2024, 12, 25))  # Wednesday

    def test_christmas_weekend_no_observance(self, euronext):
        # 2021: Dec 25 is Saturday — no Friday observance (Dec 24 is a trading day)
        assert euronext.is_session(date(2021, 12, 24))

    def test_boxing_day(self, euronext):
        assert not euronext.is_session(date(2024, 12, 26))  # Thursday

    def test_boxing_day_weekend_no_observance(self, euronext):
        # 2021: Dec 26 is Sunday — no Monday observance
        assert euronext.is_session(date(2021, 12, 27))


class TestEuronextEarlyCloses:
    def test_christmas_eve_weekday(self, euronext):
        # 2024-12-24 is Tuesday
        sched = euronext.get_schedule(date(2024, 12, 24))
        assert sched is not None
        assert sched.market_close.hour == 14
        assert sched.market_close.minute == 5

    def test_christmas_eve_weekend_no_early_close(self, euronext):
        # 2022-12-24 is Saturday — no session
        assert euronext.get_schedule(date(2022, 12, 24)) is None

    def test_new_years_eve_weekday(self, euronext):
        # 2024-12-31 is Tuesday
        sched = euronext.get_schedule(date(2024, 12, 31))
        assert sched is not None
        assert sched.market_close.hour == 14
        assert sched.market_close.minute == 5

    def test_new_years_eve_weekend_no_early_close(self, euronext):
        # 2022-12-31 is Saturday — no session
        assert euronext.get_schedule(date(2022, 12, 31)) is None


class TestEuronextTradingDayCount:
    @pytest.mark.parametrize("year", [2020, 2021, 2022, 2023, 2024])
    def test_trading_days_in_range(self, euronext, year):
        days = euronext.valid_days(date(year, 1, 1), date(year, 12, 31))
        # Euronext typically has 252-257 trading days per year
        assert 250 <= len(days) <= 258, f"Year {year}: {len(days)} trading days"
