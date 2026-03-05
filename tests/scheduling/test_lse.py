from __future__ import annotations

from datetime import date

import pytest


class TestLSEHolidays:
    def test_new_years_day(self, lse):
        assert not lse.is_session(date(2024, 1, 1))  # Monday

    def test_new_years_saturday_observed_monday(self, lse):
        # 2022-01-01 is Saturday, UK substitute Monday 2022-01-03
        assert not lse.is_session(date(2022, 1, 3))
        # Dec 31 (Friday) remains a trading day (early close)
        assert lse.is_session(date(2021, 12, 31))

    def test_new_years_sunday_observed_monday(self, lse):
        # 2023-01-01 is Sunday, observed Monday 2023-01-02
        assert not lse.is_session(date(2023, 1, 2))

    def test_good_friday(self, lse):
        assert not lse.is_session(date(2024, 3, 29))
        assert not lse.is_session(date(2025, 4, 18))

    def test_easter_monday(self, lse):
        assert not lse.is_session(date(2024, 4, 1))
        assert not lse.is_session(date(2025, 4, 21))

    def test_early_may_bank_holiday(self, lse):
        # 2024: 1st Monday of May = May 6
        assert not lse.is_session(date(2024, 5, 6))

    def test_spring_bank_holiday(self, lse):
        # 2024: last Monday of May = May 27
        assert not lse.is_session(date(2024, 5, 27))

    def test_summer_bank_holiday(self, lse):
        # 2024: last Monday of August = Aug 26
        assert not lse.is_session(date(2024, 8, 26))

    def test_christmas(self, lse):
        assert not lse.is_session(date(2024, 12, 25))  # Wednesday

    def test_christmas_saturday_observed_monday(self, lse):
        # 2021: Dec 25 is Saturday, UK substitute Monday Dec 27
        assert not lse.is_session(date(2021, 12, 27))
        # Dec 24 (Friday) remains a trading day (early close)
        assert lse.is_session(date(2021, 12, 24))

    def test_boxing_day(self, lse):
        assert not lse.is_session(date(2024, 12, 26))  # Thursday

    def test_boxing_day_christmas_saturday(self, lse):
        # 2021: Dec 25 Sat (observed Mon 27), Dec 26 Sun -> Boxing Day Tue 28
        assert not lse.is_session(date(2021, 12, 28))

    def test_boxing_day_christmas_sunday(self, lse):
        # 2022: Dec 25 Sun (observed Mon 26), Boxing Day -> Tue 27
        assert not lse.is_session(date(2022, 12, 27))


class TestLSEAdhocClosures:
    def test_queen_funeral(self, lse):
        assert not lse.is_session(date(2022, 9, 19))

    def test_platinum_jubilee(self, lse):
        assert not lse.is_session(date(2022, 6, 3))

    def test_royal_wedding_2011(self, lse):
        assert not lse.is_session(date(2011, 4, 29))


class TestLSEEarlyCloses:
    def test_christmas_eve_weekday(self, lse):
        # 2024-12-24 is Tuesday
        sched = lse.get_schedule(date(2024, 12, 24))
        assert sched is not None
        assert sched.market_close.hour == 12
        assert sched.market_close.minute == 30

    def test_christmas_eve_weekend_no_early_close(self, lse):
        # 2022-12-24 is Saturday — no session
        assert lse.get_schedule(date(2022, 12, 24)) is None

    def test_new_years_eve_weekday(self, lse):
        # 2024-12-31 is Tuesday
        sched = lse.get_schedule(date(2024, 12, 31))
        assert sched is not None
        assert sched.market_close.hour == 12
        assert sched.market_close.minute == 30

    def test_new_years_eve_weekend_no_early_close(self, lse):
        # 2022-12-31 is Saturday — no session
        assert lse.get_schedule(date(2022, 12, 31)) is None


class TestLSETradingDayCount:
    @pytest.mark.parametrize("year", [2020, 2021, 2022, 2023, 2024])
    def test_trading_days_in_range(self, lse, year):
        days = lse.valid_days(date(year, 1, 1), date(year, 12, 31))
        # LSE typically has 250-254 trading days per year
        assert (
            248 <= len(days) <= 255
        ), f"Year {year}: {len(days)} trading days"
