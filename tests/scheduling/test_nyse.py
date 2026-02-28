from __future__ import annotations

from datetime import date

import pytest

from mktlib.scheduling import get_calendar
from mktlib.scheduling.easter import easter, good_friday


@pytest.fixture
def nyse():
    return get_calendar("XNYS")


class TestEaster:
    """Verify Easter computation against known dates."""

    @pytest.mark.parametrize(
        "year, expected",
        [
            (2000, date(2000, 4, 23)),
            (2005, date(2005, 3, 27)),
            (2010, date(2010, 4, 4)),
            (2015, date(2015, 4, 5)),
            (2020, date(2020, 4, 12)),
            (2024, date(2024, 3, 31)),
            (2025, date(2025, 4, 20)),
            (2026, date(2026, 4, 5)),
        ],
    )
    def test_easter(self, year, expected):
        assert easter(year) == expected

    def test_good_friday(self):
        assert good_friday(2024) == date(2024, 3, 29)
        assert good_friday(2025) == date(2025, 4, 18)


class TestNYSEHolidays:
    def test_new_years_day(self, nyse):
        assert not nyse.is_session(date(2024, 1, 1))  # Monday

    def test_new_years_saturday_not_observed(self, nyse):
        # 2022-01-01 is Saturday — NYSE does NOT observe on prior Friday
        assert nyse.is_session(date(2021, 12, 31))

    def test_new_years_sunday_observed_monday(self, nyse):
        # 2023-01-01 is Sunday, observed Monday 2023-01-02
        assert not nyse.is_session(date(2023, 1, 2))

    def test_mlk_day(self, nyse):
        # 2024 MLK Day: Jan 15 (3rd Monday)
        assert not nyse.is_session(date(2024, 1, 15))

    def test_mlk_day_not_before_1998(self, nyse):
        # MLK Day was not observed by NYSE before 1998
        # 1997: 3rd Monday of Jan = Jan 20
        assert nyse.is_session(date(1997, 1, 20))

    def test_presidents_day(self, nyse):
        # 2024: 3rd Monday of Feb = Feb 19
        assert not nyse.is_session(date(2024, 2, 19))

    def test_good_friday(self, nyse):
        assert not nyse.is_session(date(2024, 3, 29))
        assert not nyse.is_session(date(2025, 4, 18))

    def test_memorial_day(self, nyse):
        # 2024: Last Monday of May = May 27
        assert not nyse.is_session(date(2024, 5, 27))

    def test_juneteenth(self, nyse):
        # 2023: June 19 is Monday
        assert not nyse.is_session(date(2023, 6, 19))

    def test_juneteenth_not_before_2022(self, nyse):
        # 2021: June 19 is Saturday — even if it existed, no observance
        # 2020: June 19 is Friday — should be a trading day (not observed yet)
        assert nyse.is_session(date(2020, 6, 19))

    def test_juneteenth_weekend_observance(self, nyse):
        # 2022: June 19 is Sunday, observed Monday June 20
        assert not nyse.is_session(date(2022, 6, 20))

    def test_independence_day(self, nyse):
        # 2024: July 4 is Thursday
        assert not nyse.is_session(date(2024, 7, 4))

    def test_independence_day_saturday_observed_friday(self, nyse):
        # 2020: July 4 is Saturday, observed Friday July 3
        assert not nyse.is_session(date(2020, 7, 3))

    def test_labor_day(self, nyse):
        # 2024: 1st Monday of Sep = Sep 2
        assert not nyse.is_session(date(2024, 9, 2))

    def test_thanksgiving(self, nyse):
        # 2024: 4th Thursday of Nov = Nov 28
        assert not nyse.is_session(date(2024, 11, 28))

    def test_christmas(self, nyse):
        assert not nyse.is_session(date(2024, 12, 25))  # Wednesday

    def test_christmas_saturday_observed_friday(self, nyse):
        # 2021: Dec 25 is Saturday, observed Friday Dec 24
        assert not nyse.is_session(date(2021, 12, 24))


class TestNYSEAdhocClosures:
    def test_september_11(self, nyse):
        for day in range(11, 15):
            assert not nyse.is_session(date(2001, 9, day))

    def test_hurricane_sandy(self, nyse):
        assert not nyse.is_session(date(2012, 10, 29))
        assert not nyse.is_session(date(2012, 10, 30))

    def test_bush_funeral(self, nyse):
        assert not nyse.is_session(date(2018, 12, 5))


class TestNYSEEarlyCloses:
    def test_black_friday(self, nyse):
        sched = nyse.get_schedule(date(2024, 11, 29))
        assert sched is not None
        assert sched.market_close.hour == 13

    def test_christmas_eve_weekday(self, nyse):
        # 2024-12-24 is Tuesday
        sched = nyse.get_schedule(date(2024, 12, 24))
        assert sched is not None
        assert sched.market_close.hour == 13

    def test_christmas_eve_weekend_no_early_close(self, nyse):
        # 2022-12-24 is Saturday — no session at all
        assert nyse.get_schedule(date(2022, 12, 24)) is None

    def test_independence_day_early_close(self, nyse):
        # 2024: July 4 is Thursday, early close July 3 (Wednesday)
        sched = nyse.get_schedule(date(2024, 7, 3))
        assert sched is not None
        assert sched.market_close.hour == 13

    def test_independence_day_early_close_saturday_observance(self, nyse):
        # 2020: July 4 is Saturday (observed Fri July 3), early close Thu July 2
        sched = nyse.get_schedule(date(2020, 7, 2))
        assert sched is not None
        assert sched.market_close.hour == 13


class TestNYSETradingDayCount:
    """Verify approximate number of trading days per year."""

    @pytest.mark.parametrize("year", [2020, 2021, 2022, 2023, 2024])
    def test_trading_days_in_range(self, nyse, year):
        days = nyse.valid_days(date(year, 1, 1), date(year, 12, 31))
        # NYSE typically has 250-253 trading days per year
        assert 248 <= len(days) <= 253, f"Year {year}: {len(days)} trading days"
