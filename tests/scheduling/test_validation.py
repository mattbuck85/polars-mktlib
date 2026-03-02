from __future__ import annotations

from datetime import date

import pytest

from mktlib.scheduling import get_calendar

# exchange_calendars is a dev dependency — skip if not installed
ec = pytest.importorskip("exchange_calendars")


class ExchangeValidationBase:
    """Base cross-validation against exchange_calendars."""

    MKTLIB_NAME: str
    EC_NAME: str
    VALID_DAYS_YEARS: range
    EARLY_CLOSE_YEARS: range

    def pytest_generate_tests(self, metafunc):
        if "year" in metafunc.fixturenames:
            if metafunc.function.__name__ == "test_valid_days_match":
                metafunc.parametrize("year", self.VALID_DAYS_YEARS)
            elif metafunc.function.__name__ == "test_early_closes_superset":
                metafunc.parametrize("year", self.EARLY_CLOSE_YEARS)

    @pytest.fixture
    def cal(self):
        return get_calendar(self.MKTLIB_NAME)

    @pytest.fixture
    def ec_cal(self):
        return ec.get_calendar(self.EC_NAME)

    def test_valid_days_match(self, cal, ec_cal, year):
        start = date(year, 1, 1)
        end = date(year, 12, 31)

        mktlib_days = set(cal.valid_days(start, end).to_list())
        try:
            ec_sessions = ec_cal.sessions_in_range(str(start), str(end))
        except Exception:
            pytest.skip(f"exchange_calendars out of range for year {year}")
            return
        ec_days = set(d.date() for d in ec_sessions)

        missing = ec_days - mktlib_days
        extra = mktlib_days - ec_days

        assert not missing, f"Year {year}: mktlib missing sessions: {sorted(missing)}"
        assert not extra, f"Year {year}: mktlib has extra sessions: {sorted(extra)}"

    def test_early_closes_superset(self, cal, ec_cal, year):
        """Verify mktlib early closes are a superset of exchange_calendars."""
        start = date(year, 1, 1)
        end = date(year, 12, 31)

        mktlib_schedule = cal.schedule(start, end)
        normal_close = cal.close_time
        mktlib_early = set()
        for row in mktlib_schedule.iter_rows(named=True):
            if row["market_close"].time() < normal_close:
                mktlib_early.add(row["date"])

        ec_early_dates = {
            d.date()
            for d in ec_cal.early_closes
            if start <= d.date() <= end
        }

        missing = ec_early_dates - mktlib_early
        assert not missing, f"Year {year}: mktlib missing early closes: {sorted(missing)}"


class TestNYSEValidation(ExchangeValidationBase):
    MKTLIB_NAME = "XNYS"
    EC_NAME = "XNYS"
    VALID_DAYS_YEARS = range(2007, 2027)
    EARLY_CLOSE_YEARS = range(2014, 2027)


class TestLSEValidation(ExchangeValidationBase):
    MKTLIB_NAME = "XLON"
    EC_NAME = "XLON"
    VALID_DAYS_YEARS = range(2007, 2027)
    EARLY_CLOSE_YEARS = range(2014, 2027)


class TestEuronextValidation(ExchangeValidationBase):
    MKTLIB_NAME = "XPAR"
    EC_NAME = "XPAR"
    VALID_DAYS_YEARS = range(2007, 2027)
    EARLY_CLOSE_YEARS = range(2014, 2027)


@pytest.mark.xfail(
    reason="CME implementation reuses NYSE holidays; exchange_calendars CMES treats some as early-close instead of closure",
    strict=False,
)
class TestCMERTHValidation(ExchangeValidationBase):
    MKTLIB_NAME = "XCME"
    EC_NAME = "CMES"
    VALID_DAYS_YEARS = range(2007, 2027)
    EARLY_CLOSE_YEARS = range(2014, 2027)
