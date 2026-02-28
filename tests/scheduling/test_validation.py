from __future__ import annotations

from datetime import date

import pytest

from mktlib.scheduling import get_calendar

# exchange_calendars is a dev dependency — skip if not installed
ec = pytest.importorskip("exchange_calendars")


@pytest.fixture
def nyse():
    return get_calendar("XNYS")


@pytest.fixture
def ec_nyse():
    return ec.get_calendar("XNYS")


class TestCrossValidation:
    """Cross-reference mktlib calendar against exchange_calendars."""

    @pytest.mark.parametrize("year", range(2007, 2027))
    def test_valid_days_match(self, nyse, ec_nyse, year):
        start = date(year, 1, 1)
        end = date(year, 12, 31)

        mktlib_days = set(nyse.valid_days(start, end).to_list())
        try:
            ec_sessions = ec_nyse.sessions_in_range(str(start), str(end))
        except Exception:
            pytest.skip(f"exchange_calendars out of range for year {year}")
            return
        ec_days = set(d.date() for d in ec_sessions)

        missing = ec_days - mktlib_days
        extra = mktlib_days - ec_days

        assert not missing, f"Year {year}: mktlib missing sessions: {sorted(missing)}"
        assert not extra, f"Year {year}: mktlib has extra sessions: {sorted(extra)}"

    @pytest.mark.parametrize("year", range(2014, 2027))
    def test_early_closes_superset(self, nyse, ec_nyse, year):
        """Verify mktlib early closes are a superset of exchange_calendars."""
        start = date(year, 1, 1)
        end = date(year, 12, 31)

        mktlib_schedule = nyse.schedule(start, end)
        mktlib_early = set()
        for row in mktlib_schedule.iter_rows(named=True):
            if row["market_close"].hour < 16:
                mktlib_early.add(row["date"])

        try:
            ec_early_dates = set(d.date() for d in ec_nyse.early_closes(str(start), str(end)))
        except Exception:
            pytest.skip(f"exchange_calendars out of range for year {year}")
            return

        missing = ec_early_dates - mktlib_early
        assert not missing, f"Year {year}: mktlib missing early closes: {sorted(missing)}"
