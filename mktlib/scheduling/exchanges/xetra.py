from __future__ import annotations

from datetime import date, time, timedelta

from mktlib.scheduling.easter import easter, good_friday
from mktlib.scheduling.rules import (
    AdhocClosure,
    EarlyClose,
    HolidayRule,
    weekday_before,
)

# --- Recurring German/EU holidays ---

NEW_YEARS_DAY = HolidayRule(name="New Year's Day", month=1, day=1)
LABOUR_DAY = HolidayRule(name="Labour Day", month=5, day=1)
GERMAN_UNITY_DAY = HolidayRule(
    name="German Reunification Day", month=10, day=3, start_year=2014, end_year=2021,
)
CHRISTMAS = HolidayRule(name="Christmas Day", month=12, day=25)
BOXING_DAY = HolidayRule(name="Boxing Day", month=12, day=26)

RECURRING_HOLIDAYS: list[HolidayRule] = [
    NEW_YEARS_DAY,
    LABOUR_DAY,
    GERMAN_UNITY_DAY,
    CHRISTMAS,
    BOXING_DAY,
]


def special_closures(start: date, end: date) -> list[date]:
    """Variable-date closures for Xetra:
    - Good Friday
    - Easter Monday
    - Whit Monday (2015-2021 only)
    - Christmas Eve (Dec 24) when it falls on a weekday
    - New Year's Eve (Dec 31) when it falls on a weekday
    """
    results: list[date] = []
    for year in range(start.year, end.year + 1):
        gf = good_friday(year)
        if start <= gf <= end:
            results.append(gf)

        easter_sunday = easter(year)

        easter_monday = easter_sunday + timedelta(days=1)
        if start <= easter_monday <= end:
            results.append(easter_monday)

        # Whit Monday (Pentecost Monday) = Easter + 50 days; observed 2015-2021
        if 2015 <= year <= 2021:
            whit_monday = easter_sunday + timedelta(days=50)
            if start <= whit_monday <= end:
                results.append(whit_monday)

        # Dec 24 — full closure when weekday
        dec24 = date(year, 12, 24)
        if dec24.weekday() < 5 and start <= dec24 <= end:
            results.append(dec24)

        # Dec 31 — full closure when weekday
        dec31 = date(year, 12, 31)
        if dec31.weekday() < 5 and start <= dec31 <= end:
            results.append(dec31)

    return results


# --- Ad-hoc closures ---

ADHOC_CLOSURES: list[AdhocClosure] = [
    # Whit Monday was observed as a one-off in 2007
    AdhocClosure("Whit Monday 2007", dates=[date(2007, 5, 28)]),
    AdhocClosure("Reformation Day 500th Anniversary", dates=[date(2017, 10, 31)]),
]

# --- Early closes ---
# Last working day before Dec 31 is an early close at 14:00, starting from 2010.

XETR_EARLY_CLOSE_TIME = time(14, 0)


def _last_workday_before_nye(year: int) -> date | None:
    if year < 2010:
        return None
    return weekday_before(date(year, 12, 31))


EARLY_CLOSES: list[EarlyClose] = [
    EarlyClose("Last Working Day of Year", XETR_EARLY_CLOSE_TIME, compute_fn=_last_workday_before_nye),
]

# --- Exchange constants ---

XETR_OPEN = time(9, 0)
XETR_CLOSE = time(17, 30)
XETR_TZ = "Europe/Berlin"
