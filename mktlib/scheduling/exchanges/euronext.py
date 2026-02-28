from __future__ import annotations

from datetime import date, time, timedelta

from mktlib.scheduling.easter import easter, good_friday
from mktlib.scheduling.rules import (
    AdhocClosure,
    EarlyClose,
    HolidayRule,
)

# --- Recurring holidays ---
# Euronext does NOT use nearest_workday — if a holiday falls on a weekend,
# there is no extra weekday closure.

NEW_YEARS_DAY = HolidayRule(
    name="New Year's Day",
    month=1,
    day=1,
)

LABOUR_DAY = HolidayRule(
    name="Labour Day",
    month=5,
    day=1,
)

CHRISTMAS = HolidayRule(
    name="Christmas Day",
    month=12,
    day=25,
)

BOXING_DAY = HolidayRule(
    name="Boxing Day",
    month=12,
    day=26,
)

RECURRING_HOLIDAYS: list[HolidayRule] = [
    NEW_YEARS_DAY,
    LABOUR_DAY,
    CHRISTMAS,
    BOXING_DAY,
]


def special_closures(start: date, end: date) -> list[date]:
    """Good Friday + Easter Monday closures for Euronext."""
    results: list[date] = []
    for year in range(start.year, end.year + 1):
        gf = good_friday(year)
        if start <= gf <= end:
            results.append(gf)
        easter_monday = easter(year) + timedelta(days=1)
        if start <= easter_monday <= end:
            results.append(easter_monday)
    return results


# --- Ad-hoc closures ---

ADHOC_CLOSURES: list[AdhocClosure] = []

# --- Early close helpers ---

EURONEXT_EARLY_CLOSE_TIME = time(14, 5)


def special_early_closes(start: date, end: date) -> dict[date, time]:
    """Christmas Eve and New Year's Eve early closes at 14:05."""
    result: dict[date, time] = {}
    for year in range(start.year, end.year + 1):
        dec24 = date(year, 12, 24)
        if dec24.weekday() < 5 and start <= dec24 <= end:
            result[dec24] = EURONEXT_EARLY_CLOSE_TIME
        dec31 = date(year, 12, 31)
        if dec31.weekday() < 5 and start <= dec31 <= end:
            result[dec31] = EURONEXT_EARLY_CLOSE_TIME
    return result


EARLY_CLOSES: list[EarlyClose] = []

# --- Exchange constants ---

EURONEXT_OPEN = time(9, 0)
EURONEXT_CLOSE = time(17, 30)
EURONEXT_TZ = "Europe/Paris"
