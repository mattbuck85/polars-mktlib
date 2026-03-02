from __future__ import annotations

from datetime import date, time, timedelta

from mktlib.scheduling.easter import easter, good_friday
from mktlib.scheduling.rules import (
    AdhocClosure,
    EarlyClose,
    HolidayRule,
    fixed_date_if_weekday,
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

# --- Early closes ---

EURONEXT_EARLY_CLOSE_TIME = time(14, 5)

EARLY_CLOSES: list[EarlyClose] = [
    EarlyClose("Christmas Eve", EURONEXT_EARLY_CLOSE_TIME, compute_fn=fixed_date_if_weekday(12, 24)),
    EarlyClose("New Year's Eve", EURONEXT_EARLY_CLOSE_TIME, compute_fn=fixed_date_if_weekday(12, 31)),
]

# --- Exchange constants ---

EURONEXT_OPEN = time(9, 0)
EURONEXT_CLOSE = time(17, 30)
EURONEXT_TZ = "Europe/Paris"
