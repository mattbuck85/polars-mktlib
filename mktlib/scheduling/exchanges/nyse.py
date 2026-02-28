from __future__ import annotations

from datetime import date, time, timedelta

from mktlib.scheduling.easter import good_friday
from mktlib.scheduling.rules import (
    AdhocClosure,
    EarlyClose,
    HolidayRule,
    nearest_workday,
    sunday_to_monday,
)

# --- Recurring holidays ---

# NYSE does NOT observe New Year's on the prior Friday when Jan 1 is Saturday.
# Only Sunday→Monday observance applies.
NEW_YEARS_DAY = HolidayRule(
    name="New Year's Day",
    month=1,
    day=1,
    observance=sunday_to_monday,
)

MLK_DAY = HolidayRule(
    name="Martin Luther King Jr. Day",
    month=1,
    weekday=0,  # Monday
    week=3,
    start_year=1998,
)

PRESIDENTS_DAY = HolidayRule(
    name="Presidents' Day",
    month=2,
    weekday=0,  # Monday
    week=3,
)

MEMORIAL_DAY = HolidayRule(
    name="Memorial Day",
    month=5,
    weekday=0,  # Monday
    week=-1,  # Last Monday
)

JUNETEENTH = HolidayRule(
    name="Juneteenth National Independence Day",
    month=6,
    day=19,
    observance=nearest_workday,
    start_year=2022,
)

INDEPENDENCE_DAY = HolidayRule(
    name="Independence Day",
    month=7,
    day=4,
    observance=nearest_workday,
)

LABOR_DAY = HolidayRule(
    name="Labor Day",
    month=9,
    weekday=0,  # Monday
    week=1,
)

THANKSGIVING = HolidayRule(
    name="Thanksgiving Day",
    month=11,
    weekday=3,  # Thursday
    week=4,
)

CHRISTMAS = HolidayRule(
    name="Christmas Day",
    month=12,
    day=25,
    observance=nearest_workday,
)

RECURRING_HOLIDAYS: list[HolidayRule] = [
    NEW_YEARS_DAY,
    MLK_DAY,
    PRESIDENTS_DAY,
    MEMORIAL_DAY,
    JUNETEENTH,
    INDEPENDENCE_DAY,
    LABOR_DAY,
    THANKSGIVING,
    CHRISTMAS,
]


def good_friday_closures(start: date, end: date) -> list[date]:
    """Generate Good Friday closure dates for NYSE within a range."""
    results = []
    for year in range(start.year, end.year + 1):
        gf = good_friday(year)
        if start <= gf <= end:
            results.append(gf)
    return results


# --- Ad-hoc closures (one-time historical events) ---

ADHOC_CLOSURES: list[AdhocClosure] = [
    AdhocClosure(
        name="September 11, 2001",
        dates=[date(2001, 9, 11), date(2001, 9, 12), date(2001, 9, 13), date(2001, 9, 14)],
    ),
    AdhocClosure(
        name="Hurricane Sandy",
        dates=[date(2012, 10, 29), date(2012, 10, 30)],
    ),
    AdhocClosure(name="President Ford National Day of Mourning", dates=[date(2007, 1, 2)]),
    AdhocClosure(name="President Reagan National Day of Mourning", dates=[date(2004, 6, 11)]),
    AdhocClosure(name="President Nixon National Day of Mourning", dates=[date(1994, 4, 27)]),
    AdhocClosure(name="President H.W. Bush National Day of Mourning", dates=[date(2018, 12, 5)]),
    AdhocClosure(name="President Carter National Day of Mourning", dates=[date(2025, 1, 9)]),
]

# --- Early close helpers ---

EARLY_CLOSE_TIME = time(13, 0)  # 1:00 PM

def _compute_black_friday(year: int) -> date:
    """Black Friday = day after Thanksgiving (4th Thursday of November)."""
    thanksgiving = THANKSGIVING._raw_date(year)
    assert thanksgiving is not None
    return thanksgiving + timedelta(days=1)


def _compute_independence_day_early_close(year: int) -> date | None:
    """Compute the early-close date before Independence Day.

    If July 4 falls on Sat (observed Fri Jul 3): early close Thu Jul 2.
    If July 4 falls on Sun (observed Mon Jul 5): early close Fri Jul 2.
    Otherwise: early close on July 3 (if it's a weekday).
    """
    july4 = date(year, 7, 4)
    wd = july4.weekday()

    if wd == 5:  # Saturday
        return date(year, 7, 2)
    if wd == 6:  # Sunday
        return date(year, 7, 2)
    july3 = date(year, 7, 3)
    if july3.weekday() >= 5:
        return date(year, 7, 2)
    return july3


def _compute_christmas_eve_early_close(year: int) -> date | None:
    """Compute Christmas Eve early close. Skip if Dec 24 is a weekend."""
    dec24 = date(year, 12, 24)
    if dec24.weekday() >= 5:
        return None
    return dec24


def independence_day_early_closes(start: date, end: date) -> list[date]:
    """Generate early close dates for day before Independence Day."""
    results = []
    for year in range(start.year, end.year + 1):
        d = _compute_independence_day_early_close(year)
        if d is not None and start <= d <= end:
            results.append(d)
    return results


def black_friday_early_closes(start: date, end: date) -> list[date]:
    """Generate Black Friday early close dates."""
    results = []
    for year in range(start.year, end.year + 1):
        d = _compute_black_friday(year)
        if start <= d <= end:
            results.append(d)
    return results


def christmas_eve_early_closes(start: date, end: date) -> list[date]:
    """Generate Christmas Eve early close dates (post-1993)."""
    results = []
    for year in range(max(start.year, 1993), end.year + 1):
        d = _compute_christmas_eve_early_close(year)
        if d is not None and start <= d <= end:
            results.append(d)
    return results


EARLY_CLOSES: list[EarlyClose] = []

# --- Exchange constants ---

NYSE_OPEN = time(9, 30)
NYSE_CLOSE = time(16, 0)
NYSE_TZ = "America/New_York"
