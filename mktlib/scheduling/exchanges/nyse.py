from __future__ import annotations

from datetime import date, time

from mktlib.scheduling.easter import good_friday
from mktlib.scheduling.rules import (
    AdhocClosure,
    EarlyClose,
    HolidayRule,
    day_after,
    fixed_date_if_weekday,
    holiday_eve,
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
    results: list[date] = []
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

# --- Early closes ---

EARLY_CLOSE_TIME = time(13, 0)  # 1:00 PM

EARLY_CLOSES: list[EarlyClose] = [
    EarlyClose("Independence Day Eve", EARLY_CLOSE_TIME, compute_fn=holiday_eve(7, 4)),
    EarlyClose("Black Friday", EARLY_CLOSE_TIME, compute_fn=day_after(THANKSGIVING)),
    EarlyClose("Christmas Eve", EARLY_CLOSE_TIME, compute_fn=fixed_date_if_weekday(12, 24, start_year=1993)),
]

# --- Exchange constants ---

NYSE_OPEN = time(9, 30)
NYSE_CLOSE = time(16, 0)
NYSE_TZ = "America/New_York"
