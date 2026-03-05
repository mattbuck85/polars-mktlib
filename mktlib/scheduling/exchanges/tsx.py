from __future__ import annotations

from datetime import date, time, timedelta

from mktlib.scheduling.easter import good_friday
from mktlib.scheduling.rules import (
    AdhocClosure,
    EarlyClose,
    HolidayRule,
    fixed_date_if_weekday,
)


def _weekend_to_monday(d: date) -> date:
    """Canadian statutory observance: Saturday or Sunday → following Monday."""
    if d.weekday() == 5:  # Saturday → Monday
        return d + timedelta(days=2)
    if d.weekday() == 6:  # Sunday → Monday
        return d + timedelta(days=1)
    return d


# --- Recurring Canadian holidays ---

NEW_YEARS_DAY = HolidayRule(
    name="New Year's Day",
    month=1,
    day=1,
    observance=_weekend_to_monday,
)

FAMILY_DAY = HolidayRule(
    name="Family Day",
    month=2,
    weekday=0,  # 3rd Monday of February
    week=3,
    start_year=2008,
)

# Victoria Day: Monday on or before May 24 — handled via special_closures_fn
# since it can't be expressed as "Nth weekday of month".

CANADA_DAY = HolidayRule(
    name="Canada Day",
    month=7,
    day=1,
    observance=_weekend_to_monday,
)

CIVIC_HOLIDAY = HolidayRule(
    name="Civic Holiday",
    month=8,
    weekday=0,  # 1st Monday of August
    week=1,
)

LABOUR_DAY = HolidayRule(
    name="Labour Day",
    month=9,
    weekday=0,
    week=1,
)

THANKSGIVING = HolidayRule(
    name="Thanksgiving",
    month=10,
    weekday=0,  # 2nd Monday of October
    week=2,
)

CHRISTMAS = HolidayRule(
    name="Christmas Day",
    month=12,
    day=25,
    observance=_weekend_to_monday,
)


def _boxing_day_observance(d: date) -> date:
    """Canadian Boxing Day: next available weekday after Christmas observance."""
    christmas_observed = _weekend_to_monday(date(d.year, 12, 25))
    boxing_observed = _weekend_to_monday(d)
    if boxing_observed <= christmas_observed:
        return christmas_observed + timedelta(days=1)
    return boxing_observed


BOXING_DAY = HolidayRule(
    name="Boxing Day",
    month=12,
    day=26,
    observance=_boxing_day_observance,
)


RECURRING_HOLIDAYS: list[HolidayRule] = [
    NEW_YEARS_DAY,
    FAMILY_DAY,
    # Victoria Day handled via special_closures_fn
    CANADA_DAY,
    CIVIC_HOLIDAY,
    LABOUR_DAY,
    THANKSGIVING,
    CHRISTMAS,
    BOXING_DAY,
]


def special_closures(start: date, end: date) -> list[date]:
    """Good Friday + Victoria Day closures."""
    results: list[date] = []
    for year in range(start.year, end.year + 1):
        # Good Friday
        gf = good_friday(year)
        if start <= gf <= end:
            results.append(gf)

        # Victoria Day: Monday on or before May 24
        d = date(year, 5, 24)
        while d.weekday() != 0:
            d -= timedelta(days=1)
        if start <= d <= end:
            results.append(d)

    return results


# --- Ad-hoc closures ---

ADHOC_CLOSURES: list[AdhocClosure] = [
    AdhocClosure(
        name="September 11, 2001",
        dates=[
            date(2001, 9, 11),
            date(2001, 9, 12),
            date(2001, 9, 13),
            date(2001, 9, 14),
        ],
    ),
]

# --- Early closes ---

TSX_EARLY_CLOSE_TIME = time(13, 0)

EARLY_CLOSES: list[EarlyClose] = [
    EarlyClose(
        "Christmas Eve",
        TSX_EARLY_CLOSE_TIME,
        compute_fn=fixed_date_if_weekday(12, 24),
    ),
]

# --- Exchange constants ---

TSX_OPEN = time(9, 30)
TSX_CLOSE = time(16, 0)
TSX_TZ = "America/Toronto"
