from __future__ import annotations

from datetime import date, time, timedelta

from mktlib.scheduling.easter import easter, good_friday
from mktlib.scheduling.rules import (
    AdhocClosure,
    EarlyClose,
    HolidayRule,
    last_weekday_before,
)


def _uk_substitute(d: date) -> date:
    """UK bank holiday substitution: if weekend, next Monday."""
    if d.weekday() == 5:  # Saturday → Monday
        return d + timedelta(days=2)
    if d.weekday() == 6:  # Sunday → Monday
        return d + timedelta(days=1)
    return d


# --- Recurring holidays ---

NEW_YEARS_DAY = HolidayRule(
    name="New Year's Day",
    month=1,
    day=1,
    observance=_uk_substitute,
)

EARLY_MAY_BANK_HOLIDAY = HolidayRule(
    name="Early May Bank Holiday",
    month=5,
    weekday=0,  # Monday
    week=1,
)

SPRING_BANK_HOLIDAY = HolidayRule(
    name="Spring Bank Holiday",
    month=5,
    weekday=0,  # Monday
    week=-1,  # Last Monday
)

SUMMER_BANK_HOLIDAY = HolidayRule(
    name="Summer Bank Holiday",
    month=8,
    weekday=0,  # Monday
    week=-1,  # Last Monday
)

CHRISTMAS = HolidayRule(
    name="Christmas Day",
    month=12,
    day=25,
    observance=_uk_substitute,
)


def _boxing_day_observance(d: date) -> date:
    """Boxing Day UK observance: next available weekday after Christmas substitute."""
    christmas_observed = _uk_substitute(date(d.year, 12, 25))
    boxing_observed = _uk_substitute(d)
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
    EARLY_MAY_BANK_HOLIDAY,
    SPRING_BANK_HOLIDAY,
    SUMMER_BANK_HOLIDAY,
    CHRISTMAS,
    BOXING_DAY,
]


def special_closures(start: date, end: date) -> list[date]:
    """Good Friday + Easter Monday closures for LSE."""
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

ADHOC_CLOSURES: list[AdhocClosure] = [
    AdhocClosure(
        name="Queen Elizabeth II Funeral",
        dates=[date(2022, 9, 19)],
    ),
    AdhocClosure(
        name="Queen's Diamond Jubilee",
        dates=[date(2012, 6, 4), date(2012, 6, 5)],
    ),
    AdhocClosure(
        name="Platinum Jubilee Extra Bank Holiday",
        dates=[date(2022, 6, 3)],
    ),
    AdhocClosure(
        name="Royal Wedding 2011",
        dates=[date(2011, 4, 29)],
    ),
    AdhocClosure(
        name="King's Coronation",
        dates=[date(2023, 5, 8)],
    ),
    # Spring Bank Holiday moved for Diamond Jubilee (2012), Platinum Jubilee (2022)
    # Early May Bank Holiday moved for VE Day 75th (2020)
    # These moves are handled by un-closing the original date:
]

# Dates where normally-recurring bank holidays were moved away
BANK_HOLIDAY_MOVES: dict[date, str] = {
    date(2012, 5, 28): "Spring Bank Holiday moved to June 4 for Diamond Jubilee",
    date(2020, 5, 4): "Early May Bank Holiday moved to May 8 for VE Day 75th",
    date(2022, 5, 30): "Spring Bank Holiday moved to June 2 for Platinum Jubilee",
}

# Extra closure dates from moved bank holidays (the new dates)
MOVED_HOLIDAY_CLOSURES: list[AdhocClosure] = [
    AdhocClosure(name="VE Day 75th Anniversary", dates=[date(2020, 5, 8)]),
    AdhocClosure(name="Spring Bank Holiday (Platinum Jubilee)", dates=[date(2022, 6, 2)]),
]


def special_closures_with_moves(start: date, end: date) -> list[date]:
    """Good Friday + Easter Monday + moved bank holidays, minus original moved dates."""
    results = special_closures(start, end)
    for adhoc in MOVED_HOLIDAY_CLOSURES:
        for d in adhoc.dates:
            if start <= d <= end:
                results.append(d)
    return results


# --- Early closes ---

LSE_EARLY_CLOSE_TIME = time(12, 30)

EARLY_CLOSES: list[EarlyClose] = [
    EarlyClose("Christmas Eve", LSE_EARLY_CLOSE_TIME, compute_fn=last_weekday_before(12, 25)),
    EarlyClose("New Year's Eve", LSE_EARLY_CLOSE_TIME, compute_fn=last_weekday_before(1, 1, year_offset=1)),
]

# --- Exchange constants ---

LSE_OPEN = time(8, 0)
LSE_CLOSE = time(16, 30)
LSE_TZ = "Europe/London"
