from __future__ import annotations

from datetime import date, time, timedelta

from mktlib.scheduling.rules import AdhocClosure, EarlyClose, HolidayRule

# --- Recurring Japanese holidays ---
# Japan does not use weekend observance for most holidays — when a holiday
# falls on Sunday, the following Monday is a substitute holiday ("furikae kyūjitsu").


def _sunday_to_monday(d: date) -> date:
    """Japanese substitute holiday: if Sunday, observe on Monday."""
    if d.weekday() == 6:
        return d + timedelta(days=1)
    return d


NEW_YEARS_DAY_1 = HolidayRule(name="New Year's Day", month=1, day=1)
NEW_YEARS_DAY_2 = HolidayRule(name="New Year's Day (Jan 2)", month=1, day=2)
NEW_YEARS_DAY_3 = HolidayRule(name="New Year's Day (Jan 3)", month=1, day=3)

COMING_OF_AGE_DAY = HolidayRule(
    name="Coming of Age Day",
    month=1,
    weekday=0,
    week=2,
)

NATIONAL_FOUNDATION_DAY = HolidayRule(
    name="National Foundation Day",
    month=2,
    day=11,
    observance=_sunday_to_monday,
)

EMPERORS_BIRTHDAY = HolidayRule(
    name="Emperor's Birthday",
    month=2,
    day=23,
    observance=_sunday_to_monday,
    start_year=2020,
)

# Heisei Emperor's Birthday (Akihito): Dec 23, 1990–2018.
# From 2019 onward the old date is no longer a holiday (abdication Apr 2019).
EMPEROR_BIRTHDAY_OLD = HolidayRule(
    name="Emperor's Birthday (Showa/Heisei)",
    month=12,
    day=23,
    observance=_sunday_to_monday,
    end_year=2018,
)

# Showa Emperor's Birthday (before 2019, Apr 29 was already Showa Day)
SHOWA_DAY = HolidayRule(
    name="Showa Day",
    month=4,
    day=29,
    observance=_sunday_to_monday,
)

CONSTITUTION_DAY = HolidayRule(
    name="Constitution Memorial Day",
    month=5,
    day=3,
    observance=_sunday_to_monday,
)

GREENERY_DAY = HolidayRule(
    name="Greenery Day",
    month=5,
    day=4,
    observance=_sunday_to_monday,
)

CHILDRENS_DAY = HolidayRule(
    name="Children's Day",
    month=5,
    day=5,
    observance=_sunday_to_monday,
)

MARINE_DAY = HolidayRule(
    name="Marine Day",
    month=7,
    weekday=0,
    week=3,
)

MOUNTAIN_DAY = HolidayRule(
    name="Mountain Day",
    month=8,
    day=11,
    observance=_sunday_to_monday,
    start_year=2016,
)

RESPECT_FOR_AGED_DAY = HolidayRule(
    name="Respect for the Aged Day",
    month=9,
    weekday=0,
    week=3,
)

SPORTS_DAY = HolidayRule(
    name="Sports Day",
    month=10,
    weekday=0,
    week=2,
)

CULTURE_DAY = HolidayRule(
    name="Culture Day",
    month=11,
    day=3,
    observance=_sunday_to_monday,
)

LABOUR_THANKSGIVING = HolidayRule(
    name="Labour Thanksgiving Day",
    month=11,
    day=23,
    observance=_sunday_to_monday,
)


# Vernal/Autumnal equinox dates vary — precomputed table is most reliable.
# These are determined by the National Astronomical Observatory of Japan.
_VERNAL_EQUINOX: dict[int, int] = {
    1990: 21, 1991: 21, 1992: 20, 1993: 20, 1994: 21, 1995: 21, 1996: 20,
    1997: 20, 1998: 21, 1999: 21, 2000: 20, 2001: 20, 2002: 21, 2003: 21,
    2004: 20, 2005: 20, 2006: 21, 2007: 21, 2008: 20, 2009: 20, 2010: 21,
    2011: 21, 2012: 20, 2013: 20, 2014: 21, 2015: 21, 2016: 20, 2017: 20,
    2018: 21, 2019: 21, 2020: 20, 2021: 20, 2022: 21, 2023: 21, 2024: 20,
    2025: 20, 2026: 20, 2027: 21, 2028: 20, 2029: 20, 2030: 20,
}

_AUTUMNAL_EQUINOX: dict[int, int] = {
    1990: 23, 1991: 23, 1992: 23, 1993: 23, 1994: 23, 1995: 23, 1996: 22,
    1997: 23, 1998: 23, 1999: 23, 2000: 23, 2001: 23, 2002: 23, 2003: 23,
    2004: 23, 2005: 23, 2006: 23, 2007: 23, 2008: 23, 2009: 23, 2010: 23,
    2011: 23, 2012: 22, 2013: 23, 2014: 23, 2015: 23, 2016: 22, 2017: 23,
    2018: 23, 2019: 23, 2020: 22, 2021: 23, 2022: 23, 2023: 23, 2024: 22,
    2025: 23, 2026: 23, 2027: 23, 2028: 22, 2029: 23, 2030: 23,
}

# Olympic years: dates that should be excluded from regular holiday generation
# because the holidays were moved to different dates.
# 2020 Olympics: Marine Day (3rd Mon Jul = Jul 20), Sports Day (2nd Mon Oct = Oct 12),
#                Mountain Day (Aug 11) → all moved; exclude their normal dates.
# 2021 Olympics: Marine Day (3rd Mon Jul = Jul 19), Sports Day (2nd Mon Oct = Oct 11),
#                Mountain Day (Aug 11) → all moved; exclude their normal dates.
JPX_EXCLUSIONS: set[date] = {
    # 2020 Olympics — normal holiday dates that were moved
    date(2020, 7, 20),   # Marine Day (3rd Mon Jul)
    date(2020, 8, 11),   # Mountain Day
    date(2020, 10, 12),  # Sports Day (2nd Mon Oct)
    # 2021 Olympics — normal holiday dates that were moved
    date(2021, 7, 19),   # Marine Day (3rd Mon Jul)
    date(2021, 8, 11),   # Mountain Day
    date(2021, 10, 11),  # Sports Day (2nd Mon Oct)
}


def _special_closures(start: date, end: date) -> list[date]:
    """Equinox days, Dec 31, Olympics moved holidays, and substitute holidays."""
    results: list[date] = []
    for year in range(start.year, end.year + 1):
        # Vernal Equinox Day
        ve_day = _VERNAL_EQUINOX.get(year)
        if ve_day is not None:
            d = _sunday_to_monday(date(year, 3, ve_day))
            if start <= d <= end:
                results.append(d)

        # Autumnal Equinox Day
        ae_day = _AUTUMNAL_EQUINOX.get(year)
        if ae_day is not None:
            d = _sunday_to_monday(date(year, 9, ae_day))
            if start <= d <= end:
                results.append(d)

        # Dec 31 — exchange rule, not a national holiday
        dec31 = date(year, 12, 31)
        if dec31.weekday() < 5 and start <= dec31 <= end:
            results.append(dec31)

        # Citizen's Holiday (国民の休日): when a single weekday is sandwiched
        # between two holidays, it becomes a holiday too.
        # Most common: Sep when Respect for Aged Day (Mon) and Autumnal Equinox (Wed)
        # leave Tuesday as a bridge holiday.
        if ae_day is not None:
            ae = date(year, 9, ae_day)
            # Respect for Aged Day is 3rd Monday of September
            from mktlib.scheduling.rules import _nth_weekday
            raa = _nth_weekday(year, 9, 0, 3)
            gap = (ae - raa).days
            if gap == 2:
                bridge = raa + timedelta(days=1)
                if bridge.weekday() < 5 and start <= bridge <= end:
                    results.append(bridge)

        # --- Olympics: moved holiday dates ---
        # 2020: Marine Day → Jul 23, Sports Day → Jul 24, Mountain Day → Aug 10
        if year == 2020:
            for moved in (date(2020, 7, 23), date(2020, 7, 24), date(2020, 8, 10)):
                if start <= moved <= end:
                    results.append(moved)

        # 2021: Marine Day → Jul 22, Sports Day → Jul 23, Mountain Day → Aug 9
        if year == 2021:
            for moved in (date(2021, 7, 22), date(2021, 7, 23), date(2021, 8, 9)):
                if start <= moved <= end:
                    results.append(moved)

        # --- Golden Week cascade substitute ---
        # Standard _sunday_to_monday on Constitution Day (May 3), Greenery Day (May 4),
        # and Children's Day (May 5) handles the simple case. However, when May 3 or
        # May 4 falls on Sunday the Monday substitute may itself already be a holiday,
        # so the substitute cascades to the next available weekday (May 6 or later).
        #
        # Rule applied here: if any of May 3/4/5 is Sunday and the resulting Monday
        # is already another Golden Week holiday, the substitute moves to May 6.
        #
        # Known cases:
        #   2020: May 3 = Sun → sub Mon May 4, but May 4 is Greenery Day → sub May 6 (Wed)
        #   2025: May 4 = Sun → sub Mon May 5, but May 5 is Children's Day → sub May 6 (Tue)
        may3 = date(year, 5, 3)
        may4 = date(year, 5, 4)
        may5 = date(year, 5, 5)
        gw_sub: date | None = None
        if may3.weekday() == 6:
            # May 3 (Sun) → sub would be May 4 (Mon), but May 4 is Greenery Day → sub May 6
            # (May 5 is Children's Day, so we cascade past it too if needed)
            candidate = may4
            while candidate in (may4, may5) or candidate.weekday() >= 5:
                candidate = candidate + timedelta(days=1)
            if candidate != _sunday_to_monday(may3):
                gw_sub = candidate
        elif may4.weekday() == 6:
            # May 4 (Sun) → sub would be May 5 (Mon), but May 5 is Children's Day → sub May 6
            candidate = may5
            while candidate in (may5,) or candidate.weekday() >= 5:
                candidate = candidate + timedelta(days=1)
            if candidate != _sunday_to_monday(may4):
                gw_sub = candidate
        # May 5 Sunday → sub May 6 Monday (no other holiday on May 6, simple case handled
        # by _sunday_to_monday on CHILDRENS_DAY rule — no cascade needed)

        if gw_sub is not None and start <= gw_sub <= end:
            results.append(gw_sub)

    return results


RECURRING_HOLIDAYS: list[HolidayRule] = [
    NEW_YEARS_DAY_1,
    NEW_YEARS_DAY_2,
    NEW_YEARS_DAY_3,
    COMING_OF_AGE_DAY,
    NATIONAL_FOUNDATION_DAY,
    EMPERORS_BIRTHDAY,
    EMPEROR_BIRTHDAY_OLD,
    SHOWA_DAY,
    CONSTITUTION_DAY,
    GREENERY_DAY,
    CHILDRENS_DAY,
    MARINE_DAY,
    MOUNTAIN_DAY,
    RESPECT_FOR_AGED_DAY,
    SPORTS_DAY,
    CULTURE_DAY,
    LABOUR_THANKSGIVING,
]

special_closures = _special_closures

# --- Ad-hoc closures ---
# Historical one-off closures (Emperor transitions, etc.)
# Note: 2020/2021 Olympics moved holidays are handled in _special_closures,
# with the original dates excluded via JPX_EXCLUSIONS (passed to ExchangeCalendar).

ADHOC_CLOSURES: list[AdhocClosure] = [
    # Emperor Akihito abdication / Emperor Naruhito enthronement 2019
    AdhocClosure(name="Emperor Abdication Day", dates=[date(2019, 4, 30)]),
    AdhocClosure(name="Emperor Enthronement Day", dates=[date(2019, 5, 1)]),
    AdhocClosure(name="Enthronement Ceremony", dates=[date(2019, 10, 22)]),
    # 2019 bridge holiday: May 2 sandwiched between May 1 (enthronement) and May 3 (Constitution)
    AdhocClosure(name="Citizen's Holiday (2019 enthronement bridge)", dates=[date(2019, 5, 2)]),
    # Oct 1, 2020: exchange closure (aligns with exchange_calendars XTKS)
    AdhocClosure(name="JPX adhoc closure 2020-10-01", dates=[date(2020, 10, 1)]),
]

# --- Early closes ---
# JPX has no regular early closes.

EARLY_CLOSES: list[EarlyClose] = []

# --- Exchange constants ---

JPX_OPEN = time(9, 0)
JPX_CLOSE = time(15, 0)
JPX_BREAK_START = time(11, 30)
JPX_BREAK_END = time(12, 30)
JPX_TZ = "Asia/Tokyo"
