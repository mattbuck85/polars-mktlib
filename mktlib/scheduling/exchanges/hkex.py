from __future__ import annotations

from datetime import date, time, timedelta

from mktlib.scheduling.easter import easter, good_friday
from mktlib.scheduling.rules import (
    AdhocClosure,
    EarlyClose,
    HolidayRule,
    fixed_date_if_weekday,
    sunday_to_monday,
)

# --- Hong Kong holidays ---
# HKEX follows Hong Kong public holidays. Many are lunar-calendar based
# and require precomputed date tables.


def _hk_substitute(d: date) -> date:
    """HK substitute: if Sunday, observe on Monday."""
    if d.weekday() == 6:
        return d + timedelta(days=1)
    return d


NEW_YEARS_DAY = HolidayRule(
    name="New Year's Day",
    month=1,
    day=1,
    observance=sunday_to_monday,
)

HKSAR_DAY = HolidayRule(
    name="HKSAR Establishment Day",
    month=7,
    day=1,
    observance=sunday_to_monday,
    start_year=1997,
)

NATIONAL_DAY = HolidayRule(
    name="National Day",
    month=10,
    day=1,
    observance=sunday_to_monday,
    start_year=1997,
)

LABOUR_DAY = HolidayRule(
    name="Labour Day",
    month=5,
    day=1,
    observance=sunday_to_monday,
    start_year=1999,
)

# Christmas and Boxing Day are NOT in RECURRING_HOLIDAYS — they are computed
# in _special_closures to match the HK pattern:
#   christmas()        = Dec 25 if weekday
#   weekend_christmas  = Dec 27 if Mon or Tue (covers Sat/Sun Christmas)
#   boxing_day         = Dec 26 if weekday

RECURRING_HOLIDAYS: list[HolidayRule] = [
    NEW_YEARS_DAY,
    HKSAR_DAY,
    NATIONAL_DAY,
    LABOUR_DAY,
]


# Lunar New Year dates (first day). HK observes 3 days: LNY, LNY+1, LNY+2.
# From 2013: if any of the 3 days falls on Sunday, day 4 is also a holiday.
# Pre-2013: each day individually shifts via _hk_substitute (no day-4 rule).
_LUNAR_NEW_YEAR: dict[int, tuple[int, int]] = {
    1990: (1, 27),
    1991: (2, 15),
    1992: (2, 4),
    1993: (1, 23),
    1994: (2, 10),
    1995: (1, 31),
    1996: (2, 19),
    1997: (2, 7),
    1998: (1, 28),
    1999: (2, 16),
    2000: (2, 5),
    2001: (1, 24),
    2002: (2, 12),
    2003: (2, 1),
    2004: (1, 22),
    2005: (2, 9),
    2006: (1, 29),
    2007: (2, 18),
    2008: (2, 7),
    2009: (1, 26),
    2010: (2, 14),
    2011: (2, 3),
    2012: (1, 23),
    2013: (2, 10),
    2014: (1, 31),
    2015: (2, 19),
    2016: (2, 8),
    2017: (1, 28),
    2018: (2, 16),
    2019: (2, 5),
    2020: (1, 25),
    2021: (2, 12),
    2022: (2, 1),
    2023: (1, 22),
    2024: (2, 10),
    2025: (1, 29),
    2026: (2, 17),
    2027: (2, 6),
    2028: (1, 26),
    2029: (2, 13),
    2030: (2, 3),
}

# Ching Ming (清明) — around April 4-5, solar term
_CHING_MING: dict[int, int] = {
    1990: 5,
    1991: 5,
    1992: 4,
    1993: 5,
    1994: 5,
    1995: 5,
    1996: 4,
    1997: 5,
    1998: 5,
    1999: 5,
    2000: 4,
    2001: 5,
    2002: 5,
    2003: 5,
    2004: 4,
    2005: 5,
    2006: 5,
    2007: 5,
    2008: 4,
    2009: 4,
    2010: 5,
    2011: 5,
    2012: 4,
    2013: 4,
    2014: 5,
    2015: 5,
    2016: 4,
    2017: 4,
    2018: 5,
    2019: 5,
    2020: 4,
    2021: 4,
    2022: 5,
    2023: 5,
    2024: 4,
    2025: 4,
    2026: 5,
    2027: 5,
    2028: 4,
    2029: 4,
    2030: 5,
}

# Buddha's Birthday (佛誕) — 8th day of 4th lunar month
_BUDDHAS_BIRTHDAY: dict[int, tuple[int, int]] = {
    2007: (5, 24),
    2008: (5, 12),
    2009: (5, 2),
    2010: (5, 21),
    2011: (5, 10),
    2012: (4, 28),
    2013: (5, 17),
    2014: (5, 6),
    2015: (5, 25),
    2016: (5, 14),
    2017: (5, 3),
    2018: (5, 22),
    2019: (5, 12),
    2020: (4, 30),
    2021: (5, 19),
    2022: (5, 8),
    2023: (5, 26),
    2024: (5, 15),
    2025: (5, 5),
    2026: (5, 24),
    2027: (5, 13),
    2028: (5, 2),
    2029: (5, 20),
    2030: (5, 9),
}

# Tuen Ng / Dragon Boat Festival (端午) — 5th day of 5th lunar month
_TUEN_NG: dict[int, tuple[int, int]] = {
    2007: (6, 19),
    2008: (6, 8),
    2009: (5, 28),
    2010: (6, 16),
    2011: (6, 6),
    2012: (6, 23),
    2013: (6, 12),
    2014: (6, 2),
    2015: (6, 20),
    2016: (6, 9),
    2017: (5, 30),
    2018: (6, 18),
    2019: (6, 7),
    2020: (6, 25),
    2021: (6, 14),
    2022: (6, 3),
    2023: (6, 22),
    2024: (6, 10),
    2025: (5, 31),
    2026: (6, 19),
    2027: (6, 9),
    2028: (5, 28),
    2029: (6, 16),
    2030: (6, 5),
}

# Mid-Autumn Festival (中秋) — day after the 15th of 8th lunar month.
# Note: when this date falls on Oct 1 (National Day), it is shifted to Oct 2
# so that National Day itself can be observed on Oct 1 (or Oct 2 if observed).
_MID_AUTUMN: dict[int, tuple[int, int]] = {
    2007: (9, 26),
    2008: (9, 15),
    2009: (10, 3),
    2010: (9, 23),
    2011: (9, 13),
    2012: (10, 1),
    2013: (9, 20),
    2014: (9, 9),
    2015: (9, 28),
    2016: (9, 16),
    2017: (10, 5),
    2018: (9, 25),
    2019: (9, 14),
    2020: (10, 2),
    2021: (9, 22),
    2022: (9, 12),
    2023: (9, 30),
    2024: (9, 18),
    2025: (10, 7),
    2026: (9, 26),
    2027: (9, 16),
    2028: (10, 4),
    2029: (9, 23),
    2030: (9, 13),
}

# Chung Yeung Festival (重陽) — 9th day of 9th lunar month
_CHUNG_YEUNG: dict[int, tuple[int, int]] = {
    2007: (10, 19),
    2008: (10, 7),
    2009: (10, 26),
    2010: (10, 16),
    2011: (10, 5),
    2012: (10, 23),
    2013: (10, 13),
    2014: (10, 2),
    2015: (10, 21),
    2016: (10, 9),
    2017: (10, 28),
    2018: (10, 17),
    2019: (10, 7),
    2020: (10, 26),
    2021: (10, 14),
    2022: (10, 4),
    2023: (10, 23),
    2024: (10, 11),
    2025: (10, 29),
    2026: (10, 18),
    2027: (10, 8),
    2028: (10, 26),
    2029: (10, 16),
    2030: (10, 5),
}


def _lny_closures(year: int) -> list[date]:
    """Return the set of closure dates for Lunar New Year in the given year.

    HK observes LNY day 1, 2, and 3. Each individual day applies
    _hk_substitute (Sun -> Mon). For years >= 2013, if any of the 3 LNY days
    falls on a Sunday, a 4th day of closure is added (ordinance 2011).
    """
    lny = _LUNAR_NEW_YEAR.get(year)
    if lny is None:
        return []

    day1 = date(year, lny[0], lny[1])
    days = [day1 + timedelta(days=i) for i in range(3)]

    closures: set[date] = set()
    for d in days:
        closures.add(_hk_substitute(d))

    # From 2013: if any LNY day is Sunday, the 4th day is also a holiday
    if year >= 2013 and any(d.weekday() == 6 for d in days):
        closures.add(day1 + timedelta(days=3))

    return sorted(closures)


def _mid_autumn_closure(year: int) -> date | None:
    """Return the Mid-Autumn closure date, accounting for National Day collision.

    When the day-after-mid-autumn falls on October 1 (National Day), the
    mid-autumn closure is shifted to October 2 instead (so National Day can
    be observed on Oct 1, and mid-autumn on Oct 2).
    """
    ma = _MID_AUTUMN.get(year)
    if ma is None:
        return None
    d = date(year, ma[0], ma[1])
    # If mid-autumn lands on Oct 1 (National Day), shift to Oct 2
    if d.month == 10 and d.day == 1:
        d = date(year, 10, 2)
    return _hk_substitute(d)


def _christmas_closures(year: int) -> list[date]:
    """Compute Christmas and Boxing Day closures for HK.

    Rules mirroring exchange_calendars XHKG:
    - christmas()        = Dec 25 if weekday, else skipped
    - weekend_christmas  = Dec 27 if it falls on Mon or Tue (covers Sat/Sun Christmas)
    - boxing_day         = Dec 26 if weekday, else skipped

    This yields:
    - Dec 25 normal (Tue-Fri): Dec 25 + Dec 26
    - Dec 25 Mon: Dec 25 + Dec 26
    - Dec 25 Sat: Dec 27 only (weekend_christmas)
    - Dec 25 Sun: Dec 26 + Dec 27 (boxing on Mon, weekend_christmas on Tue)
    """
    closures: set[date] = set()
    dec25 = date(year, 12, 25)
    dec26 = date(year, 12, 26)
    dec27 = date(year, 12, 27)

    # Christmas: Dec 25 if weekday
    if dec25.weekday() < 5:
        closures.add(dec25)

    # Boxing Day: Dec 26 if weekday
    if dec26.weekday() < 5:
        closures.add(dec26)

    # Weekend Christmas: Dec 27 only when it falls on Mon (Christmas=Sat) or Tue (Christmas=Sun)
    if dec27.weekday() in (0, 1):
        closures.add(dec27)

    return sorted(closures)


def _ching_ming_closure(year: int, easter_monday: date) -> date | None:
    """Ching Ming with Sunday observance and Easter Monday collision handling.

    When Ching Ming (observed) falls on Easter Monday, it is shifted one day
    later to avoid collision.
    """
    cm_day = _CHING_MING.get(year)
    if cm_day is None:
        return None
    d = _hk_substitute(date(year, 4, cm_day))
    if d == easter_monday:
        d = d + timedelta(days=1)
    return d


def _special_closures(start: date, end: date) -> list[date]:
    """Lunar/solar-term holidays + Good Friday + Easter Monday + Christmas."""
    results: list[date] = []

    for year in range(start.year, end.year + 1):
        # Good Friday
        gf = good_friday(year)
        if start <= gf <= end:
            results.append(gf)

        # Easter Monday
        em = easter(year) + timedelta(days=1)
        if start <= em <= end:
            results.append(em)

        # Lunar New Year: 3 (or 4) days with Sunday observance
        for d in _lny_closures(year):
            if start <= d <= end:
                results.append(d)

        # Ching Ming (with Easter Monday collision shift)
        cm = _ching_ming_closure(year, em)
        if cm is not None and start <= cm <= end:
            results.append(cm)

        # Buddha's Birthday
        bb = _BUDDHAS_BIRTHDAY.get(year)
        if bb is not None:
            d = _hk_substitute(date(year, bb[0], bb[1]))
            if start <= d <= end:
                results.append(d)

        # Tuen Ng (Dragon Boat)
        tn = _TUEN_NG.get(year)
        if tn is not None:
            d = _hk_substitute(date(year, tn[0], tn[1]))
            if start <= d <= end:
                results.append(d)

        # Mid-Autumn Festival (with National Day collision handling)
        ma_date = _mid_autumn_closure(year)
        if ma_date is not None and start <= ma_date <= end:
            results.append(ma_date)

        # Chung Yeung
        cy = _CHUNG_YEUNG.get(year)
        if cy is not None:
            d = _hk_substitute(date(year, cy[0], cy[1]))
            if start <= d <= end:
                results.append(d)

        # Christmas and Boxing Day
        for d in _christmas_closures(year):
            if start <= d <= end:
                results.append(d)

    return results


special_closures = _special_closures

# --- Ad-hoc closures ---
# Typhoon signal No.8+ / Black Rainstorm / special event closures (historical).
# Only full-day closures confirmed by exchange_calendars are included.
# Partial-day (morning-only) closures are excluded.

ADHOC_CLOSURES: list[AdhocClosure] = [
    AdhocClosure(
        name="Typhoon Bilis & Kaemi 2006-08-06", dates=[date(2008, 8, 6)]
    ),
    AdhocClosure(name="Typhoon Nuri 2008-08-22", dates=[date(2008, 8, 22)]),
    AdhocClosure(name="Typhoon Nasha 2011-09-29", dates=[date(2011, 9, 29)]),
    AdhocClosure(name="Typhoon Utor 2013-08-14", dates=[date(2013, 8, 14)]),
    AdhocClosure(
        name="WWII 70th Anniversary 2015-09-03", dates=[date(2015, 9, 3)]
    ),
    AdhocClosure(name="Typhoon Nida 2016-08-02", dates=[date(2016, 8, 2)]),
    AdhocClosure(name="Typhoon Haima 2016-10-21", dates=[date(2016, 10, 21)]),
    AdhocClosure(name="Typhoon Hato 2017-08-23", dates=[date(2017, 8, 23)]),
    AdhocClosure(name="Typhoon Nangka 2020-10-13", dates=[date(2020, 10, 13)]),
    AdhocClosure(
        name="Typhoon Kompasu 2021-10-13", dates=[date(2021, 10, 13)]
    ),
    AdhocClosure(name="Typhoon Talim 2023-07-17", dates=[date(2023, 7, 17)]),
    AdhocClosure(name="Typhoon Yagi 2024-09-06", dates=[date(2024, 9, 6)]),
]

# --- Early closes ---
# HK early closes: Christmas Eve, New Year's Eve, Lunar New Year's Eve at 12:00 noon

HKEX_EARLY_CLOSE_TIME = time(12, 0)


def _lny_eve(year: int) -> date | None:
    """Day before Lunar New Year — early close if weekday."""
    lny = _LUNAR_NEW_YEAR.get(year)
    if lny is None:
        return None
    day1 = date(year, lny[0], lny[1])
    eve = day1 - timedelta(days=1)
    if eve.weekday() >= 5:
        return None
    return eve


EARLY_CLOSES: list[EarlyClose] = [
    EarlyClose(
        "Christmas Eve",
        HKEX_EARLY_CLOSE_TIME,
        compute_fn=fixed_date_if_weekday(12, 24),
    ),
    EarlyClose(
        "New Year's Eve",
        HKEX_EARLY_CLOSE_TIME,
        compute_fn=fixed_date_if_weekday(12, 31),
    ),
    EarlyClose(
        "Lunar New Year's Eve", HKEX_EARLY_CLOSE_TIME, compute_fn=_lny_eve
    ),
]

# --- Exchange constants ---

HKEX_OPEN = time(9, 30)
HKEX_CLOSE = time(16, 0)
HKEX_BREAK_START = time(12, 0)
HKEX_BREAK_END = time(13, 0)
HKEX_TZ = "Asia/Hong_Kong"
