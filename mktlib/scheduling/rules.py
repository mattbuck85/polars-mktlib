from __future__ import annotations

import calendar
from dataclasses import dataclass, field
from datetime import date, time, timedelta
from typing import Callable


@dataclass(frozen=True)
class HolidayRule:
    """A recurring holiday rule.

    Either (month, day) for fixed-date holidays or (month, weekday, week)
    for relative holidays (e.g., 3rd Monday of January).
    """

    name: str
    month: int
    day: int | None = None
    weekday: int | None = None  # 0=Mon..6=Sun
    week: int | None = None  # 1-5 for Nth occurrence, -1 for last
    observance: Callable[[date], date] | None = None
    start_year: int | None = None
    end_year: int | None = None

    def dates_in_range(self, start: date, end: date) -> list[date]:
        """Generate all observed dates for this rule within [start, end].

        Checks year-1 through year+1 to catch cross-year observances
        (e.g., New Year's on Saturday observed as previous Friday Dec 31).
        """
        results: list[date] = []
        for year in range(start.year - 1, end.year + 2):
            d = self.raw_date(year)
            if d is None:
                continue
            if self.observance is not None:
                d = self.observance(d)
            if start <= d <= end:
                results.append(d)
        return results

    def raw_date(self, year: int) -> date | None:
        if self.start_year is not None and year < self.start_year:
            return None
        if self.end_year is not None and year > self.end_year:
            return None

        if self.day is not None:
            return date(year, self.month, self.day)

        if self.weekday is not None and self.week is not None:
            return _nth_weekday(year, self.month, self.weekday, self.week)

        return None


@dataclass(frozen=True)
class AdhocClosure:
    """An explicit list of closure dates (e.g., 9/11, Hurricane Sandy)."""

    name: str
    dates: list[date] = field(default_factory=lambda: [])


@dataclass(frozen=True)
class EarlyClose:
    """An early close rule — either rule-based, ad-hoc dates, or compute_fn."""

    name: str
    close_time: time
    rule: HolidayRule | None = None
    dates: list[date] = field(default_factory=lambda: [])
    compute_fn: Callable[[int], date | None] | None = None

    def dates_in_range(self, start: date, end: date) -> list[date]:
        """All early-close dates within [start, end]."""
        results: list[date] = []
        if self.rule is not None:
            results.extend(self.rule.dates_in_range(start, end))
        if self.compute_fn is not None:
            for year in range(start.year, end.year + 1):
                d = self.compute_fn(year)
                if d is not None and start <= d <= end:
                    results.append(d)
        for d in self.dates:
            if start <= d <= end and d not in results:
                results.append(d)
        return results


# --- Observance functions ---


def nearest_workday(d: date) -> date:
    """If Saturday, use Friday. If Sunday, use Monday."""
    if d.weekday() == 5:  # Saturday
        return d - timedelta(days=1)
    if d.weekday() == 6:  # Sunday
        return d + timedelta(days=1)
    return d


def sunday_to_monday(d: date) -> date:
    """If Sunday, observe on Monday."""
    if d.weekday() == 6:
        return d + timedelta(days=1)
    return d


def previous_friday(d: date) -> date:
    """Move to the previous Friday."""
    offset = (d.weekday() - 4) % 7
    return d - timedelta(days=offset) if offset else d


# --- Helpers ---


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """Find the Nth (1-5) or last (-1) occurrence of a weekday in a month.

    weekday: 0=Monday, 6=Sunday
    """
    if n == -1:
        last_day = calendar.monthrange(year, month)[1]
        d = date(year, month, last_day)
        while d.weekday() != weekday:
            d = d - timedelta(days=1)
        return d

    first = date(year, month, 1)
    diff = (weekday - first.weekday()) % 7
    first_occurrence = date(year, month, 1 + diff)
    return first_occurrence + timedelta(weeks=n - 1)


# --- Early-close compute_fn factories ---


def weekday_before(d: date) -> date:
    """Last weekday strictly before a date."""
    d = d - timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def holiday_eve(month: int, day: int) -> Callable[[int], date | None]:
    """Weekday before holiday; None if holiday falls on Sat/Sun/Mon.

    When July 4 is Sat/Sun, the observed holiday shifts to Fri/Mon — no eve.
    When July 4 is Mon, the closure itself gives a long weekend — no eve.
    Tue-Fri: early close on the weekday immediately before the holiday.
    """

    def _compute(year: int) -> date | None:
        holiday = date(year, month, day)
        if holiday.weekday() >= 5 or holiday.weekday() == 0:
            return None
        return weekday_before(holiday)

    return _compute


def fixed_date_if_weekday(
    month: int, day: int, *, start_year: int | None = None
) -> Callable[[int], date | None]:
    """Fixed date if it's a weekday; None if weekend."""

    def _compute(year: int) -> date | None:
        if start_year is not None and year < start_year:
            return None
        d = date(year, month, day)
        if d.weekday() >= 5:
            return None
        return d

    return _compute


def day_after(rule: HolidayRule) -> Callable[[int], date | None]:
    """Day after a HolidayRule's raw date (e.g., Black Friday)."""

    def _compute(year: int) -> date | None:
        d = rule.raw_date(year)
        if d is None:
            return None
        return d + timedelta(days=1)

    return _compute


def last_weekday_before(
    month: int, day: int, *, year_offset: int = 0
) -> Callable[[int], date]:
    """Always find the last weekday strictly before a date.

    year_offset=1 means the target date is in year+1 (e.g., Jan 1 of next year
    for New Year's Eve early close).
    """

    def _compute(year: int) -> date:
        target = date(year + year_offset, month, day)
        return weekday_before(target)

    return _compute
