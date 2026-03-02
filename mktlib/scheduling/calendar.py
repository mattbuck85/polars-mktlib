from __future__ import annotations

from datetime import date, datetime, time, timedelta
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

import polars as pl

from mktlib.scheduling._mixins import MinuteQueryMixin, SessionNavigationMixin, TradingIndexMixin
from mktlib.scheduling._types import MarketDailySchedule, parse_date
from mktlib.scheduling.rules import AdhocClosure, EarlyClose, HolidayRule

if TYPE_CHECKING:
    from collections.abc import Callable


class ExchangeCalendar(SessionNavigationMixin, MinuteQueryMixin, TradingIndexMixin):
    """Polars-native exchange calendar with holiday/early-close support."""

    def __init__(
        self,
        name: str,
        *,
        timezone: str,
        open_time: time,
        close_time: time,
        holidays: list[HolidayRule],
        adhoc_closures: list[AdhocClosure] | None = None,
        early_closes: list[EarlyClose] | None = None,
        special_closures_fn: Callable[[date, date], list[date]] | None = None,
        special_early_closes_fn: Callable[[date, date], dict[date, time]] | None = None,
        exclusions: set[date] | None = None,
        open_offset: int = 0,
        break_start: time | None = None,
        break_end: time | None = None,
    ):
        self.name = name
        self.timezone = timezone
        self.tz = ZoneInfo(timezone)
        self.open_time = open_time
        self.close_time = close_time
        self.holidays = holidays
        self.adhoc_closures = adhoc_closures or []
        self.early_closes = early_closes or []
        self._special_closures_fn = special_closures_fn
        self._special_early_closes_fn = special_early_closes_fn
        self._exclusions = exclusions or set()
        self.open_offset = open_offset
        self.break_start = break_start
        self.break_end = break_end

    def _closure_dates(self, start: date, end: date) -> set[date]:
        """Collect all closure dates (holidays + adhoc) within [start, end]."""
        closures: set[date] = set()
        for rule in self.holidays:
            closures.update(rule.dates_in_range(start, end))
        for adhoc in self.adhoc_closures:
            for d in adhoc.dates:
                if start <= d <= end:
                    closures.add(d)
        if self._special_closures_fn is not None:
            closures.update(self._special_closures_fn(start, end))
        closures -= self._exclusions
        return closures

    def _early_close_map(self, start: date, end: date) -> dict[date, time]:
        """Map of date -> early close time within [start, end]."""
        ec_map: dict[date, time] = {}
        for ec in self.early_closes:
            for d in ec.dates_in_range(start, end):
                ec_map[d] = ec.close_time
        if self._special_early_closes_fn is not None:
            ec_map.update(self._special_early_closes_fn(start, end))
        return ec_map

    def valid_days(self, start: date | str, end: date | str) -> pl.Series:
        """Return a Polars Series of trading days within [start, end]."""
        start_d, end_d = parse_date(start), parse_date(end)
        closures = self._closure_dates(start_d, end_d)

        days: list[date] = []
        current = start_d
        while current <= end_d:
            if current.weekday() < 5 and current not in closures:
                days.append(current)
            current += timedelta(days=1)

        return pl.Series("date", days, dtype=pl.Date)

    def schedule(self, start: date | str, end: date | str) -> pl.DataFrame:
        """Return a Polars DataFrame with columns: date, market_open, market_close.

        When the calendar has a lunch break, ``break_start`` and ``break_end``
        columns are included (null for calendars without breaks).
        """
        start_d, end_d = parse_date(start), parse_date(end)
        closures = self._closure_dates(start_d, end_d)
        ec_map = self._early_close_map(start_d, end_d)

        dates: list[date] = []
        opens: list[datetime] = []
        closes: list[datetime] = []
        break_starts: list[datetime | None] = []
        break_ends: list[datetime | None] = []

        current = start_d
        while current <= end_d:
            if current.weekday() < 5 and current not in closures:
                dates.append(current)
                open_date = current + timedelta(days=self.open_offset)
                opens.append(datetime.combine(open_date, self.open_time, tzinfo=self.tz))
                close_t = ec_map.get(current, self.close_time)
                closes.append(datetime.combine(current, close_t, tzinfo=self.tz))
                if self.break_start is not None and self.break_end is not None:
                    break_starts.append(datetime.combine(current, self.break_start, tzinfo=self.tz))
                    break_ends.append(datetime.combine(current, self.break_end, tzinfo=self.tz))
                else:
                    break_starts.append(None)
                    break_ends.append(None)
            current += timedelta(days=1)

        df = pl.DataFrame({
            "date": pl.Series(dates, dtype=pl.Date),
            "market_open": pl.Series(opens, dtype=pl.Datetime("us", self.timezone)),
            "market_close": pl.Series(closes, dtype=pl.Datetime("us", self.timezone)),
            "break_start": pl.Series(break_starts, dtype=pl.Datetime("us", self.timezone)),
            "break_end": pl.Series(break_ends, dtype=pl.Datetime("us", self.timezone)),
        })
        return df

    def is_session(self, day: date | str) -> bool:
        """Check if a given date is a trading day."""
        d = parse_date(day)
        if d.weekday() >= 5:
            return False
        closures = self._closure_dates(d, d)
        return d not in closures

    def get_schedule(self, day: date | str) -> MarketDailySchedule | None:
        """Get the schedule for a single day, or None if not a trading day."""
        d = parse_date(day)
        if not self.is_session(d):
            return None
        ec_map = self._early_close_map(d, d)
        close_t = ec_map.get(d, self.close_time)
        open_date = d + timedelta(days=self.open_offset)
        bs = datetime.combine(d, self.break_start, tzinfo=self.tz) if self.break_start else None
        be = datetime.combine(d, self.break_end, tzinfo=self.tz) if self.break_end else None
        return MarketDailySchedule(
            date=d,
            market_open=datetime.combine(open_date, self.open_time, tzinfo=self.tz),
            market_close=datetime.combine(d, close_t, tzinfo=self.tz),
            break_start=bs,
            break_end=be,
        )
