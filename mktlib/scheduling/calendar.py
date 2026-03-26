from __future__ import annotations

from datetime import date, datetime, time, timedelta
from typing import Any, TYPE_CHECKING
from zoneinfo import ZoneInfo

import polars as pl

from mktlib.scheduling._mixins import (
    MinuteQueryMixin,
    SessionNavigationMixin,
    TradingHelperMixin,
)
from mktlib.scheduling._types import MarketDailySchedule, parse_date
from mktlib.scheduling.rules import AdhocClosure, EarlyClose, HolidayRule

from mktlib.scheduling._break_mixin import BreakMixin

if TYPE_CHECKING:
    from collections.abc import Callable


class ExchangeCalendar(
    SessionNavigationMixin, MinuteQueryMixin, TradingHelperMixin
):
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
        special_early_closes_fn: (
            Callable[[date, date], dict[date, time]] | None
        ) = None,
        exclusions: set[date] | None = None,
        open_offset: int = 0,
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
        """Return a Polars Series of trading days within [start, end].

        Trading dates are weekdays (Mon-Fri) minus closures. For calendars with
        ``open_offset < 0`` (e.g. Globex, FX), a session's market_open may fall on
        the prior calendar day (e.g. Sunday 17:00 for Monday's session), but the
        session is still attributed to the weekday. Minute-level methods like
        ``is_open_on_minute`` handle the cross-day lookup.
        """
        start_d, end_d = parse_date(start), parse_date(end)
        closures = self._closure_dates(start_d, end_d)

        all_days = pl.date_range(start_d, end_d, eager=True)
        mask = all_days.dt.weekday() <= 5  # Polars ISO: 1=Mon..7=Sun
        if closures:
            mask = mask & ~all_days.is_in(list(closures))
        return all_days.filter(mask).alias("date")

    def schedule(self, start: date | str, end: date | str) -> pl.DataFrame:
        """Return a Polars DataFrame with columns: date, market_open, market_close.

        For calendars with ``open_offset < 0``, market_open falls on the prior
        calendar day (e.g. Sunday 17:00 for a Monday FX session).
        """
        start_d, end_d = parse_date(start), parse_date(end)
        days = self.valid_days(start_d, end_d)
        ec_map = self._early_close_map(start_d, end_d)

        df = days.to_frame()
        open_expr = (
            pl.col("date")
            .dt.offset_by(f"{self.open_offset}d")
            .dt.combine(self.open_time)
            .dt.replace_time_zone(self.timezone)
            .alias("market_open")
        )
        if ec_map:
            ec_df = pl.DataFrame(
                {
                    "date": pl.Series(list(ec_map.keys()), dtype=pl.Date),
                    "_ec_time": pl.Series(
                        list(ec_map.values()), dtype=pl.Time
                    ),
                }
            )
            df = df.join(ec_df, on="date", how="left")
            close_expr = (
                pl.col("date")
                .dt.combine(
                    pl.coalesce(pl.col("_ec_time"), pl.lit(self.close_time))
                )
                .dt.replace_time_zone(self.timezone)
                .alias("market_close")
            )
            return df.with_columns(open_expr, close_expr).select(
                "date", "market_open", "market_close"
            )

        close_expr = (
            pl.col("date")
            .dt.combine(self.close_time)
            .dt.replace_time_zone(self.timezone)
            .alias("market_close")
        )
        return df.with_columns(open_expr, close_expr)

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
        return MarketDailySchedule(
            date=d,
            market_open=datetime.combine(
                open_date, self.open_time, tzinfo=self.tz
            ),
            market_close=datetime.combine(d, close_t, tzinfo=self.tz),
        )


class ExchangeCalendarWithBreaks(BreakMixin, ExchangeCalendar):
    """Exchange calendar with lunch break support (e.g. JPX, HKEX)."""

    def __init__(
        self, name: str, *, break_start: time, break_end: time, **kwargs: Any
    ):
        self.break_start = break_start
        self.break_end = break_end
        super().__init__(name, **kwargs)
