from __future__ import annotations

from datetime import date, datetime, time
from typing import TYPE_CHECKING, Literal, cast

import polars as pl

from mktlib.scheduling._mixins import _ensure_aware
from mktlib.scheduling._types import MarketDailySchedule

if TYPE_CHECKING:
    from typing import Protocol

    from zoneinfo import ZoneInfo

    class _BreakCalendarProtocol(Protocol):
        tz: ZoneInfo
        timezone: str
        open_offset: int
        break_start: time
        break_end: time

        def is_session(self, day: date | str) -> bool: ...
        def valid_days(
            self, start: date | str, end: date | str
        ) -> pl.Series: ...
        def get_schedule(
            self, day: date | str
        ) -> MarketDailySchedule | None: ...
        def schedule(
            self, start: date | str, end: date | str
        ) -> pl.DataFrame: ...
        def next_session(self, day: date | str) -> date: ...
        def is_open_on_minute(self, dt: datetime) -> bool: ...
        def previous_session(self, day: date | str) -> date: ...
        def filter_market_hours(
            self, df: pl.DataFrame, date_column: str = "date"
        ) -> pl.DataFrame: ...


class BreakMixin:
    """Mixin that adds lunch-break support to an ExchangeCalendar.

    Must appear before ``ExchangeCalendar`` in MRO so its overrides win.
    ``break_start`` and ``break_end`` are required (non-optional) here.

    Cooperative ``super()`` calls are cast to ``_BreakCalendarProtocol``
    because type checkers resolve ``super()`` against the static MRO
    (just ``object`` for a mixin), not the runtime MRO.
    """

    def schedule(
        self: _BreakCalendarProtocol, start: date | str, end: date | str
    ) -> pl.DataFrame:
        """Base schedule + break_start/break_end columns."""
        df = cast("_BreakCalendarProtocol", super()).schedule(start, end)

        return df.with_columns(
            pl.col("date")
            .dt.combine(self.break_start)
            .dt.replace_time_zone(self.timezone)
            .alias("break_start"),
            pl.col("date")
            .dt.combine(self.break_end)
            .dt.replace_time_zone(self.timezone)
            .alias("break_end"),
        )

    def get_schedule(
        self: _BreakCalendarProtocol, day: date | str
    ) -> MarketDailySchedule | None:
        """Base schedule + break fields populated."""
        sched = cast("_BreakCalendarProtocol", super()).get_schedule(day)
        if sched is None:
            return None
        sched.break_start = datetime.combine(
            sched.date, self.break_start, tzinfo=self.tz
        )
        sched.break_end = datetime.combine(
            sched.date, self.break_end, tzinfo=self.tz
        )
        return sched

    def is_open_on_minute(self: _BreakCalendarProtocol, dt: datetime) -> bool:
        """Base open check, then exclude the break window."""
        if not cast("_BreakCalendarProtocol", super()).is_open_on_minute(dt):
            return False
        aware = _ensure_aware(dt, self.tz)
        t = aware.time()
        if self.break_start <= t < self.break_end:
            return False
        return True

    def trading_index(
        self: _BreakCalendarProtocol,
        start: date | str,
        end: date | str,
        period: str = "1m",
        closed: Literal["left", "right", "both", "none"] = "left",
    ) -> pl.Series:
        """Two ranges per day: open->break_start, break_end->close."""
        sched = self.schedule(start, end)
        if sched.is_empty():
            return pl.Series(
                "datetime", [], dtype=pl.Datetime("us", self.timezone)
            )
        return (
            sched.with_columns(
                pl.datetime_ranges(
                    "market_open",
                    "break_start",
                    interval=period,
                    closed=closed,
                ).alias("morning"),
                pl.datetime_ranges(
                    "break_end",
                    "market_close",
                    interval=period,
                    closed=closed,
                ).alias("afternoon"),
            )
            .with_columns(
                pl.col("morning")
                .list.concat(pl.col("afternoon"))
                .alias("datetime")
            )
            .select("datetime")
            .explode("datetime")
            .to_series()
        )

    def filter_market_hours(
        self: _BreakCalendarProtocol,
        df: pl.DataFrame,
        date_column: str = "date",
    ) -> pl.DataFrame:
        """Base filter, then exclude lunch break bars."""
        filtered = cast(
            "_BreakCalendarProtocol", super()
        ).filter_market_hours(df, date_column)
        if filtered.is_empty():
            return filtered

        in_break = (
            (pl.col(date_column).dt.time() >= self.break_start)
            & (pl.col(date_column).dt.time() < self.break_end)
        )
        return filtered.filter(~in_break)
