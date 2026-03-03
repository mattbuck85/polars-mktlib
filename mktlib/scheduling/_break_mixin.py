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
        def valid_days(self, start: date | str, end: date | str) -> pl.Series: ...
        def get_schedule(self, day: date | str) -> MarketDailySchedule | None: ...
        def schedule(self, start: date | str, end: date | str) -> pl.DataFrame: ...
        def next_session(self, day: date | str) -> date: ...
        def is_open_on_minute(self, dt: datetime) -> bool: ...
        def previous_session(self, day: date | str) -> date: ...


class BreakMixin:
    """Mixin that adds lunch-break support to an ExchangeCalendar.

    Must appear before ``ExchangeCalendar`` in MRO so its overrides win.
    ``break_start`` and ``break_end`` are required (non-optional) here.

    Cooperative ``super()`` calls are cast to ``_BreakCalendarProtocol``
    because type checkers resolve ``super()`` against the static MRO
    (just ``object`` for a mixin), not the runtime MRO.
    """

    def schedule(self: _BreakCalendarProtocol, start: date | str, end: date | str) -> pl.DataFrame:
        """Base schedule + break_start/break_end columns."""
        df = cast("_BreakCalendarProtocol", super()).schedule(start, end)

        return df.with_columns(
            pl.col("date").dt.combine(self.break_start).dt.replace_time_zone(self.timezone).alias("break_start"),
            pl.col("date").dt.combine(self.break_end).dt.replace_time_zone(self.timezone).alias("break_end"),
        )

    def get_schedule(self: _BreakCalendarProtocol, day: date | str) -> MarketDailySchedule | None:
        """Base schedule + break fields populated."""
        sched = cast("_BreakCalendarProtocol", super()).get_schedule(day)
        if sched is None:
            return None
        sched.break_start = datetime.combine(sched.date, self.break_start, tzinfo=self.tz)
        sched.break_end = datetime.combine(sched.date, self.break_end, tzinfo=self.tz)
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
        parts: list[pl.Series] = []
        for row in sched.iter_rows(named=True):
            morning = pl.datetime_range(
                row["market_open"],
                row["break_start"],
                interval=period,
                eager=True,
                closed=closed,
            )
            afternoon = pl.datetime_range(
                row["break_end"],
                row["market_close"],
                interval=period,
                eager=True,
                closed=closed,
            )
            parts.append(morning)
            parts.append(afternoon)
        if not parts:
            return pl.Series("datetime", [], dtype=pl.Datetime("us", self.timezone))
        return pl.concat(parts).alias("datetime")
