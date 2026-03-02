from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING, Literal
from zoneinfo import ZoneInfo

import polars as pl

from mktlib.scheduling._types import MarketDailySchedule, parse_date

if TYPE_CHECKING:
    from typing import Protocol

    class _CalendarProtocol(Protocol):
        tz: ZoneInfo
        timezone: str
        open_offset: int
        def is_session(self, day: date | str) -> bool: ...
        def valid_days(self, start: date | str, end: date | str) -> pl.Series: ...
        def get_schedule(self, day: date | str) -> MarketDailySchedule | None: ...
        def schedule(self, start: date | str, end: date | str) -> pl.DataFrame: ...
        def next_session(self, day: date | str) -> date: ...
        def previous_session(self, day: date | str) -> date: ...


def _ensure_aware(dt: datetime, tz: ZoneInfo) -> datetime:
    """Ensure *dt* is timezone-aware in the given tz."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=tz)
    return dt.astimezone(tz)


class SessionNavigationMixin:
    """Session-level navigation: next/previous/offset/resolve/count."""

    def next_session(self: _CalendarProtocol, day: date | str) -> date:
        """First trading day strictly after *day*."""
        d = parse_date(day) + timedelta(days=1)
        while not self.is_session(d):
            d += timedelta(days=1)
        return d

    def previous_session(self: _CalendarProtocol, day: date | str) -> date:
        """Last trading day strictly before *day*."""
        d = parse_date(day) - timedelta(days=1)
        while not self.is_session(d):
            d -= timedelta(days=1)
        return d

    def session_offset(self: _CalendarProtocol, day: date | str, n: int) -> date:
        """Offset *day* by *n* trading sessions (negative = backward).

        *day* must be a session.  ``n=0`` returns *day* unchanged.
        """
        d = parse_date(day)
        if not self.is_session(d):
            raise ValueError(f"{d} is not a trading session")
        if n == 0:
            return d
        step = 1 if n > 0 else -1
        remaining = abs(n)
        while remaining > 0:
            d += timedelta(days=step)
            if self.is_session(d):
                remaining -= 1
        return d

    def date_to_session(self: _CalendarProtocol, day: date | str, direction: str = "none") -> date:
        """Resolve *day* to a session.

        *direction*: ``"none"`` (raise if not session), ``"next"``, ``"previous"``.
        """
        d = parse_date(day)
        if self.is_session(d):
            return d
        match direction:
            case "next":
                return self.next_session(d)
            case "previous":
                return self.previous_session(d)
            case "none":
                raise ValueError(f"{d} is not a trading session")
            case _:
                raise ValueError(f"Invalid direction {direction!r}")

    def sessions_in_range(self: _CalendarProtocol, start: date | str, end: date | str) -> int:
        """Count trading sessions in [start, end]."""
        return len(self.valid_days(start, end))


class MinuteQueryMixin:
    """Minute-resolution queries: open/close lookup, session membership."""

    def is_open_on_minute(self: _CalendarProtocol, dt: datetime) -> bool:
        """Check if the exchange is open at *dt*.  Uses ``[open, close)`` semantics."""
        aware = _ensure_aware(dt, self.tz)
        d = aware.date()
        sched = self.get_schedule(d)
        if sched is not None and sched.market_open <= aware < sched.market_close:
            return True
        if self.open_offset < 0:
            next_d = self.next_session(d)
            next_sched = self.get_schedule(next_d)
            if next_sched is not None and next_sched.market_open <= aware < next_sched.market_close:
                return True
        return False

    def next_open(self: _CalendarProtocol, dt: datetime) -> datetime:
        """Next market open strictly after *dt* (or today's open if before it)."""
        aware = _ensure_aware(dt, self.tz)
        d = aware.date()
        sched = self.get_schedule(d)
        if sched is not None and aware < sched.market_open:
            return sched.market_open
        d = self.next_session(d)
        sched = self.get_schedule(d)
        assert sched is not None
        while sched.market_open <= aware:
            d = self.next_session(d)
            sched = self.get_schedule(d)
            assert sched is not None
        return sched.market_open

    def next_close(self: _CalendarProtocol, dt: datetime) -> datetime:
        """Next market close at or after *dt* (today's close if still open)."""
        aware = _ensure_aware(dt, self.tz)
        d = aware.date()
        sched = self.get_schedule(d)
        if sched is not None and aware < sched.market_close:
            return sched.market_close
        next_d = self.next_session(d)
        next_sched = self.get_schedule(next_d)
        assert next_sched is not None
        return next_sched.market_close

    def previous_open(self: _CalendarProtocol, dt: datetime) -> datetime:
        """Most recent market open strictly before *dt*."""
        aware = _ensure_aware(dt, self.tz)
        d = aware.date()
        sched = self.get_schedule(d)
        if sched is not None and aware > sched.market_open:
            return sched.market_open
        prev_d = self.previous_session(d)
        prev_sched = self.get_schedule(prev_d)
        assert prev_sched is not None
        return prev_sched.market_open

    def previous_close(self: _CalendarProtocol, dt: datetime) -> datetime:
        """Most recent market close strictly before *dt*."""
        aware = _ensure_aware(dt, self.tz)
        d = aware.date()
        sched = self.get_schedule(d)
        if sched is not None and aware > sched.market_close:
            return sched.market_close
        prev_d = self.previous_session(d)
        prev_sched = self.get_schedule(prev_d)
        assert prev_sched is not None
        return prev_sched.market_close

    def minute_to_session(self: _CalendarProtocol, dt: datetime) -> date | None:
        """Return the session date that contains *dt*, or ``None`` if market is closed."""
        aware = _ensure_aware(dt, self.tz)
        d = aware.date()
        sched = self.get_schedule(d)
        if sched is not None and sched.market_open <= aware < sched.market_close:
            return d
        if self.open_offset < 0:
            next_d = self.next_session(d)
            next_sched = self.get_schedule(next_d)
            if next_sched is not None and next_sched.market_open <= aware < next_sched.market_close:
                return next_d
        return None


class TradingIndexMixin:
    """Generate intraday timestamp indices at arbitrary frequency."""

    def trading_index(
        self: _CalendarProtocol,
        start: date | str,
        end: date | str,
        period: str = "1m",
        closed: Literal["left", "right", "both", "none"] = "left",
    ) -> pl.Series:
        """Generate a Polars Series of trading timestamps at the given frequency.

        *closed*: ``"left"`` (default) = ``[open, close)``,
        ``"right"`` = ``(open, close]``,
        ``"both"`` = ``[open, close]``,
        ``"none"`` = ``(open, close)``.
        """
        sched = self.schedule(start, end)
        parts: list[pl.Series] = []
        for row in sched.iter_rows(named=True):
            ts = pl.datetime_range(
                row["market_open"],
                row["market_close"],
                interval=period,
                eager=True,
                closed=closed,
            )
            parts.append(ts)
        if not parts:
            return pl.Series("datetime", [], dtype=pl.Datetime("us", self.timezone))
        return pl.concat(parts).alias("datetime")
