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
        def previous_session(self, day: date | str) -> date: ...


def _tz(series: pl.Series) -> str | None:
    """Extract timezone from a Datetime series, or None."""
    return series.dtype.time_zone  # type: ignore[union-attr]


def _align_tz(target: pl.Series, reference: pl.Series) -> pl.Series:
    """Align *target* timezone to match *reference*."""
    ref_tz = _tz(reference)
    tgt_tz = _tz(target)
    if ref_tz is None and tgt_tz is not None:
        return target.dt.replace_time_zone(None)
    if ref_tz is not None and tgt_tz is None:
        return target.dt.replace_time_zone(ref_tz)
    if ref_tz is not None and tgt_tz is not None and ref_tz != tgt_tz:
        return target.dt.convert_time_zone(ref_tz)
    return target


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

    def session_offset(
        self: _CalendarProtocol, day: date | str, n: int
    ) -> date:
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

    def date_to_session(
        self: _CalendarProtocol, day: date | str, direction: str = "none"
    ) -> date:
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

    def sessions_in_range(
        self: _CalendarProtocol, start: date | str, end: date | str
    ) -> int:
        """Count trading sessions in [start, end]."""
        return len(self.valid_days(start, end))


class MinuteQueryMixin:
    """Minute-resolution queries: open/close lookup, session membership."""

    def is_open_on_minute(self: _CalendarProtocol, dt: datetime) -> bool:
        """Check if the exchange is open at *dt*.  Uses ``[open, close)`` semantics."""
        aware = _ensure_aware(dt, self.tz)
        d = aware.date()
        sched = self.get_schedule(d)
        if (
            sched is not None
            and sched.market_open <= aware < sched.market_close
        ):
            return True
        if self.open_offset < 0:
            next_d = self.next_session(d)
            next_sched = self.get_schedule(next_d)
            if (
                next_sched is not None
                and next_sched.market_open <= aware < next_sched.market_close
            ):
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

    def minute_to_session(
        self: _CalendarProtocol, dt: datetime
    ) -> date | None:
        """Return the session date that contains *dt*, or ``None`` if market is closed."""
        aware = _ensure_aware(dt, self.tz)
        d = aware.date()
        sched = self.get_schedule(d)
        if (
            sched is not None
            and sched.market_open <= aware < sched.market_close
        ):
            return d
        if self.open_offset < 0:
            next_d = self.next_session(d)
            next_sched = self.get_schedule(next_d)
            if (
                next_sched is not None
                and next_sched.market_open <= aware < next_sched.market_close
            ):
                return next_d
        return None


class TradingHelperMixin:
    """Trading index generation and market-hours filtering."""

    def filter_market_hours(
        self: _CalendarProtocol,
        df: pl.DataFrame,
        date_column: str = "date",
    ) -> pl.DataFrame:
        """Filter rows to market hours using a schedule join.

        More efficient than ``trading_index()`` — joins on ~252 schedule
        rows per year instead of materializing all trading minutes.

        Parameters
        ----------
        df
            DataFrame with a Datetime column for bar timestamps.
        date_column
            Name of the datetime column (default ``"date"``).

        Returns
        -------
        pl.DataFrame
            Rows within ``[market_open, market_close - 1min]``, excluding
            lunch breaks for break calendars.
        """
        dates = df[date_column]
        if df.is_empty():
            return df

        start_val = dates.min()
        end_val = dates.max()
        if start_val is None or end_val is None:
            msg = f"'{date_column}' column contains only nulls"
            raise ValueError(msg)
        start_d = (
            start_val.date()
            if isinstance(start_val, datetime)
            else date(start_val.year, start_val.month, start_val.day)  # type: ignore[union-attr]
        )
        end_d = (
            end_val.date()
            if isinstance(end_val, datetime)
            else date(end_val.year, end_val.month, end_val.day)  # type: ignore[union-attr]
        )

        sched = self.schedule(start_d, end_d)

        # Compute last tradeable minute (open-frame: close - 1min)
        sched = sched.with_columns(
            (pl.col("market_close") - pl.duration(minutes=1)).alias(
                "_last_minute"
            ),
        )

        # Build join key: bar date (Date) to match schedule's date column
        dates_df = df.with_columns(
            pl.col(date_column).dt.date().alias("_bar_date"),
        )

        # Prepare schedule columns with tz aligned to bar timestamps
        sched_join = sched.select(
            pl.col("date").alias("_bar_date"),
            _align_tz(sched["market_open"], dates).alias("_mkt_open"),
            _align_tz(sched["_last_minute"], dates).alias("_last_min"),
        )

        joined = dates_df.join(sched_join, on="_bar_date", how="left")

        # Bar is valid if within [market_open, last_minute]
        mask = (
            joined["_mkt_open"].is_not_null()
            & (joined[date_column] >= joined["_mkt_open"])
            & (joined[date_column] <= joined["_last_min"])
        )

        return df.filter(mask)

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
        if sched.is_empty():
            return pl.Series(
                "datetime", [], dtype=pl.Datetime("us", self.timezone)
            )
        return (
            sched.with_columns(
                pl.datetime_ranges(
                    "market_open",
                    "market_close",
                    interval=period,
                    closed=closed,
                ).alias("datetime")
            )
            .select("datetime")
            .explode("datetime")
            .to_series()
        )
