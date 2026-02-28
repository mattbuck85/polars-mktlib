from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import TYPE_CHECKING, Literal
from zoneinfo import ZoneInfo

import polars as pl

from mktlib.scheduling.rules import AdhocClosure, EarlyClose, HolidayRule

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass
class MarketDailySchedule:
    """Single-day market schedule — matches tradesignalcore's dataclass."""

    date: date
    market_open: datetime
    market_close: datetime


class ExchangeCalendar:
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
        start_d, end_d = _parse_date(start), _parse_date(end)
        closures = self._closure_dates(start_d, end_d)

        days: list[date] = []
        current = start_d
        while current <= end_d:
            if current.weekday() < 5 and current not in closures:
                days.append(current)
            current += timedelta(days=1)

        return pl.Series("date", days, dtype=pl.Date)

    def schedule(self, start: date | str, end: date | str) -> pl.DataFrame:
        """Return a Polars DataFrame with columns: date, market_open, market_close."""
        start_d, end_d = _parse_date(start), _parse_date(end)
        closures = self._closure_dates(start_d, end_d)
        ec_map = self._early_close_map(start_d, end_d)

        dates: list[date] = []
        opens: list[datetime] = []
        closes: list[datetime] = []

        current = start_d
        while current <= end_d:
            if current.weekday() < 5 and current not in closures:
                dates.append(current)
                opens.append(datetime.combine(current, self.open_time, tzinfo=self.tz))
                close_t = ec_map.get(current, self.close_time)
                closes.append(datetime.combine(current, close_t, tzinfo=self.tz))
            current += timedelta(days=1)

        return pl.DataFrame({
            "date": pl.Series(dates, dtype=pl.Date),
            "market_open": pl.Series(opens, dtype=pl.Datetime("us", self.timezone)),
            "market_close": pl.Series(closes, dtype=pl.Datetime("us", self.timezone)),
        })

    def is_session(self, day: date | str) -> bool:
        """Check if a given date is a trading day."""
        d = _parse_date(day)
        if d.weekday() >= 5:
            return False
        closures = self._closure_dates(d, d)
        return d not in closures

    def get_schedule(self, day: date | str) -> MarketDailySchedule | None:
        """Get the schedule for a single day, or None if not a trading day."""
        d = _parse_date(day)
        if not self.is_session(d):
            return None
        ec_map = self._early_close_map(d, d)
        close_t = ec_map.get(d, self.close_time)
        return MarketDailySchedule(
            date=d,
            market_open=datetime.combine(d, self.open_time, tzinfo=self.tz),
            market_close=datetime.combine(d, close_t, tzinfo=self.tz),
        )

    # --- Session navigation ---

    def next_session(self, day: date | str) -> date:
        """First trading day strictly after *day*."""
        d = _parse_date(day) + timedelta(days=1)
        while not self.is_session(d):
            d += timedelta(days=1)
        return d

    def previous_session(self, day: date | str) -> date:
        """Last trading day strictly before *day*."""
        d = _parse_date(day) - timedelta(days=1)
        while not self.is_session(d):
            d -= timedelta(days=1)
        return d

    def session_offset(self, day: date | str, n: int) -> date:
        """Offset *day* by *n* trading sessions (negative = backward).

        *day* must be a session.  ``n=0`` returns *day* unchanged.
        """
        d = _parse_date(day)
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

    def date_to_session(self, day: date | str, direction: str = "none") -> date:
        """Resolve *day* to a session.

        *direction*: ``"none"`` (raise if not session), ``"next"``, ``"previous"``.
        """
        d = _parse_date(day)
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

    def sessions_in_range(self, start: date | str, end: date | str) -> int:
        """Count trading sessions in [start, end]."""
        return len(self.valid_days(start, end))

    # --- Minute-level queries ---

    def _to_aware(self, dt: datetime) -> datetime:
        """Ensure *dt* is timezone-aware in exchange tz."""
        if dt.tzinfo is None:
            return dt.replace(tzinfo=self.tz)
        return dt.astimezone(self.tz)

    def is_open_on_minute(self, dt: datetime) -> bool:
        """Check if the exchange is open at *dt*.  Uses ``[open, close)`` semantics."""
        aware = self._to_aware(dt)
        sched = self.get_schedule(aware.date())
        if sched is None:
            return False
        return sched.market_open <= aware < sched.market_close

    def next_open(self, dt: datetime) -> datetime:
        """Next market open strictly after *dt* (or today's open if before it)."""
        aware = self._to_aware(dt)
        d = aware.date()
        sched = self.get_schedule(d)
        if sched is not None and aware < sched.market_open:
            return sched.market_open
        next_d = self.next_session(d)
        next_sched = self.get_schedule(next_d)
        assert next_sched is not None
        return next_sched.market_open

    def next_close(self, dt: datetime) -> datetime:
        """Next market close at or after *dt* (today's close if still open)."""
        aware = self._to_aware(dt)
        d = aware.date()
        sched = self.get_schedule(d)
        if sched is not None and aware < sched.market_close:
            return sched.market_close
        next_d = self.next_session(d)
        next_sched = self.get_schedule(next_d)
        assert next_sched is not None
        return next_sched.market_close

    def previous_open(self, dt: datetime) -> datetime:
        """Most recent market open strictly before *dt*."""
        aware = self._to_aware(dt)
        d = aware.date()
        sched = self.get_schedule(d)
        if sched is not None and aware > sched.market_open:
            return sched.market_open
        prev_d = self.previous_session(d)
        prev_sched = self.get_schedule(prev_d)
        assert prev_sched is not None
        return prev_sched.market_open

    def previous_close(self, dt: datetime) -> datetime:
        """Most recent market close strictly before *dt*."""
        aware = self._to_aware(dt)
        d = aware.date()
        sched = self.get_schedule(d)
        if sched is not None and aware > sched.market_close:
            return sched.market_close
        prev_d = self.previous_session(d)
        prev_sched = self.get_schedule(prev_d)
        assert prev_sched is not None
        return prev_sched.market_close

    def minute_to_session(self, dt: datetime) -> date | None:
        """Return the session date that contains *dt*, or ``None`` if market is closed."""
        aware = self._to_aware(dt)
        d = aware.date()
        sched = self.get_schedule(d)
        if sched is not None and sched.market_open <= aware < sched.market_close:
            return d
        return None

    # --- Trading index ---

    def trading_index(
        self,
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


# --- Registry ---

_REGISTRY: dict[str, Callable[[], ExchangeCalendar]] = {}
_ALIASES: dict[str, str] = {}


def register_exchange(name: str, factory: Callable[[], ExchangeCalendar], aliases: list[str] | None = None) -> None:
    """Register an exchange calendar factory."""
    _REGISTRY[name] = factory
    if aliases:
        for alias in aliases:
            _ALIASES[alias] = name


def get_calendar(name: str) -> ExchangeCalendar:
    """Get an exchange calendar by name or alias."""
    canonical = _ALIASES.get(name, name)
    if canonical not in _REGISTRY:
        available = sorted(set(_REGISTRY.keys()) | set(_ALIASES.keys()))
        raise ValueError(f"Unknown exchange {name!r}. Available: {available}")
    return _REGISTRY[canonical]()


def _parse_date(d: date | str) -> date:
    if isinstance(d, str):
        return date.fromisoformat(d)
    return d


# --- Register built-in exchanges ---


def _make_nyse() -> ExchangeCalendar:
    from mktlib.scheduling.exchanges.nyse import (
        ADHOC_CLOSURES,
        EARLY_CLOSE_TIME,
        EARLY_CLOSES,
        NYSE_CLOSE,
        NYSE_OPEN,
        NYSE_TZ,
        RECURRING_HOLIDAYS,
        black_friday_early_closes,
        christmas_eve_early_closes,
        good_friday_closures,
        independence_day_early_closes,
    )

    def special_closures(start: date, end: date) -> list[date]:
        return good_friday_closures(start, end)

    def special_early_closes(start: date, end: date) -> dict[date, time]:
        result: dict[date, time] = {}
        for d in black_friday_early_closes(start, end):
            result[d] = EARLY_CLOSE_TIME
        for d in independence_day_early_closes(start, end):
            result[d] = EARLY_CLOSE_TIME
        for d in christmas_eve_early_closes(start, end):
            result[d] = EARLY_CLOSE_TIME
        return result

    return ExchangeCalendar(
        name="XNYS",
        timezone=NYSE_TZ,
        open_time=NYSE_OPEN,
        close_time=NYSE_CLOSE,
        holidays=RECURRING_HOLIDAYS,
        adhoc_closures=ADHOC_CLOSURES,
        early_closes=EARLY_CLOSES,
        special_closures_fn=special_closures,
        special_early_closes_fn=special_early_closes,
    )


register_exchange("XNYS", _make_nyse, aliases=["NYSE"])


def _make_lse() -> ExchangeCalendar:
    from mktlib.scheduling.exchanges.lse import (
        ADHOC_CLOSURES,
        BANK_HOLIDAY_MOVES,
        EARLY_CLOSES,
        LSE_CLOSE,
        LSE_OPEN,
        LSE_TZ,
        RECURRING_HOLIDAYS,
        special_closures_with_moves,
        special_early_closes,
    )

    return ExchangeCalendar(
        name="XLON",
        timezone=LSE_TZ,
        open_time=LSE_OPEN,
        close_time=LSE_CLOSE,
        holidays=RECURRING_HOLIDAYS,
        adhoc_closures=ADHOC_CLOSURES,
        early_closes=EARLY_CLOSES,
        special_closures_fn=special_closures_with_moves,
        special_early_closes_fn=special_early_closes,
        exclusions=set(BANK_HOLIDAY_MOVES.keys()),
    )


register_exchange("XLON", _make_lse, aliases=["LSE", "London"])


def _make_euronext() -> ExchangeCalendar:
    from mktlib.scheduling.exchanges.euronext import (
        ADHOC_CLOSURES,
        EARLY_CLOSES,
        EURONEXT_CLOSE,
        EURONEXT_OPEN,
        EURONEXT_TZ,
        RECURRING_HOLIDAYS,
        special_closures,
        special_early_closes,
    )

    return ExchangeCalendar(
        name="XPAR",
        timezone=EURONEXT_TZ,
        open_time=EURONEXT_OPEN,
        close_time=EURONEXT_CLOSE,
        holidays=RECURRING_HOLIDAYS,
        adhoc_closures=ADHOC_CLOSURES,
        early_closes=EARLY_CLOSES,
        special_closures_fn=special_closures,
        special_early_closes_fn=special_early_closes,
    )


register_exchange("XPAR", _make_euronext, aliases=["Euronext", "Paris"])
