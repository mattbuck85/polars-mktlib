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
        """Return a Polars DataFrame with columns: date, market_open, market_close."""
        start_d, end_d = parse_date(start), parse_date(end)
        closures = self._closure_dates(start_d, end_d)
        ec_map = self._early_close_map(start_d, end_d)

        dates: list[date] = []
        opens: list[datetime] = []
        closes: list[datetime] = []

        current = start_d
        while current <= end_d:
            if current.weekday() < 5 and current not in closures:
                dates.append(current)
                open_date = current + timedelta(days=self.open_offset)
                opens.append(datetime.combine(open_date, self.open_time, tzinfo=self.tz))
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
            market_open=datetime.combine(open_date, self.open_time, tzinfo=self.tz),
            market_close=datetime.combine(d, close_t, tzinfo=self.tz),
        )


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


def _cme_special_closures(start: date, end: date) -> list[date]:
    from mktlib.scheduling.exchanges.cme import good_friday_closures

    return good_friday_closures(start, end)


def _cme_special_early_closes(start: date, end: date) -> dict[date, time]:
    from mktlib.scheduling.exchanges.cme import (
        CME_EARLY_CLOSE_TIME,
        black_friday_early_closes,
        christmas_eve_early_closes,
        independence_day_early_closes,
    )

    result: dict[date, time] = {}
    for d in black_friday_early_closes(start, end):
        result[d] = CME_EARLY_CLOSE_TIME
    for d in independence_day_early_closes(start, end):
        result[d] = CME_EARLY_CLOSE_TIME
    for d in christmas_eve_early_closes(start, end):
        result[d] = CME_EARLY_CLOSE_TIME
    return result


def _make_cme_rth() -> ExchangeCalendar:
    from mktlib.scheduling.exchanges.cme import (
        ADHOC_CLOSURES,
        CME_RTH_CLOSE,
        CME_RTH_OPEN,
        CME_TZ,
        EARLY_CLOSES,
        RECURRING_HOLIDAYS,
    )

    return ExchangeCalendar(
        name="XCME",
        timezone=CME_TZ,
        open_time=CME_RTH_OPEN,
        close_time=CME_RTH_CLOSE,
        holidays=RECURRING_HOLIDAYS,
        adhoc_closures=ADHOC_CLOSURES,
        early_closes=EARLY_CLOSES,
        special_closures_fn=_cme_special_closures,
        special_early_closes_fn=_cme_special_early_closes,
    )


def _make_cme_globex() -> ExchangeCalendar:
    from mktlib.scheduling.exchanges.cme import (
        ADHOC_CLOSURES,
        CME_GLOBEX_CLOSE,
        CME_GLOBEX_OPEN,
        CME_TZ,
        EARLY_CLOSES,
        RECURRING_HOLIDAYS,
    )

    return ExchangeCalendar(
        name="GLBX",
        timezone=CME_TZ,
        open_time=CME_GLOBEX_OPEN,
        close_time=CME_GLOBEX_CLOSE,
        holidays=RECURRING_HOLIDAYS,
        adhoc_closures=ADHOC_CLOSURES,
        early_closes=EARLY_CLOSES,
        special_closures_fn=_cme_special_closures,
        special_early_closes_fn=_cme_special_early_closes,
        open_offset=-1,
    )


register_exchange("XCME", _make_cme_rth, aliases=["CME", "CME-RTH"])
register_exchange("GLBX", _make_cme_globex, aliases=["Globex", "CME-GLOBEX"])
