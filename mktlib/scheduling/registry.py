from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from mktlib.scheduling.calendar import ExchangeCalendar

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


# --- Shared US-exchange helpers ---


def _us_special_closures(start: date, end: date) -> list[date]:
    from mktlib.scheduling.exchanges.nyse import good_friday_closures

    return good_friday_closures(start, end)


# --- CME-specific helpers ---


def _cme_special_closures(start: date, end: date) -> list[date]:
    """Good Friday — same as NYSE."""
    from mktlib.scheduling.exchanges.nyse import good_friday_closures

    return good_friday_closures(start, end)


# --- Factory functions ---


def _make_nyse() -> ExchangeCalendar:
    from mktlib.scheduling.calendar import ExchangeCalendar
    from mktlib.scheduling.exchanges.nyse import (
        ADHOC_CLOSURES,
        EARLY_CLOSES,
        NYSE_CLOSE,
        NYSE_OPEN,
        NYSE_TZ,
        RECURRING_HOLIDAYS,
    )

    return ExchangeCalendar(
        name="XNYS",
        timezone=NYSE_TZ,
        open_time=NYSE_OPEN,
        close_time=NYSE_CLOSE,
        holidays=RECURRING_HOLIDAYS,
        adhoc_closures=ADHOC_CLOSURES,
        early_closes=EARLY_CLOSES,
        special_closures_fn=_us_special_closures,
    )


def _make_lse() -> ExchangeCalendar:
    from mktlib.scheduling.calendar import ExchangeCalendar
    from mktlib.scheduling.exchanges.lse import (
        ADHOC_CLOSURES,
        BANK_HOLIDAY_MOVES,
        EARLY_CLOSES,
        LSE_CLOSE,
        LSE_OPEN,
        LSE_TZ,
        RECURRING_HOLIDAYS,
        special_closures_with_moves,
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
        exclusions=set(BANK_HOLIDAY_MOVES.keys()),
    )


def _make_euronext() -> ExchangeCalendar:
    from mktlib.scheduling.calendar import ExchangeCalendar
    from mktlib.scheduling.exchanges.euronext import (
        ADHOC_CLOSURES,
        EARLY_CLOSES,
        EURONEXT_CLOSE,
        EURONEXT_OPEN,
        EURONEXT_TZ,
        RECURRING_HOLIDAYS,
        special_closures,
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
    )


def _make_cme_rth() -> ExchangeCalendar:
    from mktlib.scheduling.calendar import ExchangeCalendar
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
    )


def _make_cme_globex() -> ExchangeCalendar:
    from mktlib.scheduling.calendar import ExchangeCalendar
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
        open_offset=-1,
    )


# --- Register built-in exchanges ---

register_exchange("XNYS", _make_nyse, aliases=["NYSE"])
register_exchange("XLON", _make_lse, aliases=["LSE", "London"])
register_exchange("XPAR", _make_euronext, aliases=["Euronext", "Paris"])
register_exchange("XCME", _make_cme_rth, aliases=["CME", "CME-RTH"])
register_exchange("GLBX", _make_cme_globex, aliases=["Globex", "CME-GLOBEX"])
