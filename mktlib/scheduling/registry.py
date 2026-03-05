from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from mktlib.scheduling.calendar import ExchangeCalendar

# --- Registry ---

_REGISTRY: dict[str, Callable[[], ExchangeCalendar]] = {}
_ALIASES: dict[str, str] = {}
_ADHOC_URLS: dict[str, str] = {}


def register_exchange(
    name: str,
    factory: Callable[[], ExchangeCalendar],
    aliases: list[str] | None = None,
    adhoc_url: str | None = None,
) -> None:
    """Register an exchange calendar factory."""
    _REGISTRY[name] = factory
    if aliases:
        for alias in aliases:
            _ALIASES[alias] = name
    if adhoc_url is not None:
        _ADHOC_URLS[name] = adhoc_url


def get_adhoc_url(name: str) -> str | None:
    """Get the ad-hoc closure announcements URL for an exchange."""
    canonical = _ALIASES.get(name, name)
    return _ADHOC_URLS.get(canonical)


def get_all_adhoc_urls() -> dict[str, str]:
    """Get all exchange ad-hoc closure URLs (canonical name -> URL)."""
    return dict(_ADHOC_URLS)


def get_calendar(name: str) -> ExchangeCalendar:
    """Get an exchange calendar by name or alias."""
    canonical = _ALIASES.get(name, name)
    if canonical not in _REGISTRY:
        available = sorted(set(_REGISTRY.keys()) | set(_ALIASES.keys()))
        raise ValueError(f"Unknown exchange {name!r}. Available: {available}")
    return _REGISTRY[canonical]()


# --- Shared helpers ---


def _good_friday_closures(start: date, end: date) -> list[date]:
    from mktlib.scheduling.exchanges._us_holidays import (
        us_good_friday_closures,
    )

    return us_good_friday_closures(start, end)


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
        special_closures_fn=_good_friday_closures,
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
        special_closures_fn=_good_friday_closures,
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
        special_closures_fn=_good_friday_closures,
        open_offset=-1,
    )


def _make_nasdaq() -> ExchangeCalendar:
    from mktlib.scheduling.calendar import ExchangeCalendar
    from mktlib.scheduling.exchanges.nasdaq import (
        ADHOC_CLOSURES,
        EARLY_CLOSES,
        NASDAQ_CLOSE,
        NASDAQ_OPEN,
        NASDAQ_TZ,
        RECURRING_HOLIDAYS,
    )

    return ExchangeCalendar(
        name="XNAS",
        timezone=NASDAQ_TZ,
        open_time=NASDAQ_OPEN,
        close_time=NASDAQ_CLOSE,
        holidays=RECURRING_HOLIDAYS,
        adhoc_closures=ADHOC_CLOSURES,
        early_closes=EARLY_CLOSES,
        special_closures_fn=_good_friday_closures,
    )


def _make_cboe() -> ExchangeCalendar:
    from mktlib.scheduling.calendar import ExchangeCalendar
    from mktlib.scheduling.exchanges.cboe import (
        ADHOC_CLOSURES,
        CBOE_CLOSE,
        CBOE_OPEN,
        CBOE_TZ,
        EARLY_CLOSES,
        RECURRING_HOLIDAYS,
    )

    return ExchangeCalendar(
        name="XCBO",
        timezone=CBOE_TZ,
        open_time=CBOE_OPEN,
        close_time=CBOE_CLOSE,
        holidays=RECURRING_HOLIDAYS,
        adhoc_closures=ADHOC_CLOSURES,
        early_closes=EARLY_CLOSES,
        special_closures_fn=_good_friday_closures,
    )


def _make_tsx() -> ExchangeCalendar:
    from mktlib.scheduling.calendar import ExchangeCalendar
    from mktlib.scheduling.exchanges.tsx import (
        ADHOC_CLOSURES,
        EARLY_CLOSES,
        RECURRING_HOLIDAYS,
        TSX_CLOSE,
        TSX_OPEN,
        TSX_TZ,
        special_closures,
    )

    return ExchangeCalendar(
        name="XTSE",
        timezone=TSX_TZ,
        open_time=TSX_OPEN,
        close_time=TSX_CLOSE,
        holidays=RECURRING_HOLIDAYS,
        adhoc_closures=ADHOC_CLOSURES,
        early_closes=EARLY_CLOSES,
        special_closures_fn=special_closures,
    )


def _make_xetra() -> ExchangeCalendar:
    from mktlib.scheduling.calendar import ExchangeCalendar
    from mktlib.scheduling.exchanges.xetra import (
        ADHOC_CLOSURES,
        EARLY_CLOSES,
        RECURRING_HOLIDAYS,
        XETR_CLOSE,
        XETR_OPEN,
        XETR_TZ,
        special_closures,
    )

    return ExchangeCalendar(
        name="XETR",
        timezone=XETR_TZ,
        open_time=XETR_OPEN,
        close_time=XETR_CLOSE,
        holidays=RECURRING_HOLIDAYS,
        adhoc_closures=ADHOC_CLOSURES,
        early_closes=EARLY_CLOSES,
        special_closures_fn=special_closures,
    )


def _make_jpx() -> ExchangeCalendar:
    from mktlib.scheduling.calendar import ExchangeCalendarWithBreaks
    from mktlib.scheduling.exchanges.jpx import (
        ADHOC_CLOSURES,
        EARLY_CLOSES,
        JPX_BREAK_END,
        JPX_BREAK_START,
        JPX_CLOSE,
        JPX_EXCLUSIONS,
        JPX_OPEN,
        JPX_TZ,
        RECURRING_HOLIDAYS,
        special_closures,
    )

    return ExchangeCalendarWithBreaks(
        name="XTKS",
        timezone=JPX_TZ,
        open_time=JPX_OPEN,
        close_time=JPX_CLOSE,
        holidays=RECURRING_HOLIDAYS,
        adhoc_closures=ADHOC_CLOSURES,
        early_closes=EARLY_CLOSES,
        special_closures_fn=special_closures,
        exclusions=JPX_EXCLUSIONS,
        break_start=JPX_BREAK_START,
        break_end=JPX_BREAK_END,
    )


def _make_fx() -> ExchangeCalendar:
    from mktlib.scheduling.calendar import ExchangeCalendar
    from mktlib.scheduling.exchanges.fx import (
        ADHOC_CLOSURES,
        EARLY_CLOSES,
        FX_CLOSE,
        FX_OPEN,
        FX_TZ,
        RECURRING_HOLIDAYS,
    )

    return ExchangeCalendar(
        name="CMES",
        timezone=FX_TZ,
        open_time=FX_OPEN,
        close_time=FX_CLOSE,
        holidays=RECURRING_HOLIDAYS,
        adhoc_closures=ADHOC_CLOSURES,
        early_closes=EARLY_CLOSES,
        open_offset=-1,
    )


def _make_hkex() -> ExchangeCalendar:
    from mktlib.scheduling.calendar import ExchangeCalendarWithBreaks
    from mktlib.scheduling.exchanges.hkex import (
        ADHOC_CLOSURES,
        EARLY_CLOSES,
        HKEX_BREAK_END,
        HKEX_BREAK_START,
        HKEX_CLOSE,
        HKEX_OPEN,
        HKEX_TZ,
        RECURRING_HOLIDAYS,
        special_closures,
    )

    return ExchangeCalendarWithBreaks(
        name="XHKG",
        timezone=HKEX_TZ,
        open_time=HKEX_OPEN,
        close_time=HKEX_CLOSE,
        holidays=RECURRING_HOLIDAYS,
        adhoc_closures=ADHOC_CLOSURES,
        early_closes=EARLY_CLOSES,
        special_closures_fn=special_closures,
        break_start=HKEX_BREAK_START,
        break_end=HKEX_BREAK_END,
    )


# --- Register built-in exchanges ---

_CME_HOLIDAYS_URL = (
    "https://www.cmegroup.com/tools-information/holiday-calendar.html"
)

register_exchange(
    "XNYS",
    _make_nyse,
    aliases=["NYSE"],
    adhoc_url="https://www.nyse.com/regulation/closings",
)
register_exchange(
    "XLON",
    _make_lse,
    aliases=["LSE", "London"],
    adhoc_url="https://www.londonstockexchange.com/equities-trading/business-days",
)
register_exchange(
    "XPAR",
    _make_euronext,
    aliases=["Euronext", "Paris"],
    adhoc_url="https://www.euronext.com/en/trade/trading-hours-holidays",
)
register_exchange(
    "XCME",
    _make_cme_rth,
    aliases=["CME", "CME-RTH"],
    adhoc_url=_CME_HOLIDAYS_URL,
)
register_exchange(
    "GLBX",
    _make_cme_globex,
    aliases=["Globex", "CME-GLOBEX"],
    adhoc_url=_CME_HOLIDAYS_URL,
)
register_exchange(
    "XNAS",
    _make_nasdaq,
    aliases=["NASDAQ"],
    adhoc_url="https://www.nasdaqtrader.com/Trader.aspx?id=Calendar",
)
register_exchange(
    "XCBO",
    _make_cboe,
    aliases=["CBOE"],
    adhoc_url="https://www.cboe.com/about/hours/us-equities/",
)
register_exchange(
    "XTSE",
    _make_tsx,
    aliases=["TSX", "Toronto"],
    adhoc_url="https://www.tsx.com/trading/calendars-and-trading-hours/calendar",
)
register_exchange(
    "XETR",
    _make_xetra,
    aliases=["Xetra", "Frankfurt"],
    adhoc_url="https://www.xetra.com/xetra-en/trading/trading-calendar-and-trading-hours",
)
register_exchange(
    "XTKS",
    _make_jpx,
    aliases=["JPX", "Tokyo", "TSE"],
    adhoc_url="https://www.jpx.co.jp/english/equities/trading/domestic/index.html",
)
register_exchange(
    "XHKG",
    _make_hkex,
    aliases=["HKEX", "HongKong"],
    adhoc_url="https://www.hkex.com.hk/Services/Trading-hours-and-டhalves?sc_lang=en",
)
register_exchange("CMES", _make_fx, aliases=["CME-FX", "FX"])
