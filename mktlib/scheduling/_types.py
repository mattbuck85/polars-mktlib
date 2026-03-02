from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime


@dataclass
class MarketDailySchedule:
    """Single-day market schedule — matches tradesignalcore's dataclass."""

    date: date
    market_open: datetime
    market_close: datetime
    break_start: datetime | None = None
    break_end: datetime | None = None


def parse_date(d: date | str) -> date:
    if isinstance(d, str):
        return date.fromisoformat(d)
    return d
