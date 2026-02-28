# mktlib

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Python 3.14+](https://img.shields.io/badge/python-3.14+-blue.svg)](https://www.python.org/downloads/)

Polars-native financial market toolkit. Zero pandas dependency.

## Installation

```bash
pip install mktlib
```

## Supported Exchanges

| Exchange | ID | Aliases | Hours | Timezone |
|-|-|-|-|-|
| NYSE | `XNYS` | `NYSE` | 09:30 - 16:00 | America/New_York |
| LSE | `XLON` | `LSE`, `London` | 08:00 - 16:30 | Europe/London |
| Euronext | `XPAR` | `Euronext`, `Paris` | 09:00 - 17:30 | Europe/Paris |

Each calendar includes holidays, ad-hoc closures, and early closes with full observance rules.

## Usage

### Schedule & Trading Days

```python
from mktlib.scheduling import get_calendar

cal = get_calendar("NYSE")

# Trading days as a Polars Series (pl.Date)
days = cal.valid_days("2024-01-01", "2024-12-31")

# Full schedule as a Polars DataFrame (date, market_open, market_close)
schedule = cal.schedule("2024-01-02", "2024-01-31")

# Single-day schedule
sched = cal.get_schedule("2024-11-29")  # Black Friday → early close at 13:00
```

### Session Navigation

```python
cal.next_session("2024-01-05")            # date(2024, 1, 8) — skips weekend
cal.previous_session("2024-12-26")        # date(2024, 12, 24) — skips Christmas
cal.session_offset("2024-01-08", 5)       # 5 trading days forward
cal.date_to_session("2024-01-06", "next") # snap non-session to next trading day
cal.sessions_in_range("2024-01-01", "2024-12-31")  # ~252
```

### Minute-Level Queries

```python
from datetime import datetime
from zoneinfo import ZoneInfo

dt = datetime(2024, 1, 2, 12, 0, tzinfo=ZoneInfo("America/New_York"))

cal.is_open_on_minute(dt)   # True — [open, close) semantics
cal.next_open(dt)            # next market open after dt
cal.next_close(dt)           # next market close at or after dt
cal.previous_open(dt)        # most recent open before dt
cal.previous_close(dt)       # most recent close before dt
cal.minute_to_session(dt)    # date(2024, 1, 2) or None if closed
```

Naive datetimes are assumed to be in the exchange's timezone. Aware datetimes in other zones are converted automatically.

### Trading Index

```python
# Intraday timestamps at any frequency
idx = cal.trading_index("2024-01-02", "2024-01-02", period="5m")
# Returns pl.Series of pl.Datetime("us", "America/New_York")

# Control interval boundaries: "left" (default), "right", "both", "none"
idx = cal.trading_index("2024-01-02", "2024-01-05", period="1m", closed="right")
```

### Custom Calendars

```python
from datetime import time
from mktlib.scheduling.calendar import ExchangeCalendar, register_exchange
from mktlib.scheduling.rules import HolidayRule, AdhocClosure, EarlyClose

cal = ExchangeCalendar(
    name="XTKS",
    timezone="Asia/Tokyo",
    open_time=time(9, 0),
    close_time=time(15, 0),
    holidays=[
        HolidayRule("New Year's Day", month=1, day=1),
        HolidayRule("Coming of Age Day", month=1, weekday=0, week=2),  # 2nd Monday
    ],
    adhoc_closures=[AdhocClosure("Special", [date(2024, 1, 4)])],
    early_closes=[EarlyClose("Half Day", close_time=time(11, 30), dates=[date(2024, 12, 31)])],
)

# Register for lookup via get_calendar()
register_exchange("XTKS", lambda: cal, aliases=["TSE", "Tokyo"])
```

### Holiday Rules

| Rule type | Description |
|-|-|
| `HolidayRule` | Recurring holiday with optional observance (`nearest_workday`, `sunday_to_monday`, `previous_friday`) |
| `AdhocClosure` | One-off closure dates (e.g. 9/11, Hurricane Sandy) |
| `EarlyClose` | Early close with specific close time, by rule or explicit dates |

## API Reference

### `ExchangeCalendar`

| Method | Returns | Description |
|-|-|-|
| `valid_days(start, end)` | `pl.Series` | Trading dates in range |
| `schedule(start, end)` | `pl.DataFrame` | Open/close times per day |
| `is_session(day)` | `bool` | Whether a date is a trading day |
| `get_schedule(day)` | `MarketDailySchedule \| None` | Single-day open/close |
| `next_session(day)` | `date` | First trading day after day |
| `previous_session(day)` | `date` | Last trading day before day |
| `session_offset(day, n)` | `date` | Offset by n sessions |
| `date_to_session(day, direction)` | `date` | Snap to session |
| `sessions_in_range(start, end)` | `int` | Count of trading days |
| `is_open_on_minute(dt)` | `bool` | Market open at datetime |
| `next_open(dt)` | `datetime` | Next market open |
| `next_close(dt)` | `datetime` | Next market close |
| `previous_open(dt)` | `datetime` | Previous market open |
| `previous_close(dt)` | `datetime` | Previous market close |
| `minute_to_session(dt)` | `date \| None` | Session containing datetime |
| `trading_index(start, end, period, closed)` | `pl.Series` | Intraday timestamp index |

All date parameters accept `date` objects or ISO-format strings (`"2024-01-02"`).

## Development

```bash
pip install -e ".[dev]"
pytest
pyright mktlib
pre-commit install  # trailing whitespace, flake8, pyright
```

## License

Apache 2.0
