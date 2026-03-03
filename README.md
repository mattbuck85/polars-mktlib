# mktlib

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Python 3.14+](https://img.shields.io/badge/python-3.14+-blue.svg)](https://www.python.org/downloads/)

Polars-native financial market toolkit. Zero pandas dependency.

## Table of Contents

- [Installation](#installation)
- [Scheduling](#scheduling)
  - [Supported Exchanges](#supported-exchanges)
  - [Schedule & Trading Days](#schedule--trading-days)
  - [Session Navigation](#session-navigation)
  - [Minute-Level Queries](#minute-level-queries)
  - [Trading Index](#trading-index)
  - [Custom Calendars](#custom-calendars)
  - [Holiday Rules](#holiday-rules)
  - [ExchangeCalendar API](#exchangecalendar)
- [Rates](#rates--treasury-yield-curves)
  - [Quick Start](#rates-quick-start)
  - [Available Instruments](#available-instruments)
  - [Caching](#caching)
  - [Rates API](#rates-api)
- [Reports](#reports--tearsheet-generation)
  - [Quick Start](#reports-quick-start)
  - [Input Types](#input-types)
  - [Auto Risk-Free Rate](#auto-risk-free-rate)
  - [Metrics](#metrics-25)
  - [Charts](#charts-8)
  - [Custom Metrics, Charts & Templates](#custom-metrics-charts--templates)
  - [Reports API](#reports-api)
  - [Migration from quantstats](#migration-from-quantstats)
- [Development](#development)
- [License](#license)

## Installation

```bash
pip install mktlib              # core (scheduling + rates)
pip install mktlib[reports]     # + tearsheet generation (plotly, jinja2)
```

## Scheduling

### Supported Exchanges

| Exchange | ID | Aliases | Hours | Timezone |
|-|-|-|-|-|
| NYSE | `XNYS` | `NYSE` | 09:30 - 16:00 | America/New_York |
| NASDAQ | `XNAS` | `NASDAQ` | 09:30 - 16:00 | America/New_York |
| CBOE Options | `XCBO` | `CBOE` | 09:30 - 16:15 | America/New_York |
| LSE | `XLON` | `LSE`, `London` | 08:00 - 16:30 | Europe/London |
| Euronext | `XPAR` | `Euronext`, `Paris` | 09:00 - 17:30 | Europe/Paris |
| Xetra | `XETR` | `Xetra`, `Frankfurt` | 09:00 - 17:30 | Europe/Berlin |
| TSX | `XTSE` | `TSX`, `Toronto` | 09:30 - 16:00 | America/Toronto |
| CME RTH | `XCME` | `CME`, `CME-RTH` | 09:30 - 16:15 | America/New_York |
| CME Globex | `GLBX` | `Globex`, `CME-GLOBEX` | 18:00 - 17:00 | America/New_York |
| JPX (Tokyo) | `XTKS` | `JPX`, `Tokyo`, `TSE` | 09:00 - 15:00 | Asia/Tokyo |
| HKEX | `XHKG` | `HKEX`, `HongKong` | 09:30 - 16:00 | Asia/Hong_Kong |
| FX (24/5) | `CMES` | `CME-FX`, `FX` | 17:00 - 17:00 | America/New_York |

FX is a pure weekday calendar (no holidays) with 24-hour sessions (5pm-5pm ET).

Each calendar includes holidays, ad-hoc closures, and early closes with full observance rules. JPX and HKEX include lunch break support — `schedule()` returns `break_start`/`break_end` columns, and `is_open_on_minute()` returns `False` during breaks.

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
from mktlib.scheduling import ExchangeCalendar, register_exchange
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

## Rates — Treasury Yield Curves

`mktlib.rates` fetches daily Treasury yield curve data from Treasury.gov with a 3-tier caching strategy and bundled historical fallback. No API key required.

### Rates Quick Start

```python
from mktlib.rates import get_risk_free_rate, TreasuryRate

# Average 3-month T-bill rate for 2024 (default instrument)
rf = get_risk_free_rate("2024-01-01", "2024-12-31")
# Returns 0.0523 (i.e. 5.23%)

# Use a different instrument
rf_10y = get_risk_free_rate("2024-01-01", "2024-12-31", TreasuryRate.TEN_YEAR)
```

### Available Instruments

| Enum Member | Treasury Field | Description |
|-|-|-|
| `TreasuryRate.THREE_MONTH` | `BC_3MONTH` | 3-month T-bill (default, standard risk-free proxy) |
| `TreasuryRate.SIX_MONTH` | `BC_6MONTH` | 6-month T-bill |
| `TreasuryRate.ONE_YEAR` | `BC_1YEAR` | 1-year Treasury |
| `TreasuryRate.TWO_YEAR` | `BC_2YEAR` | 2-year Treasury |
| `TreasuryRate.FIVE_YEAR` | `BC_5YEAR` | 5-year Treasury |
| `TreasuryRate.TEN_YEAR` | `BC_10YEAR` | 10-year Treasury |
| `TreasuryRate.THIRTY_YEAR` | `BC_30YEAR` | 30-year Treasury |

### Caching

Data is cached at three levels to minimize network requests:

1. **In-memory** — per-year data cached for the process lifetime
2. **Disk** — `~/.cache/mktlib/rates/{year}.csv` with 7-day TTL for the current year; past years never expire
3. **Bundled** — historical CSVs (2006-2026) shipped with the package for offline use

On network failure, the library falls back to stale disk cache or bundled data and emits a `UserWarning`.

### Rates API

```python
def get_risk_free_rate(
    start: date | str,
    end: date | str,
    instrument: TreasuryRate = TreasuryRate.THREE_MONTH,
) -> float: ...
```

Returns the arithmetic mean of daily Treasury yields as a decimal (e.g. `0.0436` for 4.36%).

## Reports — Tearsheet Generation

`mktlib.reports` is a Polars-native replacement for quantstats. It computes 25 performance metrics and renders an interactive HTML tearsheet with Plotly charts — no pandas, matplotlib, or seaborn required.

### Reports Quick Start

```python
from mktlib.reports import html, metrics

# From a Polars DataFrame with 'date' and 'return' columns
html(returns_df, output="tearsheet.html", title="My Strategy")

# From a bare Polars Series (synthetic dates are generated)
html(returns_series, benchmark=bench_series, output="report.html")

# Metrics only (no HTML)
result = metrics(returns_df, benchmark=bench_df, rf=0.05)
print(result.sharpe, result.max_drawdown, result.cagr)
```

### Input Types

Both `html()` and `metrics()` accept any of:

| Type | Notes |
|-|-|
| `pl.DataFrame` | Must have `date` (Date/Datetime) and `return` (Float64) columns, or columns will be inferred |
| `pl.Series` | Bare returns; synthetic business-day dates are generated starting from 2000-01-03 |
| `pd.Series` | Duck-typed via `PandasConvertible` protocol; DatetimeIndex is converted to `pl.Date` automatically |

### Auto Risk-Free Rate

Pass `rf="auto"` to automatically fetch the 3-month T-bill average for the returns period via `mktlib.rates`:

```python
html(returns_df, rf="auto", output="tearsheet.html")
result = metrics(returns_df, rf="auto")
```

### Metrics (25)

| Category | Metrics |
|-|-|
| Returns | Cumulative, CAGR, MTD, YTD, 1Y |
| Ratios | Sharpe, Sortino, Calmar, Omega, RoMaD |
| Risk | Max DD, Max DD Date, Longest DD Days, Avg DD, Volatility (ann.) |
| Tail | VaR (95%), CVaR (95%) |
| Win/Loss | Win Rate, Payoff Ratio, Profit Factor, Kelly Criterion |
| Benchmark | Alpha, Beta, R-squared, Information Ratio |

### Charts (8)

Cumulative returns (with optional benchmark overlay), drawdown underwater, monthly returns heatmap, yearly returns bar, rolling Sharpe (126d), rolling volatility (126d), daily returns scatter, returns distribution histogram.

All charts are interactive Plotly — hover for values, zoom, pan. Plotly JS is loaded via CDN.

### Custom Metrics, Charts & Templates

```python
import plotly.graph_objects as go
from pathlib import Path

# Add custom metric cards alongside the built-in 25
html(returns_df, extra_metrics={
    "Execution": [("Trades", "142"), ("Avg Slippage", "0.02%")],
    "Custom": [("Foo", "42")],
})

# Append extra Plotly charts after the built-in 8
fig = go.Figure(data=go.Scatter(x=dates, y=pnl))
fig.update_layout(title="PnL Curve")
html(returns_df, extra_charts={"pnl": fig})

# Full control with a custom Jinja2 template
html(returns_df, template=Path("my_tearsheet.j2"),
     extra_metrics={"Custom": [("Foo", "42")]},
     extra_charts={"pnl": fig})
```

Custom templates receive: `title`, `start_date`, `end_date`, `trading_days`, `metrics_groups` (list of `(category, [(label, value), ...])` tuples including extras), `charts` (built-in HTML divs), and `extra_charts` (extra HTML divs). Pass a `Path` to load from file or a `str` for inline Jinja2.

### Reports API

```python
def html(
    returns: ReturnsInput,
    *,
    benchmark: ReturnsInput | None = None,
    output: str | None = None,          # file path; None → return HTML string
    title: str = "Strategy Tearsheet",
    rf: float | str = 0.0,              # float or "auto"
    periods_per_year: int = 252,
    compounded: bool = True,
    extra_metrics: dict[str, list[tuple[str, str]]] | None = None,
    extra_charts: dict[str, go.Figure] | None = None,
    template: str | Path | None = None,
) -> str | None: ...

def metrics(
    returns: ReturnsInput,
    *,
    benchmark: ReturnsInput | None = None,
    rf: float | str = 0.0,
    periods_per_year: int = 252,
    compounded: bool = True,
) -> MetricsResult: ...
```

`MetricsResult` is a dataclass with all 25 metrics as named fields. Benchmark fields (`alpha`, `beta`, `r_squared`, `information_ratio`) are `None` when no benchmark is provided.

### Migration from quantstats

```python
# Before
import quantstats as qs
qs.reports.html(returns, benchmark=bench, output="report.html", title="My Strategy")

# After
from mktlib.reports import html
html(returns, benchmark=bench, output="report.html", title="My Strategy")
```

`pd.Series` inputs continue to work during migration. Switch to `pl.Series` or `pl.DataFrame` to eliminate the pandas dependency entirely.

## Development

```bash
pip install -e ".[dev,reports]"
pytest
pyright mktlib
pre-commit install  # trailing whitespace, flake8, pyright
```

## License

Apache 2.0
