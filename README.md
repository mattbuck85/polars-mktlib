# mktlib

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Documentation](https://readthedocs.org/projects/polars-mktlib/badge/?version=latest)](https://polars-mktlib.readthedocs.io/en/latest/)

Financial market toolkit built entirely on Polars.

### 📬 tl;dr

⚡ **Fast enough for real work** — vectorized Polars engine grid-searches thousands of parameter combos on minute-bar data without reaching for Numba or Cython

📦 **Lightweight** — pure Polars throughout, including synthetic data generation via Rust-native plugins. No pandas, no NumPy, no heavy ML stack

🧪 **Well tested** — cross-validated exchange calendars, property-based OHLCV checks, and full backtest parity tests across engines

🧰 **Swiss-army knife** — scheduling, rates, metrics, backtesting, reporting, and data generation in one package. Great for learning, prototyping, or production

📄 **Apache 2.0** — use it anywhere, fork it, vendor it, no strings attached

> **Disclaimer:** Backtesting results and any computed market returns are for **educational and research purposes only**. They do not constitute financial advice, and past performance does not indicate future results.

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
- [Metrics](#metrics--standalone-functions)
  - [Quick Start](#metrics-quick-start)
  - [Available Metrics](#available-metrics)
  - [Dispatcher](#dispatcher)
  - [Drawdown Series](#drawdown-series)
- [Backtest](#backtest--vectorized-engine)
  - [Quick Start](#backtest-quick-start)
  - [Conditions](#conditions)
  - [Calendar & Flatten EOD](#calendar--flatten-eod)
  - [Trade Side](#trade-side)
  - [Backtest API](#backtest-api)
- [Reports](#reports--tearsheet-generation)
  - [Quick Start](#reports-quick-start)
  - [Input Types](#input-types)
  - [Auto Risk-Free Rate](#auto-risk-free-rate)
  - [Metrics](#metrics-25)
  - [Charts](#charts-8)
  - [Custom Metrics, Charts & Templates](#custom-metrics-charts--templates)
  - [Reports API](#reports-api)
  - [Migration from quantstats](#migration-from-quantstats)
- [Data](#data--synthetic-generators)
- [Development](#development)
- [License](#license)

## Installation

```bash
pip install mktlib              # core (scheduling, rates, metrics, backtest)
pip install mktlib[data]        # + synthetic data generators (polars-sdist, polars-rfft)
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

### Filter Market Hours

Filter an existing DataFrame to market hours — more efficient than
`trading_index()` when you already have data:

```python
cal = get_calendar("NYSE")

# Filter intraday bars to market hours only
filtered = cal.filter_market_hours(df, date_column="date")
```

Handles early closes, timezone alignment (naive or aware), and lunch
breaks (JPX, HKEX, SSE) automatically.

### Custom Calendars

```python
from datetime import date, time
from mktlib.scheduling import ExchangeCalendar, register_exchange
from mktlib.scheduling.rules import HolidayRule, AdhocClosure, EarlyClose

cal = ExchangeCalendar(
    name="XICE",
    timezone="Atlantic/Reykjavik",
    open_time=time(9, 30),
    close_time=time(15, 30),
    holidays=[
        HolidayRule("New Year's Day", month=1, day=1),
        HolidayRule("National Day", month=6, day=17),
        HolidayRule("Commerce Day", month=8, weekday=0, week=1),  # 1st Monday
    ],
    adhoc_closures=[AdhocClosure("Special", [date(2024, 1, 4)])],
    early_closes=[EarlyClose("Half Day", close_time=time(12, 0), dates=[date(2024, 12, 31)])],
)

register_exchange("XICE", lambda: cal, aliases=["Iceland"])
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
| `filter_market_hours(df, date_column)` | `pl.DataFrame` | Filter rows to market hours |
| `trading_index(start, end, period, closed)` | `pl.Series` | Intraday timestamp index |

All date parameters accept `date` objects or ISO-format strings (`"2024-01-02"`).

## Rates — Treasury Yield Curves

`mktlib.rates` fetches daily Treasury yield curve data from Treasury.gov with a 3-tier caching strategy and bundled historical fallback. No API key required.

### Rates Quick Start

```python
from mktlib.rates import (
    TreasuryRate, MeanMethod,
    get_risk_free_rate, get_mean_treasury_rate,
    get_treasury_rates, get_treasury_spread,
)

# Average 3-month T-bill rate for 2024 (default instrument)
rf = get_risk_free_rate("2024-01-01", "2024-12-31")
# Returns 0.0523 (i.e. 5.23%)

# Geometric mean of 10-year yield
geo = get_mean_treasury_rate("2024-01-01", "2024-12-31",
                             TreasuryRate.TEN_YEAR, MeanMethod.GEOMETRIC)

# Daily rates as a Polars DataFrame
df = get_treasury_rates("2024-01-01", "2024-03-31", TreasuryRate.TEN_YEAR)
# shape: (N, 2) — columns: date, rate

# Multiple instruments → wide DataFrame
df = get_treasury_rates("2024-01-01", "2024-03-31",
                        [TreasuryRate.TWO_YEAR, TreasuryRate.TEN_YEAR])
# columns: date, two_year, ten_year

# All 14 instruments
df = get_treasury_rates("2024-01-01", "2024-03-31")
# columns: date, one_month, two_month, ..., thirty_year_display

# Yield curve spread (10Y - 2Y by default)
spread = get_treasury_spread("2024-01-01", "2024-03-31")
# columns: date, spread
```

### Available Instruments

| Enum Member | Treasury Field | Description |
|-|-|-|
| `TreasuryRate.ONE_MONTH` | `BC_1MONTH` | 1-month T-bill |
| `TreasuryRate.TWO_MONTH` | `BC_2MONTH` | 2-month T-bill |
| `TreasuryRate.THREE_MONTH` | `BC_3MONTH` | 3-month T-bill (default, standard risk-free proxy) |
| `TreasuryRate.FOUR_MONTH` | `BC_4MONTH` | 4-month T-bill |
| `TreasuryRate.SIX_MONTH` | `BC_6MONTH` | 6-month T-bill |
| `TreasuryRate.ONE_YEAR` | `BC_1YEAR` | 1-year Treasury |
| `TreasuryRate.TWO_YEAR` | `BC_2YEAR` | 2-year Treasury |
| `TreasuryRate.THREE_YEAR` | `BC_3YEAR` | 3-year Treasury |
| `TreasuryRate.FIVE_YEAR` | `BC_5YEAR` | 5-year Treasury |
| `TreasuryRate.SEVEN_YEAR` | `BC_7YEAR` | 7-year Treasury |
| `TreasuryRate.TEN_YEAR` | `BC_10YEAR` | 10-year Treasury |
| `TreasuryRate.TWENTY_YEAR` | `BC_20YEAR` | 20-year Treasury |
| `TreasuryRate.THIRTY_YEAR` | `BC_30YEAR` | 30-year Treasury |
| `TreasuryRate.THIRTY_YEAR_DISPLAY` | `BC_30YEARDISPLAY` | 30-year Treasury (display) |

### Field Discovery

The set of known Treasury instruments is defined in the bundled `_data/schema.csv` file — a year-by-field presence matrix recording which BC_* fields exist for each year. Treasury.gov has added instruments over time (e.g. BC_2MONTH appeared in 2018, BC_4MONTH in 2022).

When the refresh script (`scripts/refresh_treasury_data.py`) fetches data from Treasury.gov, any new BC_* fields discovered in the XML are automatically appended to `schema.csv`. No code changes are needed to support new instruments.

### Caching

Data is cached at three levels to minimize network requests:

1. **In-memory** — per-year data cached for the process lifetime
2. **Disk** — `~/.cache/mktlib/rates/{year}.csv` with 7-day TTL for the current year; past years never expire
3. **Bundled** — historical CSVs (2000-2026) shipped with the package for offline use

On network failure, the library falls back to stale disk cache or bundled data and emits a `UserWarning`.

### Rates API

```python
def get_risk_free_rate(
    start: date | str,
    end: date | str,
    instrument: TreasuryRate = TreasuryRate.THREE_MONTH,
) -> float: ...

def get_mean_treasury_rate(
    start: date | str,
    end: date | str,
    instrument: TreasuryRate = TreasuryRate.THREE_MONTH,
    method: MeanMethod = MeanMethod.ARITHMETIC,
) -> float: ...

def get_treasury_rates(
    start: date | str,
    end: date | str,
    instrument: TreasuryRate | Sequence[TreasuryRate] | None = None,
) -> pl.DataFrame: ...

def get_treasury_spread(
    start: date | str,
    end: date | str,
    long: TreasuryRate = TreasuryRate.TEN_YEAR,
    short: TreasuryRate = TreasuryRate.TWO_YEAR,
) -> pl.DataFrame: ...
```

| Function | Returns | Description |
|-|-|-|
| `get_risk_free_rate` | `float` | Arithmetic mean of daily yields as a decimal (e.g. `0.0436`) |
| `get_mean_treasury_rate` | `float` | Mean rate with configurable method (arithmetic or geometric) |
| `get_treasury_rates` | `pl.DataFrame` | Daily rates — single (`date`, `rate`), multi/all (wide, one col per instrument) |
| `get_treasury_spread` | `pl.DataFrame` | Daily spread (`date`, `spread`) between two instruments |

## Metrics — Standalone Functions

`mktlib.metrics` provides standalone financial metric functions operating on Polars return series. No dependencies beyond polars.

### Metrics Quick Start

```python
from mktlib.metrics import (
    sharpe, sortino, cumulative_return, cagr,
    drawdown_series, calculate_metric, Metric,
)

# Individual functions
sr = sharpe(returns_series, rf=0.05)
cr = cumulative_return(returns_series)
dd = drawdown_series(returns_series)

# Dispatcher — compute any metric by enum
sr = calculate_metric(Metric.SHARPE, returns_series, rf=0.05)
```

### Available Metrics

| Function | Signature | Description |
|-|-|-|
| `cumulative_return` | `(ret, compounded=True)` | Total cumulative return |
| `cagr` | `(ret, compounded=True, ppy=252)` | Compound annual growth rate |
| `annualized_volatility` | `(ret, ppy=252)` | Annualized std deviation |
| `sharpe` | `(ret, ppy=252, rf=0.0)` | Annualized Sharpe ratio |
| `sortino` | `(ret, ppy=252, rf=0.0)` | Annualized Sortino ratio (downside deviation) |
| `omega` | `(ret, ppy=252, rf=0.0)` | Omega ratio (gains / losses above threshold) |
| `var` | `(ret, alpha=0.05)` | Value at Risk at alpha confidence |
| `cvar` | `(ret, alpha=0.05)` | Conditional VaR (Expected Shortfall) |
| `win_rate` | `(ret)` | Fraction of positive-return bars |
| `payoff_ratio` | `(ret)` | Average win / average loss |
| `profit_factor` | `(ret)` | Sum of gains / sum of losses |
| `kelly_criterion` | `(ret)` | Kelly criterion from bar-level returns |
| `avg_drawdown` | `(dd)` | Average drawdown during episodes |
| `longest_drawdown_days` | `(dd, dates)` | Longest drawdown in calendar days |

All functions accept `pl.Series` and return `float`. Empty inputs return `0.0`.

### Dispatcher

```python
from mktlib.metrics import calculate_metric, Metric

result = calculate_metric(
    Metric.SHARPE,
    returns_series,
    ppy=252,
    rf=0.05,
    alpha=0.05,       # for VaR/CVaR
    compounded=True,   # for cumulative return / drawdown
    dd=dd_series,      # optional pre-computed drawdown
    dates=date_series, # required for LONGEST_DRAWDOWN_DAYS
)
```

Lazily computes drawdown when needed. `CALMAR` (CAGR / max DD), `ROMAD` (cum return / max DD), and `MAX_DRAWDOWN` are computed inline in the dispatcher.

### Drawdown Series

```python
dd = drawdown_series(returns_series, compounded=True)
# Returns pl.Series of drawdown values (0 = at peak, negative = below peak)
```

## Backtest — Vectorized Engine

`mktlib.backtest` provides a signal-driven backtesting engine with fill-at-next-open semantics. Strategies define entry/exit conditions as composable Polars expressions. Supports exchange calendar filtering, session-boundary position management, and long/short sides.

### Backtest Quick Start

```python
from dataclasses import dataclass
from mktlib.backtest import run, Crossover, Crossunder, BacktestResult

@dataclass(frozen=True, slots=True)
class SmaCross:
    def entry(self) -> Crossover:
        return Crossover("fast_sma", "slow_sma")

    def exit(self) -> Crossunder:
        return Crossunder("fast_sma", "slow_sma")

# df must have: date, open, close, fast_sma, slow_sma
result: BacktestResult = run(df, SmaCross())

result.returns   # DataFrame[date, return] — per-bar strategy returns
result.trades    # DataFrame[entry_date, exit_date, pnl, bars_held]
result.signals   # Full frame with _entry, _exit, _position columns
```

**Fill-at-next-open model**: signal at bar *t* generates a market order that fills at bar *t+1*'s open.

| Bar type | Return formula |
|-|-|
| Entry bar (*t+1*) | `(close - open) / open` |
| Middle bars | `close / prev_close - 1` |
| Exit bar | `(open - prev_close) / prev_close` |

### Conditions

Conditions resolve to boolean `pl.Expr` and compose with `&`, `|`, `~`:

```python
from mktlib.backtest import (
    Crossover, Crossunder, PriceIsAbove, PriceIsBelow,
    IsRising, IsFalling, All, Any_, Not,
)

# Column crosses above column or constant
entry = Crossover("fast", "slow")
entry = Crossover("rsi", 30.0)

# Compose conditions
entry = Crossover("fast", "slow") & PriceIsAbove("close", "sma_200")
exit = Crossunder("fast", "slow") | PriceIsBelow("close", "stop_loss")
```

| Condition | Description |
|-|-|
| `Crossover(a, b)` | `a` crosses above `b` (column or constant) |
| `Crossunder(a, b)` | `a` crosses below `b` |
| `PriceIsAbove(a, b)` | `a > b` |
| `PriceIsBelow(a, b)` | `a < b` |
| `IsRising(col, period)` | Value > value `period` bars ago |
| `IsFalling(col, period)` | Value < value `period` bars ago |
| `All(a, b)` / `a & b` | Both conditions true |
| `Any_(a, b)` / `a \| b` | Either condition true |
| `Not(a)` / `~a` | Invert condition |
| `Custom(expr)` | Any `pl.Expr` evaluating to boolean |

For arbitrary logic, use `Custom` or return a bare `pl.Expr` from `entry()`/`exit()` (auto-wrapped as `Custom`):

```python
from mktlib.backtest import Custom, TradeSide

# Any polars expression — with explicit short side
entry = Custom(pl.col("rsi") > 70, trade_side=TradeSide.SHORT)
exit_ = Custom(pl.col("rsi") < 30)

# Or return a bare pl.Expr — auto-wrapped as Custom with the run() default side
class RsiStrategy:
    def entry(self) -> pl.Expr:
        return pl.col("rsi") < 30

    def exit(self) -> pl.Expr:
        return pl.col("rsi") > 70
```

### Calendar & Flatten EOD

```python
from mktlib.scheduling import get_calendar

cal = get_calendar("NYSE")

# Filter to market hours only
result = run(df, SmaCross(), calendar=cal)

# Force-close positions at session end (no overnight exposure)
result = run(df, SmaCross(), calendar=cal, flatten_eod=True)
```

With `flatten_eod=True`:
- Positions are forced to 0 at each session's last bar (e.g. 15:59 for NYSE)
- Entries are suppressed on session-last bars
- Session-forced exits fill at the session-last bar's open (not next session's open)
- Overnight gaps are never captured

### Trade Side

```python
from mktlib.backtest import TradeSide

# Short side — returns are negated
result = run(df, SmaCross(), trade_side=TradeSide.SHORT)

# Or set trade_side on the entry condition (overrides run() default)
entry = Crossover("fast", "slow", trade_side=TradeSide.SHORT)
```

### Backtest API

```python
def run(
    df: pl.DataFrame,
    strategy: Strategy,
    *,
    trade_side: TradeSide = TradeSide.LONG,
    calendar: ExchangeCalendar | None = None,
    flatten_eod: bool = False,
) -> BacktestResult: ...
```

| Parameter | Description |
|-|-|
| `df` | Must contain `date`, `open`, `close`, and indicator columns |
| `strategy` | Object with `entry()` and `exit()` returning `Condition` |
| `trade_side` | `LONG` (+1) or `SHORT` (-1); overridden by condition's `trade_side` |
| `calendar` | Exchange calendar for market-hours filtering |
| `flatten_eod` | Force-close at session end; requires `calendar` |

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

## Data — Synthetic Generators

`mktlib.data` provides stochastic process generators for testing, simulation, and Monte Carlo analysis. All functions return Polars DataFrames with seeded RNG for reproducibility.

```python
from mktlib.data import (
    fractional_random_walk,
    geometric_brownian_motion,
    monte_carlo,
    ornstein_uhlenbeck,
)

# Standard random walk
walk = fractional_random_walk(1000, seed=42)

# Trending path (Hurst > 0.5)
trending = fractional_random_walk(1000, hurst=0.8, seed=42)

# GBM price path — 252 daily steps, 5% annualised drift, 20% annualised vol
gbm = geometric_brownian_motion(252, drift=0.05, volatility=0.20, seed=42)

# Mean-reverting process
ou = ornstein_uhlenbeck(500, theta=0.7, mu=100.0, sigma=1.0, seed=42)

# 1000 Monte Carlo simulations of GBM
sims = monte_carlo(geometric_brownian_motion, n_simulations=1000, n=252, seed=42)
# Returns DataFrame with columns [simulation, step, price]
```

| Function | Process | Output columns |
|-|-|-|
| `fractional_random_walk` | Fractional Brownian motion (Davies-Harte circulant embedding + RustFFT, O(n log n)) | `step`, `price` |
| `geometric_brownian_motion` | Log-normal GBM: dS = μSdt + σSdW | `step`, `price` |
| `ornstein_uhlenbeck` | Mean-reverting: dx = θ(μ−x)dt + σdW | `step`, `value` |
| `monte_carlo` | N simulations of any generator | `simulation`, `step`, ... |

The `dt` parameter defaults to `1/252` (one trading day in annualised units). Pass `drift` and `volatility` as annualised values directly — the function scales them internally. For sub-daily data generation (e.g. 1-minute OHLCV bars), see the [advanced usage guide](https://polars-mktlib.readthedocs.io/en/latest/advanced.html).

## Development

```bash
pip install -e ".[dev,data,reports]"
pytest
pyright mktlib
pre-commit install  # trailing whitespace, flake8, pyright
```

## License

Apache 2.0
