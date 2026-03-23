# Changelog

## 0.8.0

### Added

- **`ValueGT` / `ValueLT` conditions** — renamed from `PriceIsAbove` / `PriceIsBelow` for clarity. Old names remain as aliases for backward compatibility.
- **`ValueGTE` / `ValueLTE` conditions** — new `>=` and `<=` comparisons.
- **`ColExpr` comparison operators** — `Col("a") > 100` now returns `ValueGT(Col("a"), Lit(100.0))`, enabling natural expression syntax. Supports `>`, `>=`, `<`, `<=`.
- **`ColExpr` base class** — renamed from `PriceExpr`. Old name remains as alias.
- **`InitStrategy` protocol** — typed protocol for strategies that define `init(df) -> DataFrame`, extending the base `Strategy` protocol.
- **`strategy_artifact(strategy)`** — deterministic 16-char hex fingerprint for any strategy instance. Evaluates the `entry()`/`exit()` condition trees and hashes them with all `Lit` values flattened to `0`, plus the `init()` source normalized via `ast.unparse` (formatting/comment-insensitive, with same-module reference following). **Parameter-insensitive** — stable across different parameter values for the same strategy class.
- **`combined_strategy_artifact(long, short)`** — deterministic 16-char hex digest for a long+short strategy pair. Delegates to `strategy_artifact` for each side. **Parameter-insensitive** — stable across different parameter values for the same strategy class. Ideal for caching during optimization sweeps.
- **`register_alias(cls, name)`** — register user-defined wrapper classes for canonical artifact hashing.
- **Dual-strategy long/short** — `run(df, long_strategy, short_strategy=short_strategy)` runs independent long and short strategies concurrently via `ThreadPoolExecutor`, merging positions, returns, and trades. Overlap detection raises `ValueError` if both sides try to hold simultaneously. Per-trade `side` column (+1/-1) in trades output, `_side` column in signals.
- **`_side` column** — signals and trades now include side information. `_side` in signals is `+1` (long), `-1` (short), or `0` (flat). `side` in trades is `+1` or `-1`. The side is determined by the `trade_side` parameter on `run()` or the entry condition's `trade_side` field (condition-level overrides `run()`-level).

## 0.7.2

### Added

- **`EntryRef` PriceExpr** — snapshots a column value at the entry signal bar and forward-fills it through the position lifetime. Enables TP/SL exits anchored to the entry price: `PriceIsAbove("close", Pct(EntryRef("close"), 5.0))` resolves to `close > _entry_close * 1.05`. The engine detects `EntryRef` nodes via a tree walker and creates snapshot columns automatically — no manual `init()` work needed.

### Fixed

- **TP/SL exits referencing current bar instead of entry bar** — `Pct("close", 5.0)` resolves to `close * 1.05`, making `PriceIsAbove("close", Pct("close", 5.0))` always false (`close > close * 1.05`). `EntryRef` provides the correct mechanism for entry-anchored thresholds.

## 0.7.1

### Changed

- **`ticks_to_ohlcv` vectorised** — replaced multi-step `shift(-1)` + `max_horizontal` pipeline with a single `group_by` aggregation using `first`/`max`/`min`/`last`, improving performance and eliminating the N+1 tick requirement.
- **GBM/OU annualised defaults** — `geometric_brownian_motion` and `ornstein_uhlenbeck` now default to `dt=1/252` (one trading day). Pass `drift` and `volatility` as annualised values directly; the function scales them internally.
- **Monte Carlo deterministic seeding** — `monte_carlo()` now derives per-simulation child seeds from the master seed via `random.Random`, making results fully reproducible. The `seed` column is included in the output for traceability.
- **Docs updated** — README, Sphinx quickstart, and advanced guide now use annualised conventions for all data generation examples. Added sub-daily `dt` guidance.

## 0.7.0

### Added

- **`Custom` condition** — wrap any `pl.Expr` as a backtest condition via `Custom(expr)`. `Strategy.entry()` and `exit()` now also accept bare `pl.Expr` returns (auto-wrapped to `Custom`).
- **`Strategy.init(df)` hook** — optional method on strategies to enrich the DataFrame with indicator columns before signal evaluation. Existing strategies without `init` are unaffected.
- **`MultiBacktestResult`** — returned by `run()` when `instrument_col` is set. Stores per-symbol `BacktestResult` instances for O(1) access (`result["AAPL"]`). Combined views (`.returns`, `.trades`, `.signals`) are lazy-cached with the instrument column prepended. Supports `len()`, iteration, `in`, and `.items()`.
- **`instrument_col` parameter on `run()`** — pass a column name to backtest a multi-symbol DataFrame. Each symbol is backtested independently (no indicator bleed), with calendar filtering applied once before partitioning.

### Changed

- `Strategy` protocol now accepts `Condition | pl.Expr` returns from `entry()` and `exit()`.
- Renamed `symbol_col` parameter to `instrument_col` for consistency with financial data conventions.
- **`fractional_random_walk`** now uses Davies-Harte circulant embedding + RustFFT (O(n log n)), replacing the previous Cholesky decomposition. Powered by [polars-rfft](https://github.com/mattbuck85/polars-rfft) and [polars-sdist](https://github.com/mattbuck85/polars-sdist).

### Fixed

- `TreasuryRate` enum now includes `ONE_AND_HALF_MONTH` (`BC_1_5MONTH`) to match Treasury.gov schema
- **`flatten_eod` deferred entry** — crossover signals on the last bar of a session were silently dropped. They now carry forward to the first bar of the next session (e.g. signal at 15:59 → fill at next day's 09:30).
- **`_build_session_last_mask` for non-1-minute candles** — session-last detection no longer hardcodes `market_close - 1min`; it finds the actual last bar per session from the data, fixing `flatten_eod` and deferred entry for 5min, 15min, and other candle sizes.

## 0.6.3

### Fixed

- Use a separate `README_PYPI.md` for the PyPI project page to avoid broken links.

## 0.6.2

### Added

- `ticks_to_ohlcv(ticks, bar_size)` — aggregate tick-level generator output into OHLCV bars with synthetic lognormal volume. Supports all generators (`column="price"` for GBM/fRW, `column="value"` for OU).
- `docs/advanced.rst` — end-to-end grid search optimization guide: SMA crossover parameterization, TP/SL exits, two-stage search scored by Sharpe ratio with risk-free rate from `get_risk_free_rate`.
- `scripts/grid_search_sma.py` — standalone runnable version of the advanced guide.

### Changed

- Benchmark scripts (`bench_backtest.py`, `bench_macd_market.py`, `bench_single_pass.py`) simplified — removed redundant imports and unused variables.

## 0.6.1

### Added

- `ExchangeCalendar.filter_market_hours(df)` — filter a DataFrame to market hours using an efficient schedule join. Recommended over `trading_index()` for filtering existing data.

### Changed

- Backtest engine now uses `filter_market_hours()` internally, removing duplicated schedule-join logic.

## 0.6.0

### Added

- **`mktlib.backtest` subpackage** — vectorized backtesting engine with fill-at-next-open semantics.
  - `run(df, strategy)` — main entry point. Accepts a DataFrame with OHLC data and a strategy object, returns `BacktestResult` with per-bar returns, trade log, and full signal frame.
  - `Strategy` protocol — implement `entry()` and `exit()` methods returning `Condition`.
  - `BacktestResult` dataclass — `returns` (DataFrame), `trades` (DataFrame), `signals` (DataFrame).
  - `TradeSide` enum — `LONG` / `SHORT`. Settable per-run or per-condition.
  - **Composable conditions** — `Crossover`, `Crossunder`, `PriceIsAbove`, `PriceIsBelow`, `IsRising`, `IsFalling`, `All`, `Any_`, `Not`. Compose with `&`, `|`, `~` operators.
  - **Composable price expressions** — `PriceExpr`, `Col`, `Lit`, `Pct` for building dynamic exit levels (e.g. take-profit / stop-loss) with full arithmetic (`+`, `-`, `*`, `/`, `%`).
  - Exchange calendar integration — optional `calendar` parameter filters data to market hours via schedule join. `flatten_eod=True` force-closes positions at session boundaries.
- **`mktlib.metrics` subpackage** — standalone financial metric functions on Polars return series.
  - `calculate_metric(Metric, ret)` — unified dispatcher for all 17 metrics with lazy drawdown computation.
  - `Metric` enum — `CUMULATIVE_RETURN`, `CAGR`, `ANNUALIZED_VOLATILITY`, `MAX_DRAWDOWN`, `AVG_DRAWDOWN`, `LONGEST_DRAWDOWN_DAYS`, `SHARPE`, `SORTINO`, `CALMAR`, `ROMAD`, `OMEGA`, `VAR`, `CVAR`, `WIN_RATE`, `PAYOFF_RATIO`, `PROFIT_FACTOR`, `KELLY_CRITERION`.
  - Standalone functions: `sharpe()`, `sortino()`, `calmar()`, `romad()`, `omega()`, `var()`, `cvar()`, `cumulative_return()`, `cagr()`, `annualized_volatility()`, `avg_drawdown()`, `longest_drawdown_days()`, `win_rate()`, `payoff_ratio()`, `profit_factor()`, `kelly_criterion()`, `drawdown_series()`.
- **`mktlib.reports` subpackage** — Polars-native tearsheet generation behind `[reports]` optional extra (`pip install mktlib[reports]`).
  - `html(returns, *, benchmark, output, ...)` — 25-metric interactive HTML tearsheet with 8 Plotly charts. Supports `pl.DataFrame`, `pl.Series`, and `pd.Series` inputs.
  - `metrics(returns, *, benchmark, ...)` — compute all 25 metrics without HTML output, returns `MetricsResult` dataclass.
  - `rf="auto"` — automatically fetches 3-month T-bill average from `mktlib.rates` for the returns period.
  - Custom metrics, charts, and Jinja2 templates via `extra_metrics`, `extra_charts`, and `template` parameters.
- Educational disclaimer in README.

## 0.5.4

### Changed

- Lowered minimum Python requirement from 3.14 to 3.12. No code changes needed — all features used are available in 3.12+.

### Added

- Sphinx documentation with Read the Docs pipeline.

## 0.5.3

### Fixed

- `fetch_average_rate` now falls back to the last available rate when the requested date range has no trading days, instead of returning 0.0.

### Added

- `_data/schema.csv` — year-by-field presence matrix recording which BC_* instruments exist per year, replacing hardcoded `_FIELDS` lists in `_disk_cache.py` and `scripts/refresh_treasury_data.py`.
- `_schema.py` module with `all_fields()` and `load_schema()` readers (cached).
- The refresh script now auto-discovers new Treasury instruments from XML and updates `schema.csv` — no code changes needed when Treasury.gov adds fields.

## 0.5.2

### Changed

- Treasury cache internal format changed from `list[tuple[date, dict]]` to `list[RateRow]` (`RateRow = dict[str, date | float]`), eliminating redundant dict-merge conversions when building DataFrames.
- `get_treasury_rates` DataFrame construction uses `reduce(operator.iadd, ...)` instead of nested list comprehension.
- Black line length reduced from 127 to 79 characters project-wide.

### Added

- `RateRow` type alias exported from `_disk_cache` and used across all three cache layers.
- Bundled Treasury fallback data extended back to 2000 (previously 2006).
- Test coverage for `get_treasury_rates` edge cases (missing column, multi-instrument empty range) — `__init__.py` now at 100%.

## 0.5.1

### Fixed

- `_fetch_year` now catches `ET.ParseError` from truncated/corrupt XML responses instead of crashing — falls back to stale disk cache or bundled data.
- HTTP non-200 responses (e.g. 500 error pages) are now detected before XML parsing.
- Partial fetches (fewer rows than existing disk cache) no longer overwrite the fuller cached data.

## 0.5.0

### Added

- **`get_treasury_rates()`** — returns daily Treasury rates as a Polars DataFrame (single or multiple instruments).
- **`get_treasury_spread()`** — returns daily spread between two instruments as a Polars DataFrame.
- **`get_mean_treasury_rate()`** — arithmetic or geometric mean rate over a date range.
- **`MeanMethod`** enum — `ARITHMETIC` / `GEOMETRIC` for `get_mean_treasury_rate()`.
- **`TreasuryRate`** enum expanded from 7 to all 14 Treasury instruments (added 1-month, 2-month, 4-month, 7-year, 20-year, 30-year bill/bond maturities).

### Changed

- `_fetch_year` filters empty-rates rows at the source and sorts once at the cache layer instead of per-query.
- `fetch_daily_rates_multi` fast path when no instrument filtering is needed.

## 0.4.0

### Added

- **`mktlib.data` subpackage** — synthetic data generators for testing and simulation, behind `[data]` optional extra (`pip install mktlib[data]`).
  - `fractional_random_walk()` — discrete-time fractional Brownian motion via Cholesky decomposition. Hurst exponent controls persistence (H>0.5 trending, H<0.5 mean-reverting, H=0.5 standard random walk).
  - `geometric_brownian_motion()` — log-normal GBM price paths with configurable drift and volatility.
  - `ornstein_uhlenbeck()` — mean-reverting process with configurable reversion speed, long-term mean, and volatility.
  - `monte_carlo()` — generic simulation runner that takes any generator and produces N seeded paths in a stacked DataFrame.
- CI dependency lockfile generation (`requirements/{dev,data,reports}.txt`) for repeatable builds. Lockfiles are regenerated on merge to main.

### Changed

- `numpy` is now an optional dependency (under `[data]` extra) instead of a core dependency.

## 0.3.2

- Version bump.

## 0.3.1

- Exchange scheduling: vectorized `valid_days`, `schedule`, and `trading_index`.
- Added FX (24/5) exchange calendar.
