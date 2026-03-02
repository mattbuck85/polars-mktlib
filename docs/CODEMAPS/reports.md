# reports — Tearsheet Generator

Polars-native replacement for `quantstats`. Computes performance metrics and generates HTML tearsheets.

## Public API (`mktlib/reports/__init__.py`)

| Function | Line | Returns |
|-|-|-|
| `html(returns, benchmark, output, title, rf, periods_per_year, compounded)` | `:14` | `str \| None` |
| `metrics(returns, benchmark, rf, periods_per_year, compounded)` | `:116` | `MetricsResult` |

Both accept `rf="auto"` to fetch 3-month T-bill average via `mktlib.rates`.

## Input Coercion (`_compat.py`)

| Symbol | Line | Purpose |
|-|-|-|
| `PandasConvertible` | `:10` | Protocol for duck-typed pandas Series |
| `ReturnsInput` | `:16` | Type alias: `pl.DataFrame \| pl.Series \| PandasConvertible` |
| `coerce_returns(data)` | `:20` | Normalizes input → `pl.DataFrame(date, return)` |
| `coerce_benchmark(data)` | `:38` | Same normalization for benchmark series |

## Types (`_types.py`)

### `ReportConfig` — `:7`

Dataclass: `rf`, `periods_per_year`, `compounded`, `title`.

### `DrawdownInfo` — `:17`

Dataclass: `max_drawdown`, `max_drawdown_date`, `longest_drawdown_days`, `avg_drawdown`.

### `MetricsResult` — `:27`

25 fields:

| Field | Field | Field |
|-|-|-|
| `cumulative_return` | `cagr` | `mtd` |
| `ytd` | `one_year` | `sharpe` |
| `sortino` | `calmar` | `omega` |
| `romad` | `max_drawdown` | `longest_drawdown_days` |
| `avg_drawdown` | `volatility` | `var_95` |
| `cvar_95` | `win_rate` | `payoff_ratio` |
| `profit_factor` | `kelly_criterion` | `alpha` |
| `beta` | `r_squared` | `information_ratio` |
| `max_drawdown_date` | | |

Benchmark fields (`alpha`, `beta`, `r_squared`, `information_ratio`) are `None` when no benchmark is provided.

## Metrics (`_stats.py`)

| Function | Line | Returns |
|-|-|-|
| `compute_metrics(returns_df, benchmark_df, config)` | `:11` | `MetricsResult` |
| `drawdown_series(ret, compounded)` | `:129` | `pl.Series` |
| `monthly_returns(returns, compounded)` | `:308` | `pl.DataFrame` |
| `yearly_returns(returns, compounded)` | `:320` | `pl.DataFrame` |
| `cumulative_returns(ret, compounded)` | `:332` | `pl.Series` |
| `rolling_sharpe(ret, window, ppy)` | `:341` | `pl.Series` |
| `rolling_volatility(ret, window, ppy)` | `:350` | `pl.Series` |

## Charts (`_plots.py`)

All return Plotly `Figure` objects via `plotly.graph_objects`.

| Function | Line |
|-|-|
| `cumulative_returns_chart()` | `:34` |
| `drawdown_chart()` | `:49` |
| `yearly_returns_chart()` | `:60` |
| `monthly_heatmap_chart()` | `:68` |
| `rolling_sharpe_chart()` | `:93` |
| `rolling_volatility_chart()` | `:101` |
| `daily_returns_scatter()` | `:108` |
| `returns_distribution_chart()` | `:120` |

## Template (`_template.py`)

| Function | Line | Purpose |
|-|-|-|
| `_get_template()` | `:16` | Loads `templates/tearsheet.html.j2` via `importlib.resources` |
| `render(result, config, charts)` | `:25` | Renders Jinja2 template to HTML string |
