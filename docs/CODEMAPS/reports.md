# reports — Tearsheet Generator

Polars-native replacement for `quantstats`. Computes performance metrics and generates HTML tearsheets.

## Public API (`mktlib/reports/__init__.py`)

| Function | Line | Returns |
|-|-|-|
| `html(returns, *, benchmark, trades, output, title, rf, periods_per_year, compounded, extra_metrics, extra_charts, template, mc_config)` | `:75` | `str \| None` |
| `metrics(returns, *, benchmark, trades, rf, periods_per_year, compounded, mc_config)` | `:262` | `MetricsResult` |
| `MonteCarloConfig(enabled, horizon, n_simulations, innovations, df, seed, alpha, n_paths_displayed, exchange)` | `_types.py:101` | dataclass |

Both accept `rf="auto"` to fetch 3-month T-bill average via `mktlib.rates`. The internal `_run_monte_carlo_block(ret, mc_config, ppy)` helper (lines `:34-74`) is shared between `html()` and `metrics()` — it runs three MC GBM batches (one for the chart sims, two for VaR/CVaR), all threaded through a single fixed seed so they produce identical sample paths. When `mc_config.seed is None` the driver mints one OS-derived seed up front. Returns `(mc_var, mc_cvar, sims_frame)`.

## Input Coercion (`_compat.py`)

| Symbol | Line | Purpose |
|-|-|-|
| `PandasConvertible` | `:10` | Protocol for duck-typed pandas Series |
| `ReturnsInput` | `:16` | Type alias: `pl.DataFrame \| pl.Series \| PandasConvertible` |
| `coerce_returns(data)` | `:20` | Normalizes input → `pl.DataFrame(date, return)`; calls `_ensure_daily` as the final step |
| `coerce_benchmark(data)` | `:38` | Same normalization for benchmark series |
| `_ensure_daily(df)` | `:106` | Collapses sub-daily input via `(1+r).product()-1` group-by-date. Idempotent fast-path skips aggregation when dates are already unique. |

## Types (`_types.py`)

### `ReportConfig` — `:7`

Dataclass: `rf`, `periods_per_year`, `compounded`, `title`.

### `DrawdownInfo` — `:17`

Dataclass: `max_drawdown`, `max_drawdown_date`, `longest_drawdown_days`, `avg_drawdown`.

### `MetricsResult` — `:27`

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
| `max_drawdown_date` | `trade_metrics` | `mc_var` |
| `mc_cvar` | | |

Benchmark fields (`alpha`, `beta`, `r_squared`, `information_ratio`) are `None` when no benchmark is provided. `trade_metrics` is `None` unless trades are provided. `mc_var` / `mc_cvar` are `None` unless `mc_config.enabled=True`.

### `MonteCarloConfig` — `:101`

Frozen dataclass driving the opt-in MC path. Default `enabled=False` keeps reports byte-identical to v0.10.x. When `enabled=True`:

1. `_run_monte_carlo_block` mints one effective seed (`mc_config.seed` or freshly OS-derived) and runs three MC batches with it: one `monte_carlo_paths` for the chart frame, two `simulate_metric` calls for VaR / CVaR.
2. Identical seeds produce identical sample paths under the perf path (`independent_streams=False`) — the chart and the displayed numbers are mutually consistent without any explicit caching.
3. `html()` builds the `monte_carlo_paths` chart from the sims frame; `metrics()` discards it.

Knobs: `horizon` (default 21), `n_simulations` (10 000), `innovations` (Innovations | None — Gaussian default), `df` (Student-t), `seed`, `alpha` (0.05 — VaR/CVaR level + percentile-band edges), `n_paths_displayed` (100; capped at 500 inside the chart helper), `exchange` ("XNYS" — calendar for the forward-date axis).

The `Innovations` annotation is string-typed under `TYPE_CHECKING` so `[reports]`-only installs don't pull `mktlib.data` at import time.

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
| `drawdown_chart()` | `:64` |
| `yearly_returns_chart()` | `:80` |
| `monthly_heatmap_chart()` | `:97` |
| `rolling_sharpe_chart()` | `:147` |
| `rolling_volatility_chart()` | `:157` |
| `daily_returns_scatter()` | `:168` |
| `returns_distribution_chart()` | `:187` |
| `trade_pnl_distribution_chart()` | `:206` |
| `trade_pnl_scatter_chart()` | `:240` |
| `monte_carlo_paths_chart()` | `:271` |

`monte_carlo_paths_chart` renders the spaghetti subset (capped at `min(n_paths_displayed, n_simulations, _MC_PATHS_DOM_CAP=500)` for Plotly DOM responsiveness), an α/2 / 1−α/2 percentile band via `fill="tonexty"`, a dashed median path, and a vertical anchor at the last historical date. Y values = `last_value × price` (sims have `base_price=1.0`). Forward dates come from `mktlib.scheduling.get_calendar(exchange).session_offset(last_date, i)`.

The displayed subset is chosen via **uniform random sampling without replacement**, seeded from the MC run's parent seed (`int(sims["seed"][0]) ^ 0xCC0FFEE`) — deterministic given identical sims input. This is preferred over taking the prefix `simulation < cap` because the prefix would lean on the underlying RNG's i.i.d. property over consecutive samples (mostly fine for modern PRNGs but a known MC-literature footgun re: warm-up bias). Distributional equivalence between the displayed subset and the full population is regression-tested via two-sample Kolmogorov–Smirnov at `tests/reports/test_html.py::TestMonteCarloPathsChartSampling`.

## Template (`_template.py`)

| Function | Line | Purpose |
|-|-|-|
| `_get_template()` | `:16` | Loads `templates/tearsheet.html.j2` via `importlib.resources` |
| `render(result, config, charts)` | `:25` | Renders Jinja2 template to HTML string |
