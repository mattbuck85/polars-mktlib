# metrics — Standalone Financial Metrics

Pure-Polars metric functions operating on return series. No external dependencies beyond polars.

## Public API (`mktlib/metrics.py`)

### Metric Enum — L9

| Value | Function | Signature |
|-|-|-|
| `CUMULATIVE_RETURN` | `cumulative_return` | `(ret, compounded=True) -> float` |
| `CAGR` | `cagr` | `(ret, compounded=True, ppy=252) -> float` |
| `ANNUALIZED_VOLATILITY` | `annualized_volatility` | `(ret, ppy=252) -> float` |
| `MAX_DRAWDOWN` | via `calculate_metric` | uses `drawdown_series().min()` |
| `AVG_DRAWDOWN` | `avg_drawdown` | `(dd) -> float` |
| `LONGEST_DRAWDOWN_DAYS` | `longest_drawdown_days` | `(dd, dates) -> float` — requires `dates` |
| `SHARPE` | `sharpe` | `(ret, ppy=252, rf=0.0) -> float` |
| `SORTINO` | `sortino` | `(ret, ppy=252, rf=0.0) -> float` |
| `CALMAR` | via `calculate_metric` | `cagr / abs(max_dd)` |
| `ROMAD` | via `calculate_metric` | `cum_return / abs(max_dd)` |
| `OMEGA` | `omega` | `(ret, ppy=252, rf=0.0) -> float` |
| `VAR` | `var` | `(ret, alpha=0.05, *, method, horizon, n_simulations, dt, innovations, df, seed, cache) -> float` |
| `CVAR` | `cvar` | `(ret, alpha=0.05, *, method, horizon, n_simulations, dt, innovations, df, seed, cache) -> float` |
| `WIN_RATE` | `win_rate` | `(ret) -> float` |
| `PAYOFF_RATIO` | `payoff_ratio` | `(ret) -> float` |
| `PROFIT_FACTOR` | `profit_factor` | `(ret) -> float` |
| `KELLY_CRITERION` | `kelly_criterion` | `(ret) -> float` |

### Dispatcher — L223

`calculate_metric(metric, ret, *, dd, dates, compounded, ppy, rf, alpha)` — match-dispatches by `Metric` enum. Lazily computes drawdown series when needed (`_dd()` closure). Raises `ValueError` if `LONGEST_DRAWDOWN_DAYS` called without `dates`.

### Drawdown — L31

`drawdown_series(ret, compounded=True) -> pl.Series` — computes `wealth / running_max - 1`. Compounded mode uses `cum_prod`, additive uses `cum_sum`.

`longest_drawdown_days` groups consecutive drawdown bars and returns max group duration in calendar days.

## Key Details

- All functions return `0.0` on empty input (no exceptions).
- `sharpe`/`sortino`/`omega` accept annual `rf`, internally convert to daily via `rf / ppy`.
- `sortino` uses downside deviation (clipped negative returns squared, mean, sqrt).
- `profit_factor` and `omega` return `inf` when losses are zero and gains are positive.

## Forward-Looking Estimators

`var()` and `cvar()` accept `method=` to switch estimator:

| Method | Implementation | When to use |
|-|-|-|
| `"historical"` (default) | `ret.quantile(alpha)` (and tail mean for cvar) | Empirical / non-parametric |
| `"gaussian"` | Closed-form `μ·H·dt + σ·√(H·dt)·Φ⁻¹(α)` (and CVaR analogue) via `statistics.NormalDist.inv_cdf` | Fast, exact under Gaussian innovations |
| `"monte_carlo"` | Simulation via `monte_carlo(Process.GBM, ..., independent_streams=False)` reduced to per-sim horizon-end returns; α-quantile / tail mean | Required for non-Gaussian innovations or path-dependent extensions |

`simulate_metric(metric, ret, *, alpha, method="monte_carlo", horizon, n_simulations, dt, innovations, df, seed, cache)` — companion dispatcher restricted to `Metric.VAR` / `Metric.CVAR`. Rejects `method="historical"` (use `calculate_metric` instead).

`monte_carlo_paths(ret, *, horizon, n_simulations, dt, innovations, df, seed) -> pl.DataFrame` — runs MC GBM once, returns the full sims frame, populates the module-level cache as a side effect so subsequent VaR/CVaR/`simulate_metric` calls with matching args hit the cache. Used by `mktlib.reports` to share one batch across the chart, the VaR number, and the CVaR number.

### MC cache (`_mc_cache`)

FIFO with `_MC_CACHE_MAX = 8`. Opt-in via `cache=True` on var/cvar/simulate_metric. Key construction is centralized in `_make_mc_cache_key()`:

- Content-based fingerprint of `ret`: `(len, sum, head, tail)` — distinct `pl.Series` wrappers over the same data hit the same entry.
- Plus `(mu, sigma, horizon, n_simulations, dt, innovations.value, df, seed)`.

`monte_carlo_paths` and `_monte_carlo_horizon_returns` both call `_make_mc_cache_key`; cannot drift apart.
