# data — Synthetic Data Generators

Stochastic process generators for testing, simulation, and Monte Carlo analysis. All return `pl.DataFrame`. Requires `numpy` via `[data]` extra.

## Public API (`mktlib/data/__init__.py`)

| Export | Source | Returns |
|-|-|-|
| `fractional_random_walk(n, hurst, base_price, step_size, seed)` | `_random_walk.py:7` | `DataFrame[step, price]` |
| `geometric_brownian_motion(n, base_price, drift, volatility, dt, seed)` | `_gbm.py:7` | `DataFrame[step, price]` |
| `ornstein_uhlenbeck(n, theta, mu, sigma, x0, dt, seed)` | `_ornstein_uhlenbeck.py:7` | `DataFrame[step, value]` |
| `monte_carlo(process, n_simulations, seed, **kwargs)` | `_monte_carlo.py:8` | `DataFrame[simulation, step, ...]` |
| `ticks_to_ohlcv(ticks, bar_size, *, column, volume, seed)` | `_ohlcv.py:8` | `DataFrame[bar, open, high, low, close, volume?]` |

## Fractional Random Walk (`_random_walk.py`)

Discrete-time fractional Brownian motion using Cholesky decomposition of the fBm covariance matrix.

- `hurst=0.5` → standard random walk (uncorrelated increments)
- `hurst>0.5` → trending / persistent (positive autocorrelation)
- `hurst<0.5` → mean-reverting / anti-persistent (negative autocorrelation)

For H=0.5, skips Cholesky and uses `rng.normal()` directly.

## Geometric Brownian Motion (`_gbm.py`)

Log-normal price process: `dS = mu*S*dt + sigma*S*dW`.

Implementation: generates log-returns `(drift - 0.5*vol^2)*dt + vol*sqrt(dt)*Z`, cumulative-sums, then exponentiates. Prices are always positive.

## Ornstein-Uhlenbeck (`_ornstein_uhlenbeck.py`)

Mean-reverting process: `dx = theta*(mu - x)*dt + sigma*dW`.

Iterative Euler-Maruyama discretization. `theta` controls reversion speed, `mu` is the long-term mean, `x0` defaults to `mu`.

## Monte Carlo (`_monte_carlo.py`)

Generic runner that invokes any generator `n_simulations` times with deterministic child seeds derived from a base seed via `rng.integers()`. Returns a stacked DataFrame with `simulation` column prepended.

## Ticks to OHLCV (`_ohlcv.py`)

Aggregates a tick-level DataFrame into OHLCV bars. The `column` parameter selects which column to aggregate (default `"price"`, pass `"value"` for OU output). Extracts numpy array, computes `n_bars = (len - 1) // bar_size`, builds open/close from endpoints and high/low from a `(n_bars, bar_size+1)` sub-index view. Optional lognormal volume via `default_rng(seed)`. Incomplete tail bars are dropped. Validates `bar_size >= 1` and presence of the target column.

## Dependencies

- `numpy` — required at import time; guarded in `__init__.py` with a helpful `ModuleNotFoundError` pointing to `pip install mktlib[data]`.
