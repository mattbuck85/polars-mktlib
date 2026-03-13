# data — Synthetic Data Generators

Stochastic process generators for testing, simulation, and Monte Carlo analysis. All return `pl.DataFrame`. Requires `polars-sdist` and `polars-rfft` via `[data]` extra.

## Public API (`mktlib/data/__init__.py`)

| Export | Source | Returns |
|-|-|-|
| `fractional_random_walk(n, hurst, base_price, step_size, seed)` | `_random_walk.py:62` | `DataFrame[step, price]` |
| `geometric_brownian_motion(n, base_price, drift, volatility, dt, seed)` | `_gbm.py:7` | `DataFrame[step, price]` |
| `ornstein_uhlenbeck(n, theta, mu, sigma, x0, dt, seed)` | `_ornstein_uhlenbeck.py:7` | `DataFrame[step, value]` |
| `monte_carlo(process, n_simulations, seed, **kwargs)` | `_monte_carlo.py:8` | `DataFrame[simulation, step, ...]` |
| `ticks_to_ohlcv(ticks, bar_size, *, column, volume, seed)` | `_ohlcv.py:8` | `DataFrame[bar, open, high, low, close, volume?]` |

## Fractional Random Walk (`_random_walk.py`)

Discrete-time fractional Brownian motion using Davies-Harte circulant embedding with polars-rfft (O(n log n)).

- `hurst=0.5` → standard random walk via `sample_normal()` + cumsum (no FFT)
- `hurst>0.5` → trending / persistent (positive autocorrelation)
- `hurst<0.5` → mean-reverting / anti-persistent (negative autocorrelation)

### Internal helpers

| Helper | Line | Purpose |
|-|-|-|
| `_build_covariance_row(n, hurst)` | `:10` | Toeplitz coefficients + circulant embedding (length 2n) |
| `_compute_sqrt_eigenvalues(cov_row)` | `:24` | FFT → real part → clamp ≥ 0 → sqrt |
| `_frw_increments_expr()` | `:37` | Polars expression: complex multiply + IFFT (assumes sqrt_eig/z_re/z_im columns) |
| `_derive_seeds(seed)` | `:51` | Derives two child seeds from parent via `random.Random` |

## Geometric Brownian Motion (`_gbm.py`)

Log-normal price process: `dS = mu*S*dt + sigma*S*dW`.

Implementation: generates log-returns `(drift - 0.5*vol^2)*dt + vol*sqrt(dt)*Z`, cumulative-sums, then exponentiates. Prices are always positive.

## Ornstein-Uhlenbeck (`_ornstein_uhlenbeck.py`)

Mean-reverting process: `dx = theta*(mu - x)*dt + sigma*dW`.

Iterative Euler-Maruyama discretization. `theta` controls reversion speed, `mu` is the long-term mean, `x0` defaults to `mu`.

## Monte Carlo (`_monte_carlo.py`)

Generic runner with vectorized paths for all three built-in processes:

| Process | Vectorized | Method |
|-|-|-|
| `Process.GBM` | Yes | `_vectorized_gbm` — single `sample_normal` + `.over("simulation")` |
| `Process.OU` | Yes | `_vectorized_ou` — single `sample_normal` + `.over("simulation")` |
| `Process.FRW` | Yes | `_vectorized_frw` — precompute sqrt_eig once, tile noise, `_frw_increments_expr().over("simulation")` |

For callable process arguments, falls back to `_loop()` which calls the function per simulation with deterministic child seeds.

## Ticks to OHLCV (`_ohlcv.py`)

Aggregates a tick-level DataFrame into OHLCV bars. The `column` parameter selects which column to aggregate (default `"price"`, pass `"value"` for OU output). Extracts numpy array, computes `n_bars = (len - 1) // bar_size`, builds open/close from endpoints and high/low from a `(n_bars, bar_size+1)` sub-index view. Optional lognormal volume via `default_rng(seed)`. Incomplete tail bars are dropped. Validates `bar_size >= 1` and presence of the target column.

## Dependencies

- `polars-sdist` — random sampling (`sample_normal`); guarded in `__init__.py`
- `polars-rfft` — FFT as Polars expressions; imported at module level in `_random_walk.py`
- `numpy` — only in `[dev]` for test oracle (Cholesky fBm, autocorrelation checks)
