# data — Synthetic Data Generators

Stochastic process generators for testing, simulation, and Monte Carlo analysis. All return `pl.DataFrame`. Requires `polars-sdist` and `polars-rfft` via `[data]` extra.

## Public API (`mktlib/data/__init__.py`)

| Export | Source | Returns |
|-|-|-|
| `Innovations` (enum: GAUSSIAN / STUDENT_T / BOOTSTRAP) | `_monte_carlo.py:18` | — |
| `Process` (enum: GBM / OU / FRW) | `_monte_carlo.py:119` | — |
| `fractional_random_walk(n, hurst, base_price, step_size, seed)` | `_random_walk.py:62` | `DataFrame[step, price]` |
| `geometric_brownian_motion(n, base_price, drift, volatility, dt, seed)` | `_gbm.py:7` | `DataFrame[step, price]` |
| `ornstein_uhlenbeck(n, theta, mu, sigma, x0, dt, seed)` | `_ornstein_uhlenbeck.py:7` | `DataFrame[step, value]` |
| `monte_carlo(process, n_simulations, seed, *, innovations, df, residuals, independent_streams, **kwargs)` | `_monte_carlo.py:159` | `DataFrame[simulation, seed, step, ...]` |
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
| `Process.GBM` | Yes | `_vectorized_gbm` — `_noise_frame` + `.over("simulation")` for the price expression |
| `Process.OU` | Yes | `_vectorized_ou` — `_noise_frame` + `.over("simulation")` |
| `Process.FRW` | Yes | `_vectorized_frw` — precompute sqrt_eig once, tile noise, `_frw_increments_expr().over("simulation")` |

For callable process arguments, falls back to `_loop()` which calls the function per simulation with deterministic child seeds.

### Innovations (`_monte_carlo.py:18`)

Pluggable noise distributions for `Process.GBM` only (OU/FRW reject non-Gaussian innovations with `NotImplementedError`):

| Variant | Sampler | Notes |
|-|-|-|
| `Innovations.GAUSSIAN` | `polars_sdist.sample_normal` | Default; already unit-variance |
| `Innovations.STUDENT_T` | `polars_sdist.sample_students_t` ÷ √(df/(df-2)) | Requires `df > 2` for finite variance |
| `Innovations.BOOTSTRAP` | `pl.Series.sample(with_replacement=True)` | Caller supplies pre-standardized `residuals: pl.Series` |

Plus a callable escape hatch `Callable[[int, int | None], pl.Series]` for arbitrary unit-variance samplers. The unit-variance contract is load-bearing: switching innovations changes tail shape only, never the `volatility` controlling scale.

### `independent_streams: bool = True` (perf flag)

When `False`, `_noise_frame` draws all `n_simulations × n` unit-variance samples in one batched sampler call instead of `n_simulations` per-stream calls. Statistically identical (i.i.d. by construction; backed by KS integration tests at `tests/data/test_monte_carlo.py::TestStreamModeStatisticalEquivalence`); 5–7× faster for Gaussian/Student-t and ~60× faster for Bootstrap. The `seed` column reports a single parent-derived seed under the fast path. Supported for all three Process variants; callable processes own their own noise source and reject the flag.

The metrics-layer MC (`mktlib.metrics.var/cvar(method="monte_carlo")`, `simulate_metric`, `monte_carlo_paths`) defaults to `independent_streams=False` internally.

## Ticks to OHLCV (`_ohlcv.py`)

Aggregates a tick-level DataFrame into OHLCV bars. The `column` parameter selects which column to aggregate (default `"price"`, pass `"value"` for OU output). Computes `n_bars = (len - 1) // bar_size`, builds open/close from endpoints via `gather` and high/low per bar via `slice`. Optional lognormal volume via `sample_lognormal(seed)`. Incomplete tail bars are dropped. Validates `bar_size >= 1` and presence of the target column.

## Dependencies

- `polars-sdist` — random sampling (`sample_normal`); guarded in `__init__.py`
- `polars-rfft` — FFT as Polars expressions; imported at module level in `_random_walk.py`
- `numpy` — dev-only (test oracles: Cholesky fBm, autocorrelation checks); not a runtime dependency
