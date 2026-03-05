# Changelog

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
