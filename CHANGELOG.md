# Changelog

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
