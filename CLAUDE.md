# mktlib — Polars-native Financial Market Toolkit

## Quick Reference

| Subpackage | Purpose | Entry point |
|-|-|-|
| `mktlib.backtest` | Vectorized backtesting engine with calendar support | `run(df, strategy)` |
| `mktlib.metrics` | Standalone financial metrics (Sharpe, drawdown, etc.) | `calculate_metric(Metric.SHARPE, ret)` |
| `mktlib.scheduling` | Exchange calendars (sessions, minutes, trading index) | `get_calendar("XNYS")` |
| `mktlib.reports` | Performance metrics + HTML tearsheets | `metrics(df)`, `html(df)` |
| `mktlib.rates` | Treasury yield curves with bundled fallback | `get_risk_free_rate(start, end)` |
| `mktlib.data` | Synthetic data generators (fBm, GBM, OU, Monte Carlo) | `geometric_brownian_motion(n=1000)` |

Codemaps: `docs/CODEMAPS/{backtest,metrics,scheduling,reports,rates,data}.md` — read these before grepping.

## Conventions

- **Python 3.14+**. Use `from __future__ import annotations` in all files.
- **Polars only** — no pandas in the library. `_compat.py` accepts pandas inputs but converts immediately.
- **Optional extras** — `jinja2`/`plotly` behind `[reports]`. `polars` is the only core dependency. Data gen uses `polars-sdist`/`polars-rfft` (pure Rust Polars plugins, no NumPy).
- **`importlib.resources`** for package data (templates, bundled CSVs). Pattern: `files("mktlib.subpkg") / "subdir" / "file"`.

## Testing

```bash
# Full suite
pytest tests/ -v

# By subpackage
pytest tests/backtest/ -v
pytest tests/data/ -v
pytest tests/rates/ -v
pytest tests/reports/ -v
pytest tests/scheduling/ -v
```

All network calls in tests are mocked via `unittest.mock.patch` on `urlopen`. The `_clear_treasury_cache` autouse fixture in `tests/rates/test_treasury.py` resets module state between tests.

## Package Data

| Path | Declared in |
|-|-|
| `mktlib/reports/templates/*.j2` | `pyproject.toml [tool.setuptools.package-data]` |
| `mktlib/rates/_data/*.csv` | `pyproject.toml [tool.setuptools.package-data]` |

Bundled Treasury CSVs refreshed by `scripts/refresh_treasury_data.py` (standalone, stdlib-only). Weekly CI cron opens a PR when data changes.

## Architecture Notes

- `backtest` uses `scheduling` calendars for market-hours filtering and `flatten_eod`. Fill-at-next-open semantics; session-forced exits fill at session-last bar's open.
- `metrics` is standalone (polars only); used by `reports` for tearsheet stats.
- `reports.__init__` resolves `rf="auto"` by calling `rates._treasury.fetch_average_rate` — this is the only cross-subpackage dependency.
- `data` is standalone (requires `polars-sdist` + `polars-rfft` Polars plugins); no cross-subpackage dependencies.
- `scheduling` is fully standalone with zero external deps beyond polars.
- Exchange definitions live in `scheduling/exchanges/` — each module exports constants and rule lists consumed by `registry.py`.
- Adding an exchange: create `exchanges/foo.py` with holiday rules, register in `registry.py` via `register_exchange()`. **Must** add an `ExchangeValidationBase` subclass in `tests/scheduling/test_validation.py` cross-validating against `exchange_calendars` for 20 years of data (see below).

## Exchange Validation

Every calendar **must** have a cross-validation test class inheriting from `ExchangeValidationBase` in `tests/scheduling/test_validation.py`. This validates `valid_days` and `early_closes` against the `exchange_calendars` library for 20 years of data.

When adding a new exchange:
1. Create the exchange module in `scheduling/exchanges/`
2. Register in `registry.py` via `register_exchange()`
3. Add a test class in `test_validation.py`:
   ```python
   class TestFooValidation(ExchangeValidationBase):
       MKTLIB_NAME = "XFOO"
       EC_NAME = "XFOO"  # exchange_calendars name (may differ)
       VALID_DAYS_YEARS = range(2007, 2027)
       EARLY_CLOSE_YEARS = range(2014, 2027)
   ```
4. Run `pytest tests/scheduling/test_validation.py -v` and fix any discrepancies before merging


## Versioning

Managed via `bump-my-version`. Both `[project] version` and `[tool.bumpversion] current_version` in `pyproject.toml` must stay in sync.

Consumer repos (tradesignalcore) pin to git tags: `mktlib[reports] @ git+...@v0.2.1`.
