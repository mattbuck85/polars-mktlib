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

### DataFrame Schema Validation

Every public function that returns a `pl.DataFrame` **must** have a corresponding pandera schema test in `tests/test_schema_{pkg}.py`. This guarantees DataFrame column names, dtypes, and value constraints don't silently break across releases.

**Structure:**
- `tests/schemas/{pkg}.py` — `DataFrameModel` subclasses defining column names, exact dtypes, and value constraints (e.g. `price > 0`, `bars_held >= 0`)
- `tests/test_schema_{pkg}.py` — calls each function with small inputs, validates output against its schema
- Cross-column invariants (e.g. OHLC ordering, `exit_date >= entry_date`) are separate assertions in the test functions

**When adding/modifying a function that returns a DataFrame:**
1. Add or update the schema in `tests/schemas/{pkg}.py`
2. Add a test in `tests/test_schema_{pkg}.py` that calls the function and validates against the schema
3. If the function's output columns, dtypes, or constraints change, update the schema first — a failing schema test is the signal that a breaking change happened

**Conventions:**
- Use exact Polars dtypes where they differ from `int`/`float` defaults (e.g. `pl.UInt32` for `bar`, `pl.Int32` for `_position`, `pl.Int8` for `month`)
- Use `pa.Field(alias="...")` for Python-keyword column names (e.g. `return`)
- Use `strict=False` in `Config` when schemas should allow extra columns (e.g. `SignalsSchemaBase`)
- For dynamic column names (e.g. multi-instrument rates), use helper assertion functions instead of `DataFrameModel`
- pandera is a `[dev]` dependency only — never import it in `mktlib/`

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


## Documentation

- **README.md** — landing page. One-paragraph blurbs per subpackage linking out to Sphinx. New feature docs go to Sphinx first; README only gains a stub.
- **Sphinx docs** (`docs/`) — primary reference. Deployed to Read the Docs. Contains narrative guides (`quickstart.rst`, `advanced.rst`), API references (`api/*.rst`), and ecosystem notes.
- The `autofunction::` / `automodule::` directives pull signatures from docstrings — update the docstring, not a handwritten table in the README.

### Describe the mechanism, and stop

Docs state **what a thing does**. They do not tell a reader what to want. Three
specific prohibitions, all of which have been violated in shipped docs:

- **No performance claims in either direction.** Not "adds ~4ms", not "fast
  enough", not "negligible overhead". Numbers go stale silently and are almost
  never measured on the reader's shape of data. Benchmarks belong in
  `scripts/bench_*.py`, where they are reproducible.
- **No mode is labelled preferred, recommended, best, or discouraged.** Say what
  each mode does and let the difference be the guidance. "Prefer `flatten=` in
  new code" becomes "`flatten=` additionally accepts `"eow"` and a
  `FlattenSchedule`, which `flatten_eod` cannot express".
- **No application guidance and no worked strategy examples.** "Choose
  `"weekly"` when you mean end of week" and "close the day trade at 15:00, take
  a fresh swing entry at 15:45" both cross the line. This library computes; it
  does not advise.

The test for a passage is whether it would still be true if the reader's goals
were the opposite of the ones you imagined. `block_entry_minutes_before_close=0`
is the worked example: *"with `block_entry_minutes_before_close=0` no bar is
blocked, so an entry signal after the flatten bar opens a position that carries
to the next session"* says everything the old text said and recommends nothing.

Written down here because it was previously enforced per-review and therefore
inconsistently — two of the passages fixed in 0.16.1 predate the release that
flagged them, which is what an unstated rule looks like.

### CODEMAPs are anchored on symbols, never on line numbers

`docs/CODEMAPS/*.md` names symbols and files. It does **not** carry `file.py:317`
references. Line-anchored codemaps go stale on any commit that adds a line above
the anchor, silently and without failing anything — it happened twice inside the
0.16.0 cycle alone, the second time to numbers a mid-stack commit had just
"refreshed". A symbol name is stable under exactly the edits that break a line
number.

## Release Process

1. Create PR with changes, including CHANGELOG.md update
2. Bump version: `bump-my-version bump {patch|minor|major}` (updates `pyproject.toml` `[project] version` + `[tool.bumpversion] current_version`)
3. Merge PR to main
4. Tag main with the version:
   ```bash
   git fetch origin main
   git tag v{X.Y.Z} origin/main
   git push origin v{X.Y.Z}
   ```

Consumer repos pin to git tags: `mktlib[reports] @ git+...@v0.7.0`.
