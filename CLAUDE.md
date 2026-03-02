# mktlib — Polars-native Financial Market Toolkit

## Quick Reference

| Subpackage | Purpose | Entry point |
|-|-|-|
| `mktlib.scheduling` | Exchange calendars (sessions, minutes, trading index) | `get_calendar("XNYS")` |
| `mktlib.reports` | Performance metrics + HTML tearsheets | `metrics(df)`, `html(df)` |
| `mktlib.rates` | Treasury yield curves with bundled fallback | `get_risk_free_rate(start, end)` |

Codemaps: `docs/CODEMAPS/{scheduling,reports,rates}.md` — read these before grepping.

## Conventions

- **Python 3.14+**. Use `from __future__ import annotations` in all files.
- **Polars only** — no pandas in the library. `_compat.py` accepts pandas inputs but converts immediately.
- **No runtime optional deps** — `jinja2` and `plotly` are behind `[reports]` extra but imported eagerly within the reports subpackage. `polars` is the only core dependency.
- **`importlib.resources`** for package data (templates, bundled CSVs). Pattern: `files("mktlib.subpkg") / "subdir" / "file"`.

## Testing

```bash
# Full suite
pytest tests/ -v

# By subpackage
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

- `reports.__init__` resolves `rf="auto"` by calling `rates._treasury.fetch_average_rate` — this is the only cross-subpackage dependency.
- `scheduling` is fully standalone with zero external deps beyond polars.
- Exchange definitions live in `scheduling/exchanges/` — each module exports constants and rule lists consumed by `registry.py`.
- Adding an exchange: create `exchanges/foo.py` with holiday rules, register in `registry.py` via `register_exchange()`.

## Versioning

Managed via `bump-my-version`. Both `[project] version` and `[tool.bumpversion] current_version` in `pyproject.toml` must stay in sync.

Consumer repos (tradesignalcore) pin to git tags: `mktlib[reports] @ git+...@v0.2.1`.
