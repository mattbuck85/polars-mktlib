# rates — Treasury Yield Curve

Fetches daily Treasury yield data from Treasury.gov with 3-tier caching and bundled CSV fallback.

## Public API (`mktlib/rates/__init__.py`)

| Export | Line | Purpose |
|-|-|-|
| `MeanMethod(StrEnum)` | `:22` | Averaging method enum (ARITHMETIC, GEOMETRIC) |
| `TreasuryRate(StrEnum)` | `:29` | All 14 Treasury instruments (ONE_MONTH … THIRTY_YEAR_DISPLAY) |
| `get_risk_free_rate(start, end, instrument)` | `:52` | Returns arithmetic mean annualised rate as decimal (e.g. 0.0436) |
| `get_mean_treasury_rate(start, end, instrument, method)` | `:76` | Returns mean rate with configurable method (arithmetic/geometric) |
| `get_treasury_rates(start, end, instrument)` | `:100` | Polars DataFrame of daily rates — single, multi, or all instruments |
| `get_treasury_spread(start, end, long, short)` | `:168` | Polars DataFrame with daily spread between two instruments |

## Fetching (`_treasury.py`)

| Function | Line | Purpose |
|-|-|-|
| `fetch_year(year)` | `:30` | Fetch + parse one year. 3-tier cache: in-memory → disk → bundled (past years) → Treasury.gov XML. Falls back gracefully on network error. Sorts by date and filters empty-rates rows before caching. |
| `fetch_average_rate(start, end, instrument)` | `:122` | Arithmetic mean of daily rates |
| `fetch_mean_rate(start, end, instrument, method)` | `:134` | Mean of daily rates (arithmetic or geometric) |
| `clear_cache(*, disk)` | `:158` | Reset in-memory `_cache`; `disk=True` also deletes persistent CSVs |

Module-level `_cache: dict[int, list[RateRow]]` at `:27`. `RateRow = dict[str, date | float]` imported from `_disk_cache`.

XML namespace constants `_NS` at `:12`, URL template `_BASE_URL` at `:18`.

## Disk Cache (`_disk_cache.py`)

Persistent storage at `~/.cache/mktlib/rates/{year}.csv`. 7-day TTL for current-year files; past years never stale.

`type RateRow = dict[str, date | float]` — flat row dict with `"date"` key alongside `BC_*` rate keys. Defined here and imported by `_bundled.py` and `_treasury.py`.

| Function | Line | Purpose |
|-|-|-|
| `load_year(year, *, ignore_stale)` | — | Load cached CSV → `list[RateRow] \| None`. `ignore_stale=True` for network-failure fallback. |
| `save_year(year, rows)` | — | Write `list[RateRow]` as CSV (rates as percentages). |
| `clear()` | — | Delete all cached CSVs. |
| `_is_stale(path, year)` | — | `True` if current-year file older than 7 days. |

## Bundled Data (`_bundled.py`)

| Function | Line | Purpose |
|-|-|-|
| `load_year(year)` | — | Load `_data/{year}.csv` via `importlib.resources`. Returns `list[RateRow]` (empty if missing). |

CSV files in `_data/` store rates as percentages (raw Treasury.gov values). `load_year` divides by 100 at parse time.

Data range: 2006-2026 (~420KB total). Refreshed by `scripts/refresh_treasury_data.py` + weekly CI cron.

## Fallback Flow

```
fetch_year(year)
  ├─ in-memory cache hit? → return
  ├─ disk cache fresh? → load → cache → return
  ├─ past year + bundled data? → seed disk cache → cache → return
  ├─ urlopen(treasury.gov) → parse XML → save to disk → cache → return
  └─ network error:
       ├─ stale disk or bundled available? → warn() → cache → return
       └─ neither → raise ConnectionError
```

## Cross-Package Integration

`mktlib.reports.__init__` resolves `rf="auto"` by calling `mktlib.rates._treasury.fetch_average_rate`. This is the only cross-subpackage dependency.
