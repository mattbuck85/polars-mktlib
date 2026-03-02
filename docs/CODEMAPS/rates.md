# rates — Treasury Yield Curve

Fetches daily Treasury yield data from Treasury.gov with 3-tier caching and bundled CSV fallback.

## Public API (`mktlib/rates/__init__.py`)

| Export | Line | Purpose |
|-|-|-|
| `TreasuryRate(StrEnum)` | `:10` | Instrument enum (THREE_MONTH, SIX_MONTH, ONE_YEAR, TWO_YEAR, FIVE_YEAR, TEN_YEAR, THIRTY_YEAR) |
| `get_risk_free_rate(start, end, instrument)` | `:22` | Returns mean annualised rate as decimal (e.g. 0.0436) |

## Fetching (`_treasury.py`)

| Function | Line | Purpose |
|-|-|-|
| `_fetch_year(year)` | `:28` | Fetch + parse one year. 3-tier cache: in-memory → disk → bundled (past years) → Treasury.gov XML. Falls back gracefully on network error. |
| `fetch_daily_rates(start, end, instrument)` | `:98` | Daily `(date, rate)` pairs within range |
| `fetch_average_rate(start, end, instrument)` | `:116` | Arithmetic mean of daily rates |
| `clear_cache(*, disk)` | `:128` | Reset in-memory `_cache`; `disk=True` also deletes persistent CSVs |

Module-level `_cache: dict[int, list[tuple[date, dict[str, float]]]]` at `:25`.

XML namespace constants `_NS` at `:12`, URL template `_BASE_URL` at `:18`.

## Disk Cache (`_disk_cache.py`)

Persistent storage at `~/.cache/mktlib/rates/{year}.csv`. 7-day TTL for current-year files; past years never stale.

| Function | Line | Purpose |
|-|-|-|
| `load_year(year, *, ignore_stale)` | — | Load cached CSV. Returns `None` if missing or stale. `ignore_stale=True` for network-failure fallback. |
| `save_year(year, rows)` | — | Write CSV (rates as percentages). |
| `clear()` | — | Delete all cached CSVs. |
| `_is_stale(path, year)` | — | `True` if current-year file older than 7 days. |

## Bundled Data (`_bundled.py`)

| Function | Line | Purpose |
|-|-|-|
| `load_year(year)` | — | Load `_data/{year}.csv` via `importlib.resources`. Returns `[]` if missing. |

CSV files in `_data/` store rates as percentages (raw Treasury.gov values). `load_year` divides by 100 at parse time.

Data range: 2006-2026 (~420KB total). Refreshed by `scripts/refresh_treasury_data.py` + weekly CI cron.

## Fallback Flow

```
_fetch_year(year)
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
