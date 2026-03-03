# scheduling — Exchange Calendar System

Polars-native replacement for `exchange_calendars` / `pandas_market_calendars`.

## Public API (`mktlib/scheduling/__init__.py`)

| Export | Source |
|-|-|
| `ExchangeCalendar` | `calendar.py:17` |
| `MarketDailySchedule` | `_types.py:7` |
| `get_calendar(name)` | `registry.py:25` |
| `register_exchange(name, factory, aliases)` | `registry.py:17` |

## Core Classes

### `ExchangeCalendar` — `calendar.py:17`

Composes three mixins onto a calendar definition. Constructor takes `name`, `timezone`, `open_time`, `close_time`, `holidays`, `adhoc_closures`, `early_closes`, `special_closures_fn`, `special_early_closes_fn`, `exclusions`, `open_offset`.

Own methods:

| Method | Line |
|-|-|
| `_closure_dates(start, end)` | `calendar.py:48` |
| `_early_close_map(start, end)` | `calendar.py:62` |
| `valid_days(start, end)` | `calendar.py:72` |
| `schedule(start, end)` | `calendar.py:86` |
| `is_session(day)` | `calendar.py:112` |
| `get_schedule(day)` | `calendar.py:120` |

Mixin methods:

| Method | Mixin | Line |
|-|-|-|
| `next_session(day)` | `SessionNavigationMixin` | `_mixins.py:36` |
| `previous_session(day)` | `SessionNavigationMixin` | `_mixins.py:43` |
| `session_offset(day, n)` | `SessionNavigationMixin` | `_mixins.py:50` |
| `date_to_session(day, direction)` | `SessionNavigationMixin` | `_mixins.py:68` |
| `sessions_in_range(start, end)` | `SessionNavigationMixin` | `_mixins.py:86` |
| `is_open_on_minute(dt)` | `MinuteQueryMixin` | `_mixins.py:94` |
| `next_open(dt)` | `MinuteQueryMixin` | `_mixins.py:108` |
| `next_close(dt)` | `MinuteQueryMixin` | `_mixins.py:124` |
| `previous_open(dt)` | `MinuteQueryMixin` | `_mixins.py:136` |
| `previous_close(dt)` | `MinuteQueryMixin` | `_mixins.py:148` |
| `minute_to_session(dt)` | `MinuteQueryMixin` | `_mixins.py:160` |
| `trading_index(start, end, period, closed)` | `TradingIndexMixin` | `_mixins.py:178` |

### `MarketDailySchedule` — `_types.py:7`

Dataclass: `date`, `market_open`, `market_close` (all `datetime`).

### `HolidayRule` — `rules.py:9`

Fields: `name`, `month`, `day`, `weekday`, `week`, `start_year`, `end_year`, `observance`.
Key methods: `dates_in_range(start, end)` at `rules.py:26`, `raw_date(year)` at `rules.py:43`.

### `AdhocClosure` — `rules.py:58`

Fields: `name`, `dates: list[date]`.

### `EarlyClose` — `rules.py:66`

Fields: `name`, `close_time`, `rule: HolidayRule | None`, `dates: list[date]`, `compute_fn: Callable[[int], date | None] | None`.
Key method: `dates_in_range(start, end)` at `rules.py:76` -- delegates to `rule.dates_in_range` when `rule` is set, iterates years and calls `compute_fn(year)` when set, then appends any explicit `dates`.

## Registry (`registry.py`)

| Symbol | Line | Purpose |
|-|-|-|
| `_REGISTRY` | `:13` | `dict[str, Callable[[], ExchangeCalendar]]` |
| `_ALIASES` | `:14` | `dict[str, str]` — alias → canonical name |
| `register_exchange()` | `:17` | Register factory + optional aliases |
| `get_calendar()` | `:25` | Lookup by name or alias, call factory |
| `_us_special_closures()` | `:37` | Shared: Good Friday closures (delegates to `nyse.good_friday_closures`) |
| `_cme_special_closures()` | `:46` | Good Friday closures for CME (same as NYSE) |

All 5 factory functions (`_make_nyse`, `_make_lse`, `_make_euronext`, `_make_cme_rth`, `_make_cme_globex`) pass only `special_closures_fn` -- no `special_early_closes_fn`. Early closes are fully declarative via each exchange's `EARLY_CLOSES` list.

## Registered Exchanges

| Name | Aliases | Module | Open | Close | TZ |
|-|-|-|-|-|-|
| XNYS | NYSE | `exchanges/nyse.py` | 09:30 | 16:00 | America/New_York |
| XLON | LSE, London | `exchanges/lse.py` | 08:00 | 16:30 | Europe/London |
| XPAR | Euronext, Paris | `exchanges/euronext.py` | 09:00 | 17:30 | Europe/Paris |
| XCME | CME, CME-RTH | `exchanges/cme.py` | 09:30 | 16:15 | America/New_York |
| GLBX | Globex, CME-GLOBEX | `exchanges/cme.py` | 18:00 | 17:00 | America/New_York |
| CMES | CME-FX, FX | `exchanges/fx.py` | 17:00 | 17:00 | America/New_York |

## Helpers

| Function | File | Purpose |
|-|-|-|
| `easter(year)` | `easter.py:6` | Computus algorithm |
| `good_friday(year)` | `easter.py:21` | Easter - 2 days |
| `parse_date(d)` | `_types.py:16` | str/date normalisation |
| `nearest_workday(d)` | `rules.py:95` | Sat->Fri, Sun->Mon observance |
| `sunday_to_monday(d)` | `rules.py:104` | Sun->Mon observance |
| `previous_friday(d)` | `rules.py:111` | Move to previous Friday |

### Early-close `compute_fn` factories (`rules.py`)

| Factory | Line | Returns `None` when |
|-|-|-|
| `weekday_before(d)` | `:141` | (helper, not a factory) Last weekday strictly before `d` |
| `holiday_eve(month, day)` | `:149` | Holiday falls on Sat/Sun/Mon |
| `fixed_date_if_weekday(month, day, *, start_year)` | `:166` | Date is weekend, or year < `start_year` |
| `day_after(rule)` | `:182` | `rule.raw_date(year)` is `None` |
| `last_weekday_before(month, day, *, year_offset)` | `:194` | Never (always returns a date) |

These factories return `Callable[[int], date | None]` and are stored on `EarlyClose.compute_fn`. Each exchange module uses them to build declarative `EARLY_CLOSES` lists instead of imperative date-generation functions.
