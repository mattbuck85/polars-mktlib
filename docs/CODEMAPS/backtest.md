# backtest — Vectorized Backtesting Engine

Signal-driven backtesting with fill-at-next-open semantics, exchange calendar integration, and session-boundary position management.

## Public API (`mktlib/backtest/__init__.py`)

| Export | Source | Description |
|-|-|-|
| `run(df, strategy, *, trade_side, calendar, flatten_eod, instrument_col)` | `_engine.py:279` | Run vectorized backtest; returns `BacktestResult` or `MultiBacktestResult` |
| `Strategy` | `_types.py:28` | Protocol: `entry() -> Condition \| pl.Expr`, `exit() -> Condition \| pl.Expr` |
| `BacktestResult` | `_types.py:43` | Dataclass: `returns`, `trades`, `signals` DataFrames |
| `MultiBacktestResult` | `_types.py:54` | Dict-like container of per-symbol `BacktestResult`s with lazy-cached combined views |
| `TradeSide` | `_types.py:10` | IntEnum: `LONG=1`, `SHORT=-1` (used as numeric multiplier) |
| `Condition` | `_conditions.py:146` | Base class for signal conditions resolving to `pl.Expr` |
| `Custom` | `_conditions.py:241` | Wraps a bare `pl.Expr` as a Condition |
| `Crossover` | `_conditions.py:165` | `a` crosses above `b` |
| `Crossunder` | `_conditions.py:179` | `a` crosses below `b` |
| `PriceIsAbove` | `_conditions.py:193` | `a > b` |
| `PriceIsBelow` | `_conditions.py:205` | `a < b` |
| `IsRising` | `_conditions.py:217` | Value > value `period` bars ago |
| `IsFalling` | `_conditions.py:229` | Value < value `period` bars ago |
| `All`, `Any_`, `Not` | `_conditions.py:255,267,279` | Combinators (`&`, `\|`, `~`) |
| `PriceExpr` | `_conditions.py:22` | Arithmetic building block: `Col`, `Lit`, `Pct`, `EntryRef`, `_BinOp` |
| `Col`, `Lit`, `Pct` | `_conditions.py:67,77,130` | Column ref, literal, percentage offset from base |
| `EntryRef` | `_conditions.py:146` | Entry-bar snapshot — resolves to `_entry_{col}` column |

## Engine (`_engine.py`)

### `run()` — L279

Dispatches to `_run_core` (single-symbol) or `_run_multi` (multi-symbol via `instrument_col`).

**Bare `pl.Expr` support**: `entry()`/`exit()` may return a `pl.Expr` instead of a `Condition` — auto-wrapped in `Custom` at L101–102.

**`init()` hook**: If the strategy defines `init(self, df) -> pl.DataFrame`, it is called before signal evaluation (L95–97) to enrich the DataFrame with indicator columns.

### `_run_core()` — L121

Single-symbol backtest pipeline:
1. Call `strategy.init(df)` if defined
2. Resolve entry condition to `_entry` column (pass 1)
3. If exit condition contains `EntryRef` nodes: create `_entry_{col}` snapshot columns (value where `_entry` is true, forward-filled)
4. Resolve exit condition to `_exit` column (pass 2 — snapshot columns now exist)
5. Build `_position` (1=in, 0=out) with forward-fill
6. Detect clean entry/exit transitions (`_entry_clean`, `_exit_clean`)
7. Compute per-bar returns with fill-at-open adjustment
8. Extract trade log via `_extract_trades`

### EntryRef tree walker — L87

| Function | Line | Purpose |
|-|-|-|
| `_collect_entry_refs(cond)` | L91 | Return set of column names from `EntryRef` nodes in condition tree |
| `_walk_cond(cond, cols)` | L98 | Recursive walk over `Condition` tree (All/Any\_/Not/PriceIs\*) |
| `_walk_expr(node, cols)` | L111 | Recursive walk over `PriceExpr` tree (EntryRef/Pct/\_BinOp) |

**Return model** (fill-at-next-open):
- Entry bar: `(close - open) / open`
- Middle bars: `close / prev_close - 1`
- Exit bar: `(open - prev_close) / prev_close`

### `_run_multi()` — L223

Multi-symbol backtest: partitions by `instrument_col`, runs `_run_core` per partition, wraps results in `MultiBacktestResult`.

### `flatten_eod` behavior — L115

When `flatten_eod=True` (requires `calendar`):
- Entries on session-last bars deferred to next session's first bar (L118–123)
- Position forced to 0 at session-last bar
- Session-forced exits fill at session-last bar's `open` (not next session's open)

### `_build_session_last_mask()` — L55

Detects session-last bar from actual data (not fixed offset), so it works for any candle size (1min, 5min, 15min, …). Groups by session via `join_asof` on `market_open`, takes `max(date)` per session.

### `_extract_trades()` — L368

Pairs entry/exit transitions by ordinal position. Entry fills use `open.shift(-1)`. Exit fills use `open.shift(-1)` normally, or `open` (current bar) for session-forced exits.

Output schema: `(entry_date, exit_date, pnl, bars_held)`.

### Calendar helpers

| Function | Line | Purpose |
|-|-|-|
| `_get_schedule` | L45 | Get schedule for date range from calendar |
| `_build_session_last_mask` | L55 | Boolean mask: bar is last bar of session (data-driven) |
| `_align_tz` | L27 | Match timezone between two series |

## Types (`_types.py`)

### `Strategy` — L28

Protocol with `entry()` and `exit()` returning `Condition | pl.Expr`. Optional `init(self, df) -> pl.DataFrame` (duck-typed via `getattr`, not in Protocol signature to avoid breaking existing strategies).

### `MultiBacktestResult` — L54

Dict-like container: `result["AAPL"]` → `BacktestResult`. Supports `len()`, iteration, `in`, `.items()`, `.symbols`.

Combined views via `cached_property`:
- `.returns` — `(symbol, date, return)` all symbols concatenated
- `.trades` — `(symbol, entry_date, exit_date, pnl, bars_held)`
- `.signals` — `(symbol, ..., _entry, _exit, _position)`

Symbol column name set by `instrument_col` param, prepended to each combined frame.

## Conditions (`_conditions.py`)

All conditions are frozen dataclasses with `resolve() -> pl.Expr`. Support `&`, `|`, `~` operators via `All`, `Any_`, `Not` combinators. Optional `trade_side` field overrides `run()`'s default.

`Custom(expr)` wraps any `pl.Expr` — also used implicitly when `entry()`/`exit()` returns a bare `pl.Expr`.

`PriceExpr` hierarchy (`Col`, `Lit`, `Pct`, `_BinOp`) supports arithmetic (`+`, `-`, `*`, `/`, `%`, unary `-`) for building price-relative conditions like `Pct(Col("close"), 0.02)`.

`_ref(b)` helper resolves `str` → `pl.col(b)`, `float` → `pl.lit(b)`.

## Strategies (`strategies/`)

| Strategy | File | Entry | Exit |
|-|-|-|-|
| `MacdCrossover` | `_macd.py:7` | MACD crosses above signal | MACD crosses below signal |
| `MacdCrossoverShort` | `_macd.py:21` | MACD crosses below signal (SHORT) | MACD crosses above signal |

## Performance

Benchmarked on MACD crossover, 491K minute bars (5yr synthetic data). Signal resolution via Polars in all cases.

| Engine | Time | vs Polars |
|-|-|-|
| Polars (vectorized `with_columns`) | 0.025s | baseline |
| Numpy (vectorized array ops) | 0.033s | 1.3x slower |
| Pandas (vectorized) | 0.223s | 8.9x slower |
| Python for-loop | 0.206s | 8.2x slower |
| Numba JIT (warm) | 0.009s | 2.8x faster |

Calendar filtering adds ~8ms (schedule join). `flatten_eod` adds ~4ms.

Benchmark scripts: `scripts/bench_macd_market.py`, `scripts/bench_single_pass.py`, `scripts/bench_pandas.py`, `scripts/bench_numpy.py`.

## Dependencies

- `mktlib.scheduling` — optional, used only when `calendar` is provided.
