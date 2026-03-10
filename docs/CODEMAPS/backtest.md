# backtest — Vectorized Backtesting Engine

Signal-driven backtesting with fill-at-next-open semantics, exchange calendar integration, and session-boundary position management.

## Public API (`mktlib/backtest/__init__.py`)

| Export | Source | Description |
|-|-|-|
| `run(df, strategy, *, trade_side, calendar, flatten_eod)` | `_engine.py:73` | Run vectorized backtest, returns `BacktestResult` |
| `Strategy` | `_types.py:24` | Protocol: `entry() -> Condition`, `exit() -> Condition` |
| `BacktestResult` | `_types.py:31` | Dataclass: `returns`, `trades`, `signals` DataFrames |
| `TradeSide` | `_types.py:8` | IntEnum: `LONG=1`, `SHORT=-1` (used as numeric multiplier) |
| `Condition` | `_conditions.py:15` | Base class for signal conditions resolving to `pl.Expr` |
| `Crossover` | `_conditions.py:34` | `a` crosses above `b` |
| `Crossunder` | `_conditions.py:48` | `a` crosses below `b` |
| `PriceIsAbove` | `_conditions.py:62` | `a > b` |
| `PriceIsBelow` | `_conditions.py:74` | `a < b` |
| `IsRising` | `_conditions.py:86` | Value > value `period` bars ago |
| `IsFalling` | `_conditions.py:98` | Value < value `period` bars ago |
| `All`, `Any_`, `Not` | `_conditions.py:113,125,137` | Combinators (`&`, `\|`, `~`) |

## Engine (`_engine.py`)

### `run()` — L73

Core backtest loop. Pipeline:
1. Filter to market hours via `_build_market_mask` (if calendar provided)
2. Resolve entry/exit conditions to boolean columns
3. Build `_raw_position` (1=in, 0=out) with forward-fill
4. Detect clean entry/exit transitions (`_entry_clean`, `_exit_clean`)
5. Compute per-bar returns with fill-at-open adjustment
6. Extract trade log via `_extract_trades`

**Return model** (fill-at-next-open):
- Entry bar: `(close - open) / open`
- Middle bars: `close / prev_close - 1`
- Exit bar: `(open - prev_close) / prev_close`

### `flatten_eod` behavior — L131, L205

When `flatten_eod=True` (requires `calendar`):
- Entries suppressed on session-last bars (can't enter if forced to exit same bar)
- Position forced to 0 at session-last bar via `_session_last` mask
- **Per-bar returns**: same-bar entry+exit → 0; held position exit → `exit_ret` (gap to open only, no intraday)
- **Trade extraction**: session-forced exits fill at session-last bar's `open` (not next session's open)
- Phantom exit returns zeroed on bar after session-last (safety net)

### `_extract_trades()` — L235

Pairs `_entry_clean`/`_exit_clean` transitions by ordinal position. Entry fills use `open.shift(-1)`. Exit fills use `open.shift(-1)` normally, or `open` (current bar) for session-forced exits when `flatten_eod=True`.

Output schema: `(entry_date, exit_date, pnl, bars_held)`.

### Calendar helpers

| Function | Line | Purpose |
|-|-|-|
| `_build_market_mask` | L37 | Boolean mask: bar is in calendar's trading index |
| `_build_session_last_mask` | L51 | Boolean mask: bar is last minute of session (close - 1min) |
| `_align_tz` | L26 | Match timezone between two series |

## Conditions (`_conditions.py`)

All conditions are frozen dataclasses with `resolve() -> pl.Expr`. Support `&`, `|`, `~` operators via `All`, `Any_`, `Not` combinators. Optional `trade_side` field overrides `run()`'s default.

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
