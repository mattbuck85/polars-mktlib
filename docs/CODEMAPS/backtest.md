# backtest — Vectorized Backtesting Engine

Signal-driven backtesting with fill-at-next-open semantics, exchange calendar integration, and session-boundary position management.

## Public API (`mktlib/backtest/__init__.py`)

| Export | Source | Description |
|-|-|-|
| `run(df, strategy, *, trade_side, calendar, flatten_eod, instrument_col)` | `_engine.py:279` | Run vectorized backtest; returns `BacktestResult` or `MultiBacktestResult` |
| `Cost` | `_cost.py:47` | Frozen dataclass: per-side transaction cost in basis points (`commission_bps`, `slippage_bps`, `slippage_col`) |
| `Bracket` | `_bracket.py:105` | Frozen dataclass: protective TP/SL resting against every position (`take_profit`, `stop_loss`, `both_touch`) |
| `Strategy` | `_types.py:28` | Protocol: `entry() -> Condition \| pl.Expr`, `exit() -> Condition \| pl.Expr` |
| `BacktestResult` | `_types.py:43` | Dataclass: `returns`, `trades`, `signals` DataFrames |
| `MultiBacktestResult` | `_types.py:54` | Dict-like container of per-symbol `BacktestResult`s with lazy-cached combined views |
| `TradeSide` | `_types.py:10` | IntEnum: `LONG=1`, `SHORT=-1` (used as numeric multiplier) |
| `Condition` | `_conditions.py:146` | Base class for signal conditions resolving to `pl.Expr` |
| `Custom` | `_conditions.py:241` | Wraps a bare `pl.Expr` as a Condition |
| `Crossover` | `_conditions.py:165` | `a` crosses above `b` |
| `Crossunder` | `_conditions.py:179` | `a` crosses below `b` |
| `ValueGT` | `_conditions.py:228` | `a > b` (alias: `PriceIsAbove`) |
| `ValueGTE` | `_conditions.py:240` | `a >= b` |
| `ValueLT` | `_conditions.py:252` | `a < b` (alias: `PriceIsBelow`) |
| `ValueLTE` | `_conditions.py:264` | `a <= b` |
| `IsRising` | `_conditions.py:217` | Value > value `period` bars ago |
| `IsFalling` | `_conditions.py:229` | Value < value `period` bars ago |
| `All`, `Any_`, `Not` | `_conditions.py:255,267,279` | Combinators (`&`, `\|`, `~`) |
| `ColExpr` | `_conditions.py:34` | Arithmetic + comparison building block: `Col`, `Lit`, `Pct`, `EntryRef`, `_BinOp` (alias: `PriceExpr`) |
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
7. Apply `bracket=Bracket(...)` via `_apply_bracket` — levels, triggers, first-hit-per-block, `_exit_clean` redirect (all columns dropped before return)
8. Materialize `_cost_bps` when `cost=Cost(...)` was passed (dropped before return)
9. Compute per-bar returns with fill-at-open adjustment
10. Extract trade log via `_extract_trades`

### Transaction costs (`_cost.py`)

| Symbol | Purpose |
|-|-|
| `Cost` | Frozen, `slots=True`, primitives only (no callables — a closure is invisible to a consumer's cache key). Validates non-negative + finite in `__post_init__` |
| `COST_COLUMN` | `"_cost_bps"` — the internal per-bar column the engine materializes and drops |
| `cost_bps_expr(cost)` | `lit(commission_bps + slippage_bps) [+ col(slippage_col)]` |

Cost is folded into the three **fill-bar** return expressions (`_entry_ret`, `_exit_ret`, `_limit_ret`) via the local `_charge()` helper — never into the `when/then` chains, and never multiplied by `effective_side`. `_extract_trades` reads each leg's cost with the same alignment as the price that leg pays: the entry leg from `shift(-1)`, the exit leg mirroring `exit_price`'s limit / session-last / next-open priority.

Backward compatibility: `cost=None` takes a byte-for-byte unchanged code path; `cost=Cost()` exercises the real arithmetic and is pinned against the frozen Parquet baselines in `tests/backtest/test_golden_baseline.py`.

### Bracket exits (`_bracket.py`)

| Symbol | Purpose |
|-|-|
| `Bracket` | Frozen, `slots=True`, primitives only. `take_profit`/`stop_loss` are a `float` (fraction of the entry fill) or a `str` (column of absolute levels, latched at the `_entry_clean` signal bar); `both_touch` picks the same-bar policy |
| `level_expr` / `trigger_expr` / `fill_expr` | The decision table, emitted per compile-time-known side — never derived by multiplying a comparison through `effective_side` |
| `BRACKET_COLUMNS` | The internal working columns the engine materializes and drops |

Fill table (mirrors a conventional event-driven OHLC broker: a long bracket is a sell limit plus a sell stop):

| Side | Leg | Trigger | Fill |
|-|-|-|-|
| long | TP | `high >= tp` | `max(open, tp)` |
| long | SL | `low <= sl` | `min(open, sl)` |
| short | TP | `low <= tp` | `min(open, tp)` |
| short | SL | `high >= sl` | `max(open, sl)` |

`_apply_bracket` (`_engine.py`) latches the entry fill price and a per-position `_bracket_block` id, computes both legs' levels, masks triggers to `_pos_d1 == 1` (so the bracket is armed **from the entry fill bar**, and — under `flatten_eod` — never on a session-last bar, where the engine already flattened at the open), keeps only the first trigger per block via `cum_sum().over(block) == 1`, and redirects `_exit_clean` to the bracket bar. `_position` is left inconsistent, exactly as the `Limit` path does; the return expression (`_with_bracket`) and `_extract_trades` carry the truth. Bars after the bracket bar in the same block return `0.0` — **a bracketed block never re-enters**, which is a deliberate divergence from live.

`both_touch` defaults to `"stop_first"`, which **diverges from submission-order OCO on purpose**: a live bracket is commonly an OCO pair whose TP leg is submitted before the SL leg and filled in submission order, so the realized policy on a both-touch bar is `take_profit_first`. Pinned by `test_default_policy_diverges_from_submission_order_oco`.

Unsupported combinations, both `NotImplementedError`: `bracket` + `short_strategy` (the dual merge would evaluate the long leg's levels against the short leg's position) and `bracket` + a `Limit(...)` exit (both claim the same-bar fill).

Backward compatibility: `bracket=None` takes a byte-for-byte unchanged code path, pinned by `test_golden_baseline_no_bracket`.

### EntryRef tree walker — L87

| Function | Line | Purpose |
|-|-|-|
| `_collect_entry_refs(cond)` | L91 | Return set of column names from `EntryRef` nodes in condition tree |
| `_walk_cond(cond, cols)` | L98 | Recursive walk over `Condition` tree (All/Any\_/Not/Value\*) |
| `_walk_expr(node, cols)` | L111 | Recursive walk over `ColExpr` tree (EntryRef/Pct/\_BinOp) |

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

`ColExpr` hierarchy (`Col`, `Lit`, `Pct`, `_BinOp`) supports arithmetic (`+`, `-`, `*`, `/`, `%`, unary `-`) and comparison operators (`>`, `>=`, `<`, `<=`) for building conditions like `Col("rsi") > 70` or `Pct(Col("close"), 0.02)`.

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
