# backtest — Vectorized Backtesting Engine

Signal-driven backtesting with fill-at-next-open semantics, exchange calendar integration, and session-boundary position management.

## Public API (`mktlib/backtest/__init__.py`)

| Export | Source | Description |
|-|-|-|
| `run(df, strategy, *, trade_side, calendar, flatten, flatten_eod, instrument_col)` | `_engine.py:939` | Run vectorized backtest; returns `BacktestResult` or `MultiBacktestResult`. Five `@overload` stubs at `:862`–`:924` discriminate the return type |
| `FlattenSchedule` | `_flatten.py:76` | Frozen dataclass: when positions are force-closed (`days`, `minutes_before_close`, `block_entry_minutes_before_close`) |
| `Weekday` | `_flatten.py:36` | ISO IntEnum: `MON=1` … `SUN=7`, matching `dt.weekday()` / `date.isoweekday()` |
| `Cost` | `_cost.py:47` | Frozen dataclass: per-side transaction cost in basis points (`commission_bps`, `slippage_bps`, `slippage_col`) |
| `Bracket` | `_bracket.py:117` | Frozen dataclass: protective TP/SL resting against every position (`take_profit`, `stop_loss`, `both_touch`, `anchor`) |
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

### `run()` — L939 (overloads L862–L924)

Dispatches to `_run_core` (single-symbol) or `_run_multi` (multi-symbol via `instrument_col`).

**Bare `pl.Expr` support**: `entry()`/`exit()` may return a `pl.Expr` instead of a `Condition` — auto-wrapped in `Custom` at L101–102.

**`init()` hook**: If the strategy defines `init(self, df) -> pl.DataFrame`, it is called before signal evaluation (L95–97) to enrich the DataFrame with indicator columns.

### `_run_core()` — L315

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
| `Bracket` | Frozen, `slots=True`, primitives only. `take_profit`/`stop_loss` are a `float` (fraction of the entry fill) or a `str` (column of absolute levels, latched at the `_entry_clean` signal bar); `both_touch` picks the same-bar policy; `anchor` picks the level-anchoring policy |
| `Anchor` / `_ANCHOR_POLICIES` | `Literal["position", "signal"]` and its membership set — validated in `__post_init__`, mirroring `BothTouch`. Not exported; `__init__.py` exports only `Bracket` |
| `level_expr` / `trigger_expr` / `fill_expr` | The decision table, emitted per compile-time-known side — never derived by multiplying a comparison through `effective_side`. `level_expr` takes `resignal_col` (`None` under `anchor="position"`, and then emits the same expression tree as before) |
| `BRACKET_COLUMNS` | The internal working columns the engine materializes and drops — includes `ANCHOR_FILL_COLUMN` / `RESIGNAL_COLUMN`, which are materialized only under `anchor="signal"` |

Fill table (mirrors a conventional event-driven OHLC broker: a long bracket is a sell limit plus a sell stop):

| Side | Leg | Trigger | Fill |
|-|-|-|-|
| long | TP | `high >= tp` | `max(open, tp)` |
| long | SL | `low <= sl` | `min(open, sl)` |
| short | TP | `low <= tp` | `min(open, tp)` |
| short | SL | `high >= sl` | `max(open, sl)` |

`_apply_bracket` (`_engine.py`) latches the entry fill price and a per-position `_bracket_block` id, computes both legs' levels, masks triggers to `_pos_d1 == 1` (so the bracket is armed **from the entry fill bar**, and — when flattening — never on a flatten bar, where the engine already flattened at the open), keeps only the first trigger per block via `cum_sum().over(block) == 1`, and redirects `_exit_clean` to the bracket bar. `_position` is left inconsistent, exactly as the `Limit` path does; the return expression (`_with_bracket`) and `_extract_trades` carry the truth. Bars after the bracket bar in the same block return `0.0` — **a bracketed block never re-enters**, which is a deliberate divergence from live.

`both_touch` defaults to `"stop_first"`, which **diverges from submission-order OCO on purpose**: a live bracket is commonly an OCO pair whose TP leg is submitted before the SL leg and filled in submission order, so the realized policy on a both-touch bar is `take_profit_first`. Pinned by `test_default_policy_diverges_from_submission_order_oco`.

Unsupported combinations, both `NotImplementedError`: `bracket` + `short_strategy` (the dual merge would evaluate the long leg's levels against the short leg's position) and `bracket` + a `Limit(...)` exit (both claim the same-bar fill).

**Level anchoring (`anchor`).** `"position"` (default) latches both legs once, on the entry that opened the position. `"signal"` re-latches them on every later `_entry` that fires while `_pos_d1 == 1` — and, under `flatten_eod`, not on a session-last bar, where the position is already flattened at the open. `_apply_bracket` emits two extra columns for it, both dropped by `BRACKET_COLUMNS` membership and **neither materialized under `"position"`**, so default byte-identity is structural rather than asserted:

| Column | Contents |
|-|-|
| `RESIGNAL_COLUMN` (`_bracket_resignal`) | `_entry & (_pos_d1 == 1)`, masked by `~_session_last` under `flatten_eod`, `fill_null(False)` |
| `ANCHOR_FILL_COLUMN` (`_bracket_anchor_fill`) | The open of each entry-fill *or* post-re-signal bar, forward-filled. `float` legs read this instead of `ENTRY_FILL_COLUMN`; `ENTRY_FILL_COLUMN` is untouched, so trade P&L still measures from the true entry fill |

The `.shift(1)` on both the `str` re-latch and the anchor fill is the arming convention, not cosmetic: a re-signal at bar `k` is observed at `k`'s close, so the new level is in force from `k+1` and a leg tagged at `k` closes the position first. A second, post-hoc null guard runs after `BRACKET_SEEN_COLUMN` exists — a null level on a re-latch bar in a block that has not yet fired raises; at or after the block's first trigger, that block is dead and it does not.

Backward compatibility: `bracket=None` takes a byte-for-byte unchanged code path, pinned by `test_golden_baseline_no_bracket`. `anchor="position"` is byte-identical to omitting the keyword, pinned by `test_golden_baseline_explicit_position_anchor` against the frozen baselines.

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

### `_run_multi()` — L746

Multi-symbol backtest: partitions by `instrument_col`, runs `_run_core` per partition, wraps results in `MultiBacktestResult`.

### Flatten behavior — `_engine.py:382`

When a flatten schedule resolves (requires `calendar`):
- Entries on the flatten bar are deferred to the next bar (`:388`–`:392`)
- Entries inside the block window are then dropped (`:401`–`:405`) — **after** the deferral, so the deferral cannot write into the window it just cleared
- `_position` is forced to 0 on the flatten bar, structurally (`:432`–`:440`)
- Session-forced exits fill at the flatten bar's own `open`, not the next bar's
- The bar after a flatten bar returns `0.0` unconditionally (`:620`–`:642`)

`run()` accepts `flatten=` (`None | bool | "eod" | "eow" | FlattenSchedule`) and the legacy `flatten_eod: bool`. `resolve_flatten()` (`_flatten.py:188`) collapses the two; setting both raises.

### `mktlib/backtest/_flatten.py` — everything that knows what a *day* is

The engine consumes two boolean masks and never mentions a calendar concept. That is what keeps the compiled resolver out of scope: `mktlib-scan` takes the flatten mask as `Option<&[bool]>`, so changing *when* a position is force-closed never touches the FFI contract, and entry blocking never reaches it at all (`_entry` is zeroed upstream).

| Symbol | Line | Notes |
|-|-|-|
| `FLATTEN_BAR_COLUMN` | `:32` | `"_flatten_bar"`; dropped before `run()` returns, so it appears in no golden parquet |
| `Weekday` | `:36` | ISO 1–7 |
| `FlattenSchedule` | `:76` | `__post_init__` normalizes `days` to `frozenset[Weekday]` (a bare `set` would leave the frozen dataclass unhashable) and canonicalizes `block_entry_minutes_before_close` to an int |
| `resolve_flatten()` | `:188` | `flatten=` / `flatten_eod=` → one schedule or `None` |
| `_WEEK_KEY` | `:262` | ISO Monday as an epoch-day int: `date.cast(Int32) - (weekday - 1)`. Not `year*100+week` (wrong every Dec/Jan) and not `dt.truncate("1w")` (epoch-anchored on a Thursday) |
| `_flatten_sessions()` | `:272` | Restricts to the sessions the schedule flattens. Selection runs over sessions that **have bars**, which is what makes `"weekly"` holiday-aware for free |
| `build_flatten_masks()` | `:302` | Returns `(flatten_bar, entry_blocked)`. Both offsets are provable no-ops at 0: `date <= close - 0` collapses into the existing `date < close`, and `date >= close - 0` is empty by construction |

Day selection keys off `calendar.schedule()`'s own `date` column — the *trading day* — not off `market_open`, which lands on the previous calendar date on any exchange with a negative open offset (CME Globex, FX).

### `_extract_trades()` — L1139

Pairs entry/exit transitions by ordinal position. Entry fills use `open.shift(-1)`. Exit fills use `open.shift(-1)` normally, or `open` (current bar) for session-forced exits.

Output schema: `(entry_date, exit_date, pnl, bars_held)`.

### Calendar helpers

These moved out of `_engine.py` into `_flatten.py`, so the engine holds no
calendar concept of its own.

| Function | Line | Purpose |
|-|-|-|
| `_get_schedule` | `_flatten.py:59` | Cached `calendar.schedule()` for the bar range |
| `build_flatten_masks` | `_flatten.py:302` | `(flatten_bar, entry_blocked)` boolean masks (data-driven, any candle size) |
| `_align_tz` | `_flatten.py:42` | Match timezone between two series |

Note: `mktlib/scheduling/_mixins.py:38` holds a **separate** `_align_tz`; that
is the one `tests/scheduling/test_align_tz.py` covers.

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

Calendar filtering adds ~8ms (schedule join). Flattening adds ~4ms.

Benchmark scripts: `scripts/bench_macd_market.py`, `scripts/bench_single_pass.py`, `scripts/bench_pandas.py`, `scripts/bench_numpy.py`.

## Dependencies

- `mktlib.scheduling` — optional, used only when `calendar` is provided.
