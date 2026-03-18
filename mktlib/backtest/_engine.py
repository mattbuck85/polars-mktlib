from __future__ import annotations

import datetime
from typing import TYPE_CHECKING, overload

import polars as pl

from mktlib.backtest._conditions import (
    All,
    Any_,
    Condition,
    Custom,
    EntryRef,
    Not,
    Pct,
    PriceExpr,
    PriceIsAbove,
    PriceIsBelow,
    _BinOp,
)
from mktlib.backtest._types import BacktestResult, MultiBacktestResult, Strategy, TradeSide

if TYPE_CHECKING:
    from mktlib.scheduling import ExchangeCalendar


def _to_date(dt: datetime.datetime | datetime.date) -> datetime.date:
    """Convert datetime to date if needed, for calendar API compatibility."""
    if isinstance(dt, datetime.datetime):
        return dt.date()
    return dt


def _tz(series: pl.Series) -> str | None:
    """Extract timezone from a Datetime series, or None."""
    return series.dtype.time_zone  # type: ignore[union-attr]


def _align_tz(target: pl.Series, reference: pl.Series) -> pl.Series:
    """Align *target* timezone to match *reference*."""
    ref_tz = _tz(reference)
    tgt_tz = _tz(target)
    if ref_tz is None and tgt_tz is not None:
        return target.dt.replace_time_zone(None)
    if ref_tz is not None and tgt_tz is None:
        return target.dt.replace_time_zone(ref_tz)
    return target


# ---------------------------------------------------------------------------
# Schedule cache — avoids recomputing calendar.schedule() across mask calls
# ---------------------------------------------------------------------------

_schedule_cache: dict[tuple[str, datetime.date, datetime.date], pl.DataFrame] = {}


def _get_schedule(calendar: ExchangeCalendar, dates: pl.Series) -> pl.DataFrame:
    """Return cached schedule DataFrame for the calendar covering *dates*."""
    start = _to_date(dates.min())  # type: ignore[arg-type]
    end = _to_date(dates.max())  # type: ignore[arg-type]
    key = (calendar.name, start, end)
    if key not in _schedule_cache:
        _schedule_cache[key] = calendar.schedule(start, end)
    return _schedule_cache[key]


def _build_session_last_mask(
    dates: pl.Series,
    calendar: ExchangeCalendar,
) -> pl.Series:
    """Boolean series: True on the last bar of each trading session.

    Finds the actual last bar per session from the data rather than assuming
    a fixed offset, so this works for any candle size (1min, 5min, 15min, …).
    """
    sched = _get_schedule(calendar, dates)
    open_times = _align_tz(sched["market_open"], dates)
    close_times = _align_tz(sched["market_close"], dates)

    sessions = pl.DataFrame({
        "market_open": open_times,
        "market_close": close_times,
    })

    bar_df = dates.to_frame("date").with_row_index("_idx")
    joined = bar_df.join_asof(sessions, left_on="date", right_on="market_open")

    joined = joined.filter(pl.col("date") < pl.col("market_close"))
    last_per_session = joined.group_by("market_open").agg(
        pl.col("date").max().alias("last_bar"),
    )

    last_bars = last_per_session["last_bar"].to_list()
    return dates.is_in(last_bars)


# ---------------------------------------------------------------------------
# EntryRef tree walker — collects column names needed for entry-bar snapshots
# ---------------------------------------------------------------------------


def _collect_entry_refs(cond: Condition) -> set[str]:
    """Return all column names referenced by ``EntryRef`` nodes in *cond*."""
    cols: set[str] = set()
    _walk_cond(cond, cols)
    return cols


def _walk_cond(cond: Condition, cols: set[str]) -> None:
    match cond:
        case All(left, right, _) | Any_(left, right, _):
            _walk_cond(left, cols)
            _walk_cond(right, cols)
        case Not(inner, _):
            _walk_cond(inner, cols)
        case PriceIsAbove(a, b, _) | PriceIsBelow(a, b, _):
            _walk_expr(a, cols)
            _walk_expr(b, cols)
        case _:
            pass


def _walk_expr(node: str | float | PriceExpr, cols: set[str]) -> None:
    match node:
        case EntryRef(col):
            cols.add(col)
        case Pct(base, _):
            _walk_expr(base, cols)
        case _BinOp(left, right, _):
            _walk_expr(left, cols)
            _walk_expr(right, cols)
        case _:
            pass


def _run_core(
    df: pl.DataFrame,
    strategy: Strategy,
    *,
    trade_side: TradeSide,
    calendar: ExchangeCalendar | None,
    flatten_eod: bool,
) -> BacktestResult:
    """Inner engine: assumes *df* is already calendar-filtered."""
    # Let strategy enrich the DataFrame with indicator columns
    _init = getattr(strategy, "init", None)
    if _init is not None:
        df = _init(df)

    entry_raw = strategy.entry()
    exit_raw = strategy.exit()
    entry_cond = entry_raw if isinstance(entry_raw, Condition) else Custom(entry_raw)
    exit_cond = exit_raw if isinstance(exit_raw, Condition) else Custom(exit_raw)
    entry_expr = entry_cond.resolve()

    # Entry condition's side overrides the run() default
    effective_side = int(entry_cond.trade_side or trade_side)

    # Pass 1: compute _entry
    signals = df.with_columns(entry_expr.alias("_entry"))

    # Create snapshot columns for any EntryRef nodes in exit condition
    entry_refs = _collect_entry_refs(exit_cond)
    if entry_refs:
        signals = signals.with_columns(
            pl.when(pl.col("_entry")).then(pl.col(col)).otherwise(None)
            .forward_fill().alias(f"_entry_{col}")
            for col in entry_refs
        )

    # Pass 2: compute _exit (snapshot columns now exist for EntryRef.resolve())
    exit_expr = exit_cond.resolve()
    signals = signals.with_columns(exit_expr.alias("_exit"))

    # Position tracking: 1 on entry, 0 on exit, forward-fill
    if flatten_eod:
        _session_last = _build_session_last_mask(signals["date"], calendar)  # type: ignore[arg-type]
        signals = signals.with_columns(_session_last.alias("_session_last"))
        # Defer entries on session-last bars to the first bar of the next
        # session (e.g. crossover on 15:59 → enter at next day's 09:30).
        _suppressed = pl.col("_entry") & pl.col("_session_last")
        signals = signals.with_columns(
            (pl.col("_entry") | _suppressed.shift(1).fill_null(False)).alias("_entry"),
        )
        # Suppress entries on session-last bars (position opens and immediately
        # force-closes in the same bar — not a valid trade).
        signals = signals.with_columns(
            pl.when(pl.col("_entry") & ~pl.col("_session_last"))
            .then(pl.lit(1))
            .when(pl.col("_exit") | pl.col("_session_last"))
            .then(pl.lit(0))
            .otherwise(pl.lit(None))
            .forward_fill()
            .fill_null(0)
            .alias("_position"),
        )
    else:
        signals = signals.with_columns(
            pl.when(pl.col("_entry"))
            .then(pl.lit(1))
            .when(pl.col("_exit"))
            .then(pl.lit(0))
            .otherwise(pl.lit(None))
            .forward_fill()
            .fill_null(0)
            .alias("_position"),
        )

    # Materialize shared shifted expressions once
    signals = signals.with_columns(
        pl.col("_position").shift(1).fill_null(0).alias("_pos_d1"),
        pl.col("_position").shift(2).fill_null(0).alias("_pos_d2"),
        pl.col("close").shift(1).alias("_close_prev"),
    )
    # Transition detection (uses materialized _pos_d1)
    signals = signals.with_columns(
        ((pl.col("_position") == 1) & (pl.col("_pos_d1") == 0)).alias("_entry_clean"),
        ((pl.col("_position") == 0) & (pl.col("_pos_d1") == 1)).alias("_exit_clean"),
    )

    # Detect transition bars (after the 1-bar delay for fill)
    _is_entry_bar = (pl.col("_pos_d1") == 1) & (pl.col("_pos_d2") == 0)
    _is_exit_bar = (pl.col("_pos_d1") == 0) & (pl.col("_pos_d2") == 1)

    # Per-bar returns with fill-at-open adjustment
    _entry_ret = ((pl.col("close") - pl.col("open")) / pl.col("open")) * effective_side
    _normal_ret = (pl.col("close") / pl.col("_close_prev") - 1) * effective_side
    _exit_ret = (
        (pl.col("open") - pl.col("_close_prev")) / pl.col("_close_prev")
    ) * effective_side

    # Compute returns + flatten_eod overrides in minimal with_columns calls
    if flatten_eod:
        # Base returns + session-last override in one pass
        signals = signals.with_columns(
            pl.when(pl.col("_session_last") & _is_entry_bar)
            .then(0.0)
            .when(pl.col("_session_last") & (pl.col("_pos_d1") == 1))
            .then(_exit_ret)
            .when(_is_entry_bar)
            .then(_entry_ret)
            .when(_is_exit_bar)
            .then(_exit_ret)
            .when(pl.col("_pos_d1") == 1)
            .then(_normal_ret)
            .otherwise(0.0)
            .fill_null(0.0)
            .alias("return"),
        )
        # Post-session-last bar zeroing
        signals = signals.with_columns(
            pl.when(
                pl.col("_session_last").shift(1).fill_null(False)
                & ~_is_entry_bar
            )
            .then(0.0)
            .otherwise(pl.col("return"))
            .alias("return"),
        )
    else:
        signals = signals.with_columns(
            pl.when(_is_entry_bar)
            .then(_entry_ret)
            .when(_is_exit_bar)
            .then(_exit_ret)
            .when(pl.col("_pos_d1") == 1)
            .then(_normal_ret)
            .otherwise(0.0)
            .fill_null(0.0)
            .alias("return"),
        )

    # Drop internal columns before return
    signals = signals.drop("_pos_d1", "_pos_d2", "_close_prev")

    returns = signals.select("date", "return")

    # Build trade log from entry/exit transitions
    trades = _extract_trades(signals, effective_side, flatten_eod=flatten_eod)

    return BacktestResult(returns=returns, trades=trades, signals=signals)


def _run_multi(
    df: pl.DataFrame,
    strategy: Strategy,
    *,
    instrument_col: str,
    trade_side: TradeSide,
    calendar: ExchangeCalendar | None,
    flatten_eod: bool,
) -> MultiBacktestResult:
    """Run independent backtests per instrument and combine results."""
    if instrument_col not in df.columns:
        msg = f"instrument_col={instrument_col!r} not found in DataFrame columns"
        raise ValueError(msg)

    # Calendar filter once on full df
    if calendar is not None:
        df = calendar.filter_market_hours(df, "date")

    by_instrument: dict[str, BacktestResult] = {}
    for inst_df in df.partition_by(instrument_col, maintain_order=True):
        instrument = inst_df[instrument_col][0]
        by_instrument[instrument] = _run_core(
            inst_df.drop(instrument_col),
            strategy,
            trade_side=trade_side,
            calendar=calendar,
            flatten_eod=flatten_eod,
        )

    return MultiBacktestResult(by_instrument, instrument_col=instrument_col)


@overload
def run(
    df: pl.DataFrame,
    strategy: Strategy,
    *,
    trade_side: TradeSide = ...,
    calendar: ExchangeCalendar | None = ...,
    flatten_eod: bool = ...,
    instrument_col: str,
) -> MultiBacktestResult: ...


@overload
def run(
    df: pl.DataFrame,
    strategy: Strategy,
    *,
    trade_side: TradeSide = ...,
    calendar: ExchangeCalendar | None = ...,
    flatten_eod: bool = ...,
    instrument_col: None = ...,
) -> BacktestResult: ...


def run(
    df: pl.DataFrame,
    strategy: Strategy,
    *,
    trade_side: TradeSide = TradeSide.LONG,
    calendar: ExchangeCalendar | None = None,
    flatten_eod: bool = False,
    instrument_col: str | None = None,
) -> BacktestResult | MultiBacktestResult:
    """Run a vectorized backtest with fill-at-next-open semantics.

    Parameters
    ----------
    df
        Must contain ``date``, ``open``, ``close``, and any indicator
        columns referenced by the strategy.
    strategy
        Object with ``entry()`` and ``exit()`` returning Conditions.
        May optionally define ``init(df) -> pl.DataFrame`` to enrich the
        DataFrame with indicator columns before signal evaluation.
    trade_side
        Trade direction. Overridden by the entry condition's ``trade_side``
        if set.
    calendar
        Exchange calendar for market-hours filtering. When provided, the
        DataFrame is filtered to market hours before signal computation.
    flatten_eod
        Force-close positions at each session's last bar, eliminating
        overnight exposure. Requires *calendar*.
    instrument_col
        Column name identifying the symbol/ticker in a multi-symbol
        DataFrame. When provided, returns a
        :class:`~mktlib.backtest.MultiBacktestResult` that stores
        per-symbol results for O(1) access (``result["AAPL"]``).
        Combined DataFrames with the symbol column prepended are
        available via ``.returns``, ``.trades``, ``.signals`` properties.

    Returns
    -------
    BacktestResult
        When *instrument_col* is ``None`` (default).
    MultiBacktestResult
        When *instrument_col* is set. Supports ``result[symbol]`` for O(1)
        per-symbol access, iteration, and lazy-cached combined views.

    Notes
    -----
    Signal at bar *t* → market order fills at bar *t+1*'s open.

    - **Entry bar** (*t+1*): return = ``(close - open) / open``
    - **Middle bars**: return = ``close / prev_close - 1``
    - **Exit bar** (first bar where position drops to 0): return =
      ``(open - prev_close) / prev_close`` (gap to fill price only)

    When *instrument_col* is set, each symbol is backtested independently —
    indicators (e.g. rolling SMA) do not bleed across symbols. Calendar
    filtering is applied once on the full DataFrame before partitioning.
    Per-symbol aggregation (e.g. equal-weight portfolio returns) is left
    to the caller::

        result.returns.group_by("date").agg(pl.col("return").mean())
    """
    if flatten_eod and calendar is None:
        msg = "flatten_eod=True requires a calendar"
        raise ValueError(msg)

    if instrument_col is not None:
        return _run_multi(
            df,
            strategy,
            instrument_col=instrument_col,
            trade_side=trade_side,
            calendar=calendar,
            flatten_eod=flatten_eod,
        )

    # Single-symbol path: calendar filter → _run_core
    if calendar is not None:
        df = calendar.filter_market_hours(df, "date")

    return _run_core(
        df,
        strategy,
        trade_side=trade_side,
        calendar=calendar,
        flatten_eod=flatten_eod,
    )


def _extract_trades(
    signals: pl.DataFrame,
    side: int = 1,
    *,
    flatten_eod: bool = False,
) -> pl.DataFrame:
    """Extract per-trade PnL from position transitions.

    Fill prices use the *next* bar's open (fill-at-next-open model).
    For session-forced exits (flatten_eod), the exit fill is the
    session-last bar's own open (can't trade during the close minute).
    """
    # Pre-compute next bar's open for fill price
    signals_with_next = signals.with_columns(
        pl.col("open").shift(-1).alias("_next_open"),
    )
    entries = signals_with_next.filter(pl.col("_entry_clean")).select(
        pl.col("date").alias("entry_date"),
        pl.col("_next_open").alias("entry_price"),
        pl.int_range(pl.len()).alias("_entry_idx"),
    )

    # Exit fill price: session-forced exits use current bar's open,
    # normal exits use next bar's open (fill-at-next-open).
    if flatten_eod:
        exit_price_expr = (
            pl.when(pl.col("_session_last"))
            .then(pl.col("open"))
            .otherwise(pl.col("_next_open"))
            .alias("exit_price")
        )
    else:
        exit_price_expr = pl.col("_next_open").alias("exit_price")

    exits = signals_with_next.filter(pl.col("_exit_clean")).select(
        pl.col("date").alias("exit_date"),
        exit_price_expr,
        pl.int_range(pl.len()).alias("_exit_idx"),
    )

    # Pair entries with exits by ordinal position
    n_trades = min(entries.height, exits.height)
    if n_trades == 0:
        return pl.DataFrame(
            schema={
                "entry_date": signals["date"].dtype,
                "exit_date": signals["date"].dtype,
                "pnl": pl.Float64,
                "bars_held": pl.Int64,
            }
        )

    entries = entries.head(n_trades)
    exits = exits.head(n_trades)

    trades = pl.DataFrame(
        {
            "entry_date": entries["entry_date"],
            "exit_date": exits["exit_date"],
            "entry_price": entries["entry_price"],
            "exit_price": exits["exit_price"],
        }
    )

    trades = trades.with_columns(
        (side * (pl.col("exit_price") / pl.col("entry_price") - 1)).alias("pnl"),
        (
            (pl.col("exit_date").cast(pl.Date) - pl.col("entry_date").cast(pl.Date)).dt.total_days()
        ).alias("bars_held"),
    ).select("entry_date", "exit_date", "pnl", "bars_held")

    return trades
