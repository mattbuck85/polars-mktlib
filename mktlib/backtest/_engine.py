from __future__ import annotations

import datetime
from typing import TYPE_CHECKING

import polars as pl

from mktlib.backtest._types import BacktestResult, Strategy

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


def _build_market_mask(
    dates: pl.Series,
    calendar: ExchangeCalendar,
) -> pl.Series:
    """Boolean series: True where date is in the calendar's trading index."""
    trading_idx = calendar.trading_index(
        _to_date(dates.min()),  # type: ignore[arg-type]
        _to_date(dates.max()),  # type: ignore[arg-type]
        period="1m",
    )
    trading_idx = _align_tz(trading_idx, dates)
    return dates.is_in(trading_idx.to_list())


def _build_session_last_mask(
    dates: pl.Series,
    calendar: ExchangeCalendar,
) -> pl.Series:
    """Boolean series: True on the last bar of each trading session.

    Uses open-frame convention: last bar timestamp = market_close - 1min.
    E.g. NYSE 16:00 close -> last bar at 15:59:00.
    """
    sched = calendar.schedule(
        _to_date(dates.min()),  # type: ignore[arg-type]
        _to_date(dates.max()),  # type: ignore[arg-type]
    )
    close_times = _align_tz(sched["market_close"], dates)
    # Last bar = market_close - 1min (open-frame: bar at 15:59 covers 15:59-16:00)
    last_minutes = (
        close_times.to_frame("c")
        .select(pl.col("c") - pl.duration(minutes=1))["c"]
    )
    return dates.is_in(last_minutes.to_list())


def run(
    df: pl.DataFrame,
    strategy: Strategy,
    *,
    trade_on: str = "close",
    calendar: ExchangeCalendar | None = None,
    prefilter_market_data: bool = True,
    flatten_eod: bool = False,
) -> BacktestResult:
    """Run a vectorized backtest with fill-at-next-open semantics.

    Parameters
    ----------
    df
        Must contain ``date``, ``open``, the *trade_on* price column, and
        any indicator columns referenced by the strategy.
    strategy
        Object with ``entry()`` and ``exit()`` returning Conditions.
    trade_on
        Price column used for return calculation (default ``close``).
    calendar
        Exchange calendar for market-hours filtering.
    prefilter_market_data
        If True (default), filter DataFrame to market hours before signal
        computation. If False, compute indicators on all data but zero out
        non-market returns.
    flatten_eod
        Force-close positions at each session's last bar, eliminating
        overnight exposure. Requires *calendar*.

    Notes
    -----
    Signal at bar *t* → market order fills at bar *t+1*'s open.

    - **Entry bar** (*t+1*): return = ``(close - open) / open``
    - **Middle bars**: return = ``close / prev_close - 1``
    - **Exit bar** (first bar where position drops to 0): return =
      ``(open - prev_close) / prev_close`` (gap to fill price only)
    """
    if flatten_eod and calendar is None:
        msg = "flatten_eod=True requires a calendar"
        raise ValueError(msg)

    # Pre-filter to market hours if requested
    if calendar is not None and prefilter_market_data:
        mask = _build_market_mask(df["date"], calendar)
        df = df.filter(mask)

    entry_expr = strategy.entry().resolve()
    exit_expr = strategy.exit().resolve()

    signals = df.with_columns(
        entry_expr.alias("_entry"),
        exit_expr.alias("_exit"),
    )

    # Position tracking: 1 on entry, 0 on exit, forward-fill
    if flatten_eod:
        _session_last = _build_session_last_mask(signals["date"], calendar)  # type: ignore[arg-type]
        signals = signals.with_columns(_session_last.alias("_session_last"))
        signals = signals.with_columns(
            pl.when(pl.col("_entry") & ~pl.col("_session_last"))
            .then(pl.lit(1))
            .when(pl.col("_exit") | pl.col("_session_last"))
            .then(pl.lit(0))
            .otherwise(pl.lit(None))
            .forward_fill()
            .fill_null(0)
            .alias("_raw_position"),
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
            .alias("_raw_position"),
        )

    # Suppress re-entry: only count 0→1 transitions as real entries
    signals = signals.with_columns(
        pl.when(
            (pl.col("_raw_position") == 1)
            & (pl.col("_raw_position").shift(1).fill_null(0) == 0)
        )
        .then(pl.lit(True))
        .otherwise(pl.lit(False))
        .alias("_entry_clean"),
        pl.when(
            (pl.col("_raw_position") == 0)
            & (pl.col("_raw_position").shift(1).fill_null(0) == 1)
        )
        .then(pl.lit(True))
        .otherwise(pl.lit(False))
        .alias("_exit_clean"),
    )

    # Use raw_position as _position (already handles forward-fill dedup)
    signals = signals.with_columns(
        pl.col("_raw_position").alias("_position"),
    )

    # Delayed position: position(t-1) tells us if we're in a trade this bar
    _pos_delayed = pl.col("_position").shift(1).fill_null(0)
    _pos_delayed2 = pl.col("_position").shift(2).fill_null(0)

    # Detect transition bars (after the 1-bar delay for fill)
    _is_entry_bar = (_pos_delayed == 1) & (_pos_delayed2 == 0)
    _is_exit_bar = (_pos_delayed == 0) & (_pos_delayed2 == 1)

    # Per-bar returns with fill-at-open adjustment
    _entry_ret = (pl.col(trade_on) - pl.col("open")) / pl.col("open")
    _normal_ret = pl.col(trade_on) / pl.col(trade_on).shift(1) - 1
    _exit_ret = (pl.col("open") - pl.col(trade_on).shift(1)) / pl.col(trade_on).shift(1)

    signals = signals.with_columns(
        pl.when(_is_entry_bar)
        .then(_entry_ret)
        .when(_is_exit_bar)
        .then(_exit_ret)
        .when(_pos_delayed == 1)
        .then(_normal_ret)
        .otherwise(0.0)
        .fill_null(0.0)
        .alias("return"),
    )

    # flatten_eod: zero out phantom exit returns at session boundaries.
    # The fill-at-next-open model would otherwise compute an exit return
    # on the first bar of the next session, capturing the overnight gap.
    if flatten_eod:
        signals = signals.with_columns(
            pl.when(pl.col("_session_last").shift(1).fill_null(False))
            .then(0.0)
            .otherwise(pl.col("return"))
            .alias("return"),
        )

    # Mask mode: zero out returns outside market hours
    if calendar is not None and not prefilter_market_data:
        _in_market = _build_market_mask(signals["date"], calendar)
        signals = signals.with_columns(
            (pl.col("return") * _in_market.cast(pl.Int8)).alias("return"),
        )

    returns = signals.select("date", "return")

    # Build trade log from entry/exit transitions
    trades = _extract_trades(signals, trade_on)

    return BacktestResult(returns=returns, trades=trades, signals=signals)


def _extract_trades(signals: pl.DataFrame, trade_on: str) -> pl.DataFrame:
    """Extract per-trade PnL from position transitions.

    Fill prices use the *next* bar's open (fill-at-next-open model).
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
    exits = signals_with_next.filter(pl.col("_exit_clean")).select(
        pl.col("date").alias("exit_date"),
        pl.col("_next_open").alias("exit_price"),
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
        ((pl.col("exit_price") / pl.col("entry_price")) - 1).alias("pnl"),
        (
            (pl.col("exit_date").cast(pl.Date) - pl.col("entry_date").cast(pl.Date)).dt.total_days()
        ).alias("bars_held"),
    ).select("entry_date", "exit_date", "pnl", "bars_held")

    return trades
