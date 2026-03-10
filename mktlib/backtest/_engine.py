from __future__ import annotations

import datetime
from typing import TYPE_CHECKING

import polars as pl

from mktlib.backtest._types import BacktestResult, Strategy, TradeSide

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


def _build_market_mask(
    dates: pl.Series,
    calendar: ExchangeCalendar,
) -> pl.Series:
    """Boolean series: True where date falls within market hours.

    Uses a schedule join (~1,260 rows for 5 years) instead of materializing
    all 472K+ trading minutes via ``trading_index``.
    """
    sched = _get_schedule(calendar, dates)

    # Compute last tradeable minute (open-frame: close - 1min)
    sched = sched.with_columns(
        (pl.col("market_close") - pl.duration(minutes=1)).alias("_last_minute"),
    )

    # Build join key: bar date (Date) to match schedule's date column
    dates_df = dates.to_frame("date").with_columns(
        pl.col("date").dt.date().alias("_bar_date"),
    )

    # Prepare schedule columns with tz aligned to bar timestamps
    sched_join = sched.select(
        pl.col("date").alias("_bar_date"),
        _align_tz(sched["market_open"], dates).alias("_mkt_open"),
        _align_tz(sched["_last_minute"], dates).alias("_last_min"),
    )
    if "break_start" in sched.columns:
        sched_join = sched_join.with_columns(
            _align_tz(sched["break_start"], dates).alias("_brk_start"),
            _align_tz(sched["break_end"], dates).alias("_brk_end"),
        )

    joined = dates_df.join(sched_join, on="_bar_date", how="left")

    # Bar is valid if within [market_open, last_minute]
    mask = (
        joined["_mkt_open"].is_not_null()
        & (joined["date"] >= joined["_mkt_open"])
        & (joined["date"] <= joined["_last_min"])
    )

    # Break calendars: exclude [break_start, break_end)
    if "break_start" in sched.columns:
        in_break = (joined["date"] >= joined["_brk_start"]) & (
            joined["date"] < joined["_brk_end"]
        )
        mask = mask & ~in_break

    return mask


def _build_session_last_mask(
    dates: pl.Series,
    calendar: ExchangeCalendar,
) -> pl.Series:
    """Boolean series: True on the last bar of each trading session.

    Uses open-frame convention: last bar timestamp = market_close - 1min.
    E.g. NYSE 16:00 close -> last bar at 15:59:00.
    """
    sched = _get_schedule(calendar, dates)
    close_times = _align_tz(sched["market_close"], dates)
    # Last bar = market_close - 1min (open-frame: bar at 15:59 covers 15:59-16:00)
    last_minutes = close_times.to_frame("c").select(
        pl.col("c") - pl.duration(minutes=1)
    )["c"]
    return dates.is_in(last_minutes.to_list())


def run(
    df: pl.DataFrame,
    strategy: Strategy,
    *,
    trade_side: TradeSide = TradeSide.LONG,
    calendar: ExchangeCalendar | None = None,
    flatten_eod: bool = False,
) -> BacktestResult:
    """Run a vectorized backtest with fill-at-next-open semantics.

    Parameters
    ----------
    df
        Must contain ``date``, ``open``, ``close``, and any indicator
        columns referenced by the strategy.
    strategy
        Object with ``entry()`` and ``exit()`` returning Conditions.
    trade_side
        Trade direction. Overridden by the entry condition's ``trade_side``
        if set.
    calendar
        Exchange calendar for market-hours filtering. When provided, the
        DataFrame is filtered to market hours before signal computation.
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

    # Filter to market hours when calendar is provided
    if calendar is not None:
        mask = _build_market_mask(df["date"], calendar)
        df = df.filter(mask)

    entry_cond = strategy.entry()
    exit_cond = strategy.exit()
    entry_expr = entry_cond.resolve()
    exit_expr = exit_cond.resolve()

    # Entry condition's side overrides the run() default
    effective_side = int(entry_cond.trade_side or trade_side)

    signals = df.with_columns(
        entry_expr.alias("_entry"),
        exit_expr.alias("_exit"),
    )

    # Position tracking: 1 on entry, 0 on exit, forward-fill
    if flatten_eod:
        _session_last = _build_session_last_mask(signals["date"], calendar)  # type: ignore[arg-type]
        signals = signals.with_columns(_session_last.alias("_session_last"))
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
