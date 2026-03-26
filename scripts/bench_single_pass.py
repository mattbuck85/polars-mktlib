#!/usr/bin/env python3
"""Benchmark: Single-pass numpy loop vs Polars expression engine.

Reimplements the backtest logic from _engine.py as a single Python for-loop
over numpy arrays to measure whether avoiding intermediate DataFrame
allocations helps.
"""
from __future__ import annotations

import math
import time

import numba as nb
import numpy as np
import polars as pl
import polars_talib as plta

from mktlib.backtest._engine import run
from mktlib.backtest._types import BacktestResult, TradeSide
from mktlib.backtest.strategies import MacdCrossover
from mktlib.data import geometric_brownian_motion, ticks_to_ohlcv

# ---------------------------------------------------------------------------
# Data generation — uses mktlib.data generators
# ---------------------------------------------------------------------------


def generate_minute_ohlcv(
    n_years: int = 5,
    trading_days_per_year: int = 252,
    minutes_per_day: int = 390,
    start_price: float = 100.0,
    annual_drift: float = 0.08,
    annual_vol: float = 0.20,
    sub_steps: int = 10,
    seed: int = 42,
) -> pl.DataFrame:
    """Generate minute-resolution OHLCV using geometric Brownian motion."""
    n_bars = n_years * trading_days_per_year * minutes_per_day
    dt_per_sub = 1.0 / (trading_days_per_year * minutes_per_day * sub_steps)

    gbm = geometric_brownian_motion(
        n=n_bars * sub_steps + 1,
        base_price=start_price,
        drift=annual_drift,
        volatility=annual_vol,
        dt=dt_per_sub,
        seed=seed,
    )
    ohlcv = ticks_to_ohlcv(gbm, bar_size=sub_steps, seed=seed + 1)

    # Build business-day minute timestamps (09:30-16:00 ET)
    n_days = n_years * trading_days_per_year
    all_dates = pl.date_range(
        pl.date(2019, 1, 1), pl.date(2019 + n_years + 1, 12, 31), eager=True
    )
    weekdays = all_dates.filter(all_dates.dt.weekday() <= 5)
    trading_dates = weekdays.head(n_days)

    trading_dates_np = trading_dates.to_numpy()[:n_days]
    minute_offsets_us = (
        (np.arange(minutes_per_day) * 60 + 9 * 3600 + 30 * 60)
        * 1_000_000
    ).astype("timedelta64[us]")

    dates_as_dt = trading_dates_np.astype("datetime64[us]")
    datetimes_np = (
        dates_as_dt.reshape(-1, 1) + minute_offsets_us.reshape(1, -1)
    ).ravel()[:n_bars]

    return pl.DataFrame(
        {
            "date": pl.Series("date", datetimes_np),
            "open": ohlcv["open"],
            "high": ohlcv["high"],
            "low": ohlcv["low"],
            "close": ohlcv["close"],
            "volume": ohlcv["volume"],
        }
    )


# ---------------------------------------------------------------------------
# Single-pass numpy implementation
# ---------------------------------------------------------------------------


def run_single_pass(
    df: pl.DataFrame,
    strategy: MacdCrossover,
    trade_side: TradeSide = TradeSide.LONG,
) -> BacktestResult:
    """Reimplement the backtest engine as a single Python for-loop over numpy."""
    side = int(trade_side)

    # --- Signal resolution (still Polars — 1 with_columns) ----------------
    entry_expr = strategy.entry().resolve()
    exit_expr = strategy.exit().resolve()
    signals_df = df.with_columns(
        entry_expr.alias("_entry"),
        exit_expr.alias("_exit"),
    )

    # --- Extract to numpy arrays ------------------------------------------
    entry = signals_df["_entry"].to_numpy()
    exit_ = signals_df["_exit"].to_numpy()
    open_ = signals_df["open"].to_numpy()
    close = signals_df["close"].to_numpy()
    n = len(entry)

    # --- Single for-loop --------------------------------------------------
    position = np.empty(n, dtype=np.int64)
    returns = np.empty(n, dtype=np.float64)
    entry_indices: list[int] = []
    exit_indices: list[int] = []

    pos = 0
    pos_d1 = 0
    pos_d2 = 0
    close_prev = math.nan

    for i in range(n):
        if entry[i]:
            pos = 1
        elif exit_[i]:
            pos = 0
        position[i] = pos

        entry_clean = (pos == 1) and (pos_d1 == 0)
        exit_clean = (pos == 0) and (pos_d1 == 1)
        if entry_clean:
            entry_indices.append(i)
        if exit_clean:
            exit_indices.append(i)

        is_entry_bar = (pos_d1 == 1) and (pos_d2 == 0)
        is_exit_bar = (pos_d1 == 0) and (pos_d2 == 1)

        if is_entry_bar:
            ret = (close[i] - open_[i]) / open_[i] * side
        elif is_exit_bar:
            ret = (open_[i] - close_prev) / close_prev * side
        elif pos_d1 == 1:
            ret = (close[i] / close_prev - 1) * side
        else:
            ret = 0.0
        returns[i] = ret

        pos_d2 = pos_d1
        pos_d1 = pos
        close_prev = close[i]

    # --- Trade extraction (vectorized post-loop) --------------------------
    n_trades = min(len(entry_indices), len(exit_indices))
    entry_idx_arr = np.array(entry_indices[:n_trades], dtype=np.int64)
    exit_idx_arr = np.array(exit_indices[:n_trades], dtype=np.int64)

    date_np = signals_df["date"].to_numpy()

    if n_trades > 0:
        # Fill at next open: idx + 1
        entry_fill_idx = np.minimum(entry_idx_arr + 1, n - 1)
        exit_fill_idx = np.minimum(exit_idx_arr + 1, n - 1)

        entry_prices = open_[entry_fill_idx]
        exit_prices = open_[exit_fill_idx]
        pnl = side * (exit_prices / entry_prices - 1)

        entry_dates = date_np[entry_idx_arr]
        exit_dates = date_np[exit_idx_arr]

        # bars_held as date diffs (cast to date)
        entry_dates_date = entry_dates.astype("datetime64[D]")
        exit_dates_date = exit_dates.astype("datetime64[D]")
        bars_held = (exit_dates_date - entry_dates_date).astype(np.int64)

        trades_df = pl.DataFrame(
            {
                "entry_date": pl.Series("entry_date", entry_dates),
                "exit_date": pl.Series("exit_date", exit_dates),
                "pnl": pnl,
                "bars_held": bars_held,
            }
        )
    else:
        trades_df = pl.DataFrame(
            schema={
                "entry_date": signals_df["date"].dtype,
                "exit_date": signals_df["date"].dtype,
                "pnl": pl.Float64,
                "bars_held": pl.Int64,
            }
        )

    # --- Construct BacktestResult -----------------------------------------
    returns_df = pl.DataFrame(
        {
            "date": signals_df["date"],
            "return": returns,
        }
    )

    # Build entry_clean / exit_clean boolean arrays for signals
    entry_clean_arr = np.zeros(n, dtype=bool)
    exit_clean_arr = np.zeros(n, dtype=bool)
    for idx in entry_indices:
        entry_clean_arr[idx] = True
    for idx in exit_indices:
        exit_clean_arr[idx] = True

    signals_out = signals_df.with_columns(
        pl.Series("_position", position),
        pl.Series("_entry_clean", entry_clean_arr),
        pl.Series("_exit_clean", exit_clean_arr),
    )

    return BacktestResult(returns=returns_df, trades=trades_df, signals=signals_out)


# ---------------------------------------------------------------------------
# Numba-jitted single-pass kernel
# ---------------------------------------------------------------------------

# Max trades we'll ever collect — sized generously so we never overflow.
_MAX_TRADES = 100_000


@nb.njit(cache=True)
def _numba_loop(
    entry: np.ndarray,  # bool
    exit_: np.ndarray,  # bool
    open_: np.ndarray,  # float64
    close: np.ndarray,  # float64
    side: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Core single-pass loop compiled with numba.

    Returns
    -------
    position, returns, entry_indices, exit_indices, n_entries, n_exits
    """
    n = len(entry)
    position = np.empty(n, dtype=np.int64)
    returns = np.empty(n, dtype=np.float64)
    entry_indices = np.empty(_MAX_TRADES, dtype=np.int64)
    exit_indices = np.empty(_MAX_TRADES, dtype=np.int64)
    n_entries = 0
    n_exits = 0

    pos = 0
    pos_d1 = 0
    pos_d2 = 0
    close_prev = np.nan

    for i in range(n):
        if entry[i]:
            pos = 1
        elif exit_[i]:
            pos = 0
        position[i] = pos

        entry_clean = (pos == 1) and (pos_d1 == 0)
        exit_clean = (pos == 0) and (pos_d1 == 1)
        if entry_clean and n_entries < _MAX_TRADES:
            entry_indices[n_entries] = i
            n_entries += 1
        if exit_clean and n_exits < _MAX_TRADES:
            exit_indices[n_exits] = i
            n_exits += 1

        is_entry_bar = (pos_d1 == 1) and (pos_d2 == 0)
        is_exit_bar = (pos_d1 == 0) and (pos_d2 == 1)

        if is_entry_bar:
            ret = (close[i] - open_[i]) / open_[i] * side
        elif is_exit_bar:
            ret = (open_[i] - close_prev) / close_prev * side
        elif pos_d1 == 1:
            ret = (close[i] / close_prev - 1) * side
        else:
            ret = 0.0
        returns[i] = ret

        pos_d2 = pos_d1
        pos_d1 = pos
        close_prev = close[i]

    return position, returns, entry_indices, exit_indices, n_entries, n_exits


def run_numba(
    df: pl.DataFrame,
    strategy: MacdCrossover,
    trade_side: TradeSide = TradeSide.LONG,
) -> BacktestResult:
    """Single-pass backtest using numba-jitted loop."""
    side = int(trade_side)

    # Signal resolution (Polars)
    entry_expr = strategy.entry().resolve()
    exit_expr = strategy.exit().resolve()
    signals_df = df.with_columns(
        entry_expr.alias("_entry"),
        exit_expr.alias("_exit"),
    )

    entry = signals_df["_entry"].to_numpy()
    exit_ = signals_df["_exit"].to_numpy()
    open_ = signals_df["open"].to_numpy()
    close = signals_df["close"].to_numpy()
    n = len(entry)

    # Numba kernel
    position, returns, entry_idx_buf, exit_idx_buf, n_entries, n_exits = _numba_loop(
        entry, exit_, open_, close, side,
    )

    # Trade extraction
    n_trades = min(n_entries, n_exits)
    entry_idx_arr = entry_idx_buf[:n_trades]
    exit_idx_arr = exit_idx_buf[:n_trades]

    date_np = signals_df["date"].to_numpy()

    if n_trades > 0:
        entry_fill_idx = np.minimum(entry_idx_arr + 1, n - 1)
        exit_fill_idx = np.minimum(exit_idx_arr + 1, n - 1)

        entry_prices = open_[entry_fill_idx]
        exit_prices = open_[exit_fill_idx]
        pnl = side * (exit_prices / entry_prices - 1)

        entry_dates = date_np[entry_idx_arr]
        exit_dates = date_np[exit_idx_arr]
        entry_dates_date = entry_dates.astype("datetime64[D]")
        exit_dates_date = exit_dates.astype("datetime64[D]")
        bars_held = (exit_dates_date - entry_dates_date).astype(np.int64)

        trades_df = pl.DataFrame(
            {
                "entry_date": pl.Series("entry_date", entry_dates),
                "exit_date": pl.Series("exit_date", exit_dates),
                "pnl": pnl,
                "bars_held": bars_held,
            }
        )
    else:
        trades_df = pl.DataFrame(
            schema={
                "entry_date": signals_df["date"].dtype,
                "exit_date": signals_df["date"].dtype,
                "pnl": pl.Float64,
                "bars_held": pl.Int64,
            }
        )

    returns_df = pl.DataFrame(
        {"date": signals_df["date"], "return": returns}
    )

    entry_clean_arr = np.zeros(n, dtype=bool)
    exit_clean_arr = np.zeros(n, dtype=bool)
    entry_clean_arr[entry_idx_buf[:n_entries]] = True
    exit_clean_arr[exit_idx_buf[:n_exits]] = True

    signals_out = signals_df.with_columns(
        pl.Series("_position", position),
        pl.Series("_entry_clean", entry_clean_arr),
        pl.Series("_exit_clean", exit_clean_arr),
    )

    return BacktestResult(returns=returns_df, trades=trades_df, signals=signals_out)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate(polars_result: BacktestResult, numpy_result: BacktestResult) -> None:
    """Compare returns and trades between polars and numpy results."""
    pol_ret = polars_result.returns["return"].to_numpy()
    np_ret = numpy_result.returns["return"].to_numpy()

    returns_match = np.allclose(pol_ret, np_ret, atol=1e-12, equal_nan=True)
    trades_match = polars_result.trades.height == numpy_result.trades.height

    cum_ret_polars = float((1 + polars_result.returns["return"]).product() - 1)  # type: ignore[operator]
    cum_ret_numpy = float((1 + numpy_result.returns["return"]).product() - 1)  # type: ignore[operator]

    print("\n\u2014 Validation \u2014")
    print(f"  Returns match:      {returns_match}")
    print(f"  Trades match:       {trades_match}")
    print(f"  Cum return (polars):  {cum_ret_polars:+.4%}")
    print(f"  Cum return (numpy):   {cum_ret_numpy:+.4%}")

    if not returns_match:
        diffs = np.abs(pol_ret - np_ret)
        max_diff_idx = int(np.nanargmax(diffs))
        print(f"  Max diff:           {diffs[max_diff_idx]:.2e} at index {max_diff_idx}")
        # Show a few surrounding bars for debugging
        lo = max(0, max_diff_idx - 2)
        hi = min(len(pol_ret), max_diff_idx + 3)
        print(f"  Polars [{lo}:{hi}]: {pol_ret[lo:hi]}")
        print(f"  Numpy  [{lo}:{hi}]: {np_ret[lo:hi]}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 64)
    print("Benchmark: Single-Pass Numpy Loop vs Polars Engine")
    print("=" * 64)

    t0 = time.perf_counter()
    df = generate_minute_ohlcv()
    t_gen = time.perf_counter() - t0
    print(f"\nData generation:   {t_gen:.3f}s  ({df.height:,} rows)")

    t0 = time.perf_counter()
    df = (
        df.with_columns(plta.macd(pl.col("close")).alias("_macd"))
        .unnest("_macd")
        .rename({"macdsignal": "macd_signal"})
    )
    t_ind = time.perf_counter() - t0
    print(f"MACD indicators:   {t_ind:.3f}s")

    strategy = MacdCrossover()

    # Polars engine
    t0 = time.perf_counter()
    polars_result = run(df, strategy)
    t_polars = time.perf_counter() - t0
    print(f"\nPolars engine:     {t_polars:.3f}s")

    # Single-pass numpy (pure Python)
    t0 = time.perf_counter()
    numpy_result = run_single_pass(df, strategy)
    t_numpy = time.perf_counter() - t0
    print(f"Single-pass numpy: {t_numpy:.3f}s")

    speedup_np = t_polars / t_numpy if t_numpy > 0 else float("inf")
    print(f"Speedup vs polars: {speedup_np:.1f}x")

    # Numba JIT — warmup (includes compilation)
    print("\nNumba JIT warmup...", end=" ", flush=True)
    t0 = time.perf_counter()
    _warmup = run_numba(df, strategy)
    t_warmup = time.perf_counter() - t0
    print(f"{t_warmup:.3f}s (includes compilation)")

    # Numba JIT — timed
    t0 = time.perf_counter()
    numba_result = run_numba(df, strategy)
    t_numba = time.perf_counter() - t0
    print(f"Numba JIT:         {t_numba:.3f}s")

    speedup_nb = t_polars / t_numba if t_numba > 0 else float("inf")
    print(f"Speedup vs polars: {speedup_nb:.1f}x")

    validate(polars_result, numpy_result)
    validate(polars_result, numba_result)


if __name__ == "__main__":
    main()
