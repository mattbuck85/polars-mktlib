#!/usr/bin/env python3
"""Benchmark: Vectorized numpy engine vs Polars engine.

Reimplements the backtest logic using pure numpy vectorized operations
(no loops, no DataFrames for the core computation). Signal resolution
still uses Polars since the Condition DSL depends on it.
"""
from __future__ import annotations

import time

import numpy as np
import polars as pl
import polars_talib as plta

from mktlib.backtest._engine import run
from mktlib.backtest._types import BacktestResult, TradeSide
from mktlib.backtest.strategies import MacdCrossover

from scripts.bench_macd_market import generate_minute_ohlcv


# ---------------------------------------------------------------------------
# Vectorized numpy engine
# ---------------------------------------------------------------------------


def _forward_fill_position(entry: np.ndarray, exit_: np.ndarray) -> np.ndarray:
    """Compute position array: 1 on entry, 0 on exit, forward-fill.

    Equivalent to: when(entry).then(1).when(exit).then(0).otherwise(null).ffill().fill_null(0)

    Uses a cumulative-max trick: at each signal event, stamp an increasing
    sequence number. The most recent event's value (entry=1, exit=0)
    determines position at every bar.
    """
    n = len(entry)
    # Event sequence: monotonically increasing at each entry or exit
    events = entry | exit_
    event_seq = np.cumsum(events)  # 0 where no event yet

    # Value at each event: 1 for entry, 0 for exit
    # Where both are true, entry takes priority (matches Polars when/then order)
    event_val = np.where(entry, 1, 0).astype(np.int64)

    # Build a "last event value" array using the event sequence as scatter keys
    # For bars with no event, we forward-fill from the last event.
    position = np.zeros(n, dtype=np.int64)

    if event_seq[-1] == 0:
        # No events at all
        return position

    # At each event bar, record its value. Between events, ffill.
    # We can do this by: create array indexed by event_seq, scatter event_val,
    # then gather by event_seq.
    max_seq = int(event_seq[-1])
    seq_values = np.zeros(max_seq + 1, dtype=np.int64)  # seq 0 = "no event" = 0

    # Scatter: last event at each sequence number wins
    event_mask = events
    seq_values[event_seq[event_mask]] = event_val[event_mask]

    # Forward-fill the seq_values (some seq numbers may not have events due to
    # simultaneous entry+exit, but cumsum ensures every seq is hit exactly once)
    # Actually each seq number is hit exactly once by construction, but let's
    # ffill anyway for safety.
    for i in range(1, max_seq + 1):
        if seq_values[i] == 0 and i > 0:
            # This shouldn't happen with cumsum, but guard it
            pass  # already 0

    # Gather: position[i] = seq_values[event_seq[i]]
    position = seq_values[event_seq]

    return position


def run_numpy(
    df: pl.DataFrame,
    strategy: MacdCrossover,
    trade_side: TradeSide = TradeSide.LONG,
) -> BacktestResult:
    """Vectorized backtest using pure numpy array operations."""
    side = int(strategy.entry().trade_side or trade_side)

    # Signal resolution (Polars — single with_columns)
    entry_expr = strategy.entry().resolve()
    exit_expr = strategy.exit().resolve()
    signals_pl = df.with_columns(
        entry_expr.alias("_entry"),
        exit_expr.alias("_exit"),
    )

    # Extract to numpy
    entry = signals_pl["_entry"].to_numpy()
    exit_ = signals_pl["_exit"].to_numpy()
    open_ = signals_pl["open"].to_numpy()
    close = signals_pl["close"].to_numpy()
    n = len(entry)

    # Position tracking (vectorized forward-fill)
    position = _forward_fill_position(entry, exit_)

    # Shifted arrays
    pos_d1 = np.empty(n, dtype=np.int64)
    pos_d1[0] = 0
    pos_d1[1:] = position[:-1]

    pos_d2 = np.empty(n, dtype=np.int64)
    pos_d2[:2] = 0
    pos_d2[2:] = position[:-2]

    close_prev = np.empty(n, dtype=np.float64)
    close_prev[0] = np.nan
    close_prev[1:] = close[:-1]

    # Transition detection
    entry_clean = (position == 1) & (pos_d1 == 0)
    exit_clean = (position == 0) & (pos_d1 == 1)

    # Transition bars (1-bar delay for fill)
    is_entry_bar = (pos_d1 == 1) & (pos_d2 == 0)
    is_exit_bar = (pos_d1 == 0) & (pos_d2 == 1)
    in_position = pos_d1 == 1

    # Per-bar returns
    entry_ret = (close - open_) / open_ * side
    normal_ret = (close / close_prev - 1) * side
    exit_ret = (open_ - close_prev) / close_prev * side

    returns = np.where(
        is_entry_bar, entry_ret,
        np.where(
            is_exit_bar, exit_ret,
            np.where(in_position, normal_ret, 0.0),
        ),
    )
    returns = np.nan_to_num(returns, nan=0.0)

    # --- Trade extraction (vectorized) ------------------------------------
    entry_idx = np.flatnonzero(entry_clean)
    exit_idx = np.flatnonzero(exit_clean)
    n_trades = min(len(entry_idx), len(exit_idx))

    date_np = signals_pl["date"].to_numpy()

    if n_trades > 0:
        ei = entry_idx[:n_trades]
        xi = exit_idx[:n_trades]

        entry_fill = np.minimum(ei + 1, n - 1)
        exit_fill = np.minimum(xi + 1, n - 1)

        entry_prices = open_[entry_fill]
        exit_prices = open_[exit_fill]
        pnl = side * (exit_prices / entry_prices - 1)

        entry_dates = date_np[ei]
        exit_dates = date_np[xi]
        bars_held = (exit_dates.astype("datetime64[D]") - entry_dates.astype("datetime64[D]")).astype(np.int64)

        trades_df = pl.DataFrame({
            "entry_date": pl.Series("entry_date", entry_dates),
            "exit_date": pl.Series("exit_date", exit_dates),
            "pnl": pnl,
            "bars_held": bars_held,
        })
    else:
        trades_df = pl.DataFrame(
            schema={
                "entry_date": signals_pl["date"].dtype,
                "exit_date": signals_pl["date"].dtype,
                "pnl": pl.Float64,
                "bars_held": pl.Int64,
            }
        )

    # --- Build result -----------------------------------------------------
    returns_df = pl.DataFrame({
        "date": signals_pl["date"],
        "return": returns,
    })

    signals_out = signals_pl.with_columns(
        pl.Series("_position", position),
        pl.Series("_entry_clean", entry_clean),
        pl.Series("_exit_clean", exit_clean),
    )

    return BacktestResult(returns=returns_df, trades=trades_df, signals=signals_out)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate(
    label: str,
    polars_result: BacktestResult,
    other_result: BacktestResult,
) -> None:
    pol_ret = polars_result.returns["return"].to_numpy()
    oth_ret = other_result.returns["return"].to_numpy()

    returns_match = np.allclose(pol_ret, oth_ret, atol=1e-12, equal_nan=True)
    trades_match = polars_result.trades.height == other_result.trades.height

    cum_polars = float((1 + polars_result.returns["return"]).product() - 1)  # type: ignore[operator]
    cum_other = float((1 + other_result.returns["return"]).product() - 1)  # type: ignore[operator]

    print(f"\n\u2014 Validation ({label}) \u2014")
    print(f"  Returns match:      {returns_match}")
    print(f"  Trades match:       {trades_match}")
    print(f"  Cum return (polars):  {cum_polars:+.4%}")
    print(f"  Cum return ({label:6s}): {cum_other:+.4%}")

    if not returns_match:
        diffs = np.abs(pol_ret - oth_ret)
        idx = int(np.nanargmax(diffs))
        print(f"  Max diff:           {diffs[idx]:.2e} at index {idx}")
        lo, hi = max(0, idx - 2), min(len(pol_ret), idx + 3)
        print(f"  Polars [{lo}:{hi}]: {pol_ret[lo:hi]}")
        print(f"  Numpy  [{lo}:{hi}]: {oth_ret[lo:hi]}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 64)
    print("Benchmark: Vectorized Numpy vs Polars Engine")
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

    # Numpy engine
    t0 = time.perf_counter()
    numpy_result = run_numpy(df, strategy)
    t_numpy = time.perf_counter() - t0
    print(f"Numpy engine:      {t_numpy:.3f}s")

    if t_numpy > 0 and t_polars > 0:
        if t_polars < t_numpy:
            print(f"Polars is {t_numpy / t_polars:.1f}x faster")
        else:
            print(f"Numpy is {t_polars / t_numpy:.1f}x faster")

    validate("numpy", polars_result, numpy_result)


if __name__ == "__main__":
    main()
