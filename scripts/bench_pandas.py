#!/usr/bin/env python3
"""Benchmark: Pandas vectorized engine vs Polars engine.

Reimplements the same backtest logic (no calendar, no flatten_eod) using
pandas vectorized operations, then compares timing and correctness against
the Polars engine.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import polars as pl
import polars_talib as plta

from mktlib.backtest._engine import run
from mktlib.backtest._types import BacktestResult, TradeSide
from mktlib.backtest.strategies import MacdCrossover

# ---------------------------------------------------------------------------
# Data generation — reuse from bench_macd_market
# ---------------------------------------------------------------------------
from scripts.bench_macd_market import generate_minute_ohlcv


# ---------------------------------------------------------------------------
# Pandas engine
# ---------------------------------------------------------------------------


def run_pandas(
    df: pl.DataFrame,
    strategy: MacdCrossover,
    trade_side: TradeSide = TradeSide.LONG,
) -> BacktestResult:
    """Vectorized backtest using pandas, matching Polars engine semantics."""
    side = int(strategy.entry().trade_side or trade_side)

    # Signal resolution — use Polars for the crossover expressions, then
    # convert to pandas for the rest (mirrors what the Polars engine does
    # for signal resolution).
    entry_expr = strategy.entry().resolve()
    exit_expr = strategy.exit().resolve()
    signals_pl = df.with_columns(
        entry_expr.alias("_entry"),
        exit_expr.alias("_exit"),
    )
    pdf = signals_pl.to_pandas()

    # Position tracking: 1 on entry, 0 on exit, forward-fill
    pos_raw = pd.array([pd.NA] * len(pdf), dtype=pd.Int64Dtype())
    pos_raw[pdf["_entry"].values] = 1
    pos_raw[pdf["_exit"].values] = 0
    pdf["_position"] = pd.Series(pos_raw, dtype=pd.Int64Dtype()).ffill().fillna(0).astype(np.int64)

    # Shifted columns
    pdf["_pos_d1"] = pdf["_position"].shift(1, fill_value=0)
    pdf["_pos_d2"] = pdf["_position"].shift(2, fill_value=0)
    pdf["_close_prev"] = pdf["close"].shift(1)

    # Transition detection
    pdf["_entry_clean"] = (pdf["_position"] == 1) & (pdf["_pos_d1"] == 0)
    pdf["_exit_clean"] = (pdf["_position"] == 0) & (pdf["_pos_d1"] == 1)

    # Transition bars (1-bar delay)
    is_entry_bar = (pdf["_pos_d1"] == 1) & (pdf["_pos_d2"] == 0)
    is_exit_bar = (pdf["_pos_d1"] == 0) & (pdf["_pos_d2"] == 1)
    in_position = pdf["_pos_d1"] == 1

    # Returns
    entry_ret = (pdf["close"] - pdf["open"]) / pdf["open"] * side
    normal_ret = (pdf["close"] / pdf["_close_prev"] - 1) * side
    exit_ret = (pdf["open"] - pdf["_close_prev"]) / pdf["_close_prev"] * side

    ret = pd.Series(np.zeros(len(pdf)), index=pdf.index)
    # Order matters: most specific first, np.where-chain equivalent
    ret = ret.where(~in_position, normal_ret)
    ret = ret.where(~is_exit_bar, exit_ret)
    ret = ret.where(~is_entry_bar, entry_ret)
    # Zero out bars not in position and not transition bars
    ret = ret.where(in_position | is_entry_bar | is_exit_bar, 0.0)
    pdf["return"] = ret.fillna(0.0)

    # --- Trade extraction -------------------------------------------------
    entry_mask = pdf["_entry_clean"].values
    exit_mask = pdf["_exit_clean"].values

    entry_idx = np.flatnonzero(entry_mask)
    exit_idx = np.flatnonzero(exit_mask)
    n_trades = min(len(entry_idx), len(exit_idx))

    open_arr = pdf["open"].values
    date_arr = pdf["date"].values

    if n_trades > 0:
        ei = entry_idx[:n_trades]
        xi = exit_idx[:n_trades]

        # Fill at next open
        entry_fill = np.minimum(ei + 1, len(pdf) - 1)
        exit_fill = np.minimum(xi + 1, len(pdf) - 1)

        entry_prices = open_arr[entry_fill]
        exit_prices = open_arr[exit_fill]
        pnl = side * (exit_prices / entry_prices - 1)

        entry_dates = date_arr[ei]
        exit_dates = date_arr[xi]
        ed = entry_dates.astype("datetime64[D]")
        xd = exit_dates.astype("datetime64[D]")
        bars_held = (xd - ed).astype(np.int64)

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

    # --- Convert back to Polars for BacktestResult ------------------------
    returns_df = pl.DataFrame({
        "date": signals_pl["date"],
        "return": pdf["return"].values,
    })

    # Drop internal shifted columns, keep _position, _entry_clean, _exit_clean
    pdf.drop(columns=["_pos_d1", "_pos_d2", "_close_prev"], inplace=True)
    signals_out = pl.from_pandas(pdf)

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
        print(f"  Other  [{lo}:{hi}]: {oth_ret[lo:hi]}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 64)
    print("Benchmark: Pandas vs Polars Engine")
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

    # Pandas engine
    t0 = time.perf_counter()
    pandas_result = run_pandas(df, strategy)
    t_pandas = time.perf_counter() - t0
    print(f"Pandas engine:     {t_pandas:.3f}s")

    ratio = t_pandas / t_polars if t_polars > 0 else float("inf")
    print(f"Polars/Pandas:     {ratio:.1f}x {'faster' if ratio > 1 else 'slower'} (Polars)")

    validate("pandas", polars_result, pandas_result)


if __name__ == "__main__":
    main()
