#!/usr/bin/env python3
"""Benchmark: mktlib.backtest on 5-year 1-minute OHLCV (~491k bars).

Generates synthetic data via GBM, adds MACD indicators, runs MacdCrossover,
and reports wall-clock timing + basic stats.
"""
from __future__ import annotations

import time

import numpy as np
import polars as pl
import polars_talib as plta

from mktlib.backtest._engine import run
from mktlib.backtest.strategies._macd import MacdCrossover
from mktlib.data import geometric_brownian_motion, ticks_to_ohlcv


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


def main() -> None:
    print("=" * 60)
    print("Benchmark: mktlib.backtest — 5-year 1-minute OHLCV")
    print("=" * 60)

    # 1. Generate data
    t0 = time.perf_counter()
    df = generate_minute_ohlcv()
    t_gen = time.perf_counter() - t0
    print(f"\nData generation:  {t_gen:.3f}s  ({df.height:,} rows x {df.width} cols)")

    # 2. Add MACD indicators
    t0 = time.perf_counter()
    df = (
        df.with_columns(plta.macd(pl.col("close")).alias("_macd"))
        .unnest("_macd")
        .rename({"macdsignal": "macd_signal"})
    )
    t_ind = time.perf_counter() - t0
    print(f"MACD indicators:  {t_ind:.3f}s")

    # 3. Run backtest
    strategy = MacdCrossover()
    t0 = time.perf_counter()
    result = run(df, strategy)
    t_bt = time.perf_counter() - t0
    print(f"Backtest engine:  {t_bt:.3f}s")

    # 3b. Run with calendar
    from mktlib.scheduling import get_calendar

    cal = get_calendar("XNYS")
    t0 = time.perf_counter()
    result_cal = run(df, strategy, calendar=cal)
    t_cal = time.perf_counter() - t0
    print(f"Backtest + calendar: {t_cal:.3f}s  ({result_cal.signals.height:,} bars after filter)")

    # 3c. Run with calendar + flatten_eod
    t0 = time.perf_counter()
    result_flat = run(df, strategy, calendar=cal, flatten_eod=True)
    t_flat = time.perf_counter() - t0
    print(f"Backtest + flatten_eod:  {t_flat:.3f}s  ({result_flat.trades.height:,} trades)")

    # 4. Summary stats
    rets = result.returns["return"]
    cum_ret = float((1 + rets).product() - 1)  # type: ignore[operator]
    equity = (1 + rets).cum_prod()
    running_max = equity.cum_max()
    dd = (equity - running_max) / running_max
    max_dd = float(dd.min())  # type: ignore[arg-type]
    n_trades = result.trades.height

    print(f"\n{'— Results —':^60}")
    print(f"Total bars:       {df.height:,}")
    print(f"Trade count:      {n_trades:,}")
    print(f"Cumulative return: {cum_ret:+.4%}")
    print(f"Max drawdown:     {max_dd:+.4%}")
    print(f"Total wall-clock: {t_gen + t_ind + t_bt:.3f}s")

    # 5. Lookahead bias spot-check
    if n_trades > 0:
        # Find first entry signal bar via _entry column
        signal_idx_s = result.signals.with_row_index("_idx").filter(
            pl.col("_entry").fill_null(False)
            & (pl.col("_position") == 1)
            & (pl.col("_position").shift(1).fill_null(0) == 0)
        )["_idx"]
        if signal_idx_s.len() > 0:
            sig_idx = signal_idx_s[0]
            signal_close = result.signals["close"][sig_idx]
            next_open = result.signals["open"][sig_idx + 1]

            # The entry bar's return should use open, not prior close
            entry_bar_ret = result.returns["return"][sig_idx + 1]
            expected_ret = (result.signals["close"][sig_idx + 1] - next_open) / next_open
            print(f"\n{'— Lookahead Bias Check —':^60}")
            print(f"First entry signal bar close: {signal_close:.6f}")
            print(f"Next bar open (fill price):   {next_open:.6f}")
            print(f"Entry bar return (actual):    {entry_bar_ret:.8f}")
            print(f"Entry bar return (expected):  {expected_ret:.8f}")
            ok = abs(entry_bar_ret - expected_ret) < 1e-12
            print(f"Match: {'YES — no lookahead bias' if ok else 'NO — LOOKAHEAD DETECTED'}")

    print()


if __name__ == "__main__":
    main()
