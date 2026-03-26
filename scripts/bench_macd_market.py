#!/usr/bin/env python3
"""Benchmark: MACD crossover market-order fast path (_run_market).

Tests both with and without flatten_eod to measure the pure market-order path.
"""
from __future__ import annotations

import time

import numpy as np
import polars as pl
import polars_talib as plta

from mktlib.backtest._engine import run
from mktlib.backtest.strategies import MacdCrossover
from mktlib.data import geometric_brownian_motion, ticks_to_ohlcv
from mktlib.scheduling import get_calendar


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


def bench(
    df: pl.DataFrame,
    label: str,
    *,
    calendar: bool = False,
    flatten_eod: bool = False,
) -> None:
    cal = get_calendar("XNYS") if calendar or flatten_eod else None
    strategy = MacdCrossover()

    t0 = time.perf_counter()
    result = run(df, strategy, calendar=cal, flatten_eod=flatten_eod)
    t_bt = time.perf_counter() - t0

    rets = result.returns["return"]
    cum_ret = float((1 + rets).product() - 1)  # type: ignore[operator]
    n_trades = result.trades.height

    print(f"\n{label}")
    print(f"  Backtest engine:  {t_bt:.3f}s")
    print(f"  Total bars:       {result.signals.height:,}")
    print(f"  Trade count:      {n_trades:,}")
    print(f"  Cumulative return: {cum_ret:+.4%}")


def main() -> None:
    print("=" * 64)
    print("Benchmark: MACD Market-Order Fast Path (_run_market)")
    print("=" * 64)

    t0 = time.perf_counter()
    df = generate_minute_ohlcv()
    t_gen = time.perf_counter() - t0
    print(f"\nData generation:  {t_gen:.3f}s  ({df.height:,} rows)")

    t0 = time.perf_counter()
    df = (
        df.with_columns(plta.macd(pl.col("close")).alias("_macd"))
        .unnest("_macd")
        .rename({"macdsignal": "macd_signal"})
    )
    t_ind = time.perf_counter() - t0
    print(f"MACD indicators:  {t_ind:.3f}s")

    bench(df, "No calendar:", calendar=False)
    bench(df, "With calendar:", calendar=True)
    bench(df, "With calendar + flatten_eod:", calendar=True, flatten_eod=True)


if __name__ == "__main__":
    main()
