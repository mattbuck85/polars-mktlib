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
    rng = np.random.default_rng(seed)
    n_bars = n_years * trading_days_per_year * minutes_per_day

    dt = 1.0 / (trading_days_per_year * minutes_per_day)
    drift_per_step = (annual_drift - 0.5 * annual_vol**2) * dt / sub_steps
    vol_per_step = annual_vol * np.sqrt(dt / sub_steps)

    log_returns = drift_per_step + vol_per_step * rng.standard_normal(
        (n_bars, sub_steps)
    )
    cum_log = np.cumsum(log_returns, axis=1)

    bar_log_return = cum_log[:, -1]
    cum_bar_log = np.cumsum(bar_log_return)
    open_prices = start_price * np.exp(
        np.concatenate([[0.0], cum_bar_log[:-1]])
    )

    open_arr = open_prices
    close_arr = open_prices * np.exp(cum_log[:, -1])
    high_arr = open_prices * np.exp(np.max(cum_log, axis=1))
    low_arr = open_prices * np.exp(np.min(cum_log, axis=1))

    high_arr = np.maximum(high_arr, np.maximum(open_arr, close_arr))
    low_arr = np.minimum(low_arr, np.minimum(open_arr, close_arr))

    volume = rng.lognormal(mean=10.0, sigma=1.0, size=n_bars).astype(np.int64)

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
            "open": open_arr[:n_bars],
            "high": high_arr[:n_bars],
            "low": low_arr[:n_bars],
            "close": close_arr[:n_bars],
            "volume": volume[:n_bars],
        }
    )


def bench(df: pl.DataFrame, label: str, *, flatten_eod: bool) -> None:
    cal = get_calendar("XNYS") if flatten_eod else None
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

    bench(df, "Without flatten_eod:", flatten_eod=False)
    bench(df, "With flatten_eod:", flatten_eod=True)


if __name__ == "__main__":
    main()
