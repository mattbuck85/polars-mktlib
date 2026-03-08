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

    # Generate sub-step returns for OHLC derivation
    log_returns = drift_per_step + vol_per_step * rng.standard_normal(
        (n_bars, sub_steps)
    )
    # Cumulative within each bar (relative to bar open)
    cum_log = np.cumsum(log_returns, axis=1)

    # Build price series: each bar's open = previous bar's close
    bar_log_return = cum_log[:, -1]  # total log-return per bar
    cum_bar_log = np.cumsum(bar_log_return)
    # open[0] = start_price, open[i] = start_price * exp(sum of bar_log_returns up to i-1)
    open_prices = start_price * np.exp(
        np.concatenate([[0.0], cum_bar_log[:-1]])
    )

    # Derive OHLC from sub-steps
    open_arr = open_prices
    close_arr = open_prices * np.exp(cum_log[:, -1])
    high_arr = open_prices * np.exp(np.max(cum_log, axis=1))
    low_arr = open_prices * np.exp(np.min(cum_log, axis=1))

    # Ensure high >= max(open, close) and low <= min(open, close)
    high_arr = np.maximum(high_arr, np.maximum(open_arr, close_arr))
    low_arr = np.minimum(low_arr, np.minimum(open_arr, close_arr))

    # Volume: lognormal
    volume = rng.lognormal(mean=10.0, sigma=1.0, size=n_bars).astype(np.int64)

    # Datetime index: business days x minute range 09:30-16:00
    n_days = n_years * trading_days_per_year
    # Generate enough business days
    all_dates = pl.date_range(
        pl.date(2019, 1, 1), pl.date(2019 + n_years + 1, 12, 31), eager=True
    )
    # Filter to weekdays only (Mon=1..Fri=5 in polars)
    weekdays = all_dates.filter(all_dates.dt.weekday() <= 5)
    trading_dates = weekdays.head(n_days)

    # Build minute timestamps using numpy for speed
    trading_dates_np = trading_dates.to_numpy()[:n_days]
    minute_offsets_us = (
        (np.arange(minutes_per_day) * 60 + 9 * 3600 + 30 * 60)
        * 1_000_000  # microseconds
    ).astype("timedelta64[us]")

    # Broadcast: (n_days, 1) + (1, minutes_per_day) → (n_days, minutes_per_day)
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

    # 3b. Run with calendar (prefilter mode)
    from mktlib.scheduling import get_calendar

    cal = get_calendar("XNYS")
    t0 = time.perf_counter()
    result_cal = run(df, strategy, calendar=cal, prefilter_market_data=True)
    t_cal = time.perf_counter() - t0
    print(f"Backtest + calendar (prefilter): {t_cal:.3f}s  ({result_cal.signals.height:,} bars after filter)")

    # 3c. Run with calendar + flatten_eod
    t0 = time.perf_counter()
    result_flat = run(df, strategy, calendar=cal, prefilter_market_data=True, flatten_eod=True)
    t_flat = time.perf_counter() - t0
    print(f"Backtest + flatten_eod:  {t_flat:.3f}s  ({result_flat.trades.height:,} trades)")

    # 3d. Run with calendar (mask mode)
    t0 = time.perf_counter()
    result_mask = run(df, strategy, calendar=cal, prefilter_market_data=False)
    t_mask = time.perf_counter() - t0
    print(f"Backtest + calendar (mask mode): {t_mask:.3f}s")

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
        signal_idx_s = result.signals.with_row_index("_idx").filter(
            pl.col("_entry_clean")
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
