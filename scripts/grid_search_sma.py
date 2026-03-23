#!/usr/bin/env python3
"""Grid search optimization: SMA crossover with take-profit / stop-loss.

Two-stage optimization:
1. Search over fast/slow SMA periods, scoring by Sharpe ratio.
2. Fix best SMA params, then search over TP/SL percentages.

Requires: pip install mktlib[data] polars-talib
"""
from __future__ import annotations

import itertools

import numpy as np
import polars as pl
import polars_talib as plta

from mktlib.backtest import (
    Condition,
    Crossover,
    Crossunder,
    Pct,
    ValueGT,
    ValueLT,
    run,
)
from mktlib.data import geometric_brownian_motion, ticks_to_ohlcv
from mktlib.metrics import cumulative_return, sharpe
from mktlib.rates import get_risk_free_rate

_MINUTES_PER_YEAR = 252 * 390  # trading minutes in a year

# ---------------------------------------------------------------------------
# 1. Generate synthetic 1-minute OHLCV data
# ---------------------------------------------------------------------------


def generate_minute_ohlcv(
    n_years: int = 5,
    trading_days_per_year: int = 252,
    minutes_per_day: int = 390,
    start_price: float = 100.0,
    annual_drift: float = 0.08,
    annual_vol: float = 1.0,  # intentionally high — see docs/advanced.rst
    seed: int = 42,
) -> pl.DataFrame:
    """Generate minute-resolution OHLCV from ``mktlib.data.geometric_brownian_motion``.

    Uses GBM at second-level resolution (1 tick = 1 second of trading time),
    then aggregates into 1-minute OHLC bars via ``ticks_to_ohlcv``.
    """
    ticks_per_bar = 60  # 60 seconds per minute
    n_bars = n_years * trading_days_per_year * minutes_per_day
    dt_1s = 1 / (trading_days_per_year * 6.5 * 3600)

    # Generate a single GBM path at second-level resolution
    gbm = geometric_brownian_motion(
        n=n_bars * ticks_per_bar,
        base_price=start_price,
        drift=annual_drift,
        volatility=annual_vol,
        dt=dt_1s,
        seed=seed,
    )
    ohlcv = ticks_to_ohlcv(gbm, bar_size=ticks_per_bar, seed=seed + 1)

    # Build business-day minute timestamps (09:30-16:00 ET)
    n_days = n_years * trading_days_per_year
    all_dates = pl.date_range(
        pl.date(2019, 1, 1), pl.date(2019 + n_years + 1, 12, 31), eager=True
    )
    weekdays = all_dates.filter(all_dates.dt.weekday() <= 5)
    trading_dates = weekdays.head(n_days)

    trading_dates_np = trading_dates.to_numpy()[:n_days]
    minute_offsets_us = (
        (np.arange(minutes_per_day) * 60 + 9 * 3600 + 30 * 60) * 1_000_000
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
# 2. Parameterized SMA crossover strategy
# ---------------------------------------------------------------------------

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SmaCross:
    fast_period: int = 20
    slow_period: int = 50

    def init(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(
            plta.sma(pl.col("close"), timeperiod=self.fast_period).alias(
                "fast_sma"
            ),
            plta.sma(pl.col("close"), timeperiod=self.slow_period).alias(
                "slow_sma"
            ),
        )

    def entry(self) -> Crossover:
        return Crossover("fast_sma", "slow_sma")

    def exit(self) -> Crossunder:
        return Crossunder("fast_sma", "slow_sma")


@dataclass(frozen=True, slots=True)
class SmaCrossWithExits:
    """SMA crossover with percentage TP/SL relative to slow SMA."""

    fast_period: int = 15
    slow_period: int = 20
    tp_pct: float = 1.0
    sl_pct: float = 0.5

    def init(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(
            plta.sma(pl.col("close"), timeperiod=self.fast_period).alias(
                "fast_sma"
            ),
            plta.sma(pl.col("close"), timeperiod=self.slow_period).alias(
                "slow_sma"
            ),
        )

    def entry(self) -> Crossover:
        return Crossover("fast_sma", "slow_sma")

    def exit(self) -> Condition:
        tp = ValueGT("close", Pct("slow_sma", self.tp_pct))
        sl = ValueLT("close", Pct("slow_sma", -self.sl_pct))
        return Crossunder("fast_sma", "slow_sma") | tp | sl


# ---------------------------------------------------------------------------
# 3. Grid search: SMA periods
# ---------------------------------------------------------------------------


def search_sma_periods(df: pl.DataFrame, rf: float) -> pl.DataFrame:
    """Grid search over fast/slow SMA periods, scored by Sharpe ratio."""
    fast_range = range(5, 100, 5)
    slow_range = range(20, 250, 10)

    results: list[dict[str, object]] = []
    for fast, slow in itertools.product(fast_range, slow_range):
        if fast >= slow:
            continue

        strategy = SmaCross(fast_period=fast, slow_period=slow)
        result = run(df, strategy)
        ret = result.returns["return"]

        results.append(
            {
                "fast_period": fast,
                "slow_period": slow,
                "sharpe": round(sharpe(ret, ppy=_MINUTES_PER_YEAR, rf=rf), 4),
                "cumulative_return": round(cumulative_return(ret), 4),
                "n_trades": len(result.trades),
            }
        )

    return pl.DataFrame(results).sort("sharpe", descending=True)


# ---------------------------------------------------------------------------
# 4. Grid search: TP/SL percentages (best SMA params fixed)
# ---------------------------------------------------------------------------


def search_tp_sl(
    df: pl.DataFrame, best_fast: int, best_slow: int, rf: float
) -> pl.DataFrame:
    """Grid search over TP/SL pcts with fixed SMA periods."""
    tp_range = [i / 10 for i in range(0, 11)]  # 0.1% to 1.0% step 0.1%
    sl_range = [i / 10 for i in range(0, 11)]  # 0.1% to 1.0% step 0.1%

    results: list[dict[str, object]] = []
    for tp_pct, sl_pct in itertools.product(tp_range, sl_range):
        strategy = SmaCrossWithExits(
            fast_period=best_fast,
            slow_period=best_slow,
            tp_pct=tp_pct,
            sl_pct=sl_pct,
        )
        result = run(df, strategy)
        ret = result.returns["return"]

        results.append(
            {
                "tp_pct": tp_pct,
                "sl_pct": sl_pct,
                "sharpe": round(sharpe(ret, ppy=_MINUTES_PER_YEAR, rf=rf), 4),
                "cumulative_return": round(cumulative_return(ret), 4),
                "n_trades": len(result.trades),
            }
        )

    return pl.DataFrame(results).sort("sharpe", descending=True)


# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("Generating 5 years of 1-minute OHLCV data...")
    df = generate_minute_ohlcv()
    print(f"  {df.shape[0]:,} bars\n")

    # Fetch the average 3-month T-bill yield for the period
    rf = get_risk_free_rate(
        df["date"].dt.date().min(), df["date"].dt.date().max()
    )
    print(f"Risk-free rate for period: {rf:.4f}\n")

    # Stage 1: SMA period search
    print("Stage 1: Grid search over SMA periods...")
    sma_results = search_sma_periods(df, rf)
    print("\nTop 5 SMA parameter combos by Sharpe:")
    print(sma_results.head(5))

    best = sma_results.row(0, named=True)
    best_fast = int(best["fast_period"])
    best_slow = int(best["slow_period"])
    print(
        f"\nBest: fast={best_fast}, slow={best_slow}, Sharpe={best['sharpe']}\n"
    )

    # Stage 2: TP/SL search with best SMA params
    print("Stage 2: Grid search over TP/SL percentages...")
    tp_sl_results = search_tp_sl(df, best_fast, best_slow, rf)
    print("\nTop 5 TP/SL combos by Sharpe:")
    print(tp_sl_results.head(5))

    best_exit = tp_sl_results.row(0, named=True)
    print(
        f"\nBest: TP={best_exit['tp_pct']}%, SL={best_exit['sl_pct']}%, "
        f"Sharpe={best_exit['sharpe']}"
    )


if __name__ == "__main__":
    main()
