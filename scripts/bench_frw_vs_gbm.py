#!/usr/bin/env python3
"""Benchmark: Fractional Random Walk vs Geometric Brownian Motion.

Compares single-path and Monte Carlo generation times across signal sizes.
FRW (Davies-Harte) uses FFT/IFFT so has higher overhead than GBM (cumsum only).
"""
from __future__ import annotations

import time

import polars as pl

from mktlib.data import (
    fractional_random_walk,
    geometric_brownian_motion,
    monte_carlo,
)
from mktlib.data._monte_carlo import Process


def _bench(label: str, fn, *args, warmup: int = 2, repeats: int = 7, **kwargs):
    for _ in range(warmup):
        fn(*args, **kwargs)

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        times.append(time.perf_counter() - t0)

    times.sort()
    median = times[len(times) // 2]
    print(f"  {label:45s} {median*1000:8.2f} ms")
    return median


def _print_ratio(t_frw: float, t_gbm: float) -> None:
    if t_gbm > 0:
        ratio = t_frw / t_gbm
        print(f"  {'FRW / GBM ratio':>45s} {ratio:7.1f}x")


def main() -> None:
    print("=" * 68)
    print("Benchmark: Fractional Random Walk vs Geometric Brownian Motion")
    print("=" * 68)

    gbm_kwargs = dict(base_price=100.0, drift=0.05, volatility=0.2, dt=1 / 252)

    # ── Single path: FRW H=0.5 (standard RW, no FFT) vs GBM ──
    print("\n" + "-" * 68)
    print("Single path: FRW (H=0.5, standard RW) vs GBM")
    print("-" * 68)

    for n in (1_000, 10_000, 100_000, 1_000_000):
        print(f"\n  n = {n:,}")
        t_gbm = _bench("GBM", geometric_brownian_motion, n, seed=42, **gbm_kwargs)
        t_frw = _bench("FRW (H=0.5)", fractional_random_walk, n, hurst=0.5, seed=42)
        _print_ratio(t_frw, t_gbm)

    # ── Single path: FRW H=0.7 (trending, uses FFT) vs GBM ──
    print("\n" + "-" * 68)
    print("Single path: FRW (H=0.7, Davies-Harte FFT) vs GBM")
    print("-" * 68)

    for n in (1_000, 10_000, 100_000, 1_000_000):
        print(f"\n  n = {n:,}")
        t_gbm = _bench("GBM", geometric_brownian_motion, n, seed=42, **gbm_kwargs)
        t_frw = _bench("FRW (H=0.7, FFT)", fractional_random_walk, n, hurst=0.7, seed=42)
        _print_ratio(t_frw, t_gbm)

    # ── Single path: FRW H=0.3 (mean-reverting, uses FFT) vs GBM ──
    print("\n" + "-" * 68)
    print("Single path: FRW (H=0.3, mean-reverting FFT) vs GBM")
    print("-" * 68)

    for n in (1_000, 10_000, 100_000):
        print(f"\n  n = {n:,}")
        t_gbm = _bench("GBM", geometric_brownian_motion, n, seed=42, **gbm_kwargs)
        t_frw = _bench("FRW (H=0.3, FFT)", fractional_random_walk, n, hurst=0.3, seed=42)
        _print_ratio(t_frw, t_gbm)

    # ── Monte Carlo: FRW vs GBM ──
    print("\n" + "-" * 68)
    print("Monte Carlo: FRW vs GBM (n=1,000 per path)")
    print("-" * 68)

    n = 1_000
    for n_sims in (100, 500, 1_000):
        print(f"\n  {n_sims:,} simulations x {n:,} steps")
        t_gbm = _bench(
            "GBM (vectorized MC)",
            monte_carlo, Process.GBM, n_sims, seed=42, n=n, **gbm_kwargs,
        )
        t_frw = _bench(
            "FRW H=0.7 (MC, loop per sim)",
            monte_carlo, Process.FRW, n_sims, seed=42, n=n, hurst=0.7,
        )
        _print_ratio(t_frw, t_gbm)


if __name__ == "__main__":
    main()
