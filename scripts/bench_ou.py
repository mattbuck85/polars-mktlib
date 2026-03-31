#!/usr/bin/env python3
"""Benchmark: Vectorized OU & GBM vs iterative loops, single path and Monte Carlo."""
from __future__ import annotations

import math
import random
import time

import polars as pl
from polars_sdist import sample_normal

from mktlib.data import geometric_brownian_motion, monte_carlo, ornstein_uhlenbeck
from mktlib.data._monte_carlo import Process


# ---------------------------------------------------------------------------
# Iterative baselines
# ---------------------------------------------------------------------------


def _iterative_ou(
    n: int,
    theta: float = 0.7,
    mu: float = 100.0,
    sigma: float = 1.0,
    x0: float | None = None,
    dt: float = 1.0,
    seed: int | None = None,
) -> pl.DataFrame:
    """Euler-Maruyama loop (the old implementation)."""
    z = sample_normal(n, seed=seed).to_list()
    start = mu if x0 is None else x0
    noise_scale = sigma * math.sqrt(dt)

    x = [start]
    for i in range(1, n):
        x_prev = x[-1]
        x.append(x_prev + theta * (mu - x_prev) * dt + noise_scale * z[i])

    return pl.DataFrame({"step": range(n), "value": x})


def _iterative_gbm(
    n: int,
    base_price: float = 100.0,
    drift: float = 0.0,
    volatility: float = 0.01,
    dt: float = 1.0,
    seed: int | None = None,
) -> pl.DataFrame:
    """GBM via iterative log-return accumulation."""
    z = sample_normal(n, seed=seed).to_list()
    mu_dt = (drift - 0.5 * volatility**2) * dt
    sigma_sqrt_dt = volatility * math.sqrt(dt)
    log_base = math.log(base_price)

    log_prices = [log_base]
    for i in range(1, n):
        log_prices.append(log_prices[-1] + mu_dt + sigma_sqrt_dt * z[i])

    prices = [math.exp(lp) for lp in log_prices]
    return pl.DataFrame({"step": range(n), "price": prices})


def _iterative_monte_carlo_gbm(
    n_simulations: int,
    seed: int | None = None,
    **kwargs: float | int | None,
) -> pl.DataFrame:
    """Loop-based Monte Carlo: call _iterative_gbm per simulation."""
    rng = random.Random(seed)
    child_seeds = [rng.randrange(2**63) for _ in range(n_simulations)]

    frames: list[pl.DataFrame] = []
    for i, s in enumerate(child_seeds):
        df = _iterative_gbm(**kwargs, seed=s)  # type: ignore[arg-type]
        frames.append(df.with_columns(pl.lit(i).alias("simulation")))

    result = pl.concat(frames)
    cols = result.columns
    return result.select(["simulation"] + [c for c in cols if c != "simulation"])


def _iterative_monte_carlo_ou(
    n_simulations: int,
    seed: int | None = None,
    **kwargs: float | int | None,
) -> pl.DataFrame:
    """Loop-based Monte Carlo: call _iterative_ou per simulation."""
    rng = random.Random(seed)
    child_seeds = [rng.randrange(2**63) for _ in range(n_simulations)]

    frames: list[pl.DataFrame] = []
    for i, s in enumerate(child_seeds):
        df = _iterative_ou(**kwargs, seed=s)  # type: ignore[arg-type]
        frames.append(df.with_columns(pl.lit(i).alias("simulation")))

    result = pl.concat(frames)
    cols = result.columns
    return result.select(["simulation"] + [c for c in cols if c != "simulation"])


# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------


def _bench(label: str, fn, *args, warmup: int = 1, repeats: int = 5, **kwargs):
    """Time *fn* and print median elapsed time."""
    for _ in range(warmup):
        fn(*args, **kwargs)

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        times.append(time.perf_counter() - t0)

    times.sort()
    median = times[len(times) // 2]
    print(f"  {label:40s} {median*1000:8.1f} ms")
    return median, result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 64)
    print("Benchmark: Vectorized vs iterative (OU & GBM)")
    print("=" * 64)

    # ---- GBM ----
    print("\n" + "=" * 64)
    print("GBM")
    print("=" * 64)

    gbm_kwargs: dict[str, float] = dict(base_price=100.0, drift=0.05, volatility=0.2, dt=1 / 252)

    for n in (1_000, 10_000, 100_000):
        print(f"\n--- Single path, n={n:,} ---")
        t_vec, _ = _bench("vectorized (polars_sdist)", geometric_brownian_motion, n, seed=42, **gbm_kwargs)
        t_loop, _ = _bench("iterative  (python loop)", _iterative_gbm, n, seed=42, **gbm_kwargs)
        _print_speedup(t_vec, t_loop)

    n = 1_000
    for n_sims in (100, 500, 1_000):
        print(f"\n--- Monte Carlo, n={n:,} x {n_sims:,} simulations ---")
        t_vec, _ = _bench(
            "vectorized (single alloc)",
            monte_carlo, Process.GBM, n_sims, seed=42, n=n, **gbm_kwargs,
        )
        t_loop, _ = _bench(
            "iterative  (loop per sim)",
            _iterative_monte_carlo_gbm, n_sims, seed=42, n=n, **gbm_kwargs,
        )
        _print_speedup(t_vec, t_loop)

    # ---- OU ----
    print("\n" + "=" * 64)
    print("OU")
    print("=" * 64)

    ou_kwargs: dict[str, float] = dict(theta=0.7, mu=100.0, sigma=1.0, x0=200.0, dt=0.1)

    for n in (1_000, 10_000, 100_000):
        print(f"\n--- Single path, n={n:,} ---")
        t_vec, _ = _bench("vectorized (polars_sdist)", ornstein_uhlenbeck, n, seed=42, **ou_kwargs)
        t_loop, _ = _bench("iterative  (python loop)", _iterative_ou, n, seed=42, **ou_kwargs)
        _print_speedup(t_vec, t_loop)

    n = 1_000
    for n_sims in (100, 500, 1_000):
        print(f"\n--- Monte Carlo, n={n:,} x {n_sims:,} simulations ---")
        t_vec, _ = _bench(
            "vectorized (single alloc)",
            monte_carlo, Process.OU, n_sims, seed=42, n=n, **ou_kwargs,
        )
        t_loop, _ = _bench(
            "iterative  (loop per sim)",
            _iterative_monte_carlo_ou, n_sims, seed=42, n=n, **ou_kwargs,
        )
        _print_speedup(t_vec, t_loop)


def _print_speedup(t_vec: float, t_loop: float) -> None:
    if t_vec > 0:
        ratio = t_loop / t_vec
        print(f"  {'speedup':>40s} {ratio:7.1f}x")


if __name__ == "__main__":
    main()
