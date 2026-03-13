from __future__ import annotations

import enum
import math
import random
from typing import Any, Callable, overload

import polars as pl
from polars_sdist import sample_normal

from mktlib.data._gbm import _gbm_price_expr
from mktlib.data._ornstein_uhlenbeck import _ou_value_expr


class Process(enum.Enum):
    GBM = "gbm"
    OU = "ou"
    FRW = "frw"


@overload
def monte_carlo(
    process: Process,
    n_simulations: int = 1000,
    *,
    seed: int | None = None,
    **process_kwargs: Any,
) -> pl.DataFrame: ...


@overload
def monte_carlo(
    process: Callable[..., pl.DataFrame],
    n_simulations: int = 1000,
    *,
    seed: int | None = None,
    **process_kwargs: Any,
) -> pl.DataFrame: ...


def monte_carlo(
    process: Process | Callable[..., pl.DataFrame],
    n_simulations: int = 1000,
    *,
    seed: int | None = None,
    **process_kwargs: Any,
) -> pl.DataFrame:
    """Run multiple simulations of a stochastic process.

    Returns a stacked DataFrame with a ``simulation`` column prepended.
    Each simulation uses a deterministic seed derived from *seed*.
    """
    if n_simulations < 1:
        raise ValueError("n_simulations must be >= 1")

    if isinstance(process, Process):
        if process is Process.GBM:
            return _vectorized_gbm(n_simulations, seed=seed, **process_kwargs)
        if process is Process.OU:
            return _vectorized_ou(n_simulations, seed=seed, **process_kwargs)

    return _loop(process, n_simulations, seed=seed, **process_kwargs)


def _vectorized_gbm(
    n_simulations: int,
    *,
    seed: int | None = None,
    n: int = 100,
    base_price: float = 100.0,
    drift: float = 0.0,
    volatility: float = 0.01,
    dt: float = 1.0,
) -> pl.DataFrame:
    if n < 1:
        raise ValueError("n must be >= 1")

    log_base = math.log(base_price)
    mu_dt = (drift - 0.5 * volatility**2) * dt
    sigma_sqrt_dt = volatility * math.sqrt(dt)

    total = n_simulations * n
    z = sample_normal(total, seed=seed)
    idx = pl.arange(0, total, eager=True)

    return (
        pl.DataFrame({"simulation": idx // n, "step": idx % n, "z": z})
        .with_columns(
            _gbm_price_expr(log_base, mu_dt, sigma_sqrt_dt).over("simulation")
        )
        .select("simulation", "step", "price")
    )


def _vectorized_ou(
    n_simulations: int,
    *,
    seed: int | None = None,
    n: int = 100,
    theta: float = 0.7,
    mu: float = 100.0,
    sigma: float = 1.0,
    x0: float | None = None,
    dt: float = 1.0,
) -> pl.DataFrame:
    if n < 1:
        raise ValueError("n must be >= 1")

    start = mu if x0 is None else x0
    alpha = 1.0 - theta * dt
    beta = theta * mu * dt
    noise_scale = sigma * math.sqrt(dt)

    total = n_simulations * n
    z = sample_normal(total, seed=seed)
    idx = pl.arange(0, total, eager=True)

    return (
        pl.DataFrame({"simulation": idx // n, "step": idx % n, "z": z})
        .with_columns(
            _ou_value_expr(start, alpha, beta, noise_scale).over("simulation")
        )
        .select("simulation", "step", "value")
    )


def _loop(
    process: Process | Callable[..., pl.DataFrame],
    n_simulations: int,
    *,
    seed: int | None = None,
    **process_kwargs: Any,
) -> pl.DataFrame:
    rng = random.Random(seed)
    child_seeds = [rng.randrange(2**63) for _ in range(n_simulations)]

    if isinstance(process, Process):
        match process:
            case Process.GBM:
                from mktlib.data._gbm import geometric_brownian_motion

                func = geometric_brownian_motion
            case Process.OU:
                from mktlib.data._ornstein_uhlenbeck import ornstein_uhlenbeck

                func = ornstein_uhlenbeck
            case Process.FRW:
                from mktlib.data._random_walk import fractional_random_walk

                func = fractional_random_walk
    else:
        func = process

    frames: list[pl.DataFrame] = []
    for i, s in enumerate(child_seeds):
        df = func(**process_kwargs, seed=s)
        frames.append(df.with_columns(pl.lit(i).alias("simulation")))

    result = pl.concat(frames)
    cols = result.columns
    return result.select(
        ["simulation"] + [c for c in cols if c != "simulation"]
    )
