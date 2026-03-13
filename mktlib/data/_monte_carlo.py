"""Vectorized Monte Carlo simulation of stochastic processes.

All built-in processes are discretizations of the general SDE::

    dX(t) = a(X, t) dt + b(X, t) dW(t)

where ``dW(t) = Z·√dt`` and ``Z ~ N(0,1)``.

+---------+------------------+------------------+
| Process | Drift a(X, t)    | Diffusion b(X,t) |
+---------+------------------+------------------+
| GBM     | μX               | σX               |
| OU      | θ(μ − X)         | σ                |
| FRW     | 0                | step_size         |
+---------+------------------+------------------+

FRW with ``H = 0.5`` reduces to a standard random walk (pure cumsum).
For ``H ≠ 0.5``, increments are generated via the Davies-Harte spectral
method rather than SDE discretization::

    γ(k) = ½(|k-1|^{2H} − 2|k|^{2H} + |k+1|^{2H})
    Λ    = FFT(γ_circ)
    ΔX   = Re[ IFFT( √Λ · (Z_re + i·Z_im) ) ]

where ``γ_circ`` is the circulant embedding of the autocovariance,
``Λ`` are its eigenvalues, and ``Z_re, Z_im ~ N(0,1)`` i.i.d.

For each simulation, ``n`` i.i.d. standard normal samples are drawn
in bulk via ``polars-sdist``, then partitioned by simulation index
using ``.over("simulation")``.
"""
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
        if process is Process.FRW:
            return _vectorized_frw(n_simulations, seed=seed, **process_kwargs)

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


def _vectorized_frw(
    n_simulations: int,
    *,
    seed: int | None = None,
    n: int = 100,
    hurst: float = 0.5,
    base_price: float = 100.0,
    step_size: float = 1.0,
) -> pl.DataFrame:
    if n < 1:
        raise ValueError("n must be >= 1")

    if hurst == 0.5:
        # Simple cumsum path — no FFT needed
        total = n_simulations * n
        z = sample_normal(total, seed=seed)
        idx = pl.arange(0, total, eager=True)
        return (
            pl.DataFrame({
                "simulation": idx // n,
                "step": idx % n,
                "z": z * step_size,
            })
            .with_columns(
                (
                    base_price
                    + pl.col("z")
                    .cum_sum()
                    .shift(1)
                    .fill_null(0.0)
                    .over("simulation")
                ).alias("price")
            )
            .select("simulation", "step", "price")
        )

    from mktlib.data._random_walk import (
        _build_covariance_row,
        _compute_sqrt_eigenvalues,
        _derive_seeds,
        _frw_increments_expr,
    )

    cov_row = _build_covariance_row(n, hurst)
    sqrt_eig = _compute_sqrt_eigenvalues(cov_row)
    m = len(sqrt_eig)  # 2n

    # Derive two child seeds for re/im noise
    seed_re, seed_im = _derive_seeds(seed)

    total_noise = n_simulations * m
    z_re_all = sample_normal(total_noise, seed=seed_re)
    z_im_all = sample_normal(total_noise, seed=seed_im)

    # Tile sqrt_eig across simulations
    sqrt_eig_tiled = pl.Series(
        "sqrt_eig", sqrt_eig.to_list() * n_simulations
    )

    idx = pl.arange(0, total_noise, eager=True)

    df = pl.DataFrame({
        "simulation": idx // m,
        "embed_idx": idx % m,
        "sqrt_eig": sqrt_eig_tiled,
        "z_re": z_re_all,
        "z_im": z_im_all,
    })

    # Apply IFFT per simulation group
    df = df.with_columns(
        _frw_increments_expr().over("simulation")
    )

    # Filter to first n rows per simulation (discard circulant padding)
    df = df.filter(pl.col("embed_idx") < n)

    # Scale and cumsum → prices
    return (
        df.with_columns(
            (
                base_price
                + (pl.col("increment") * (m**0.5 * step_size))
                .cum_sum()
                .shift(1)
                .fill_null(0.0)
                .over("simulation")
            ).alias("price")
        )
        .rename({"embed_idx": "step"})
        .select("simulation", "step", "price")
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
