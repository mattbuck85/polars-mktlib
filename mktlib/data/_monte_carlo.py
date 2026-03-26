from __future__ import annotations

import enum
import math
import random
from typing import Any, Callable, overload

import polars as pl
from polars_sdist import sample_normal

from mktlib.data._gbm import _gbm_price_expr
from mktlib.data._ornstein_uhlenbeck import _ou_value_expr

from polars_sdist import SdistNamespace as sdist


def _child_seeds(n_simulations: int, seed: int | None) -> list[int]:
    rng = random.Random(seed)
    return [rng.randrange(2**63) for _ in range(n_simulations)]


def _noise_frame(
    n_simulations: int,
    n: int,
    *,
    seed: int | None = None,
) -> pl.DataFrame:
    """Build ``simulation | seed | step | z`` frame with per-sim normal draws."""
    child_seeds = _child_seeds(n_simulations, seed)
    z = pl.concat(
        [sample_normal(n, seed=s) for s in child_seeds],
        rechunk=True,
    )
    total = n_simulations * n
    idx = pl.arange(0, total, eager=True)
    sim = idx // n
    return pl.DataFrame(
        {
            "simulation": sim,
            "seed": pl.Series("seed", child_seeds).gather(sim),
            "step": idx % n,
            "z": z,
        }
    )


class Process(enum.Enum):
    """Built-in stochastic processes for :func:`monte_carlo`.

    Each variant maps to a vectorized implementation that draws all normal
    samples in a single ``polars-sdist`` call and partitions them across
    simulations with ``.over("simulation")`` — no Python loop.

    Pass the enum member as the first argument to :func:`monte_carlo`,
    along with any keyword arguments accepted by the underlying single-path
    generator (e.g. ``n``, ``drift``, ``volatility`` for GBM).
    """

    GBM = "gbm"
    """Geometric Brownian motion — lognormal price paths."""
    OU = "ou"
    """Ornstein-Uhlenbeck — mean-reverting process."""
    FRW = "frw"
    """Fractional random walk — fBm increments via Davies-Harte."""


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
    r"""Run multiple simulations of a stochastic process.

    Generates *n_simulations* independent paths from the same process and
    returns them stacked in a single DataFrame with a ``simulation`` index
    column prepended.

    Parameters
    ----------
    process
        Which process to simulate.  A :class:`Process` enum member selects the
        vectorized fast path; a callable with signature
        ``(*, seed: int, **kwargs) -> pl.DataFrame`` uses a serial loop
        fallback.
    n_simulations
        Number of independent paths to generate.
    seed
        Parent RNG seed.  Deterministic child seeds are derived so that each
        simulation is reproducible and independent.
    **process_kwargs
        Forwarded to the underlying generator.  For :class:`Process` members
        these match the single-path function signatures (e.g. ``n``, ``drift``,
        ``volatility`` for GBM).

    Returns
    -------
    pl.DataFrame
        Columns ``simulation`` (int), ``seed`` (int), ``step`` (int), and a
        value column whose name depends on the process — ``price`` for GBM /
        FRW, ``value`` for OU.  The ``seed`` column contains the child seed
        used for that simulation — constant within each simulation group.

    Notes
    -----
    All built-in processes are discretizations of the general SDE::

        dX(t) = a(X, t) dt + b(X, t) dW(t)

    where :math:`dW(t) = Z \sqrt{dt}`, :math:`Z \sim N(0,1)`.  The drift and
    diffusion coefficients for each :class:`Process` variant are:

    ===== ======================== ===================
    Enum  Drift *a(X, t)*          Diffusion *b(X, t)*
    ===== ======================== ===================
    GBM   :math:`\mu X`            :math:`\sigma X`
    OU    :math:`\theta(\mu - X)`  :math:`\sigma`
    FRW   0                        step_size
    ===== ======================== ===================

    **Vectorized path (Process enum).**  A single ``sample_normal(n_simulations * n)``
    call draws all noise upfront.  Per-simulation recurrences are computed via
    Polars ``.over("simulation")`` expressions, keeping the entire operation in
    Rust with no Python loop.  For FRW with :math:`H \neq 0.5`, the
    sqrt-eigenvalues of the circulant covariance are computed once and tiled
    across simulations.

    **Serial fallback (callable).**  The callable is invoked once per
    simulation with a deterministic child seed derived from *seed*.  Results
    are concatenated with :func:`polars.concat`.  This path is slower but
    supports arbitrary user-defined generators.

    **Seed derivation.**  Child seeds are produced by a ``random.Random(seed)``
    instance, so results are fully reproducible for a given *seed* /
    *n_simulations* pair.
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
    volatility: float = 1.0,
    dt: float = 1 / 252,
) -> pl.DataFrame:
    if n < 1:
        raise ValueError("n must be >= 1")

    log_base = math.log(base_price)
    mu_dt = (drift - 0.5 * volatility**2) * dt
    sigma_sqrt_dt = volatility * math.sqrt(dt)

    return (
        _noise_frame(n_simulations, n, seed=seed)
        .with_columns(
            _gbm_price_expr(log_base, mu_dt, sigma_sqrt_dt).over("simulation")
        )
        .select("simulation", "seed", "step", "price")
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
    dt: float = 1 / 252,
) -> pl.DataFrame:
    if n < 1:
        raise ValueError("n must be >= 1")

    start = mu if x0 is None else x0
    alpha = 1.0 - theta * dt
    beta = theta * mu * dt
    noise_scale = sigma * math.sqrt(dt)

    return (
        _noise_frame(n_simulations, n, seed=seed)
        .with_columns(
            _ou_value_expr(start, alpha, beta, noise_scale).over("simulation")
        )
        .select("simulation", "seed", "step", "value")
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
        return (
            _noise_frame(n_simulations, n, seed=seed)
            .with_columns(
                (
                    base_price
                    + (pl.col("z") * step_size)
                    .cum_sum()
                    .shift(1)
                    .fill_null(0.0)
                    .over("simulation")
                ).alias("price")
            )
            .select("simulation", "seed", "step", "price")
        )

    from mktlib.data._random_walk import (
        _build_covariance_row,
        _derive_seeds,
        _frw_increments_expr,
        _sqrt_eigenvalue_expr,
    )

    cov_row = _build_covariance_row(n, hurst)
    m = len(cov_row)  # 2n

    # Per-simulation child seeds → per-sim re/im seeds via _derive_seeds
    child_seeds = _child_seeds(n_simulations, seed)
    z_re_parts: list[pl.Series] = []
    z_im_parts: list[pl.Series] = []
    for cs in child_seeds:
        s_re, s_im = _derive_seeds(cs)
        z_re_parts.append(sample_normal(m, seed=s_re))
        z_im_parts.append(sample_normal(m, seed=s_im))
    z_re_all = pl.concat(z_re_parts, rechunk=True)
    z_im_all = pl.concat(z_im_parts, rechunk=True)

    # Compute sqrt_eig once, then tile across simulations
    sqrt_eig = (
        pl.DataFrame({"cov_row": cov_row})
        .select(_sqrt_eigenvalue_expr())
        .to_series()
    )
    sqrt_eig_tiled = pl.Series("sqrt_eig", sqrt_eig.to_list() * n_simulations)

    total_noise = n_simulations * m
    idx = pl.arange(0, total_noise, eager=True)
    seed_col = pl.Series("seed", child_seeds).gather(idx // m)

    df = pl.DataFrame(
        {
            "simulation": idx // m,
            "seed": seed_col,
            "embed_idx": idx % m,
            "sqrt_eig": sqrt_eig_tiled,
            "z_re": z_re_all,
            "z_im": z_im_all,
        }
    )

    # Apply IFFT per simulation group
    df = df.with_columns(_frw_increments_expr().over("simulation"))

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
        .select("simulation", "seed", "step", "price")
    )


def _loop(
    process: Process | Callable[..., pl.DataFrame],
    n_simulations: int,
    *,
    seed: int | None = None,
    **process_kwargs: Any,
) -> pl.DataFrame:
    child_seeds = _child_seeds(n_simulations, seed)

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
        frames.append(
            df.with_columns(
                pl.lit(i).alias("simulation"),
                pl.lit(s).alias("seed"),
            )
        )

    result = pl.concat(frames)
    cols = result.columns
    return result.select(
        ["simulation", "seed"]
        + [c for c in cols if c not in ("simulation", "seed")]
    )
