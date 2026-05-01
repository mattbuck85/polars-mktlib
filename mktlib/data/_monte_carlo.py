from __future__ import annotations

import enum
import math
import random
from typing import Any, Callable, overload

import polars as pl
from polars_sdist import sample_normal, sample_students_t

from mktlib.data._gbm import _gbm_price_expr
from mktlib.data._ornstein_uhlenbeck import _ou_value_expr

from polars_sdist import SdistNamespace as sdist


class Innovations(enum.Enum):
    """Noise distribution for stochastic-process simulations.

    All variants produce **unit-variance** i.i.d. samples — the
    ``volatility`` (or analogous) parameter of the host process remains
    the controlling scale.  Switching innovations changes only the
    *shape* of the tails, not the second moment of bar-level
    log-returns.

    Currently honoured by :class:`Process.GBM`; using a non-Gaussian
    member with :class:`Process.OU` or :class:`Process.FRW` raises
    :class:`NotImplementedError` (FRW's Davies–Harte construction is
    only meaningful under Gaussian noise; OU's direct-``sigma``
    parameterisation tangles with the unit-variance contract).
    """

    GAUSSIAN = "gaussian"
    """Standard normal :math:`Z \\sim N(0, 1)` via ``polars_sdist.sample_normal``."""
    STUDENT_T = "students_t"
    """Student-t with degrees-of-freedom *df* (>2), rescaled to unit variance."""
    BOOTSTRAP = "bootstrap"
    """Resample with replacement from a caller-supplied unit-variance residual series."""


def _child_seeds(n_simulations: int, seed: int | None) -> list[int]:
    rng = random.Random(seed)
    return [rng.randrange(2**63) for _ in range(n_simulations)]


def _draw_unit_variance(
    n: int,
    seed: int | None,
    *,
    innovations: Innovations | Callable[[int, int | None], pl.Series],
    df: float | None,
    residuals: pl.Series | None,
) -> pl.Series:
    """Single-simulation unit-variance noise draw.

    Dispatches on *innovations*; called once per child seed and
    concatenated upstream.  All return paths produce a Series of length
    *n* with (asymptotic) unit variance.
    """
    if callable(innovations):
        return innovations(n, seed)
    match innovations:
        case Innovations.GAUSSIAN:
            return sample_normal(n, seed=seed)
        case Innovations.STUDENT_T:
            if df is None:
                msg = "innovations=STUDENT_T requires df= (degrees of freedom)"
                raise ValueError(msg)
            if df <= 2:
                msg = f"Student-t df must be > 2 for finite variance (got {df})"
                raise ValueError(msg)
            scale = math.sqrt(df / (df - 2.0))
            return sample_students_t(n, df=df, seed=seed) / scale
        case Innovations.BOOTSTRAP:
            if residuals is None:
                msg = "innovations=BOOTSTRAP requires residuals= (unit-variance series)"
                raise ValueError(msg)
            return residuals.sample(n=n, with_replacement=True, seed=seed)


def _noise_frame(
    n_simulations: int,
    n: int,
    *,
    seed: int | None = None,
    innovations: Innovations | Callable[[int, int | None], pl.Series] = Innovations.GAUSSIAN,
    df: float | None = None,
    residuals: pl.Series | None = None,
    independent_streams: bool = True,
) -> pl.DataFrame:
    """Build ``simulation | seed | step | z`` frame with per-sim noise draws.

    The *z* column is unit-variance regardless of *innovations* — the
    distributional choice only changes tail shape.  See
    :class:`Innovations` for the contract.

    Parameters
    ----------
    independent_streams
        When ``True`` (default), each simulation draws from its own
        deterministically-derived child RNG and the *seed* column reports
        that per-stream seed.  Preserves byte-for-byte equivalence with
        pre-v0.11.x behaviour.

        When ``False``, draws ``n_simulations * n`` unit-variance samples
        in a single batched call to the underlying sampler.  The result
        is statistically identical (i.i.d. samples by construction) but
        runs **5–7× faster for Gaussian / Student-t and ~200× faster for
        bootstrap** because per-call RNG construction overhead dominates
        the wall-clock at typical (10k sims) scales.  The *seed* column
        is filled with one parent-derived seed shared by every row —
        callers that introspect per-simulation seeds must opt out.
    """
    total = n_simulations * n

    if independent_streams:
        child_seeds = _child_seeds(n_simulations, seed)
        z = pl.concat(
            [
                _draw_unit_variance(
                    n, s, innovations=innovations, df=df, residuals=residuals,
                )
                for s in child_seeds
            ],
            rechunk=True,
        )
        seeds_per_sim = pl.Series("seed", child_seeds)
    else:
        # One concrete parent-derived seed drives both the batched draw
        # and the seed column, so the column stays meaningful even when
        # the caller passes seed=None (we materialize an OS-time-based
        # int rather than leaving it null).
        parent_seed = _child_seeds(1, seed)[0]
        z = _draw_unit_variance(
            total, parent_seed, innovations=innovations, df=df, residuals=residuals,
        )
        seeds_per_sim = pl.Series("seed", [parent_seed] * n_simulations)

    idx = pl.arange(0, total, eager=True)
    sim = idx // n
    return pl.DataFrame(
        {
            "simulation": sim,
            "seed": seeds_per_sim.gather(sim),
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
    innovations: Innovations | Callable[[int, int | None], pl.Series] = Innovations.GAUSSIAN,
    df: float | None = None,
    residuals: pl.Series | None = None,
    independent_streams: bool = True,
    **process_kwargs: Any,
) -> pl.DataFrame: ...


@overload
def monte_carlo(
    process: Callable[..., pl.DataFrame],
    n_simulations: int = 1000,
    *,
    seed: int | None = None,
    innovations: Innovations | Callable[[int, int | None], pl.Series] = Innovations.GAUSSIAN,
    df: float | None = None,
    residuals: pl.Series | None = None,
    independent_streams: bool = True,
    **process_kwargs: Any,
) -> pl.DataFrame: ...


def monte_carlo(
    process: Process | Callable[..., pl.DataFrame],
    n_simulations: int = 1000,
    *,
    seed: int | None = None,
    innovations: Innovations | Callable[[int, int | None], pl.Series] = Innovations.GAUSSIAN,
    df: float | None = None,
    residuals: pl.Series | None = None,
    independent_streams: bool = True,
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
    innovations
        Noise distribution feeding the SDE.  Currently honoured by
        :class:`Process.GBM`; passing a non-Gaussian member with
        :class:`Process.OU`, :class:`Process.FRW`, or a callable *process*
        raises :class:`NotImplementedError`.  All variants emit
        unit-variance i.i.d. samples — the host process's ``volatility``
        parameter remains the controlling scale.  See :class:`Innovations`.
    df
        Required when ``innovations=Innovations.STUDENT_T``; degrees of
        freedom (must be > 2 for finite variance).
    residuals
        Required when ``innovations=Innovations.BOOTSTRAP``; a
        unit-variance ``pl.Series`` of empirical residuals to resample
        with replacement.
    independent_streams
        Default ``True`` — preserves the per-simulation child-seed
        contract.  Set to ``False`` to draw all noise in one batched
        sampler call (statistically identical i.i.d. samples,
        **5×–200× faster** at typical scales).  Trade-off: the *seed*
        column reports a single parent-derived seed instead of one per
        simulation.  GBM only — passing ``False`` with OU / FRW raises
        :class:`NotImplementedError`.
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

    non_gaussian = (
        callable(innovations) or innovations is not Innovations.GAUSSIAN
    )

    if isinstance(process, Process):
        if process is Process.GBM:
            return _vectorized_gbm(
                n_simulations,
                seed=seed,
                innovations=innovations,
                df=df,
                residuals=residuals,
                independent_streams=independent_streams,
                **process_kwargs,
            )
        if non_gaussian:
            msg = (
                f"innovations={innovations!r} is only supported for "
                "Process.GBM in v0.11.0; OU and FRW require Gaussian noise"
            )
            raise NotImplementedError(msg)
        if process is Process.OU:
            return _vectorized_ou(
                n_simulations, seed=seed,
                independent_streams=independent_streams, **process_kwargs,
            )
        if process is Process.FRW:
            return _vectorized_frw(
                n_simulations, seed=seed,
                independent_streams=independent_streams, **process_kwargs,
            )

    if non_gaussian:
        msg = (
            "custom innovations are not supported for callable processes; "
            "the user-supplied generator owns its own noise source"
        )
        raise NotImplementedError(msg)
    if not independent_streams:
        # Callable processes own their own noise source; we have no way to
        # batch a user-defined generator, so the flag has no effect there.
        msg = (
            "independent_streams=False is not supported for callable processes; "
            "the user-supplied generator owns its own noise source"
        )
        raise NotImplementedError(msg)
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
    innovations: Innovations | Callable[[int, int | None], pl.Series] = Innovations.GAUSSIAN,
    df: float | None = None,
    residuals: pl.Series | None = None,
    independent_streams: bool = True,
) -> pl.DataFrame:
    if n < 1:
        raise ValueError("n must be >= 1")

    log_base = math.log(base_price)
    mu_dt = (drift - 0.5 * volatility**2) * dt
    sigma_sqrt_dt = volatility * math.sqrt(dt)

    return (
        _noise_frame(
            n_simulations,
            n,
            seed=seed,
            innovations=innovations,
            df=df,
            residuals=residuals,
            independent_streams=independent_streams,
        )
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
    geometric: bool = False,
    independent_streams: bool = True,
) -> pl.DataFrame:
    if n < 1:
        raise ValueError("n must be >= 1")

    start = mu if x0 is None else x0
    alpha = 1.0 - theta * dt
    beta = theta * mu * dt
    noise_scale = sigma * math.sqrt(dt)

    df = (
        _noise_frame(
            n_simulations, n, seed=seed,
            independent_streams=independent_streams,
        )
        .with_columns(
            _ou_value_expr(start, alpha, beta, noise_scale).over("simulation")
        )
    )
    if geometric:
        return df.with_columns(pl.col("value").exp().alias("price")).select("simulation", "seed", "step", "price")
    return df.select("simulation", "seed", "step", "value")


def _vectorized_frw(
    n_simulations: int,
    *,
    seed: int | None = None,
    n: int = 100,
    hurst: float = 0.5,
    base_price: float = 100.0,
    step_size: float = 1.0,
    geometric: bool = False,
    independent_streams: bool = True,
) -> pl.DataFrame:
    if n < 1:
        raise ValueError("n must be >= 1")

    if hurst == 0.5:
        # Simple cumsum path — no FFT needed
        cumulative_expr = (
            (pl.col("z") * step_size)
            .cum_sum()
            .shift(1)
            .fill_null(0.0)
            .over("simulation")
        )
        if geometric:
            price_expr = (base_price * cumulative_expr.exp()).alias("price")
        else:
            price_expr = (base_price + cumulative_expr).alias("price")
        return (
            _noise_frame(
                n_simulations, n, seed=seed,
                independent_streams=independent_streams,
            )
            .with_columns(price_expr)
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

    if independent_streams:
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
        seeds_per_sim = pl.Series("seed", child_seeds)
    else:
        # Single-batch: derive one parent-derived seed pair (re, im) and
        # draw all 2*n_simulations*m unit normals in two sampler calls.
        # Statistically identical (i.i.d. unit-normal pairs across sims)
        # because the Davies-Harte construction only requires per-sim
        # independence of the real/imaginary parts, which holds trivially
        # under i.i.d. draws.
        parent_seed = _child_seeds(1, seed)[0]
        parent_re, parent_im = _derive_seeds(parent_seed)
        z_re_all = sample_normal(n_simulations * m, seed=parent_re)
        z_im_all = sample_normal(n_simulations * m, seed=parent_im)
        seeds_per_sim = pl.Series("seed", [parent_seed] * n_simulations)

    # Compute sqrt_eig once, then tile across simulations
    sqrt_eig = (
        pl.DataFrame({"cov_row": cov_row})
        .select(_sqrt_eigenvalue_expr())
        .to_series()
    )
    sqrt_eig_tiled = pl.Series("sqrt_eig", sqrt_eig.to_list() * n_simulations)

    total_noise = n_simulations * m
    idx = pl.arange(0, total_noise, eager=True)
    seed_col = seeds_per_sim.gather(idx // m)

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
    cumulative_expr = (
        (pl.col("increment") * (m**0.5 * step_size))
        .cum_sum()
        .shift(1)
        .fill_null(0.0)
        .over("simulation")
    )
    if geometric:
        price_expr = (base_price * cumulative_expr.exp()).alias("price")
    else:
        price_expr = (base_price + cumulative_expr).alias("price")
    return (
        df.with_columns(price_expr)
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
