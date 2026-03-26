from __future__ import annotations

import math

import polars as pl
from polars_sdist import sample_normal


def _gbm_price_expr(log_base: float, mu_dt: float, sigma_sqrt_dt: float) -> pl.Expr:
    """GBM price expression. Assumes columns ``step`` and ``z`` exist."""
    return (
        pl.when(pl.col("step") == 0)
        .then(0.0)
        .otherwise(pl.col("z") * sigma_sqrt_dt + mu_dt)
        .cum_sum()
        .add(log_base)
        .exp()
        .alias("price")
    )


def geometric_brownian_motion(
    n: int,
    base_price: float = 100.0,
    drift: float = 0.0,
    volatility: float = 0.01,
    dt: float = 1 / 252,
    seed: int | None = None,
) -> pl.DataFrame:
    r"""Geometric Brownian motion price path.

    Simulates the SDE :math:`dS = \mu S \, dt + \sigma S \, dW` using the
    exact log-space solution derived via Itô's lemma.

    Parameters
    ----------
    n
        Number of time steps to generate.
    base_price
        Initial price :math:`S_0`.
    drift
        Annualised expected return :math:`\mu`.
    volatility
        Annualised volatility :math:`\sigma`.
    dt
        Time step size in annualised units.  Defaults to ``1/252`` (one
        trading day).  For sub-daily granularity, divide further — e.g.
        ``1/(252*6.5*3600)`` for one-second ticks during US equity hours.
    seed
        RNG seed for reproducibility.

    Returns
    -------
    pl.DataFrame
        Columns ``step`` (int) and ``price`` (float).

    Notes
    -----
    **Why log-space?**  The SDE is multiplicative in *S*, so direct
    Euler-Maruyama discretisation introduces bias.  Applying Itô's lemma
    to :math:`X = \ln S` yields an additive SDE::

        dX = (μ − ½σ²) dt + σ dW

    whose discrete-time exact solution is::

        ln S[i] = ln S[i−1] + (μ − ½σ²)·dt + σ·√dt·Z[i],  Z ~ N(0,1)

    The implementation draws all *Z* values in one ``sample_normal`` call,
    computes log-increments, and uses ``cum_sum`` + ``exp`` to recover
    prices — no Python loop required.

    **Itô correction (−½σ²).**  Because :math:`\exp` is convex, Jensen's
    inequality gives :math:`E[\exp(X)] > \exp(E[X])`.  The −½σ² drift
    adjustment compensates for this so that the expected price path
    satisfies :math:`E[S(t)] = S_0 \exp(\mu t)`.

    **Distributional properties.**  Log-returns
    :math:`\ln(S_{i}/S_{i-1})` are i.i.d. normal; prices :math:`S_i` are
    lognormally distributed.
    """
    if n < 1:
        raise ValueError("n must be >= 1")

    log_base = math.log(base_price)
    mu_dt = (drift - 0.5 * volatility**2) * dt
    sigma_sqrt_dt = volatility * math.sqrt(dt)

    z = sample_normal(n, seed=seed)
    step = pl.arange(0, n, eager=True).alias("step")

    return (
        pl.DataFrame({"step": step, "z": z})
        .with_columns(_gbm_price_expr(log_base, mu_dt, sigma_sqrt_dt))
        .select("step", "price")
    )
