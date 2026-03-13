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
    dt: float = 1.0,
    seed: int | None = None,
) -> pl.DataFrame:
    """Geometric Brownian motion price path: dS = mu*S*dt + sigma*S*dW."""
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
