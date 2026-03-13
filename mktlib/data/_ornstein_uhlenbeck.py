from __future__ import annotations

import math

import polars as pl
from polars_sdist import sample_normal


def ornstein_uhlenbeck(
    n: int,
    theta: float = 0.7,
    mu: float = 100.0,
    sigma: float = 1.0,
    x0: float | None = None,
    dt: float = 1.0,
    seed: int | None = None,
) -> pl.DataFrame:
    """Discrete Ornstein-Uhlenbeck process via vectorized Euler-Maruyama.

    The Euler-Maruyama recurrence is::

        x[i] = x[i-1] + θ(μ - x[i-1])dt + σ√dt·z[i]
             = α·x[i-1] + β + σ√dt·z[i]

    where ``α = 1 - θ·dt`` and ``β = θ·μ·dt``. This is an AR(1) process
    whose closed-form unrolling is::

        x[i] = αⁱ·x₀ + Σⱼ₌₁ⁱ α^(i-j)·(β + σ√dt·z[j])

    Factoring out ``αⁱ`` gives::

        x[i] = αⁱ · (x₀ + Σⱼ₌₁ⁱ (β + σ√dt·z[j]) / αʲ)

    The inner sum is a prefix sum of ``innovation[j] / αʲ``, which is
    computed with :meth:`polars.Expr.cum_sum` — no Python loop required.
    """
    if n < 1:
        raise ValueError("n must be >= 1")

    start = mu if x0 is None else x0
    alpha = 1.0 - theta * dt
    beta = theta * mu * dt
    noise_scale = sigma * math.sqrt(dt)

    z = sample_normal(n, seed=seed)
    step = pl.arange(0, n, eager=True).alias("step")

    return (
        pl.DataFrame({"step": step, "z": z})
        .with_columns(_ou_value_expr(start, alpha, beta, noise_scale))
        .select("step", "value")
    )


def _ou_value_expr(
    start: float, alpha: float, beta: float, noise_scale: float
) -> pl.Expr:
    """OU value expression. Assumes columns ``step`` and ``z`` exist."""
    innov = (
        pl.when(pl.col("step") == 0)
        .then(0.0)
        .otherwise(beta + noise_scale * pl.col("z"))
    )
    alpha_pow = pl.lit(alpha).pow(pl.col("step"))
    return (
        (innov / alpha_pow).cum_sum() * alpha_pow + start * alpha_pow
    ).alias("value")
