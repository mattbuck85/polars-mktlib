from __future__ import annotations

import numpy as np
import polars as pl


def fractional_random_walk(
    n: int,
    hurst: float = 0.5,
    base_price: float = 100.0,
    step_size: float = 1.0,
    seed: int | None = None,
) -> pl.DataFrame:
    """Discrete-time fractional random walk using Cholesky decomposition of fBm covariance.

    For ``hurst=0.5`` this is a standard random walk.  ``hurst > 0.5`` produces
    trending (persistent) paths; ``hurst < 0.5`` produces mean-reverting
    (anti-persistent) paths.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    if not 0 < hurst < 1:
        raise ValueError("hurst must be in (0, 1)")

    rng = np.random.default_rng(seed)

    if hurst == 0.5:
        increments = rng.normal(0, step_size, n)
    else:
        # Build fBm covariance matrix and generate correlated increments
        # via Hosking / Cholesky method
        h2 = 2 * hurst
        # Covariance of fBm increments: C(i,j) = 0.5*(|i-j+1|^2H - 2|i-j|^2H + |i-j-1|^2H)
        cov = np.empty((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(i, n):
                d = abs(i - j)
                val = 0.5 * (abs(d + 1) ** h2 - 2 * abs(d) ** h2 + abs(d - 1) ** h2)
                cov[i, j] = val
                cov[j, i] = val

        L = np.linalg.cholesky(cov)
        z = rng.standard_normal(n)
        increments = (L @ z) * step_size

    prices = base_price + np.cumsum(increments)
    prices = np.insert(prices, 0, base_price)[:n]

    return pl.DataFrame({
        "step": range(n),
        "price": prices,
    })
