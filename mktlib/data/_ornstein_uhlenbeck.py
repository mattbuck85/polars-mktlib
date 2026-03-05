from __future__ import annotations

import numpy as np
import polars as pl


def ornstein_uhlenbeck(
    n: int,
    theta: float = 0.7,
    mu: float = 100.0,
    sigma: float = 1.0,
    x0: float | None = None,
    dt: float = 1.0,
    seed: int | None = None,
) -> pl.DataFrame:
    """Discrete Ornstein-Uhlenbeck process: dx = theta*(mu - x)*dt + sigma*dW."""
    if n < 1:
        raise ValueError("n must be >= 1")

    rng = np.random.default_rng(seed)
    x = np.empty(n)
    x[0] = mu if x0 is None else x0
    noise = rng.normal(0, sigma * np.sqrt(dt), n - 1)

    for i in range(1, n):
        x[i] = x[i - 1] + theta * (mu - x[i - 1]) * dt + noise[i - 1]

    return pl.DataFrame({
        "step": range(n),
        "value": x,
    })
