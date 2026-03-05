from __future__ import annotations

import numpy as np
import polars as pl


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

    rng = np.random.default_rng(seed)
    # Log-normal increments: ln(S_{t+1}/S_t) = (mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z
    z = rng.standard_normal(n - 1)
    log_returns = (drift - 0.5 * volatility**2) * dt + volatility * np.sqrt(
        dt
    ) * z
    log_prices = np.concatenate(
        [[np.log(base_price)], np.log(base_price) + np.cumsum(log_returns)]
    )
    prices = np.exp(log_prices)

    return pl.DataFrame(
        {
            "step": range(n),
            "price": prices,
        }
    )
