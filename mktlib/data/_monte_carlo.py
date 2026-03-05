from __future__ import annotations

from typing import Any, Callable

import polars as pl


def monte_carlo(
    process: Callable[..., pl.DataFrame],
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

    import numpy as np

    rng = np.random.default_rng(seed)
    child_seeds = rng.integers(0, 2**63, size=n_simulations)

    frames: list[pl.DataFrame] = []
    for i, s in enumerate(child_seeds):
        df = process(**process_kwargs, seed=int(s))
        frames.append(df.with_columns(pl.lit(i).alias("simulation")))

    result = pl.concat(frames)
    # Reorder so simulation is first
    cols = result.columns
    return result.select(["simulation"] + [c for c in cols if c != "simulation"])
