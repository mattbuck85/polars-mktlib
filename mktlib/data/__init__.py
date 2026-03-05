from __future__ import annotations

try:
    import numpy as _np  # noqa: F401
except ModuleNotFoundError as _e:
    raise ModuleNotFoundError(
        "mktlib.data requires numpy. Install it with: pip install mktlib[data]"
    ) from _e

from mktlib.data._gbm import geometric_brownian_motion
from mktlib.data._monte_carlo import monte_carlo
from mktlib.data._ornstein_uhlenbeck import ornstein_uhlenbeck
from mktlib.data._random_walk import fractional_random_walk

__all__ = [
    "fractional_random_walk",
    "geometric_brownian_motion",
    "monte_carlo",
    "ornstein_uhlenbeck",
]
