Data
====

Synthetic data generators for testing, simulation, and Monte Carlo analysis.

Requires the ``data`` extra: ``pip install mktlib[data]``, which installs
`polars-sdist <https://github.com/mattbuck85/polars-sdist>`_ and
`polars-rfft <https://github.com/mattbuck85/polars-rfft>`_ (pure Rust Polars plugins).

All functions return Polars DataFrames with seeded RNG for reproducibility.

Stochastic Differential Equations
----------------------------------

Geometric Brownian Motion (GBM)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Log-normal price paths: :math:`dS = \mu S \, dt + \sigma S \, dW`

Suitable for equity price simulation. The ``drift`` parameter controls the expected
return and ``volatility`` controls dispersion.

.. autofunction:: mktlib.data.geometric_brownian_motion
   :no-index:

Ornstein–Uhlenbeck (OU)
~~~~~~~~~~~~~~~~~~~~~~~~

Mean-reverting process: :math:`dx = \theta(\mu - x) \, dt + \sigma \, dW`

Useful for modeling interest rates, volatility, or pairs-trading spreads where
the process reverts to a long-run mean ``mu`` at speed ``theta``.

.. autofunction:: mktlib.data.ornstein_uhlenbeck
   :no-index:

Fractional Brownian Motion (fBm)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Generated via the Davies-Harte circulant embedding method using FFT
(O(n log n)). The Hurst exponent *H* controls path behavior:

- **H = 0.5** — standard random walk (fast path, no FFT)
- **H > 0.5** — trending (persistent) paths
- **H < 0.5** — mean-reverting (anti-persistent) paths

Powered by `polars-rfft <https://github.com/mattbuck85/polars-rfft>`_ (RustFFT) and
`polars-sdist <https://github.com/mattbuck85/polars-sdist>`_ for normal sampling.

.. autofunction:: mktlib.data.fractional_random_walk
   :no-index:

OHLCV Aggregation
-----------------

.. autofunction:: mktlib.data.ticks_to_ohlcv
   :no-index:

Usage:

.. code-block:: python

   from mktlib.data import geometric_brownian_motion, ticks_to_ohlcv

   # Generate tick-level data, then aggregate to OHLCV bars
   ticks = geometric_brownian_motion(n=25200, seed=42)  # 1 tick/sec for 7 hours
   ohlcv = ticks_to_ohlcv(ticks, bar_size=60)           # 1-minute bars

Monte Carlo
-----------

.. autofunction:: mktlib.data.monte_carlo
   :no-index:

Usage:

.. code-block:: python

   from mktlib.data import geometric_brownian_motion, monte_carlo

   # 1000 simulations of 252-step GBM
   sims = monte_carlo(geometric_brownian_motion, n_simulations=1000, n=252, seed=42)
   # Returns DataFrame with columns [simulation, step, price]

Full Module Reference
---------------------

.. automodule:: mktlib.data
   :members:
   :undoc-members:
   :show-inheritance:
