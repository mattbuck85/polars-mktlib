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

Ornstein–Uhlenbeck (OU)
~~~~~~~~~~~~~~~~~~~~~~~~

Mean-reverting process: :math:`dx = \theta(\mu - x) \, dt + \sigma \, dW`

Useful for modeling interest rates, volatility, or pairs-trading spreads where
the process reverts to a long-run mean ``mu`` at speed ``theta``.

.. autofunction:: mktlib.data.ornstein_uhlenbeck

Fractional Brownian Motion (fBm)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Generated via the Davies-Harte circulant embedding method using FFT
(O(n log n)). The Hurst exponent *H* controls path behavior:

- **H = 0.5** — standard random walk (fast path, no FFT)
- **H > 0.5** — trending (persistent) paths
- **H < 0.5** — mean-reverting (anti-persistent) paths

.. autofunction:: mktlib.data.fractional_random_walk

OHLCV Aggregation
-----------------

.. autofunction:: mktlib.data.ticks_to_ohlcv

Usage:

.. code-block:: python

   from mktlib.data import geometric_brownian_motion, ticks_to_ohlcv

   # Generate tick-level data, then aggregate to OHLCV bars
   ticks = geometric_brownian_motion(n=25200, seed=42)  # 1 tick/sec for 7 hours
   ohlcv = ticks_to_ohlcv(ticks, bar_size=60)           # 1-minute bars

Monte Carlo
-----------

.. autoclass:: mktlib.data.Process
   :members:
   :undoc-members:

.. autofunction:: mktlib.data.monte_carlo

**Vectorized enum path (recommended)** — bulk-samples all normals upfront and
partitions via ``.over("simulation")``, avoiding per-simulation Python loops:

.. code-block:: python

   from mktlib.data import Process, monte_carlo

   # 1000 GBM simulations, 252 steps each
   gbm_sims = monte_carlo(Process.GBM, n_simulations=1000, n=252, seed=42)
   # → DataFrame[simulation, step, price]

   # 500 Ornstein–Uhlenbeck simulations
   ou_sims = monte_carlo(Process.OU, n_simulations=500, n=252, theta=0.7, mu=100.0, seed=1)
   # → DataFrame[simulation, step, value]

   # 200 fractional random walk simulations (trending, H > 0.5)
   frw_sims = monte_carlo(Process.FRW, n_simulations=200, n=252, hurst=0.7, seed=2)
   # → DataFrame[simulation, step, price]

**Callable fallback** — pass any function with signature
``(*, seed: int, **kwargs) -> pl.DataFrame``. Runs a serial loop with
deterministic child seeds:

.. code-block:: python

   from mktlib.data import geometric_brownian_motion, monte_carlo

   # Equivalent to Process.GBM, but uses the serial loop path
   sims = monte_carlo(geometric_brownian_motion, n_simulations=1000, n=252, seed=42)
   # → DataFrame[simulation, step, price]

.. note::

   The ``Process`` enum path is significantly faster for large simulation counts
   because it draws all random samples in a single ``polars-sdist`` call and
   computes paths with ``.over("simulation")`` expressions. The callable path
   loops in Python and concatenates individual DataFrames.
