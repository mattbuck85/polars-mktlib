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

   # Generate tick-level data at 1-second resolution, then aggregate to 1-minute bars
   dt_1s = 1 / (252 * 6.5 * 3600)  # 1 second in annualised units
   ticks = geometric_brownian_motion(n=23_400, volatility=1.0, dt=dt_1s, seed=42)
   ohlcv = ticks_to_ohlcv(ticks, bar_size=60, seed=43)  # 60 seconds → 1-minute bar

See :doc:`/advanced` for a full walkthrough generating multi-year 1-minute OHLCV data.

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
   # → DataFrame[simulation, seed, step, price]

   # 500 Ornstein–Uhlenbeck simulations
   ou_sims = monte_carlo(Process.OU, n_simulations=500, n=252, theta=0.7, mu=100.0, seed=1)
   # → DataFrame[simulation, seed, step, value]

   # 200 fractional random walk simulations (trending, H > 0.5)
   frw_sims = monte_carlo(Process.FRW, n_simulations=200, n=252, hurst=0.7, seed=2)
   # → DataFrame[simulation, seed, step, price]

**Callable fallback** — pass any function with signature
``(*, seed: int, **kwargs) -> pl.DataFrame``. Runs a serial loop with
deterministic child seeds:

.. code-block:: python

   from mktlib.data import geometric_brownian_motion, monte_carlo

   # Equivalent to Process.GBM, but uses the serial loop path
   sims = monte_carlo(geometric_brownian_motion, n_simulations=1000, n=252, seed=42)
   # → DataFrame[simulation, seed, step, price]

.. note::

   The ``Process`` enum path is significantly faster for large simulation counts
   because it draws all random samples in a single ``polars-sdist`` call and
   computes paths with ``.over("simulation")`` expressions. The callable path
   loops in Python and concatenates individual DataFrames.

Pluggable Innovations
~~~~~~~~~~~~~~~~~~~~~

The default GBM noise source is standard-normal, but ``monte_carlo()``
accepts arbitrary unit-variance innovations via the ``innovations``
argument. This is what makes the simulation engine useful for
forward-looking risk numbers under non-Gaussian assumptions — see
:doc:`metrics` for the VaR / CVaR estimators that consume these.

.. autoclass:: mktlib.data.Innovations
   :members:
   :undoc-members:

**Unit-variance contract** (load-bearing): every innovation must produce
i.i.d. samples with unit variance. The host process's ``volatility``
parameter remains the controlling scale; switching innovations changes
the *tail shape*, not the second moment. Concretely:

* :data:`Innovations.GAUSSIAN` — :math:`Z \sim N(0, 1)` via
  ``polars-sdist``'s ``sample_normal``. Already unit-variance.
* :data:`Innovations.STUDENT_T` — :math:`T_\nu` rescaled to unit
  variance by dividing by :math:`\sqrt{\nu/(\nu-2)}`. Requires ``df=ν``
  with :math:`\nu > 2` (strictly — the divisor explodes at
  :math:`\nu = 2`, and Cauchy at :math:`\nu = 1` has undefined variance).
* :data:`Innovations.BOOTSTRAP` — resamples a caller-supplied
  ``residuals: pl.Series`` (which the caller is responsible for
  pre-standardizing to unit variance) with replacement. Distribution-free.

A callable noise source ``Callable[[int, int | None], pl.Series]`` is
also accepted as an escape hatch for skew-normal, mixture-of-normals,
or any other custom sampler — the function receives ``(n, seed)`` and
must return a unit-variance ``pl.Series``.

.. note::

   In v0.11.0 ``innovations`` is honoured only by :data:`Process.GBM`.
   Passing a non-Gaussian member with :data:`Process.OU`,
   :data:`Process.FRW`, or a callable process raises
   :class:`NotImplementedError` (FRW's Davies–Harte construction is
   only meaningful under Gaussian noise; OU's direct-σ parameterization
   tangles with the unit-variance contract).

.. code-block:: python

   from mktlib.data import Innovations, Process, monte_carlo

   # Heavy-tailed Student-t (df=5)
   sims = monte_carlo(
       Process.GBM, n_simulations=10_000, n=22, seed=42,
       drift=0.05, volatility=0.2, dt=1/252,
       innovations=Innovations.STUDENT_T, df=5,
   )

   # Empirical-residual bootstrap (distribution-free)
   import polars as pl
   residuals = (some_log_returns - some_log_returns.mean()) / some_log_returns.std()
   sims = monte_carlo(
       Process.GBM, n_simulations=10_000, n=22, seed=42,
       drift=0.05, volatility=0.2, dt=1/252,
       innovations=Innovations.BOOTSTRAP, residuals=residuals,
   )

Performance: ``independent_streams``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By default ``monte_carlo()`` derives one child RNG per simulation and
concatenates their outputs (``independent_streams=True``). This costs
~30 ms of per-call construction overhead at 10 000 simulations, which
typically dwarfs the actual sample-generation time. Setting
``independent_streams=False`` draws all noise in *one* batched sampler
call and reshapes — statistically identical (i.i.d. samples by
construction; backed by Kolmogorov–Smirnov integration tests across
every Process × Innovations combination), but **5–7× faster for
Gaussian / Student-t and ~60× faster for Bootstrap** at typical scales.

The trade-off is that the ``seed`` column reports a single
parent-derived seed shared across all simulations rather than one per
stream — fine for any consumer that doesn't introspect per-simulation
seeds. The metrics layer (:doc:`metrics`) defaults to
``independent_streams=False`` internally; reports and live trading
should too unless they specifically need per-stream replayability.

.. note::

   The two modes are statistically equivalent under i.i.d. innovations:
   the underlying RNG's i.i.d. property is what TestU01 BigCrush
   validates over long sequences, which the single-batch path uses
   directly. The per-stream approach instead relies on cycle-spacing
   independence across child seeds — a probabilistic argument the
   single-batch path doesn't need.
