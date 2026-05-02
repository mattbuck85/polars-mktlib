Metrics
=======

Standalone financial metric functions operating on Polars return series.
No dependencies beyond polars — included in the base ``pip install mktlib``.

Overview
--------

Two usage patterns:

1. **Direct functions** — call individual metric functions with a ``pl.Series``
2. **Dispatcher** — use ``calculate_metric()`` with a ``Metric`` enum value

.. code-block:: python

   from mktlib.metrics import sharpe, cumulative_return, drawdown_series

   sr = sharpe(returns, rf=0.05)
   cr = cumulative_return(returns)
   dd = drawdown_series(returns)

.. code-block:: python

   from mktlib.metrics import calculate_metric, Metric

   sr = calculate_metric(Metric.SHARPE, returns, rf=0.05)

Metric Enum
-----------

.. autoclass:: mktlib.metrics.Metric
   :members:
   :undoc-members:

Dispatcher
----------

.. autofunction:: mktlib.metrics.calculate_metric

Return Metrics
--------------

.. autofunction:: mktlib.metrics.cumulative_return

.. autofunction:: mktlib.metrics.cagr

.. autofunction:: mktlib.metrics.annualized_volatility

Risk-Adjusted Ratios
--------------------

.. autofunction:: mktlib.metrics.sharpe

.. autofunction:: mktlib.metrics.sortino

.. autofunction:: mktlib.metrics.omega

Tail Risk
---------

The :func:`mktlib.metrics.var` and :func:`mktlib.metrics.cvar` functions
support three estimators via the ``method=`` kwarg:

* ``"historical"`` (default) — empirical α-quantile of the input series.
  Non-parametric, no distributional assumption, no horizon — answers
  *"how bad were the worst α of bars?"*.
* ``"gaussian"`` — closed-form parametric estimator under fitted GBM.
  Cheap and exact under Gaussian innovations.
* ``"monte_carlo"`` — simulation-based; required for non-Gaussian
  innovations or when path-dependent extensions are added.

**Closed-form Gaussian VaR / CVaR (the math).**  Fit
:math:`\hat{\mu} = \overline{\log(1+r_t)} / \Delta t` and
:math:`\hat{\sigma} = \mathrm{std}(\log(1+r_t)) / \sqrt{\Delta t}` from
the input series, then the :math:`H`-bar log-return is

.. math::

   \log(1 + R_H) \;\sim\; N\!\bigl(\hat{\mu} \, H \, \Delta t, \;
                                    \hat{\sigma}^2 \, H \, \Delta t\bigr)

so the analytic VaR / CVaR (in *simple-return* units) are

.. math::

   \mathrm{VaR}_\alpha
     &= \exp\!\bigl(\hat{\mu} H \Delta t
                    + \hat{\sigma} \sqrt{H \Delta t} \, \Phi^{-1}(\alpha)\bigr) - 1 \\
   \mathrm{CVaR}_\alpha
     &= \exp\!\bigl(\hat{\mu} H \Delta t
                    - \hat{\sigma} \sqrt{H \Delta t} \;
                      \tfrac{\phi(z_\alpha)}{\alpha}\bigr) - 1

where :math:`\Phi^{-1}` is the standard-normal inverse CDF (computed
via :class:`statistics.NormalDist`), :math:`\phi` is the standard-normal
PDF, and :math:`z_\alpha = \Phi^{-1}(\alpha)`. Both are typically negative.

**Square-root-of-time scaling.** The variance of the cumulative
log-return is *linear* in :math:`H` under i.i.d. innovations
(:math:`\mathrm{Var}(\sum r_t) = H \sigma^2 \Delta t`), so the
volatility component scales as :math:`\sqrt{H}` — Basel's
"square-root-of-time" rule. Drift scales linearly in :math:`H`. At
short horizons :math:`\hat{\sigma}\sqrt{H \Delta t}` dominates; at long
horizons drift catches up and the signal-to-noise ratio improves as
:math:`\sqrt{H}\cdot \hat{\mu} / \hat{\sigma}`.

**When does Monte Carlo earn its compute budget?** Under pure GBM with
Gaussian innovations, ``method="monte_carlo"`` returns the *closed-form*
number plus sampling noise — running 10 000 paths to recover an analytic
quantity is theatre. MC pays off when:

1. **Innovations stop being Gaussian** — Student-t (heavier tails),
   bootstrapped empirical residuals, or any callable noise source. Sums
   of Student-t aren't Student-t (only the Gaussian is stable under
   summation), so no analytic form exists.
2. **Path-dependent measures** — drawdown over horizon, time-to-barrier,
   expected shortfall conditional on a within-horizon event. These are
   functionals of the whole path, not the endpoint.
3. **Autocorrelated returns / fBm** — variance of the H-bar sum is no
   longer :math:`H \sigma^2`. Under fBm with Hurst :math:`h`, variance
   scales as :math:`H^{2h}`: sub-diffusive (:math:`h < 0.5`) tightens
   long-horizon tails; super-diffusive (:math:`h > 0.5`) fattens them.

**Practical caveats.**

- Single-bar VaR (``horizon=1``) under Gaussian innovations: the
  ``"monte_carlo"`` and ``"gaussian"`` paths return the same number
  modulo sampling noise. Use ``"gaussian"`` to skip the simulation tax.
- Tail-size rule of thumb: at small :math:`\alpha` (e.g. ``0.01``) use
  ``n_simulations >= 200/alpha`` so the CVaR tail (the worst
  :math:`\alpha` fraction of paths) is well-populated.
- Identical *seed* values across :func:`var` and :func:`cvar` calls
  produce identical sample paths under the perf path
  (``independent_streams=False``).  Use the same seed in both calls
  if you want their tail numbers to come from the same simulation —
  no shared cache needed; the deterministic RNG does the work.

.. autofunction:: mktlib.metrics.var

.. autofunction:: mktlib.metrics.cvar

Forward-Looking Estimators
--------------------------

.. autofunction:: mktlib.metrics.simulate_metric

.. autofunction:: mktlib.metrics.monte_carlo_paths

The :func:`monte_carlo_paths` helper is the entry point used by
:doc:`reports` to render the Monte Carlo simulation-paths chart. It
runs MC once and returns the *full* sims frame (long-form
``simulation, seed, step, price`` — base price 1.0, scale by initial
equity for absolute units).  No caching: at the perf-path defaults
(10 k × 22 ≈ 10–15 ms per batch) re-running on every call is cheaper
than maintaining a content-fingerprint cache.  Callers who want the
chart paths and the VaR / CVaR numbers to come from the *same*
simulation should pass an identical *seed* to every call —
deterministic seeding gives byte-for-byte identical samples without
any explicit batching.

Win/Loss
--------

.. autofunction:: mktlib.metrics.win_rate

.. autofunction:: mktlib.metrics.payoff_ratio

.. autofunction:: mktlib.metrics.profit_factor

.. autofunction:: mktlib.metrics.kelly_criterion

Drawdown
--------

.. autofunction:: mktlib.metrics.drawdown_series

.. autofunction:: mktlib.metrics.avg_drawdown

.. autofunction:: mktlib.metrics.longest_drawdown_days
