Reports
=======

Performance metrics and HTML tearsheet generation.

Two entry points:

* :func:`mktlib.reports.html` — generates an interactive HTML tearsheet
  (returns the string or writes to disk).
* :func:`mktlib.reports.metrics` — computes the same metric set without
  the rendering overhead, returns a :class:`MetricsResult` dataclass.

Both accept a daily ``returns`` series; sub-daily input is automatically
collapsed to one geometrically-compounded value per date. The accepted
input types — ``pl.Series``, ``pl.DataFrame`` with ``date`` and ``return``
columns, or any pandas-like object with a ``DatetimeIndex`` — are
documented under :func:`mktlib.reports.html`.

.. autofunction:: mktlib.reports.html

.. autofunction:: mktlib.reports.metrics

Result and Configuration
------------------------

.. autoclass:: mktlib.reports.MetricsResult
   :members:
   :undoc-members:

.. autoclass:: mktlib.reports.ReportConfig
   :members:
   :undoc-members:

.. autoclass:: mktlib.reports.TradeMetrics
   :members:
   :undoc-members:

.. autoclass:: mktlib.reports.DrawdownInfo
   :members:
   :undoc-members:

Forward-Looking Risk (Monte Carlo)
----------------------------------

The :class:`MonteCarloConfig` dataclass enables an opt-in Monte Carlo
estimation path on top of the standard tearsheet. When ``enabled=True``,
:func:`html` and :func:`metrics` populate two new
:class:`MetricsResult` fields — ``mc_var`` and ``mc_cvar`` — using the
simulation-based estimator from :doc:`metrics`, and (in :func:`html`)
render a *Monte Carlo Forward Paths* chart showing the spaghetti subset,
α/2 / 1−α/2 percentile band, and median path anchored at the last
historical equity value over forward business days.

.. autoclass:: mktlib.reports.MonteCarloConfig
   :members:
   :undoc-members:

**Three MC runs per report, one shared seed.**  Internally :func:`html`
and :func:`metrics` run :func:`mktlib.metrics.monte_carlo_paths` for
the chart frame plus two :func:`mktlib.metrics.simulate_metric` calls
(VaR + CVaR), all threaded through one fixed seed so they produce
identical sample paths.  When ``mc_config.seed`` is ``None`` the
driver mints one OS-derived seed up front and reuses it across all
three calls; otherwise the user-supplied seed is honoured.  This
keeps the chart and the displayed numbers mutually consistent without
the complexity of a content-fingerprint cache — at the perf-path
defaults (~10–15 ms per MC batch) the triplet adds 30–45 ms total,
invisible inside the typical tearsheet render.

.. code-block:: python

   from mktlib.data import Innovations
   from mktlib.reports import html, MonteCarloConfig

   # Heavy-tailed Student-t innovations, 21-bar (≈1 month) horizon
   html(
       returns,
       title="Forward-Looking Tearsheet",
       output="tearsheet.html",
       mc_config=MonteCarloConfig(
           enabled=True,
           horizon=21,
           n_simulations=10_000,
           innovations=Innovations.STUDENT_T,
           df=5,
           seed=42,
           alpha=0.05,
           n_paths_displayed=100,
           exchange="XNYS",
       ),
   )

The chart's x-axis spans roughly 60 historical bars plus the forecast
horizon, with forward dates generated via
``mktlib.scheduling.get_calendar(exchange).session_offset(...)`` so
weekends and holidays are skipped.

**Displayed-subset sampling.** When ``n_simulations > n_paths_displayed``
the chart renders only a subset of paths.  The subset is drawn via
*uniform random sampling without replacement*, seeded from the MC
run's parent seed for reproducibility — this is preferred over taking
the prefix ``simulation < n_paths_displayed`` because the prefix
relies on the underlying RNG's i.i.d. property over consecutive
samples (mostly fine for modern PRNGs but a known MC-literature
footgun re: warm-up bias).  Distributional equivalence between the
displayed subset and the full population is regression-tested via
two-sample Kolmogorov–Smirnov.

.. note::

   When ``enabled=False`` (the default), the ``mc_*`` fields stay
   ``None`` and the tearsheet renders identically to v0.10.x. The
   ``MonteCarloConfig`` import has zero runtime cost on
   ``[reports]``-only installs — the ``mktlib.data`` dependency is
   loaded lazily inside the MC code path.
