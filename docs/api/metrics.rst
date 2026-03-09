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

.. autofunction:: mktlib.metrics.var

.. autofunction:: mktlib.metrics.cvar

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
