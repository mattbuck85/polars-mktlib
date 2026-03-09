Backtest
========

Vectorized backtesting engine with fill-at-next-open semantics, exchange calendar
integration, and session-boundary position management.

Included in the base ``pip install mktlib`` — no extra dependencies.

Overview
--------

The engine runs a signal-driven backtest where:

- A **Strategy** defines ``entry()`` and ``exit()`` methods returning composable **Conditions**
- Conditions resolve to boolean Polars expressions evaluated over the full DataFrame
- Fills use **next-bar-open** semantics: signal at bar *t* → market order fills at bar *t+1*'s open
- Optional **calendar** filters to market hours; ``flatten_eod=True`` force-closes positions at session end

Return Model
~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Bar type
     - Formula
   * - Entry bar (*t+1*)
     - ``(close - open) / open``
   * - Middle bars
     - ``close / prev_close - 1``
   * - Exit bar
     - ``(open - prev_close) / prev_close``
   * - Session-forced exit (``flatten_eod``)
     - ``(open - prev_close) / prev_close`` for held positions; ``0`` for same-bar entry+exit

Engine
------

.. autofunction:: mktlib.backtest.run

Types
-----

.. autoclass:: mktlib.backtest.BacktestResult
   :members:

.. autoclass:: mktlib.backtest.Strategy
   :members:

.. autoclass:: mktlib.backtest.TradeSide
   :members:
   :undoc-members:

Conditions
----------

Conditions are frozen dataclasses that resolve to boolean ``pl.Expr``. They compose
with ``&`` (All), ``|`` (Any\_), and ``~`` (Not) operators.

.. code-block:: python

   from mktlib.backtest import Crossover, PriceIsAbove

   # Compose with operators
   entry = Crossover("fast", "slow") & PriceIsAbove("close", "sma_200")

.. autoclass:: mktlib.backtest.Condition
   :members:

.. autoclass:: mktlib.backtest.Crossover
   :members:

.. autoclass:: mktlib.backtest.Crossunder
   :members:

.. autoclass:: mktlib.backtest.PriceIsAbove
   :members:

.. autoclass:: mktlib.backtest.PriceIsBelow
   :members:

.. autoclass:: mktlib.backtest.IsRising
   :members:

.. autoclass:: mktlib.backtest.IsFalling
   :members:

Combinators
~~~~~~~~~~~

.. autoclass:: mktlib.backtest.All
   :members:

.. autoclass:: mktlib.backtest.Any_
   :members:

.. autoclass:: mktlib.backtest.Not
   :members:
