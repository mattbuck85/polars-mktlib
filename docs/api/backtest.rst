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

Price Expressions
-----------------

Price expressions build composable numeric ``pl.Expr`` trees for use with
``PriceIsAbove`` and ``PriceIsBelow``. They support standard arithmetic
(``+``, ``-``, ``*``, ``/``, ``%``, unary ``-``) and mix freely with plain
``str`` column names and ``float`` literals.

.. code-block:: python

   from mktlib.backtest import (
       Col, Lit, Pct, PriceIsAbove, PriceIsBelow, Crossover,
   )

   # Take-profit / stop-loss as an OR-combined exit
   tp = PriceIsAbove("close", Pct("entry_sma", 5))   # close > sma * 1.05
   sl = PriceIsBelow("close", Col("sma") - Col("vol") * 2)  # 2x vol below SMA
   exit_cond = tp | sl

   # Arithmetic expressions on both sides
   PriceIsAbove(Col("fast") - Col("slow"), Lit(0.0))

.. autoclass:: mktlib.backtest.PriceExpr
   :members:

.. autoclass:: mktlib.backtest.Col
   :members:

.. autoclass:: mktlib.backtest.Lit
   :members:

.. autoclass:: mktlib.backtest.Pct
   :members:

Performance
-----------

Benchmark results for a MACD crossover strategy on synthetic minute-resolution
OHLCV data (491,400 rows / 5 years). Signal resolution uses Polars in all cases;
only the position-tracking / returns computation differs.

.. list-table::
   :header-rows: 1

   * - Engine
     - Time
     - vs Polars
   * - **Polars** (vectorized ``with_columns``)
     - 0.025s
     - baseline
   * - Numpy (vectorized array ops)
     - 0.033s
     - 1.3x slower
   * - Pandas (vectorized)
     - 0.223s
     - 8.9x slower
   * - Python for-loop over numpy arrays
     - 0.206s
     - 8.2x slower
   * - Numba JIT (warm, ``@njit``)
     - 0.009s
     - 2.8x faster

Calendar filtering adds ~8ms for schedule-join market-hours masking.
``flatten_eod`` adds ~4ms on top.

.. note::

   Numba requires ahead-of-time compilation (~0.6s on first call, cached to disk
   thereafter). The Polars engine is the best default — no extra dependencies and
   competitive performance. Benchmark scripts live in ``scripts/bench_*.py``.

Combinators
~~~~~~~~~~~

.. autoclass:: mktlib.backtest.All
   :members:

.. autoclass:: mktlib.backtest.Any_
   :members:

.. autoclass:: mktlib.backtest.Not
   :members:
