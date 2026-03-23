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

Multi-Symbol Backtesting
~~~~~~~~~~~~~~~~~~~~~~~~

Pass ``instrument_col`` to :func:`run` to backtest multiple instruments in a single
call. Returns a :class:`MultiBacktestResult` with O(1) per-instrument access:

.. code-block:: python

   # df has columns: symbol, date, open, close
   result = run(df, SmaCross(), instrument_col="symbol")

   # O(1) per-symbol access — returns a BacktestResult
   aapl = result["AAPL"]
   aapl.returns.columns   # ["date", "return"]

   # Iterate over symbols
   for symbol, bt in result.items():
       print(symbol, bt.trades.height)

   # Combined views (lazy-cached, symbol column first)
   result.returns.columns   # ["symbol", "date", "return"]

   # Equal-weight portfolio
   portfolio = result.returns.group_by("date").agg(pl.col("return").mean())

Types
-----

.. autoclass:: mktlib.backtest.BacktestResult
   :members:

.. autoclass:: mktlib.backtest.MultiBacktestResult
   :members:
   :special-members: __getitem__, __len__, __contains__

.. autoclass:: mktlib.backtest.Strategy
   :members:

   .. note::

      Strategies may optionally define an ``init(self, df) -> pl.DataFrame``
      method to enrich the DataFrame with indicator columns before signal
      evaluation. This hook is called after calendar filtering (if any) and
      before ``entry()``/``exit()`` resolution. It is **not** part of the
      Protocol — existing strategies without ``init`` continue to work unchanged.

.. autoclass:: mktlib.backtest.TradeSide
   :members:
   :undoc-members:

Conditions
----------

Conditions are frozen dataclasses that resolve to boolean ``pl.Expr``. They compose
with ``&`` (All), ``|`` (Any\_), and ``~`` (Not) operators.

.. code-block:: python

   from mktlib.backtest import Crossover, ValueGT

   # Compose with operators
   entry = Crossover("fast", "slow") & ValueGT("close", "sma_200")

.. autoclass:: mktlib.backtest.Condition
   :members:

.. autoclass:: mktlib.backtest.Crossover
   :members:

.. autoclass:: mktlib.backtest.Crossunder
   :members:

.. autoclass:: mktlib.backtest.ValueGT
   :members:

.. autoclass:: mktlib.backtest.ValueGTE
   :members:

.. autoclass:: mktlib.backtest.ValueLT
   :members:

.. autoclass:: mktlib.backtest.ValueLTE
   :members:

.. autoclass:: mktlib.backtest.IsRising
   :members:

.. autoclass:: mktlib.backtest.IsFalling
   :members:

.. autoclass:: mktlib.backtest.Custom
   :members:

Combinators
~~~~~~~~~~~

.. autoclass:: mktlib.backtest.All
   :members:

.. autoclass:: mktlib.backtest.Any_
   :members:

.. autoclass:: mktlib.backtest.Not
   :members:

Column Expressions
------------------

Column expressions build composable numeric ``pl.Expr`` trees for use with
``ValueGT``, ``ValueLT``, and their ``>=``/``<=`` variants. They support standard
arithmetic (``+``, ``-``, ``*``, ``/``, ``%``, unary ``-``), comparison operators
(``>``, ``>=``, ``<``, ``<=``), and mix freely with plain ``str`` column names
and ``float`` literals.

.. code-block:: python

   from mktlib.backtest import (
       Col, Lit, Pct, ValueGT, ValueLT, Crossover,
   )

   # Take-profit / stop-loss as an OR-combined exit
   tp = ValueGT("close", Pct("entry_sma", 5))   # close > sma * 1.05
   sl = ValueLT("close", Col("sma") - Col("vol") * 2)  # 2x vol below SMA
   exit_cond = tp | sl

   # Arithmetic expressions on both sides
   ValueGT(Col("fast") - Col("slow"), Lit(0.0))

   # Comparison operators on ColExpr return conditions directly
   entry = Col("rsi") > 70  # equivalent to ValueGT(Col("rsi"), Lit(70.0))

.. autoclass:: mktlib.backtest.ColExpr
   :members:

.. autoclass:: mktlib.backtest.Col
   :members:

.. autoclass:: mktlib.backtest.Lit
   :members:

.. autoclass:: mktlib.backtest.Pct
   :members:

Entry-Bar Anchoring with ``EntryRef``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When building TP/SL exits relative to the **entry price**, a plain column
reference doesn't work:

.. code-block:: python

   # BUG: resolves to close > close * 1.05 — always false
   ValueGT("close", Pct("close", 5.0))

The threshold needs to reference the entry bar's close, not the current bar's.
``EntryRef`` solves this by snapshotting a column at the entry signal bar and
forward-filling it through the position's lifetime:

.. code-block:: python

   from mktlib.backtest import EntryRef, Pct, ValueGT, ValueLT

   # TP: close > entry_close * 1.05
   tp = ValueGT("close", Pct(EntryRef("close"), 5.0))

   # SL: close < entry_close * 0.97
   sl = ValueLT("close", Pct(EntryRef("close"), -3.0))

Under the hood, the engine:

1. Detects ``EntryRef`` nodes in the exit condition tree
2. Computes ``_entry`` signals (pass 1)
3. Creates ``_entry_{col}`` snapshot columns: the column value where ``_entry``
   is true, ``null`` elsewhere, then ``forward_fill()``
4. Resolves the exit condition against the snapshot columns (pass 2)

``EntryRef`` composes freely with other expressions:

.. code-block:: python

   # ATR-based stop: 2 ATR below entry close
   sl = ValueLT("close", EntryRef("close") - Col("atr") * 2)

   # Multiple snapshots: entry close for TP, entry ATR for SL
   tp = ValueGT("close", Pct(EntryRef("close"), 5.0))
   sl = ValueLT("close", EntryRef("close") - EntryRef("atr") * 2)

.. autoclass:: mktlib.backtest.EntryRef
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
