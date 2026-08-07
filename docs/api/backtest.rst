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
- Optional **calendar** filters to market hours; ``flatten=...`` force-closes positions on a schedule

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
   * - Session-forced exit (``flatten=...``)
     - ``(open - prev_close) / prev_close`` for held positions; ``0`` for same-bar entry+exit
   * - Bracket exit bar (``bracket=Bracket(...)``)
     - ``(bracket_fill - prev_close) / prev_close``; ``(bracket_fill - open) / open`` when the bracket closes the position on the bar it opened
   * - Bars after a bracket exit, up to the signal exit
     - ``0`` — the position is closed and does not re-open within the block
   * - Any fill bar, with ``cost=Cost(...)``
     - the formula above, minus ``cost_bps / 1e4`` (holding bars unchanged; a bracket exit on the entry bar pays twice, since two fills land there)

Flatten Schedules
~~~~~~~~~~~~~~~~~

Pass ``flatten=...`` to :func:`run` to force-close open positions on a
schedule. Requires a *calendar* — sessions, and therefore session closes, are
defined by the calendar.

.. code-block:: python

   from mktlib.backtest import FlattenSchedule, Weekday, run
   from mktlib.scheduling import get_calendar

   cal = get_calendar("XNYS")

   run(df, strat, calendar=cal, flatten="eod")   # every session's last bar
   run(df, strat, calendar=cal, flatten="eow")   # last session of each week

   run(df, strat, calendar=cal, flatten=FlattenSchedule(
       days={Weekday.FRI},
       minutes_before_close=30,             # flatten at 15:30
       block_entry_minutes_before_close=60, # open nothing after 15:00
   ))

``"eod"`` and ``True`` are :class:`FlattenSchedule` ``(days="daily")``;
``"eow"`` is ``(days="weekly")``. ``False`` and ``None`` disable flattening.

How the offsets behave:

**The offsets are wall-clock, not bar counts.** ``minutes_before_close=N``
selects the last bar *starting at or before* ``close - N``. On a 15-minute
grid whose final bar starts at 15:45 against a 16:00 close, every ``N`` below
15 resolves to that same bar. Both offsets are measured against each session's
own close, so an early-closing half-day needs no special-casing.

**``block_entry_minutes_before_close`` defaults to ``minutes_before_close``.**
At that default every in-session bar at or after the flatten bar's cutoff is
blocked, so no position can open after the flatten. With
``block_entry_minutes_before_close=0`` no bar is blocked, so an entry signal
after the flatten bar opens a position that carries to the next session.

**A signal inside the block window is dropped, not deferred.** Deferring it
would fill on the next session's open, hours stale. For an edge condition such
as :class:`Crossover`, which fires once, that means the signal is gone.

``days="weekly"`` selects the session on which the **exchange calendar** closes
each ISO week, which makes it holiday-aware for free: when Friday is a holiday,
Thursday is the week's closing session and takes the flatten. An explicit
``days={Weekday.FRI}`` is deliberately literal — that same week does not flatten
at all. ``"weekly"`` selects the week's closing session; the weekday set selects
those weekdays and nothing else.

The distinction that matters for walk-forward work is that selection asks the
*calendar*, not the data. If a week's closing session carries no bars, that week
is not flattened — the flatten does not fall back to whatever bar happens to be
last. Otherwise a fold ending on a Wednesday would flatten there while the
pooled run over the same bars would not, and the two would disagree over
identical inputs. An interior gap of that kind is logged at ``WARNING``; a frame
that simply ends mid-week is not, because from inside the engine the two cases
are indistinguishable.

.. autoclass:: mktlib.backtest.FlattenSchedule
   :members:

.. autoclass:: mktlib.backtest.Weekday
   :members:

.. note::

   ``flatten_eod=True`` remains fully supported and is exactly equivalent to
   ``flatten="eod"``. ``flatten=`` additionally accepts ``"eow"`` and a
   :class:`FlattenSchedule`, which ``flatten_eod`` cannot express. Setting both
   raises rather than picking a winner.

Transaction Costs
~~~~~~~~~~~~~~~~~

Pass ``cost=Cost(...)`` to :func:`run` to charge per-side transaction costs.
Costs are stated in **basis points of notional** because the engine is
share-count free — it composes price relatives only, so a per-share or
per-order fee schedule cannot be expressed without a quantity. Convert at
the call site: ``bps = 1e4 * fee_per_share / expected_fill_price``.

.. code-block:: python

   from mktlib.backtest import Cost, run

   # 1 bp commission + 0.5 bp assumed slippage, each side
   result = run(df, strategy, cost=Cost(commission_bps=1.0, slippage_bps=0.5))

   # flat commission plus a per-bar slippage column the strategy computed
   result = run(df, strategy, cost=Cost(commission_bps=1.0, slippage_col="half_spread_bps"))

The charge is applied **at the fill** — on the entry bar, the exit bar, the
limit-fill bar and the session-forced-exit bar — in both ``returns`` and
``trades.pnl``. It is never applied as a post-hoc transform on the returns
series, which cannot see trade boundaries, and never multiplied by the trade
side: a short pays the same haircut a long does.

``Cost()`` with its all-zero defaults is an exact no-op, so adding the
parameter cannot move an existing backtest.

.. note::

   Costs are primitives only — floats and a column name, never a callable.
   A closure has no stable identity, so a callable cost model would be
   invisible to a caller's cache key and two runs with colliding keys could
   serve each other's results.

.. note::

   Two known reconciliation gaps between ``returns`` and ``trades.pnl``:
   a position still open on the final bar produces no trade row (pre-existing),
   and where a limit exit fires on the entry-fill bar the returns series
   charges one side while ``trades.pnl`` charges two.

.. autoclass:: mktlib.backtest.Cost
   :members:

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

Portfolio Weights
^^^^^^^^^^^^^^^^^

Pass ``instrument_weights`` to collapse per-symbol results into a single
weighted ``(date, return)`` portfolio series:

.. code-block:: python

   result = run(
       df_multi, strategy,
       instrument_weights={"TQQQ": 0.5, "MSFT": 0.1, "AAPL": 0.1, ...},
   )
   result.returns   # (date, return) — weighted portfolio series

Weights accept either a ``Mapping[str, float]`` or a ``pl.DataFrame``
with columns ``(instrument, weight)``. Proportional and normalized
inputs are equivalent — mktlib renormalizes at aggregation. When a
symbol is missing on a given date, its weight drops from that date's
denominator (dynamic renormalization), keeping the portfolio series
continuous across alignment gaps.

When ``instrument_weights`` is supplied without an explicit
``instrument_col``, mktlib defaults to ``"instrument"`` (matching the
canonical portfolio-weights schema). Public schema constants
(``PORTFOLIO_WEIGHTS_COLUMNS``, ``INSTRUMENT_COLUMN``, ``WEIGHT_COLUMN``)
live in :mod:`mktlib.backtest._weights`.

.. autoexception:: mktlib.backtest.InvalidPortfolioWeights
   :members:

.. autofunction:: mktlib.backtest.to_portfolio_weights_df

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

Same-Bar Fills: Take-Profit / Stop-Loss
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Wrap an exit condition in :class:`Limit` to fill on the *same* bar the
condition fires, at the limit price — instead of the default fill-at-
next-open. Designed for TP/SL strategies where the fill price is known
in advance.

.. code-block:: python

   from mktlib.backtest import Col, Limit, Lit, ValueGTE, ValueLTE

   # Take-profit: exit when high >= 103, fill at 103
   tp_exit = Limit(ValueGTE(Col("high"), Lit(103.0)))

   # Stop-loss: exit when low <= 95, fill at 95
   sl_exit = Limit(ValueLTE(Col("low"), Lit(95.0)))

The fill price defaults to the RHS of the wrapped comparison (TP/SL
idiom ``high >= TP`` → fill at ``TP``). Pass ``price=`` explicitly for
trailing stops or decoupled trigger/fill:

.. code-block:: python

   trailing_exit = Limit(
       ValueLTE(Col("low"), Col("trailing_stop")),
       price=Col("trailing_stop"),
   )

.. note::

   Only the *top-level* ``Limit`` wrapper is recognized. Nested use
   inside ``All`` / ``Any_`` / ``Not`` behaves as a plain boolean.

   ``Limit`` expresses **one** same-bar exit. For a *pair* of protective
   levels resting against every position, use ``bracket=Bracket(...)``
   below — it owns the position lifecycle rather than just the fill
   price, and resolves the same-bar both-touch case explicitly. The two
   cannot be combined; passing both raises ``NotImplementedError``.

.. autoclass:: mktlib.backtest.Limit
   :members:

Bracket Exits
~~~~~~~~~~~~~

Pass ``bracket=Bracket(...)`` to :func:`run` to rest a take-profit and a
stop-loss against every position from its **entry fill bar** onward. The
first leg to be tagged closes the position *inside* that bar, ahead of
whatever the strategy's ``exit()`` condition would have done at the next
bar's open.

.. code-block:: python

   from mktlib.backtest import Bracket, run

   # 2% target, 1% stop, both as fractions of the entry fill price
   result = run(df, strategy, bracket=Bracket(take_profit=0.02, stop_loss=0.01))

   # Absolute levels from a column the strategy computed, latched at the
   # entry signal bar (e.g. close + atr * mult)
   result = run(df, strategy, bracket=Bracket(stop_loss="atr_stop"))

   # Re-latch both legs on every entry signal that fires while held
   result = run(
       df, strategy, bracket=Bracket(take_profit=0.02, anchor="signal")
   )

Requires ``high`` and ``low`` columns. The fill table mirrors a
conventional event-driven OHLC broker exactly — a long bracket is a
sell limit plus a sell stop, a short bracket the mirror:

.. list-table::
   :header-rows: 1

   * - Side
     - Leg
     - Trigger
     - Fill price
   * - long
     - take-profit
     - ``high >= tp``
     - ``max(open, tp)``
   * - long
     - stop-loss
     - ``low <= sl``
     - ``min(open, sl)``
   * - short
     - take-profit
     - ``low <= tp``
     - ``min(open, tp)``
   * - short
     - stop-loss
     - ``high >= sl``
     - ``max(open, sl)``

Clamping against the bar's own open is what makes a gap honest: a long
stop at 95 on a bar that opens at 90 fills at 90, not at 95.

.. warning::

   **Within-bar ordering is unknowable from OHLC.** A bar that reaches
   both levels records nothing about which came first. ``both_touch`` is
   a stated assumption, not a measurement; re-run with both settings to
   bound the true result.

   The default ``both_touch="stop_first"`` **deliberately diverges from
   submission-order OCO**: a live bracket is commonly an OCO pair whose
   take-profit leg is submitted first and filled in submission order, so
   the realized policy on such a bar is ``"take_profit_first"``. mktlib
   defaults to the pessimistic resolution; pass
   ``both_touch="take_profit_first"`` to reproduce it.

.. note::

   A bracket exit does **not** re-arm the entry signal. The position stays
   closed for the rest of the block the entry opened; the next trade needs
   the ``exit()`` condition to fire and a fresh entry signal after it.
   Live, a still-true entry condition would re-enter on the following bar.

   Not supported together with ``short_strategy=`` (the dual path merges
   only ``_position`` and ``_side``, so the long leg's levels would be
   evaluated against the short leg's position) or with a ``Limit(...)``
   exit condition. Both raise ``NotImplementedError``.

Level Anchoring
^^^^^^^^^^^^^^^

``anchor`` states which entry signal the levels are measured from. It is a
property of the bracket, not of the position machinery: both policies open,
hold and close exactly the same positions, and only the protective levels
differ.

``anchor="position"`` (the default)
   Both legs are latched once, on the entry that opened the position, and
   hold for the life of the trade. An entry signal that fires while the
   position is already open is suppressed by the position machinery and
   does not move them.

``anchor="signal"``
   Every later entry signal that fires while the position is still open
   re-latches both legs.

.. code-block:: python

   Bracket(take_profit=0.02, stop_loss=0.01, anchor="signal")

A re-signal on bar *k* is observed at that bar's close, so the level in force
*during* bar *k* is still the previous one and the new level applies from
*k + 1*. A ``float`` leg re-anchors to ``open[k + 1]`` — the price a fresh
entry on that signal would have filled at, by the engine's own
fill-at-next-open rule — and a ``str`` leg re-reads its column on bar *k*.
A leg tagged on bar *k* therefore closes the position before the re-anchor
takes effect.

Re-anchoring moves the levels and nothing else. The position opened once, so
trade P&L still measures from the **original** entry fill, and a re-signal
never re-opens or extends a block — block boundaries are fixed before the
bracket is applied.

.. note::

   Under a flatten schedule, a re-signal on a **flatten bar** does not
   re-anchor: the engine has already force-closed the position at that bar's
   open. The signal is deferred to the next bar, where it opens a fresh
   position and latches that position's own levels.

   The gate is the flatten bar, not the session's last bar. They coincide
   only at ``minutes_before_close=0``. With
   ``FlattenSchedule(minutes_before_close=30)`` the flatten bar is 15:30 and
   the session's last bar is 15:59, and a re-signal at 15:45 on a held
   position re-anchors like any other. With no flatten, a re-signal on a
   session's last bar anchors a ``float`` leg to the next session's opening
   price, so an overnight gap carries the levels with it.

   An entry condition that is **level**-triggered rather than edge-triggered
   stays true on consecutive bars, so under ``anchor="signal"`` it re-latches
   on every one of them and the bracket becomes a trailing one.
   ``Crossover`` / ``Crossunder`` are edge-triggered; ``ValueGT`` and its
   siblings are not. This is the same edge-versus-level distinction that
   governs whether an ``EntryRef`` level is fixed or trailing.

   ``anchor`` governs bracket levels only. ``EntryRef`` snapshots are
   unaffected and continue to latch at the entry that opened the position,
   under either policy.

.. autoclass:: mktlib.backtest.Bracket
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
3. Resolves which of those signals actually **open a position**, by walking the
   entry/exit chain forward — a signal arriving while a position is still open
   opens nothing and must not anchor
4. Creates ``_entry_{col}`` snapshot columns: the column value at each realized
   entry, ``null`` elsewhere, then ``forward_fill()``
5. Resolves the exit condition against the snapshot columns (pass 2)

.. note::

   The snapshot latches on entry signals that actually **open a position**. A
   signal that fires while a position is already open is suppressed by the
   position machinery, and the anchor does not follow it — the level stays put
   for the life of the trade, which is what makes an ``EntryRef`` take-profit a
   *fixed* target rather than a trailing one.

   Two consequences worth knowing:

   * If an entry signal lands on the very bar an exit would have fired, the
     entry wins and the position stays open. The exit does not happen and the
     trade continues on its **original** anchor.
   * Under ``flatten_eod``, an entry deferred across a session boundary anchors
     on the bar it actually opens on, not the one the signal fired on.

   Before 0.14.0 the snapshot was filled from every raw signal, so it moved
   mid-trade. Pin ``mktlib==0.13.2`` to reproduce results from that behaviour.

.. tip::

   Everything on the threshold side is read **once, at the entry bar** — so
   arithmetic over several snapshots stays a fixed level:

   .. code-block:: python

      # Fixed stop: 2 ATR below the entry bar's close.
      sl = ValueLT("close", EntryRef("close") - EntryRef("atr") * 2)

      # TRAILING stop: re-reads atr every bar, so the level moves with it.
      sl = ValueLT("close", EntryRef("close") - Col("atr") * 2)

   Both are supported and both are correct; they are simply different
   instruments. The first can be resolved by a faster search, because the
   barrier does not move.

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

Calendar filtering adds a schedule join for market-hours masking. A flatten
schedule adds the mask construction on top of that join.

.. note::

   Numba requires ahead-of-time compilation (~0.6s on first call, cached to disk
   thereafter). The Polars engine requires no extra dependencies. Benchmark
   scripts live in ``scripts/bench_*.py``.
