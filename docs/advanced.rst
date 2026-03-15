Advanced Usage: Grid Search Optimization
=========================================

This guide walks through a realistic end-to-end workflow: optimizing an SMA
crossover strategy by grid-searching over indicator periods and exit
parameters, using ``mktlib.backtest``, ``mktlib.metrics``, and
`polars-talib <https://github.com/pola-rs/polars-talib>`_.

Prerequisites:

.. code-block:: bash

   pip install mktlib[data] polars-talib

The full runnable script is at ``scripts/grid_search_sma.py`` in the
repository.

Generate Synthetic Data
-----------------------

We generate 5 years of 1-minute OHLCV bars (~491k rows) by running
``mktlib.data.geometric_brownian_motion`` at second-level resolution, then
aggregating to 1-minute bars with ``ticks_to_ohlcv``.

The ``dt`` parameter controls the time step in annualised units. The default
``1/252`` gives one trading day per step. For 1-minute bars, set ``dt`` so
each tick represents one second of trading time:

.. code-block:: python

   from mktlib.data import geometric_brownian_motion, ticks_to_ohlcv

   # 1 tick = 1 second of US equity trading time
   #   252 trading days × 6.5 hours × 3600 seconds
   dt_1s = 1 / (252 * 6.5 * 3600)

   n_days = 5 * 252          # 5 years of trading days
   ticks_per_day = 390 * 60  # 390 minutes × 60 seconds

   gbm = geometric_brownian_motion(
       n=n_days * ticks_per_day + 1,
       base_price=100.0,
       drift=0.08,
       volatility=1.0,
       dt=dt_1s,
       seed=42,
   )
   ohlcv = ticks_to_ohlcv(gbm, bar_size=60, seed=43)  # 60 ticks → 1-minute bar

Pass ``drift`` and ``volatility`` as annualised values — the function scales
them internally via ``dt``. A ``volatility`` of ``1.0`` produces realistic
intra-minute price variation for typical equity-like data.

``ticks_to_ohlcv`` takes any DataFrame with a numeric column (the output of
any generator) and aggregates every ``bar_size`` steps into one OHLCV bar.
Pass ``column="value"`` for Ornstein-Uhlenbeck output; the default is
``"price"`` (GBM / fractional random walk).
Open/close are the first/last price in each bar, high/low span all intermediate
prices, and volume is synthetic lognormal (disable with ``volume=False``).
Bars that would be incomplete at the tail are dropped.

The result has columns ``bar, open, high, low, close, volume``. Timestamp
assignment is left to the caller — ``scripts/grid_search_sma.py`` shows how to
add business-day minute timestamps on top.
See :doc:`api/data` for the full data generation API.

Define a Parameterized Strategy
-------------------------------

The strategy is a frozen dataclass with ``fast_period`` and ``slow_period``
parameters. The optional ``init`` hook lets the strategy add its own indicator
columns to the DataFrame before signal evaluation — making it self-contained:

.. code-block:: python

   import polars as pl
   import polars_talib as plta
   from dataclasses import dataclass
   from mktlib.backtest import Crossover, Crossunder

   @dataclass(frozen=True, slots=True)
   class SmaCross:
       fast_period: int = 20
       slow_period: int = 50

       def init(self, df: pl.DataFrame) -> pl.DataFrame:
           return df.with_columns(
               plta.sma(pl.col("close"), timeperiod=self.fast_period).alias("fast_sma"),
               plta.sma(pl.col("close"), timeperiod=self.slow_period).alias("slow_sma"),
           )

       def entry(self) -> Crossover:
           return Crossover("fast_sma", "slow_sma")

       def exit(self) -> Crossunder:
           return Crossunder("fast_sma", "slow_sma")

With ``init``, the caller just passes raw OHLCV data — no external indicator step:

.. code-block:: python

   result = run(df, SmaCross(fast_period=10, slow_period=50))

Fetch the Risk-Free Rate
~~~~~~~~~~~~~~~~~~~~~~~~

To compute a meaningful Sharpe ratio we need the risk-free rate for the period.
``get_risk_free_rate`` returns the average 3-month T-bill yield (annualized
decimal) over the given date range:

.. code-block:: python

   from mktlib.rates import get_risk_free_rate

   rf = get_risk_free_rate(df["date"].min(), df["date"].max())

Grid Search over SMA Periods
-----------------------------

Search over fast periods (5--50, step 5) and slow periods (20--200, step 10),
skipping invalid combos where fast >= slow. For each combo, run the backtest
and score by Sharpe ratio:

.. code-block:: python

   import itertools
   from mktlib.backtest import run
   from mktlib.metrics import sharpe, cumulative_return  # sharpe accepts rf= kwarg

   fast_range = range(5, 55, 5)
   slow_range = range(20, 210, 10)

   results = []
   for fast, slow in itertools.product(fast_range, slow_range):
       if fast >= slow:
           continue

       strategy = SmaCross(fast_period=fast, slow_period=slow)
       result = run(df, strategy)
       ret = result.returns["return"]

       results.append({
           "fast_period": fast,
           "slow_period": slow,
           "sharpe": round(sharpe(ret, rf=rf), 4),
           "cumulative_return": round(cumulative_return(ret), 4),
           "n_trades": len(result.trades),
       })

   sma_results = pl.DataFrame(results).sort("sharpe", descending=True)
   print(sma_results.head(5))

Extract the best parameters for the next stage:

.. code-block:: python

   best = sma_results.row(0, named=True)
   best_fast = int(best["fast_period"])
   best_slow = int(best["slow_period"])

Add Take-Profit / Stop-Loss Optimization
-----------------------------------------

Extend the strategy with percentage-based exits using ``Pct``,
``PriceIsAbove``, and ``PriceIsBelow``. The TP triggers when price rises a
given percentage above the slow SMA; the SL triggers when price falls below:

.. code-block:: python

   from mktlib.backtest import Condition, Pct, PriceIsAbove, PriceIsBelow

   @dataclass(frozen=True, slots=True)
   class SmaCrossWithExits:
       fast_period: int = 20
       slow_period: int = 50
       tp_pct: float = 5.0
       sl_pct: float = 3.0

       def init(self, df: pl.DataFrame) -> pl.DataFrame:
           return df.with_columns(
               plta.sma(pl.col("close"), timeperiod=self.fast_period).alias("fast_sma"),
               plta.sma(pl.col("close"), timeperiod=self.slow_period).alias("slow_sma"),
           )

       def entry(self) -> Crossover:
           return Crossover("fast_sma", "slow_sma")

       def exit(self) -> Condition:
           tp = PriceIsAbove("close", Pct("slow_sma", self.tp_pct))
           sl = PriceIsBelow("close", Pct("slow_sma", -self.sl_pct))
           return Crossunder("fast_sma", "slow_sma") | tp | sl

``Pct("slow_sma", 5)`` resolves to ``slow_sma * 1.05`` — 5% above.
``Pct("slow_sma", -3)`` resolves to ``slow_sma * 0.97`` — 3% below.
Conditions compose with ``|`` (any) and ``&`` (all).

Now grid-search TP/SL percentages with the best SMA periods fixed:

.. code-block:: python

   tp_range = [i / 10 for i in range(1, 11)]   # 0.1% to 1.0%, step 0.1%
   sl_range = [i / 10 for i in range(1, 11)]

   results = []
   for tp_pct, sl_pct in itertools.product(tp_range, sl_range):
       strategy = SmaCrossWithExits(
           fast_period=best_fast,
           slow_period=best_slow,
           tp_pct=tp_pct,
           sl_pct=sl_pct,
       )
       result = run(df, strategy)
       ret = result.returns["return"]

       results.append({
           "tp_pct": tp_pct,
           "sl_pct": sl_pct,
           "sharpe": round(sharpe(ret, rf=rf), 4),
           "cumulative_return": round(cumulative_return(ret), 4),
           "n_trades": len(result.trades),
       })

   tp_sl_results = pl.DataFrame(results).sort("sharpe", descending=True)
   print(tp_sl_results.head(5))

Analyze Results
---------------

The two-stage approach keeps the search space manageable: ~160 combos for SMA
periods, then ~361 combos for TP/SL — instead of ~58,000 for a single
combined grid.

A few things to keep in mind:

- **Overfitting risk**: optimizing on the same data you evaluate on will
  overestimate real performance. Split your data into in-sample (for
  optimization) and out-of-sample (for validation).
- **Transaction costs**: the backtest engine uses fill-at-next-open semantics
  but does not model commissions or slippage. Strategies with many trades may
  look better than they are.
- **Metric choice**: Sharpe rewards consistency. Consider ``sortino`` (downside
  risk only) or ``omega`` as alternatives. See :doc:`api/metrics` for the full
  list.

For generating a complete tearsheet of the winning strategy, see
``mktlib.reports.html()`` in :doc:`api/reports`.

Multi-Symbol Grid Search
------------------------

When optimizing a strategy across multiple symbols, use ``instrument_col`` to
run all symbols in a single backtest call. Per-symbol returns let you
evaluate each ticker independently or aggregate into a portfolio:

.. code-block:: python

   import itertools
   import polars as pl
   from mktlib.backtest import run
   from mktlib.metrics import sharpe

   symbols_df = ...  # DataFrame with columns: symbol, date, open, close

   fast_range = range(5, 55, 5)
   slow_range = range(20, 210, 10)

   results = []
   for fast, slow in itertools.product(fast_range, slow_range):
       if fast >= slow:
           continue

       strategy = SmaCross(fast_period=fast, slow_period=slow)
       result = run(symbols_df, strategy, instrument_col="symbol")

       # Per-symbol Sharpe — O(1) access via result[symbol]
       for sym in result.symbols:
           sym_ret = result[sym].returns["return"]
           results.append({
               "symbol": sym,
               "fast_period": fast,
               "slow_period": slow,
               "sharpe": round(sharpe(sym_ret), 4),
           })

       # Or equal-weight portfolio Sharpe
       portfolio = result.returns.group_by("date").agg(
           pl.col("return").mean()
       )["return"]
       results.append({
           "symbol": "PORTFOLIO",
           "fast_period": fast,
           "slow_period": slow,
           "sharpe": round(sharpe(portfolio), 4),
       })

   grid = pl.DataFrame(results).sort("sharpe", descending=True)

This avoids the outer loop over symbols that single-symbol backtesting
would require, while keeping each symbol's indicator computation isolated.

See Also
--------

- :doc:`quickstart` — basic API usage
- :doc:`api/backtest` — full backtest API reference
- :doc:`api/metrics` — all available financial metrics
- :doc:`api/data` — synthetic data generators
