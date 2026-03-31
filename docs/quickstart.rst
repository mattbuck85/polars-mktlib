Quick Start
===========

This guide covers the subpackages in mktlib with short, runnable examples.

Vectorized Backtesting
----------------------

Run signal-driven backtests with fill-at-next-open semantics:

.. code-block:: python

   import polars as pl
   import polars_talib as plta
   from dataclasses import dataclass
   from mktlib.backtest import run, Crossover, Crossunder

   @dataclass(frozen=True, slots=True)
   class SmaCross:
       fast_period: int = 20
       slow_period: int = 50

       def init(self, df: pl.DataFrame) -> pl.DataFrame:
           """Add SMA indicator columns before signal evaluation."""
           return df.with_columns(
               plta.sma(pl.col("close"), timeperiod=self.fast_period).alias("fast_sma"),
               plta.sma(pl.col("close"), timeperiod=self.slow_period).alias("slow_sma"),
           )

       def entry(self) -> Crossover:
           return Crossover("fast_sma", "slow_sma")

       def exit(self) -> Crossunder:
           return Crossunder("fast_sma", "slow_sma")

   # df needs only: date, open, close — init() adds the indicators
   result = run(df, SmaCross())
   result.returns   # DataFrame[date, return]
   result.trades    # DataFrame[entry_date, exit_date, pnl, bars_held]

With exchange calendar and session-boundary management:

.. code-block:: python

   from mktlib.scheduling import get_calendar

   cal = get_calendar("NYSE")

   # Filter to market hours, force-close at session end
   result = run(df, SmaCross(), calendar=cal, flatten_eod=True)

Conditions compose with ``&``, ``|``, ``~``:

.. code-block:: python

   from mktlib.backtest import ValueGT

   entry = Crossover("fast", "slow") & ValueGT("close", "sma_200")

Column expressions (``Col``, ``Lit``, ``Pct``) let you build dynamic exit
levels without any engine changes — combine ``ValueGT`` and
``ValueLT`` with ``|`` for a vectorized take-profit / stop-loss:

.. code-block:: python

   from dataclasses import dataclass
   from mktlib.backtest import (
       run, Crossover, Col, Pct, ValueGT, ValueLT, Condition,
   )

   @dataclass(frozen=True, slots=True)
   class SmaCrossWithExits:
       """SMA crossover entry with percentage TP and volatility-based SL."""

       def entry(self) -> Crossover:
           return Crossover("fast_sma", "slow_sma")

       def exit(self) -> Condition:
           tp = ValueGT("close", Pct("slow_sma", 5))        # 5% above slow SMA
           sl = ValueLT("close", Col("slow_sma") - Col("vol") * 2)  # 2x vol below
           return tp | sl

   # df must have: date, open, close, fast_sma, slow_sma, vol
   result = run(df, SmaCrossWithExits())

Multi-symbol backtesting — run the same strategy across multiple tickers in
one call:

.. code-block:: python

   # df has columns: symbol, date, open, close (for AAPL, TSLA, SPY)
   result = run(df, SmaCross(), instrument_col="symbol")

   # O(1) per-symbol access
   aapl = result["AAPL"]
   aapl.returns   # DataFrame[date, return] — no symbol column
   aapl.trades    # DataFrame[entry_date, exit_date, pnl, bars_held]

   # Combined views (symbol column prepended)
   result.returns   # DataFrame[symbol, date, return]

   # Equal-weight portfolio returns
   portfolio = result.returns.group_by("date").agg(pl.col("return").mean())

Each symbol is backtested independently — indicators do not bleed across
symbols, and ``init()`` is called once per symbol.

See :doc:`api/backtest` for the full API.

Financial Metrics
-----------------

Compute standalone metrics from return series (included in base install):

.. code-block:: python

   from mktlib.metrics import (
       sharpe, sortino, cumulative_return, cagr,
       drawdown_series, calculate_metric, Metric,
   )

   sr = sharpe(returns_series, rf=0.05)
   cr = cumulative_return(returns_series)
   dd = drawdown_series(returns_series)

   # Or use the dispatcher
   sr = calculate_metric(Metric.SHARPE, returns_series, rf=0.05)

Available: Sharpe, Sortino, Omega, VaR, CVaR, CAGR, max/avg drawdown, win rate, payoff ratio, profit factor, Kelly criterion, and more. See :doc:`api/metrics` for details.

Exchange Scheduling
-------------------

Get trading calendars for major exchanges:

.. code-block:: python

   from mktlib.scheduling import get_calendar

   cal = get_calendar("NYSE")

   # Trading days in a range
   days = cal.valid_days("2024-01-01", "2024-12-31")

   # Full schedule with open/close times
   schedule = cal.schedule("2024-01-02", "2024-01-31")

   # Session navigation
   cal.next_session("2024-01-05")       # skips weekend -> Jan 8
   cal.session_offset("2024-01-08", 5)  # 5 trading days forward

   # Minute-level: is the market open?
   from datetime import datetime
   from zoneinfo import ZoneInfo

   dt = datetime(2024, 1, 2, 12, 0, tzinfo=ZoneInfo("America/New_York"))
   cal.is_open_on_minute(dt)  # True

   # Intraday trading index
   idx = cal.trading_index("2024-01-02", "2024-01-02", period="5m")

   # Filter existing DataFrame to market hours (more efficient than trading_index)
   filtered = cal.filter_market_hours(df, date_column="date")

Supported exchanges: NYSE, NASDAQ, CBOE, LSE, Euronext, Xetra, TSX, CME (RTH & Globex), JPX, HKEX, FX (24/5). See :doc:`api/scheduling` for details.

Treasury Rates
--------------

Fetch Treasury yield curve data with automatic caching:

.. code-block:: python

   from mktlib.rates import (
       TreasuryRate,
       get_risk_free_rate,
       get_treasury_rates,
       get_treasury_spread,
   )

   # Average 3-month T-bill rate for 2024
   rf = get_risk_free_rate("2024-01-01", "2024-12-31")

   # Daily rates as a Polars DataFrame
   df = get_treasury_rates("2024-01-01", "2024-03-31", TreasuryRate.TEN_YEAR)

   # Yield curve spread (10Y - 2Y)
   spread = get_treasury_spread("2024-01-01", "2024-03-31")

Data is cached in memory, on disk (``~/.cache/mktlib/rates/``), and bundled with the package for offline use. See :doc:`api/rates` for the full API.

Performance Reports
-------------------

Generate tearsheets with 25 metrics and 8 interactive charts (requires ``pip install mktlib[reports]``):

.. code-block:: python

   from mktlib.reports import html, metrics

   # HTML tearsheet from a Polars DataFrame with 'date' and 'return' columns
   html(returns_df, output="tearsheet.html", title="My Strategy")

   # With benchmark and auto risk-free rate
   html(returns_df, benchmark=bench_df, rf="auto", output="report.html")

   # Metrics only (no HTML)
   result = metrics(returns_df)
   print(result.sharpe, result.max_drawdown, result.cagr)

Accepts ``pl.DataFrame``, ``pl.Series``, or ``pd.Series`` inputs.

Per-trade metrics are computed automatically when ``trades`` is provided:

.. code-block:: python

   # From a backtest result with trades
   result = run(df, SmaCross())

   # HTML tearsheet with per-trade analysis
   html(result.returns, trades=result.trades, output="tearsheet.html")

   # Metrics only — includes TradeMetrics
   m = metrics(result.returns, trades=result.trades)
   print(m.trade_metrics.trade_win_rate)     # e.g. 0.55
   print(m.trade_metrics.profit_factor)      # e.g. 1.8
   print(m.trade_metrics.trade_sharpe)       # risk-adjusted per-trade
   print(m.trade_metrics.kelly_criterion)    # optimal bet fraction

Available trade metrics: win rate, payoff ratio, profit factor, Kelly criterion,
avg/largest winner and loser, max consecutive wins/losses, trade Sharpe, trade
Sortino, and trades per year. The HTML tearsheet includes a PnL distribution
histogram.

See :doc:`api/reports` for all options.

Synthetic Data Generators
-------------------------

Generate stochastic process data for testing and simulation (requires ``pip install mktlib[data]``):

.. code-block:: python

   from mktlib.data import (
       fractional_random_walk,
       geometric_brownian_motion,
       monte_carlo,
       ornstein_uhlenbeck,
   )

   # GBM price path — 252 daily steps, 5% annualised drift, 20% annualised vol
   gbm = geometric_brownian_motion(252, drift=0.05, volatility=0.20, seed=42)

   # Mean-reverting process
   ou = ornstein_uhlenbeck(500, theta=0.7, mu=100.0, sigma=1.0, seed=42)

   # 1000 Monte Carlo simulations
   sims = monte_carlo(geometric_brownian_motion, n_simulations=1000, n=252, seed=42)

All generators return Polars DataFrames with seeded RNG for reproducibility. See :doc:`api/data` for details.
