Quick Start
===========

This guide covers the subpackages in mktlib with short, runnable examples.

Vectorized Backtesting
----------------------

Run signal-driven backtests with fill-at-next-open semantics:

.. code-block:: python

   from dataclasses import dataclass
   from mktlib.backtest import run, Crossover, Crossunder

   @dataclass(frozen=True, slots=True)
   class SmaCross:
       def entry(self) -> Crossover:
           return Crossover("fast_sma", "slow_sma")

       def exit(self) -> Crossunder:
           return Crossunder("fast_sma", "slow_sma")

   # df must have: date, open, close, fast_sma, slow_sma
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

   from mktlib.backtest import PriceIsAbove

   entry = Crossover("fast", "slow") & PriceIsAbove("close", "sma_200")

Price expressions (``Col``, ``Lit``, ``Pct``) let you build dynamic exit
levels without any engine changes — combine ``PriceIsAbove`` and
``PriceIsBelow`` with ``|`` for a vectorized take-profit / stop-loss:

.. code-block:: python

   from dataclasses import dataclass
   from mktlib.backtest import (
       run, Crossover, Col, Pct, PriceIsAbove, PriceIsBelow, Condition,
   )

   @dataclass(frozen=True, slots=True)
   class SmaCrossWithExits:
       """SMA crossover entry with percentage TP and volatility-based SL."""

       def entry(self) -> Crossover:
           return Crossover("fast_sma", "slow_sma")

       def exit(self) -> Condition:
           tp = PriceIsAbove("close", Pct("slow_sma", 5))        # 5% above slow SMA
           sl = PriceIsBelow("close", Col("slow_sma") - Col("vol") * 2)  # 2x vol below
           return tp | sl

   # df must have: date, open, close, fast_sma, slow_sma, vol
   result = run(df, SmaCrossWithExits())

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

Accepts ``pl.DataFrame``, ``pl.Series``, or ``pd.Series`` inputs. See :doc:`api/reports` for all options.

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

   # GBM price path
   gbm = geometric_brownian_motion(252, drift=0.05/252, volatility=0.20/252**0.5, seed=42)

   # Mean-reverting process
   ou = ornstein_uhlenbeck(500, theta=0.7, mu=100.0, sigma=1.0, seed=42)

   # 1000 Monte Carlo simulations
   sims = monte_carlo(geometric_brownian_motion, n_simulations=1000, n=252, seed=42)

All generators return Polars DataFrames with seeded RNG for reproducibility. See :doc:`api/data` for details.
