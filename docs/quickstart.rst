Quick Start
===========

This guide covers the four subpackages in mktlib with short, runnable examples.

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
