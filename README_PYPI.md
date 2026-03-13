# mktlib

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Documentation](https://readthedocs.org/projects/polars-mktlib/badge/?version=latest)](https://polars-mktlib.readthedocs.io/en/latest/)

Financial market toolkit built entirely on Polars.

### tl;dr

- **Fast enough for real work** — vectorized Polars engine grid-searches thousands of parameter combos on minute-bar data without reaching for Numba or Cython
- **Lightweight** — pure Polars throughout, including synthetic data generation via Rust-native plugins. No pandas, no NumPy, no heavy ML stack
- **Well tested** — cross-validated exchange calendars, property-based OHLCV checks, and full backtest parity tests across engines
- **Swiss-army knife** — scheduling, rates, metrics, backtesting, reporting, and data generation in one package. Great for learning, prototyping, or production
- **Apache 2.0** — use it anywhere, fork it, vendor it, no strings attached

> **Disclaimer:** Backtesting results and any computed market returns are for **educational and research purposes only**. They do not constitute financial advice, and past performance does not indicate future results.

For full documentation, see [polars-mktlib.readthedocs.io](https://polars-mktlib.readthedocs.io).
