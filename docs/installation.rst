Installation
============

Requirements
------------

- Python 3.12 or later
- `polars <https://pola.rs/>`_ (installed automatically)

Install from PyPI
-----------------

.. code-block:: bash

   pip install mktlib

Optional extras
---------------

mktlib has optional extras for additional functionality:

.. code-block:: bash

   pip install mktlib[data]        # synthetic data generators (adds polars-sdist, polars-rfft)
   pip install mktlib[reports]     # tearsheet generation (adds plotly, jinja2)
   pip install mktlib[fast]        # compiled chain resolver (adds mktlib-scan)
   pip install mktlib[data,reports]  # several extras

The ``fast`` extra
------------------

``mktlib[fast]`` installs `mktlib-scan
<https://github.com/mattbuck85/mktlib-scan>`_, a compiled implementation of the
``EntryRef`` chain resolver. It changes **speed and nothing else**: output is
required to be bit-identical to the pure-Python resolver, and mktlib's full
equivalence corpus plus every frozen golden baseline run against both
implementations in CI.

It is worth installing for strategies whose exit condition uses
:class:`~mktlib.backtest.EntryRef` — a fixed take-profit / stop-loss measured
from the entry bar. Measured on a KVO crossover strategy with a 5% / -3% band:

.. list-table::
   :header-rows: 1

   * - bars
     - pure Python
     - ``[fast]``
     - whole-run
   * - 500,000
     - 127.0 ms
     - 28.4 ms
     - 4.5x
   * - 1,000,000
     - 240.8 ms
     - 58.0 ms
     - 4.2x
   * - 2,000,000
     - 492.0 ms
     - 128.0 ms
     - 3.8x

Strategies with no ``EntryRef`` in the exit tree never call the resolver, so the
extra makes no difference to them.

Wheels are published for linux (x86_64, aarch64), macOS (x86_64, arm64) and
Windows (x86_64). On any other platform mktlib silently uses the pure-Python
resolver — correct, just not accelerated. Pin a backend explicitly with
:func:`~mktlib.backtest.set_scan_backend` or ``MKTLIB_SCAN_BACKEND``, and ask
which one is active with :func:`~mktlib.backtest.active_scan_backend`.

Install from source
-------------------

.. code-block:: bash

   git clone https://github.com/mattbuck85/polars-mktlib.git
   cd polars-mktlib
   pip install -e ".[dev,data,reports]"

Verify installation
-------------------

.. code-block:: python

   import mktlib
   from mktlib.scheduling import get_calendar

   cal = get_calendar("NYSE")
   print(cal.valid_days("2024-01-01", "2024-01-31"))
