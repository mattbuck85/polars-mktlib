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

   pip install mktlib[data]        # synthetic data generators (adds numpy)
   pip install mktlib[reports]     # tearsheet generation (adds plotly, jinja2)
   pip install mktlib[data,reports]  # both extras

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
