See Also
========

Libraries that complement mktlib, grouped by role.

Polars Plugins (mktlib dependencies)
-------------------------------------

`polars-rfft <https://github.com/mattbuck85/polars-rfft>`_
   Polars plugin wrapping `RustFFT <https://github.com/ejmahler/RustFFT>`_
   for FFT/IFFT as Polars expressions.
   Used by :mod:`mktlib.data` (``fractional_random_walk``) for Davies-Harte
   circulant embedding.

`polars-sdist <https://github.com/mattbuck85/polars-sdist>`_
   Polars plugin wrapping `statrs <https://github.com/statrs-dev/statrs>`_
   (PDF/CDF/PPF) and `rand_distr <https://docs.rs/rand_distr>`_ (sampling)
   for statistical distributions as Polars expressions.
   Used by :mod:`mktlib.data` generators and ``ticks_to_ohlcv`` volume
   synthesis.

Technical Analysis
------------------

`polars-talib <https://github.com/Yvictor/polars_ta_extension>`_ (PyPI: ``polars_talib``)
   150+ TA-Lib indicators as native Polars expressions via Rust bindings.
   ~150x faster than pandas + talib for multi-symbol ``.over()`` workflows.

Data Validation
---------------

`pandera[polars] <https://github.com/pandera-dev/pandera>`_ (PyPI: ``pip install pandera[polars]``)
   DataFrame schema validation and type hints for Polars.
   Define column types, value ranges, and custom checks as declarative
   schemas.
