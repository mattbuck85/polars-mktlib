from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version as _pkg_version

try:
    __version__ = _pkg_version("mktlib")
except PackageNotFoundError:
    # Editable install where the dist-info isn't on sys.path yet, or
    # source checkout being imported without `pip install -e .` first.
    __version__ = "0.0.0+unknown"

from mktlib.metrics import Metric, calculate_metric, drawdown_series
