"""What an installed mktlib does with each set of extras, and nothing else.

This module imports only the standard library, pytest, polars and mktlib,
so it runs in an environment that holds none of the dev dependencies.  The
``clean-install`` CI job installs the built package with one set of extras
plus pytest, copies this file out of the checkout and runs it there with
``MKTLIB_CLEAN_INSTALL`` set to the extras it installed (``""`` for none).

Under that variable each case is held to the extras named: an entry point
whose extras are installed must work, and one whose extras are missing must
raise ``ModuleNotFoundError``.  The environment is also checked to be the
one the job meant: mktlib imported from the installed package, and no
dev-only module importable.

In an ordinary test run the variable is unset: an extra counts as installed
when the modules it brings are importable (the dev extra installs all of
them, and one CI arm removes the accelerator again), and the environment
checks are skipped.
"""

from __future__ import annotations

import datetime as dt
import importlib.util
import os
from collections.abc import Callable

import polars as pl
import pytest

import mktlib

_ENV = "MKTLIB_CLEAN_INSTALL"
_CLEAN = _ENV in os.environ
_ALL_EXTRAS = frozenset({"data", "reports", "fast"})

#: Importable only when the dev extra is installed.
_DEV_ONLY = (
    "numpy",
    "pandas",
    "scipy",
    "pyarrow",
    "pandera",
    "exchange_calendars",
)

#: The module each extra installs, as evidence the extra is present.
_EXTRA_MODULES = {
    "data": ("polars_sdist", "polars_rfft"),
    "reports": ("plotly", "jinja2", "markupsafe"),
    "fast": ("mktlib_scan",),
}


def _importable(extra: str) -> bool:
    return all(importlib.util.find_spec(m) for m in _EXTRA_MODULES[extra])


#: Under the job's variable, what the job says it installed; otherwise
#: what this environment holds.
_INSTALLED = (
    frozenset(e for e in os.environ[_ENV].split(",") if e)
    if _CLEAN
    else frozenset(e for e in _ALL_EXTRAS if _importable(e))
)

_N = 400
_DATES = pl.date_range(
    dt.date(2023, 1, 2),
    dt.date(2023, 1, 2) + dt.timedelta(days=_N - 1),
    eager=True,
)
_RETURNS = pl.Series("r", [0.001 * ((i % 7) - 3) for i in range(_N)])
_FRAME = pl.DataFrame({"date": _DATES, "return": _RETURNS})


def _bars() -> pl.DataFrame:
    close = [100.0 + ((i * 7) % 13) - 6.0 for i in range(_N)]
    return pl.DataFrame(
        {
            "date": _DATES,
            "open": close,
            "high": [c + 1.0 for c in close],
            "low": [c - 1.0 for c in close],
            "close": close,
            "volume": [1_000.0] * _N,
        }
    )


def _backtest() -> object:
    from mktlib.backtest import Crossover, Crossunder, run

    class Cross:
        name = "cross"

        def entry(self) -> Crossover:
            return Crossover("close", 100.0)

        def exit(self) -> Crossunder:
            return Crossunder("close", 100.0)

    result = run(_bars(), Cross())
    assert result.returns.height == _N
    return result


def _metrics() -> object:
    from mktlib.metrics import cumulative_return, cvar, sharpe, var

    values = (
        sharpe(_RETURNS),
        cumulative_return(_RETURNS),
        var(_RETURNS, 0.05),
        cvar(_RETURNS, 0.05),
    )
    assert all(isinstance(v, float) for v in values)
    return values


def _metrics_monte_carlo() -> object:
    from mktlib.metrics import cvar

    value = cvar(
        _RETURNS,
        0.05,
        method="monte_carlo",
        horizon=5,
        n_simulations=200,
        seed=1,
    )
    assert isinstance(value, float)
    return value


def _scheduling() -> object:
    from mktlib.scheduling import get_calendar

    calendar = get_calendar("XNYS")
    schedule = calendar.schedule("2024-11-25", "2024-12-02")
    index = calendar.trading_index("2024-11-25", "2024-11-26")
    assert schedule.height > 0 and index.len() > 0
    return schedule


def _rates() -> object:
    # The bundled yield-curve files ship as package data; the network is
    # not touched for a year that is bundled.
    from importlib.resources import files

    data = files("mktlib.rates") / "_data"
    bundled = [p.name for p in data.iterdir() if p.name.endswith(".csv")]
    assert bundled, "no bundled Treasury CSV in the installed package"
    return bundled


def _data() -> object:
    from mktlib.data import fractional_random_walk, geometric_brownian_motion

    gbm = geometric_brownian_motion(n=100, seed=1)
    frw = fractional_random_walk(100, hurst=0.7, seed=1)
    assert gbm.height == 100 and frw.height == 100
    return gbm


def _reports_metrics() -> object:
    from mktlib.reports import metrics

    return metrics(_FRAME, rf=0.0)


def _reports_html() -> object:
    # The templates ship as package data.
    from mktlib.reports import html

    out = html(_FRAME, rf=0.0)
    assert isinstance(out, str) and "<html" in out.lower()
    return out


def _reports_html_monte_carlo() -> object:
    from mktlib.reports import MonteCarloConfig, html

    out = html(_FRAME, rf=0.0, mc_config=MonteCarloConfig(enabled=True))
    assert isinstance(out, str) and "<html" in out.lower()
    return out


def _scan_backend() -> object:
    from mktlib.backtest import (
        active_scan_backend,
        get_scan_backend,
        set_scan_backend,
    )

    before = get_scan_backend()
    set_scan_backend("native")
    try:
        assert active_scan_backend() == "native"
        return _backtest()
    finally:
        set_scan_backend(before)


#: Entry point, and the extras it needs.
CASES: dict[str, tuple[Callable[[], object], frozenset[str]]] = {
    "backtest.run": (_backtest, frozenset()),
    "metrics": (_metrics, frozenset()),
    "scheduling": (_scheduling, frozenset()),
    "rates package data": (_rates, frozenset()),
    "metrics monte carlo": (_metrics_monte_carlo, frozenset({"data"})),
    "data generators": (_data, frozenset({"data"})),
    "reports.metrics": (_reports_metrics, frozenset({"reports"})),
    "reports.html": (_reports_html, frozenset({"reports"})),
    "reports.html monte carlo": (
        _reports_html_monte_carlo,
        frozenset({"reports", "data"}),
    ),
    "backtest.run native": (_scan_backend, frozenset({"fast"})),
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_entry_point_follows_its_extras(name: str) -> None:
    call, needs = CASES[name]
    if needs <= _INSTALLED:
        call()
        return
    # A missing extra is reported as a missing module, never as an abort
    # from native code or as a wrong result.
    with pytest.raises((ModuleNotFoundError, RuntimeError)) as caught:
        call()
    if isinstance(caught.value, RuntimeError):
        # The scan backend states its own reason when pinned and absent.
        assert "mktlib-scan" in str(caught.value)


@pytest.mark.skipif(not _CLEAN, reason=f"{_ENV} is not set")
def test_environment_is_the_one_the_job_installed() -> None:
    assert "site-packages" in (mktlib.__file__ or ""), mktlib.__file__

    leaked = [m for m in _DEV_ONLY if importlib.util.find_spec(m)]
    assert leaked == [], f"dev-only modules importable: {leaked}"

    for extra, modules in _EXTRA_MODULES.items():
        present = [m for m in modules if importlib.util.find_spec(m)]
        if extra in _INSTALLED:
            assert present == list(modules), (extra, present)
        else:
            assert present == [], (extra, present)


def test_extras_in_this_module_are_the_declared_ones() -> None:
    """A new extra in pyproject.toml gets cases here before it ships."""
    from importlib.metadata import metadata

    declared = set(metadata("mktlib").get_all("Provides-Extra") or [])
    assert declared - {"dev"} == set(_ALL_EXTRAS)
