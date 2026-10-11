"""The scenarios the engine-parity tests run, and how they are compared.

A scenario is a function of one argument, a *layout*: a callable applied to
every input frame or series the scenario hands to the library.  The
identity layout leaves the input as built; :func:`in_chunks` rebuilds it
from 7-row pieces.  A scenario returns whatever the library returned
(frames, series, dataclasses, scalars, strings, or containers of those),
and :func:`assert_exact` compares two such results with no tolerance.

The library keeps two process-wide caches of calendar-derived values.
:func:`run_fresh` empties them before a scenario runs, so that a value
computed on one run is never served to the run it is compared against.
"""

from __future__ import annotations

import dataclasses
import datetime
import math
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any, TypeVar

import polars as pl
from polars.testing import assert_frame_equal, assert_series_equal

from mktlib import data as mdata
from mktlib import metrics as mmetrics
from mktlib import reports
from mktlib.backtest import (
    Cost,
    FlattenSchedule,
    run,
)
from mktlib.backtest import _flatten
from mktlib.reports import MonteCarloConfig
from mktlib.scheduling import get_calendar
from mktlib.scheduling import registry as _registry
from tests.backtest import test_golden_baseline as golden

Frameish = TypeVar("Frameish", pl.DataFrame, pl.Series)
Layout = Callable[[Any], Any]
Scenario = Callable[[Layout], object]

CHUNK_ROWS = 7


def as_built(obj: Frameish) -> Frameish:
    """The identity layout."""
    return obj


def in_chunks(obj: Frameish) -> Frameish:
    """The same rows, held in ``CHUNK_ROWS``-row chunks."""
    n = len(obj)
    if n <= CHUNK_ROWS:
        return obj
    # Slices of a single chunk, so that each piece is one chunk whatever the
    # layout the input arrived in.
    whole = obj.rechunk()
    pieces = [whole.slice(i, CHUNK_ROWS) for i in range(0, n, CHUNK_ROWS)]
    out = pl.concat(pieces, rechunk=False)  # type: ignore[type-var]
    assert out.n_chunks() == len(pieces), "the layout did not survive concat"
    return out  # type: ignore[return-value]


def run_fresh(scenario: Scenario, layout: Layout = as_built) -> object:
    """Run *scenario* with the library's calendar caches emptied first."""
    _flatten._schedule_cache.clear()
    _flatten._week_last_cache.clear()
    return scenario(layout)


# ---------------------------------------------------------------------------
# Exact comparison
# ---------------------------------------------------------------------------


def assert_exact(
    left: object, right: object, where: str = "result", *, rel_tol: float = 0.0
) -> None:
    """Assert two scenario results are identical, to the bit for floats.

    Frames and series: exact values, dtypes, column order and row order.
    Floats: ``==``, with NaN equal to NaN.  Everything else: ``==``.

    A non-zero *rel_tol* relaxes the float comparisons, and only those, to
    that relative tolerance.
    """
    assert type(left) is type(right), (
        f"{where}: {type(left).__name__} != {type(right).__name__}"
    )
    exact = rel_tol == 0.0
    tolerance: dict[str, Any] = {} if exact else {"rel_tol": rel_tol, "abs_tol": 0.0}
    if isinstance(left, pl.DataFrame):
        try:
            assert_frame_equal(
                left,
                right,  # type: ignore[arg-type]
                check_exact=exact,
                check_dtypes=True,
                check_column_order=True,
                check_row_order=True,
                **tolerance,
            )
        except AssertionError as exc:
            raise AssertionError(f"{where}: {exc}") from None
    elif isinstance(left, pl.Series):
        try:
            assert_series_equal(
                left,
                right,  # type: ignore[arg-type]
                check_exact=exact,
                check_dtypes=True,
                check_names=True,
                check_order=True,
                **tolerance,
            )
        except AssertionError as exc:
            raise AssertionError(f"{where}: {exc}") from None
    elif dataclasses.is_dataclass(left) and not isinstance(left, type):
        for f in dataclasses.fields(left):
            assert_exact(
                getattr(left, f.name),
                getattr(right, f.name),
                f"{where}.{f.name}",
                rel_tol=rel_tol,
            )
    elif isinstance(left, dict):
        assert isinstance(right, dict)
        assert list(left) == list(right), f"{where}: keys differ"
        for key in left:
            assert_exact(left[key], right[key], f"{where}[{key!r}]", rel_tol=rel_tol)
    elif isinstance(left, (list, tuple)):
        assert isinstance(right, (list, tuple))
        assert len(left) == len(right), f"{where}: lengths differ"
        for i, (a, b) in enumerate(zip(left, right, strict=True)):
            assert_exact(a, b, f"{where}[{i}]", rel_tol=rel_tol)
    elif isinstance(left, float):
        assert isinstance(right, float)
        same = (
            left == right
            or (math.isnan(left) and math.isnan(right))
            or (not exact and math.isclose(left, right, rel_tol=rel_tol, abs_tol=0.0))
        )
        assert same, f"{where}: {left!r} != {right!r}"
    else:
        assert left == right, f"{where}: {left!r} != {right!r}"


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------


@contextmanager
def _golden_inputs(layout: Layout) -> Iterator[None]:
    """Route every frame a golden scenario passes to ``run`` through *layout*."""
    original = golden.run

    def run_with_layout(df: pl.DataFrame, *args: Any, **kwargs: Any) -> Any:
        return original(layout(df), *args, **kwargs)

    golden.run = run_with_layout  # type: ignore[assignment]
    try:
        yield
    finally:
        golden.run = original  # type: ignore[assignment]


def _golden(name: str) -> Scenario:
    def scenario(layout: Layout) -> object:
        with _golden_inputs(layout):
            return golden.SCENARIOS[name]()

    return scenario


#: Symbols and the seed and start price of each one's synthetic series.
_SYMBOLS = {
    "AAA": (11111, 100.0),
    "BBB": (22222, 55.0),
    "CCC": (33333, 240.0),
    "DDD": (44444, 17.5),
    "EEE": (55555, 910.0),
}


def multi_frame(n_instruments: int, *, seed_offset: int = 0) -> pl.DataFrame:
    """Daily bars for the first *n_instruments* symbols, one block each."""
    return pl.concat(
        [
            golden._daily_frame(seed=seed + seed_offset, start=start).with_columns(
                pl.lit(symbol).alias("symbol")
            )
            for symbol, (seed, start) in list(_SYMBOLS.items())[:n_instruments]
        ]
    )


def weights_for(n_instruments: int) -> dict[str, float]:
    """Unequal weights summing to one over the first *n_instruments* symbols."""
    raw = [0.37, 0.23, 0.19, 0.13, 0.08][:n_instruments]
    total = sum(raw)
    return {
        symbol: w / total
        for symbol, w in zip(list(_SYMBOLS)[:n_instruments], raw, strict=True)
    }


def weighted_run(
    n_instruments: int, *, seed_offset: int = 0, layout: Layout = as_built
) -> pl.DataFrame:
    """Weighted portfolio returns of a multi-instrument run."""
    result = run(
        layout(multi_frame(n_instruments, seed_offset=seed_offset)),
        golden._CrossStrategy(),
        instrument_col="symbol",
        instrument_weights=weights_for(n_instruments),
        cost=Cost(commission_bps=0.7, slippage_bps=0.3),
    )
    return result.returns


def wide_weights(n_instruments: int, n_extra: int = 600) -> dict[str, float]:
    """The weights of :func:`weights_for` inside a much longer mapping.

    The mapping lists *n_extra* symbols the frame does not hold and then
    the frame's own symbols in the reverse of their input order, so the
    order of the mapping and the order of the input disagree.
    """
    own = weights_for(n_instruments)
    extras = {f"ZZ{i:04d}": 0.01 for i in range(n_extra)}
    return {**extras, **dict(reversed(own.items()))}


def wide_weighted_run(
    n_instruments: int, *, n_extra: int = 600, seed_offset: int = 0
) -> pl.DataFrame:
    """Weighted returns where the weights mapping dwarfs the instruments."""
    import logging
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        logging.disable(logging.WARNING)
        try:
            result = run(
                multi_frame(n_instruments, seed_offset=seed_offset),
                golden._CrossStrategy(),
                instrument_col="symbol",
                instrument_weights=wide_weights(n_instruments, n_extra),
                cost=Cost(commission_bps=0.7, slippage_bps=0.3),
            )
        finally:
            logging.disable(logging.NOTSET)
    return result.returns


def input_order_weighted_sum(
    n_instruments: int, *, seed_offset: int = 0
) -> pl.DataFrame:
    """The weighted returns of each date, summed in input order.

    Each instrument is run on its own and its returns are given their
    weight as a literal, so no join is involved; the frames are stacked
    in the order the instruments appear in the input and each date's rows
    are summed in that order.
    """
    weights = weights_for(n_instruments)
    frame = multi_frame(n_instruments, seed_offset=seed_offset)
    parts = []
    for (symbol,), part in frame.partition_by(
        "symbol", maintain_order=True, as_dict=True
    ).items():
        returns = run(
            part.drop("symbol"),
            golden._CrossStrategy(),
            cost=Cost(commission_bps=0.7, slippage_bps=0.3),
        ).returns
        parts.append(
            returns.select(
                "date",
                (pl.col("return") * weights[str(symbol)]).alias("_wr"),
                pl.lit(weights[str(symbol)]).alias("weight"),
            )
        )
    return (
        pl.concat(parts)
        .group_by("date", maintain_order=True)
        .agg((pl.col("_wr").sum() / pl.col("weight").sum()).alias("return"))
        .sort("date")
    )


def _weighted(n_instruments: int) -> Scenario:
    def scenario(layout: Layout) -> object:
        return weighted_run(n_instruments, layout=layout)

    return scenario


def _intraday_weeks_frame() -> pl.DataFrame:
    """30-minute bars around the clock over three weeks holding a holiday."""
    first = datetime.datetime(2024, 11, 18)
    stamps = [first + datetime.timedelta(minutes=30 * i) for i in range(48 * 21)]
    return golden._frame(stamps, seed=20260801)


def _intraday_multi_frame() -> pl.DataFrame:
    base = _intraday_weeks_frame()
    return pl.concat(
        [
            base.with_columns(
                (pl.col(c) * scale for c in ("open", "high", "low", "close", "fast", "slow", "tp_level")),
            ).with_columns(pl.lit(symbol).alias("symbol"))
            for symbol, scale in (("AAA", 1.0), ("BBB", 0.5), ("CCC", 2.0))
        ]
    )


def _scenario_weekly_block(layout: Layout) -> object:
    """Weekly flatten with an entry-block window, through the calendar filter."""
    result = run(
        layout(_intraday_weeks_frame()),
        golden._CrossStrategy(),
        calendar=get_calendar("XNYS"),
        flatten=FlattenSchedule(
            days="weekly", minutes_before_close=30, block_entry_minutes_before_close=60
        ),
    )
    return golden._artifacts(result)


def _scenario_block_whole_session(layout: Layout) -> object:
    """An entry-block window as long as a session: no position may open."""
    result = run(
        layout(_intraday_weeks_frame()),
        golden._CrossStrategy(),
        calendar=get_calendar("XNYS"),
        flatten=FlattenSchedule(block_entry_minutes_before_close=390),
    )
    return golden._artifacts(result)


def _scenario_intraday_multi_weighted(layout: Layout) -> object:
    """Calendar filter, daily flatten and weighted returns on three symbols."""
    result = run(
        layout(_intraday_multi_frame()),
        golden._CrossStrategy(),
        instrument_col="symbol",
        instrument_weights=weights_for(3),
        calendar=get_calendar("XNYS"),
        flatten_eod=True,
    )
    return golden._artifacts(result)


def _scenario_dual_trades(layout: Layout) -> object:
    """A long and a short strategy merged into one result."""
    result = run(
        layout(_intraday_weeks_frame()),
        golden._CrossStrategy(),
        short_strategy=golden._ShortCrossStrategy(),
        calendar=get_calendar("XNYS"),
        flatten_eod=True,
    )
    return golden._artifacts(result)


#: Scenarios run on each scan backend.
BACKTEST_SCENARIOS: dict[str, Scenario] = {
    **{f"golden/{name}": _golden(name) for name in sorted(golden.SCENARIOS)},
    "weighted/2": _weighted(2),
    "weighted/3": _weighted(3),
    "weighted/5": _weighted(5),
    "weekly_block": _scenario_weekly_block,
    "block_whole_session": _scenario_block_whole_session,
    "intraday_multi_weighted": _scenario_intraday_multi_weighted,
    "dual_intraday": _scenario_dual_trades,
}


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------


def _business_days(n: int) -> list[datetime.date]:
    out: list[datetime.date] = []
    day = datetime.date(2021, 1, 4)
    while len(out) < n:
        if day.weekday() < 5:
            out.append(day)
        day += datetime.timedelta(days=1)
    return out


def returns_frame(n: int = 600, *, seed: int = 4242) -> pl.DataFrame:
    """Daily ``(date, return)`` rows with a slight positive drift."""
    u = golden._uniforms(n, seed)
    return pl.DataFrame(
        {
            "date": _business_days(n),
            "return": [(x - 0.48) * 0.03 for x in u],
        }
    )


def _sub_daily_returns(n_days: int = 300) -> pl.DataFrame:
    """Four timestamped returns per date: the frame is compounded per day."""
    u = golden._uniforms(4 * n_days, 777)
    stamps = [
        datetime.datetime(d.year, d.month, d.day, 10 + 2 * k)
        for d in _business_days(n_days)
        for k in range(4)
    ]
    return pl.DataFrame({"date": stamps, "return": [(x - 0.49) * 0.01 for x in u]})


def _trades_frame(n: int = 80) -> pl.DataFrame:
    days = _business_days(4 * n)
    u = golden._uniforms(n, 99)
    return pl.DataFrame(
        {
            "entry_date": [days[4 * i] for i in range(n)],
            "exit_date": [days[4 * i + 2] for i in range(n)],
            "side": pl.Series([1 if i % 3 else -1 for i in range(n)], dtype=pl.Int8),
            "pnl": [(x - 0.45) * 0.04 for x in u],
            "bars_held": pl.Series([2 + i % 5 for i in range(n)], dtype=pl.Int64),
        }
    )


def _mc_config(innovations: mdata.Innovations) -> MonteCarloConfig:
    return MonteCarloConfig(
        enabled=True,
        horizon=21,
        n_simulations=400,
        innovations=innovations,
        df=5.0 if innovations is mdata.Innovations.STUDENT_T else None,
        seed=20260811,
        n_paths_displayed=25,
    )


def _report_metrics(innovations: mdata.Innovations) -> Scenario:
    def scenario(layout: Layout) -> object:
        return reports.metrics(
            layout(returns_frame()),
            benchmark=layout(returns_frame(seed=1717)),
            trades=layout(_trades_frame()),
            rf=0.02,
            mc_config=_mc_config(innovations),
        )

    return scenario


def _report_html(innovations: mdata.Innovations) -> Scenario:
    def scenario(layout: Layout) -> object:
        return _stable_html(
            reports.html(
                layout(returns_frame()),
                benchmark=layout(returns_frame(seed=1717)),
                trades=layout(_trades_frame()),
                rf=0.02,
                mc_config=_mc_config(innovations),
            )
        )

    return scenario


def _stable_html(html: str | None) -> str:
    """The tearsheet with the per-render element ids of its charts removed."""
    import re

    assert html is not None
    return re.sub(
        r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", "ID", html
    )


def _scenario_sub_daily_metrics(layout: Layout) -> object:
    return reports.metrics(layout(_sub_daily_returns()))


def _scenario_sub_daily_html(layout: Layout) -> object:
    return _stable_html(reports.html(layout(_sub_daily_returns())))


def _scenario_unsorted_metrics(layout: Layout) -> object:
    """One row per date, latest first: the frame is sorted by the library."""
    return reports.metrics(layout(returns_frame().reverse()))


def _scenario_series_metrics(layout: Layout) -> object:
    return reports.metrics(layout(returns_frame()["return"]), compounded=False)


REPORT_SCENARIOS: dict[str, Scenario] = {
    **{
        f"metrics/mc_{inn.value}": _report_metrics(inn) for inn in mdata.Innovations
    },
    **{f"html/mc_{inn.value}": _report_html(inn) for inn in mdata.Innovations},
    "metrics/sub_daily": _scenario_sub_daily_metrics,
    "html/sub_daily": _scenario_sub_daily_html,
    "metrics/series": _scenario_series_metrics,
    "metrics/unsorted": _scenario_unsorted_metrics,
}


# ---------------------------------------------------------------------------
# Synthetic data generators
# ---------------------------------------------------------------------------


def _loop_process(*, seed: int, n: int) -> pl.DataFrame:
    return mdata.geometric_brownian_motion(n, seed=seed)


def _residuals() -> pl.Series:
    u = golden._uniforms(300, 31)
    centred = pl.Series("r", [x - 0.5 for x in u])
    return (centred - centred.mean()) / centred.std()


def _ticks(layout: Layout) -> pl.DataFrame:
    return layout(mdata.geometric_brownian_motion(700, volatility=0.2, seed=5))


#: One entry per name in ``mktlib.data.__all__`` that is a function, keyed
#: ``<function>/<variant>``.
DATA_SCENARIOS: dict[str, Scenario] = {
    "geometric_brownian_motion/default": lambda layout: mdata.geometric_brownian_motion(
        600, drift=0.05, volatility=0.2, seed=1
    ),
    "ornstein_uhlenbeck/arithmetic": lambda layout: mdata.ornstein_uhlenbeck(
        600, seed=2
    ),
    "ornstein_uhlenbeck/geometric": lambda layout: mdata.ornstein_uhlenbeck(
        600, mu=4.0, sigma=0.3, seed=2, geometric=True
    ),
    "fractional_random_walk/h05": lambda layout: mdata.fractional_random_walk(
        600, seed=3
    ),
    "fractional_random_walk/h07": lambda layout: mdata.fractional_random_walk(
        600, hurst=0.7, seed=3
    ),
    "fractional_random_walk/h03_geometric": lambda layout: mdata.fractional_random_walk(
        600, hurst=0.3, step_size=0.01, seed=3, geometric=True
    ),
    "ticks_to_ohlcv/volume": lambda layout: mdata.ticks_to_ohlcv(
        _ticks(layout), 13, seed=8
    ),
    "ticks_to_ohlcv/no_volume": lambda layout: mdata.ticks_to_ohlcv(
        _ticks(layout), 13, volume=False
    ),
    "monte_carlo/gbm_gaussian": lambda layout: mdata.monte_carlo(
        mdata.Process.GBM, 40, seed=4, n=50
    ),
    "monte_carlo/gbm_gaussian_batched": lambda layout: mdata.monte_carlo(
        mdata.Process.GBM, 40, seed=4, n=50, independent_streams=False
    ),
    "monte_carlo/gbm_students_t": lambda layout: mdata.monte_carlo(
        mdata.Process.GBM,
        40,
        seed=4,
        n=50,
        innovations=mdata.Innovations.STUDENT_T,
        df=4.0,
    ),
    "monte_carlo/gbm_bootstrap": lambda layout: mdata.monte_carlo(
        mdata.Process.GBM,
        40,
        seed=4,
        n=50,
        innovations=mdata.Innovations.BOOTSTRAP,
        residuals=layout(_residuals()),
    ),
    "monte_carlo/ou": lambda layout: mdata.monte_carlo(
        mdata.Process.OU, 40, seed=4, n=50
    ),
    "monte_carlo/frw": lambda layout: mdata.monte_carlo(
        mdata.Process.FRW, 40, seed=4, n=50, hurst=0.7
    ),
    "monte_carlo/frw_h05": lambda layout: mdata.monte_carlo(
        mdata.Process.FRW, 40, seed=4, n=50
    ),
    "monte_carlo/callable": lambda layout: mdata.monte_carlo(
        _loop_process, 12, seed=4, n=50
    ),
}


def data_functions() -> set[str]:
    """The public callables of ``mktlib.data`` that are not classes."""
    return {
        name
        for name in mdata.__all__
        if callable(getattr(mdata, name)) and not isinstance(getattr(mdata, name), type)
    }


# ---------------------------------------------------------------------------
# Calendars
# ---------------------------------------------------------------------------

#: A range holding, for every registered calendar, at least one weekday
#: closure, and an early close wherever the calendar has any.
CALENDAR_START = datetime.date(2024, 11, 18)
CALENDAR_END = datetime.date(2025, 1, 10)

CALENDAR_NAMES = sorted(_registry._REGISTRY)


def calendar_bars(name: str, *, utc: bool = False) -> pl.DataFrame:
    """30-minute bars around the clock over the calendar range.

    ``date`` is the bar's label and ``end`` the instant it closes.  The
    stamps are in the calendar's own zone, or in UTC when *utc* is set.
    """
    cal = get_calendar(name)
    zone = "UTC" if utc else cal.timezone
    first = datetime.datetime.combine(
        CALENDAR_START - datetime.timedelta(days=1), datetime.time(0)
    )
    last = datetime.datetime.combine(
        CALENDAR_END + datetime.timedelta(days=1), datetime.time(0)
    )
    stamps = pl.datetime_range(
        first, last, interval="30m", time_zone=zone, eager=True
    ).alias("date")
    n = len(stamps)
    return pl.DataFrame(
        {
            "date": stamps,
            "end": stamps.dt.offset_by("30m"),
            "row": pl.int_range(n, eager=True),
            "x": pl.int_range(n, eager=True).cast(pl.Float64) * 0.1 + 0.3,
        }
    )


def _calendar_scenarios(name: str) -> dict[str, Scenario]:
    def schedule(layout: Layout) -> object:
        return get_calendar(name).schedule(CALENDAR_START, CALENDAR_END)

    def trading_index(layout: Layout) -> object:
        cal = get_calendar(name)
        return [
            cal.trading_index(CALENDAR_START, CALENDAR_END),
            cal.trading_index(CALENDAR_START, CALENDAR_END, "30m", closed="both"),
        ]

    def filter_label(layout: Layout) -> object:
        cal = get_calendar(name)
        return [
            cal.filter_market_hours(layout(calendar_bars(name))),
            cal.filter_market_hours(layout(calendar_bars(name, utc=True))),
        ]

    def filter_end_column(layout: Layout) -> object:
        cal = get_calendar(name)
        return [
            cal.filter_market_hours(layout(calendar_bars(name)), end_column="end"),
            cal.filter_market_hours(
                layout(calendar_bars(name, utc=True)), end_column="end"
            ),
        ]

    return {
        f"{name}/schedule": schedule,
        f"{name}/trading_index": trading_index,
        f"{name}/filter_market_hours": filter_label,
        f"{name}/filter_market_hours_end_column": filter_end_column,
    }


CALENDAR_SCENARIOS: dict[str, Scenario] = {
    key: scenario
    for name in CALENDAR_NAMES
    for key, scenario in _calendar_scenarios(name).items()
}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _metric(metric: mmetrics.Metric) -> Scenario:
    def scenario(layout: Layout) -> object:
        frame = returns_frame()
        return mmetrics.calculate_metric(
            metric,
            layout(frame["return"]),
            dates=layout(frame["date"]),
            rf=0.02,
        )

    return scenario


def mc_tail(
    metric: mmetrics.Metric,
    innovations: mdata.Innovations,
    *,
    seed: int = 20260811,
    n_simulations: int = 400,
    horizon: int = 21,
    layout: Layout = as_built,
) -> float:
    """Monte Carlo VaR or CVaR of the corpus return series."""
    return mmetrics.simulate_metric(
        metric,
        layout(returns_frame()["return"]),
        horizon=horizon,
        n_simulations=n_simulations,
        innovations=innovations,
        df=5.0 if innovations is mdata.Innovations.STUDENT_T else None,
        seed=seed,
    )


def _simulated(metric: mmetrics.Metric, innovations: mdata.Innovations) -> Scenario:
    def scenario(layout: Layout) -> object:
        return [
            mc_tail(metric, innovations, seed=seed, n_simulations=n, layout=layout)
            for seed in (1, 2, 3)
            for n in (200, 1000)
        ]

    return scenario


def _scenario_paths(layout: Layout) -> object:
    return mmetrics.monte_carlo_paths(
        layout(returns_frame()["return"]), horizon=21, n_simulations=200, seed=9
    )


METRIC_SCENARIOS: dict[str, Scenario] = {
    **{f"calculate/{m.value}": _metric(m) for m in mmetrics.Metric},
    **{
        f"simulate/{m.value}/{inn.value}": _simulated(m, inn)
        for m in (mmetrics.Metric.VAR, mmetrics.Metric.CVAR)
        for inn in mdata.Innovations
    },
    "monte_carlo_paths": _scenario_paths,
}
