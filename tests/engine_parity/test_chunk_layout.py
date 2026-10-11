"""Results do not depend on how the input is split into chunks.

A frame built in one piece and the same rows held in many small chunks are
the same data.  Each scenario runs on its inputs as built and again with
every input frame and series rebuilt from 7-row pieces (``pl.concat`` with
``rechunk=False``); the two results must be identical: exact floats,
dtypes, column order and row order.

That holds for the backtest, the calendars and the generators.  It does not
hold for the scalar metrics, nor for the reports built from them.  A polars
float reduction (``mean``, ``var``, ``std``) over a series held in several
chunks can differ in the last digits from the same reduction over one
chunk, and the metrics reduce the caller's series as it arrives.  Those
scenarios are run to the exact comparison as expected failures, and to a
relative tolerance as a bound on the size of the difference.
"""

from __future__ import annotations

import polars as pl
import pytest

from tests.engine_parity import _corpus


def _assert_layout_independent(scenario: _corpus.Scenario) -> None:
    single = _corpus.run_fresh(scenario, _corpus.as_built)
    chunked = _corpus.run_fresh(scenario, _corpus.in_chunks)
    _corpus.assert_exact(chunked, single)


def test_layout_splits_a_frame_into_seven_row_chunks() -> None:
    frame = _corpus.returns_frame(100)
    assert frame.n_chunks() == 1
    chunked = _corpus.in_chunks(frame)
    assert chunked.n_chunks() == 15
    assert chunked.equals(frame)
    assert [c.height for c in chunked.iter_slices(7)][:2] == [7, 7]


def test_layout_splits_a_series_into_seven_row_chunks() -> None:
    series = _corpus.returns_frame(100)["return"]
    chunked = _corpus.in_chunks(series)
    assert chunked.n_chunks() == 15
    assert chunked.equals(series)
    assert [len(c) for c in chunked.get_chunks()] == [7] * 14 + [2]


def test_layout_reaches_the_library(monkeypatch: pytest.MonkeyPatch) -> None:
    """A golden scenario run under the layout hands ``run`` a chunked frame."""
    from tests.backtest import test_golden_baseline as golden

    seen: list[int] = []
    real_run = golden.run

    def spy(df: pl.DataFrame, *args: object, **kwargs: object) -> object:
        seen.append(df.n_chunks())
        return real_run(df, *args, **kwargs)  # type: ignore[call-overload]

    monkeypatch.setattr(golden, "run", spy)
    _corpus.run_fresh(_corpus.BACKTEST_SCENARIOS["golden/multi"], _corpus.in_chunks)
    assert seen and min(seen) > 1


@pytest.mark.parametrize("name", sorted(_corpus.BACKTEST_SCENARIOS))
def test_backtest_does_not_depend_on_chunk_layout(name: str, scan_backend: str) -> None:
    _assert_layout_independent(_corpus.BACKTEST_SCENARIOS[name])


@pytest.mark.parametrize("name", sorted(_corpus.CALENDAR_SCENARIOS))
def test_calendar_does_not_depend_on_chunk_layout(name: str) -> None:
    _assert_layout_independent(_corpus.CALENDAR_SCENARIOS[name])


_REDUCTION = (
    "a float reduction over a series in several chunks can differ in the "
    "last digits from the reduction over one chunk"
)

#: The largest relative difference a chunk layout may make to a metric.
CHUNK_REL_TOL = 1e-12


@pytest.mark.xfail(reason=_REDUCTION, strict=False)
@pytest.mark.parametrize("name", sorted(_corpus.REPORT_SCENARIOS))
def test_report_does_not_depend_on_chunk_layout(name: str) -> None:
    pytest.importorskip("plotly")
    _assert_layout_independent(_corpus.REPORT_SCENARIOS[name])


@pytest.mark.xfail(reason=_REDUCTION, strict=False)
@pytest.mark.parametrize("name", sorted(_corpus.METRIC_SCENARIOS))
def test_metric_does_not_depend_on_chunk_layout(name: str) -> None:
    _assert_layout_independent(_corpus.METRIC_SCENARIOS[name])


@pytest.mark.parametrize(
    "name", sorted(n for n in _corpus.REPORT_SCENARIOS if n.startswith("metrics/"))
)
def test_report_metrics_move_by_rounding_only_under_chunk_layout(name: str) -> None:
    scenario = _corpus.REPORT_SCENARIOS[name]
    single = _corpus.run_fresh(scenario, _corpus.as_built)
    chunked = _corpus.run_fresh(scenario, _corpus.in_chunks)
    _corpus.assert_exact(chunked, single, rel_tol=CHUNK_REL_TOL)


@pytest.mark.parametrize("name", sorted(_corpus.METRIC_SCENARIOS))
def test_metric_moves_by_rounding_only_under_chunk_layout(name: str) -> None:
    scenario = _corpus.METRIC_SCENARIOS[name]
    single = _corpus.run_fresh(scenario, _corpus.as_built)
    chunked = _corpus.run_fresh(scenario, _corpus.in_chunks)
    _corpus.assert_exact(chunked, single, rel_tol=CHUNK_REL_TOL)


@pytest.mark.parametrize(
    "name",
    sorted(
        name
        for name in _corpus.DATA_SCENARIOS
        if name.startswith(("ticks_to_ohlcv/", "monte_carlo/gbm_bootstrap"))
    ),
)
def test_generator_does_not_depend_on_chunk_layout(name: str) -> None:
    """The generators that take a frame or a series as input."""
    _assert_layout_independent(_corpus.DATA_SCENARIOS[name])
