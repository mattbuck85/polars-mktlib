"""What a result owes to how the input is split into chunks.

A frame built in one piece and the same rows held in many small chunks are
the same data.  Each scenario runs on its inputs as built and again with
every input frame and series rebuilt from 7-row pieces (``pl.concat`` with
``rechunk=False``).

For the backtest, the calendars and the generators the two results are
identical: exact floats, dtypes, column order and row order.

For the scalar metrics and the reports built from them they need not be.
polars sums each chunk of a series separately, so a float reduction
(``mean``, ``var``, ``std``) over several chunks can differ in the last
digits from the same reduction over one, and the metrics reduce the
caller's series as it arrives.  Those scenarios fall into two named sets:

* the ones that are identical on this corpus, which are exact tests; and
* :data:`DIFFERING`, the ones that are not.  They have one kind of test:
  the two results have the same structure (types, keys, lengths, dtypes,
  column and row order, every non-float value, and for a tearsheet the
  same text around its numbers), and each float is within
  :data:`CORPUS_REL_BOUND` of its counterpart.

``CORPUS_REL_BOUND`` is a bound on this corpus, 600 daily returns in 7-row
chunks.  It is not a bound on the library: the difference grows with the
length of the series.
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


#: The calendar scenarios that take a frame.  ``schedule`` and
#: ``trading_index`` take dates and have no input to lay out.
_CALENDAR_FRAME_SCENARIOS = sorted(
    name
    for name in _corpus.CALENDAR_SCENARIOS
    if "/filter_market_hours" in name
)


def test_calendar_frame_scenarios_are_the_filters() -> None:
    assert len(_CALENDAR_FRAME_SCENARIOS) == 2 * len(_corpus.CALENDAR_NAMES)


@pytest.mark.parametrize("name", _CALENDAR_FRAME_SCENARIOS)
def test_calendar_does_not_depend_on_chunk_layout(name: str) -> None:
    _assert_layout_independent(_corpus.CALENDAR_SCENARIOS[name])


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


# ---------------------------------------------------------------------------
# Metrics and reports
# ---------------------------------------------------------------------------

_SCALAR_SCENARIOS: dict[str, _corpus.Scenario] = {
    **_corpus.REPORT_SCENARIOS,
    **_corpus.METRIC_SCENARIOS,
}

#: The metric and report scenarios whose result differs in the last digits
#: between the input as built and the input in 7-row chunks.
DIFFERING = frozenset(
    {
        "calculate/kelly_criterion",
        "calculate/omega",
        "calculate/payoff_ratio",
        "calculate/profit_factor",
        "calculate/sharpe",
        "calculate/sortino",
        "html/mc_bootstrap",
        "html/mc_gaussian",
        "html/mc_students_t",
        "metrics/mc_bootstrap",
        "metrics/mc_gaussian",
        "metrics/mc_students_t",
        "metrics/series",
        "monte_carlo_paths",
        "simulate/cvar/bootstrap",
        "simulate/cvar/gaussian",
        "simulate/cvar/students_t",
        "simulate/var/gaussian",
        "simulate/var/students_t",
    }
)

#: Relative bound on the difference between two floats of a ``DIFFERING``
#: result.  It sits well above the largest difference seen on this corpus
#: and is not a property of the library.
CORPUS_REL_BOUND = 1e-12


def _needs_plotly(name: str) -> None:
    if name.startswith("html/"):
        pytest.importorskip("plotly")


def test_differing_scenarios_are_in_the_corpus() -> None:
    assert len(_SCALAR_SCENARIOS) == len(_corpus.REPORT_SCENARIOS) + len(
        _corpus.METRIC_SCENARIOS
    )
    assert DIFFERING <= set(_SCALAR_SCENARIOS)


@pytest.mark.parametrize("name", sorted(set(_SCALAR_SCENARIOS) - DIFFERING))
def test_scalar_result_does_not_depend_on_chunk_layout(name: str) -> None:
    _needs_plotly(name)
    _assert_layout_independent(_SCALAR_SCENARIOS[name])


@pytest.mark.parametrize("name", sorted(DIFFERING))
def test_scalar_result_keeps_its_structure_under_chunk_layout(
    name: str,
) -> None:
    """Same structure, and floats within the bound set for this corpus."""
    _needs_plotly(name)
    scenario = _SCALAR_SCENARIOS[name]
    single = _corpus.run_fresh(scenario, _corpus.as_built)
    chunked = _corpus.run_fresh(scenario, _corpus.in_chunks)
    _corpus.assert_exact(chunked, single, rel_tol=CORPUS_REL_BOUND)


# ---------------------------------------------------------------------------
# The structural comparison
# ---------------------------------------------------------------------------


def _packed(values: list[float]) -> str:
    """A float array as a tearsheet chart holds it."""
    import base64
    import struct

    raw = struct.pack(f"<{len(values)}d", *values)
    encoded = base64.b64encode(raw).decode().replace("/", "\\u002f")
    return f'{{"dtype":"f8","bdata":"{encoded}"}}'


def _sheet(
    sharpe: float = 1.25,
    label: str = "Sharpe",
    rows: int = 2,
    curve: tuple[float, ...] = (1.0, 1.5, 0.3333333333333333),
) -> str:
    body = f"<tr><td>{label}</td><td>{sharpe!r}</td></tr>"
    body += "<tr><td>Trades</td><td>80</td></tr>" * (rows - 1)
    return f"<table>{body}</table><script>{_packed(list(curve))}</script>"


def test_text_structure_reads_written_and_packed_numbers() -> None:
    text, numbers = _corpus.text_structure(_sheet())
    # The packed values, then the written ones: the two cells, and the
    # length the packed array is replaced by.
    assert numbers == [1.0, 1.5, 0.3333333333333333, 1.25, 80.0, 3.0]
    assert "<td>Sharpe</td>" in "".join(text)
    assert "bdata" in "".join(text) and "AAAA" not in "".join(text)


def _differs(left: str, right: str) -> bool:
    try:
        _corpus.assert_exact(left, right, rel_tol=CORPUS_REL_BOUND)
    except AssertionError:
        return True
    return False


def test_structural_comparison_accepts_a_last_digit() -> None:
    nudged = _sheet(
        sharpe=1.2500000000000002, curve=(1.0, 1.5, 0.33333333333333337)
    )
    assert nudged != _sheet()
    assert not _differs(nudged, _sheet())


@pytest.mark.parametrize(
    "other",
    [
        _sheet(rows=1),
        _sheet(rows=3),
        _sheet(label="Sortino"),
        _sheet(sharpe=1.2500001),
        _sheet(curve=(1.0, 1.5)),
        _sheet(curve=(1.0, 1.5, 0.3333334)),
        _sheet().replace("80", "81"),
    ],
    ids=[
        "missing_row",
        "extra_row",
        "label",
        "written_number",
        "packed_length",
        "packed_number",
        "integer",
    ],
)
def test_structural_comparison_rejects(other: str) -> None:
    assert _differs(other, _sheet())


def test_structural_comparison_keeps_frames_exact_outside_floats() -> None:
    frame = pl.DataFrame({"n": [1, 2], "x": [0.1, 0.2], "s": ["a", "b"]})
    nudged = frame.with_columns(pl.col("x") * (1 + 2e-16))
    bound = CORPUS_REL_BOUND
    _corpus.assert_exact(nudged, frame, rel_tol=bound)
    for other in (
        frame.with_columns(pl.col("n") + 1),
        frame.with_columns(pl.col("s").str.to_uppercase()),
        frame.with_columns(pl.col("x") * 1.001),
        frame.with_columns(pl.col("x").cast(pl.Float32)),
        frame.select("x", "n", "s"),
        frame.head(1),
        frame.reverse(),
    ):
        with pytest.raises(AssertionError):
            _corpus.assert_exact(other, frame, rel_tol=bound)
