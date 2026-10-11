"""Results do not depend on a row order polars leaves unspecified.

Each scenario runs twice: once as is, and once with every unpinned join and
group_by returning its rows reversed (the ``unspecified_order`` fixture).
The two results must be identical: exact floats, dtypes, column order and
row order.  A difference means some result is read off an order that is an
accident of the engine that produced it.

Sorts are not perturbed by that fixture.  The ``tie_order`` fixture is its
counterpart for them: it reverses the input order of rows that are equal on
a sort key, and the tests at the end of this module run under it.
"""

from __future__ import annotations

import dataclasses
import datetime
from collections.abc import Callable
from contextlib import AbstractContextManager

import polars as pl
import pytest

from mktlib import data as mdata
from mktlib.backtest import Condition, Custom, FlattenSchedule, run
from mktlib.backtest._flatten import build_flatten_masks
from mktlib.scheduling import get_calendar
from tests.backtest import test_golden_baseline as golden
from tests.engine_parity import _corpus
from tests.engine_parity._chaos import Perturbation

Chaos = Callable[[], AbstractContextManager[Perturbation]]


def _assert_order_independent(scenario: _corpus.Scenario, chaos: Chaos) -> None:
    plain = _corpus.run_fresh(scenario)
    with chaos():
        perturbed = _corpus.run_fresh(scenario)
    _corpus.assert_exact(perturbed, plain)


# ---------------------------------------------------------------------------
# The fixture itself
# ---------------------------------------------------------------------------


def _left() -> pl.DataFrame:
    return pl.DataFrame({"k": [1, 2, 3, 4], "a": [10.0, 20.0, 30.0, 40.0]})


def _right() -> pl.DataFrame:
    return pl.DataFrame({"k": [1, 2, 3, 4], "b": ["w", "x", "y", "z"]})


def test_fixture_reverses_a_join_that_does_not_pin_order(
    unspecified_order: Chaos,
) -> None:
    expected = _left().join(_right(), on="k", how="left", maintain_order="left")
    with unspecified_order() as seen:
        bare = _left().join(_right(), on="k", how="left")
        none = _left().join(_right(), on="k", maintain_order="none")
    assert bare["k"].to_list() == [4, 3, 2, 1]
    assert none["k"].to_list() == [4, 3, 2, 1]
    assert bare.equals(expected.reverse())
    assert seen.reversed_calls == 2


@pytest.mark.parametrize("order", ["left", "right", "left_right", "right_left"])
def test_fixture_leaves_a_join_that_pins_order(
    unspecified_order: Chaos, order: str
) -> None:
    expected = _left().join(_right(), on="k", maintain_order=order)  # type: ignore[arg-type]
    with unspecified_order() as seen:
        pinned = _left().join(_right(), on="k", maintain_order=order)  # type: ignore[arg-type]
    assert pinned.equals(expected)
    assert pinned["k"].to_list() == [1, 2, 3, 4]
    assert seen.reversed_calls == 0


def _groups() -> pl.DataFrame:
    return pl.DataFrame({"g": [3, 3, 1, 2, 2, 2], "v": [1, 2, 3, 4, 5, 6]})


def test_fixture_reverses_a_group_by_that_does_not_pin_order(
    unspecified_order: Chaos,
) -> None:
    ordered = _groups().group_by("g", maintain_order=True).agg(pl.col("v").sum())
    lens = _groups().group_by("g", maintain_order=True).len()
    with unspecified_order() as seen:
        bare_agg = _groups().group_by("g").agg(pl.col("v").sum())
        bare_len = _groups().group_by("g").len()
    # The engine's own order is unspecified, so the check is on the sorted
    # frame for content and on the reversal count for the perturbation.
    assert bare_agg.sort("g").equals(ordered.sort("g"))
    assert bare_len.sort("g").equals(lens.sort("g"))
    assert seen.reversed_calls == 2


def test_fixture_reversal_of_a_group_by_is_a_reversal(
    unspecified_order: Chaos, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Against a group_by whose own order is known, the result is reversed."""
    from polars.dataframe.group_by import GroupBy

    ordered = _groups().group_by("g", maintain_order=True)
    expected_agg = ordered.agg(pl.col("v").sum())
    expected_len = ordered.len()
    real_agg = GroupBy.agg

    def first_seen(self: GroupBy, *aggs: object, **named: object) -> pl.DataFrame:
        """``agg`` in first-seen order, standing in for 'some engine order'."""
        pinned = self.maintain_order
        self.maintain_order = True
        try:
            return real_agg(self, *aggs, **named)  # type: ignore[arg-type]
        finally:
            self.maintain_order = pinned

    monkeypatch.setattr(GroupBy, "agg", first_seen)
    with unspecified_order():
        bare_agg = _groups().group_by("g").agg(pl.col("v").sum())
        bare_len = _groups().group_by("g").len()
    assert bare_agg.equals(expected_agg.reverse())
    assert bare_agg["g"].to_list() == [2, 1, 3]
    assert bare_len.equals(expected_len.reverse())


def test_fixture_leaves_a_group_by_that_pins_order(unspecified_order: Chaos) -> None:
    expected_agg = _groups().group_by("g", maintain_order=True).agg(pl.col("v").sum())
    expected_len = _groups().group_by("g", maintain_order=True).len()
    with unspecified_order() as seen:
        agg = _groups().group_by("g", maintain_order=True).agg(pl.col("v").sum())
        length = _groups().group_by("g", maintain_order=True).len()
    assert agg.equals(expected_agg)
    assert agg["g"].to_list() == [3, 1, 2]
    assert length.equals(expected_len)
    assert seen.reversed_calls == 0


def test_fixture_leaves_join_asof_and_sorts(unspecified_order: Chaos) -> None:
    quotes = pl.DataFrame({"t": [1, 4, 7], "q": [0.1, 0.4, 0.7]})
    bars = pl.DataFrame({"t": [2, 5, 8, 9]})
    expected = bars.join_asof(quotes, on="t")
    with unspecified_order() as seen:
        asof = bars.join_asof(quotes, on="t")
        ordered = _groups().sort("g", maintain_order=True)
    assert asof.equals(expected)
    assert ordered["v"].to_list() == [3, 4, 5, 6, 1, 2]
    assert seen.reversed_calls == 0


def test_fixture_is_removed_on_exit(unspecified_order: Chaos) -> None:
    before = _left().join(_right(), on="k", how="left")
    with unspecified_order():
        pass
    after = _left().join(_right(), on="k", how="left")
    assert after.equals(before)


def test_fixture_records_the_library_call_site(unspecified_order: Chaos) -> None:
    with unspecified_order() as seen:
        get_calendar("XNYS").schedule(_corpus.CALENDAR_START, _corpus.CALENDAR_END)
    sites = seen.library_sites(pinned=True) + seen.library_sites(pinned=False)
    assert [(s.file, s.kind) for s in sites] == [
        ("mktlib/scheduling/calendar.py", "join")
    ]


# ---------------------------------------------------------------------------
# The corpus
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(_corpus.BACKTEST_SCENARIOS))
def test_backtest_does_not_depend_on_unspecified_order(
    name: str, scan_backend: str, unspecified_order: Chaos
) -> None:
    _assert_order_independent(_corpus.BACKTEST_SCENARIOS[name], unspecified_order)


@pytest.mark.parametrize("name", sorted(_corpus.REPORT_SCENARIOS))
def test_report_does_not_depend_on_unspecified_order(
    name: str, unspecified_order: Chaos
) -> None:
    pytest.importorskip("plotly")
    _assert_order_independent(_corpus.REPORT_SCENARIOS[name], unspecified_order)


@pytest.mark.parametrize("name", sorted(_corpus.DATA_SCENARIOS))
def test_generator_does_not_depend_on_unspecified_order(
    name: str, unspecified_order: Chaos
) -> None:
    _assert_order_independent(_corpus.DATA_SCENARIOS[name], unspecified_order)


@pytest.mark.parametrize("name", sorted(_corpus.CALENDAR_SCENARIOS))
def test_calendar_does_not_depend_on_unspecified_order(
    name: str, unspecified_order: Chaos
) -> None:
    _assert_order_independent(_corpus.CALENDAR_SCENARIOS[name], unspecified_order)


@pytest.mark.parametrize("name", sorted(_corpus.METRIC_SCENARIOS))
def test_metric_does_not_depend_on_unspecified_order(
    name: str, unspecified_order: Chaos
) -> None:
    _assert_order_independent(_corpus.METRIC_SCENARIOS[name], unspecified_order)


# ---------------------------------------------------------------------------
# The corpus covers what it claims to
# ---------------------------------------------------------------------------


def test_every_golden_scenario_is_in_the_corpus() -> None:
    assert {f"golden/{name}" for name in golden.SCENARIOS} <= set(
        _corpus.BACKTEST_SCENARIOS
    )


def test_every_public_generator_is_in_the_corpus() -> None:
    covered = {key.split("/")[0] for key in _corpus.DATA_SCENARIOS}
    assert covered == _corpus.data_functions()
    processes = {p.name.lower() for p in mdata.Process}
    assert processes <= {
        key.split("/")[1].split("_")[0]
        for key in _corpus.DATA_SCENARIOS
        if key.startswith("monte_carlo/")
    }


@pytest.mark.parametrize("name", _corpus.CALENDAR_NAMES)
def test_calendar_range_holds_a_holiday_and_an_early_close(name: str) -> None:
    cal = get_calendar(name)
    start, end = _corpus.CALENDAR_START, _corpus.CALENDAR_END
    weekdays = pl.date_range(start, end, eager=True).filter(
        pl.date_range(start, end, eager=True).dt.weekday() <= 5
    )
    # A calendar with no weekday closure, or no early close, anywhere in a
    # decade has none for the range to cover.
    decade = (datetime.date(2016, 1, 1), datetime.date(2026, 12, 31))
    all_weekdays = pl.date_range(*decade, eager=True)
    all_weekdays = all_weekdays.filter(all_weekdays.dt.weekday() <= 5)
    closes_ever = len(cal.valid_days(*decade)) < len(all_weekdays)
    closes_in_range = cal.schedule(start, end).height < len(weekdays)
    assert closes_in_range == closes_ever, f"{name}: no weekday closure in range"
    early_ever = bool(cal._early_close_map(*decade))
    early_in_range = bool(cal._early_close_map(start, end))
    assert early_in_range == early_ever, f"{name}: no early close in range"
    filtered = cal.filter_market_hours(_corpus.calendar_bars(name))
    assert 0 < filtered.height < _corpus.calendar_bars(name).height


def test_most_calendars_have_a_holiday_and_an_early_close_in_range() -> None:
    start, end = _corpus.CALENDAR_START, _corpus.CALENDAR_END
    n_weekdays = sum(
        (start + datetime.timedelta(days=i)).weekday() < 5
        for i in range((end - start).days + 1)
    )
    calendars = [get_calendar(name) for name in _corpus.CALENDAR_NAMES]
    with_holiday = [c.name for c in calendars if len(c.valid_days(start, end)) < n_weekdays]
    with_early = [c.name for c in calendars if c._early_close_map(start, end)]
    assert len(with_holiday) >= 11, with_holiday
    assert len(with_early) >= 10, with_early


# ---------------------------------------------------------------------------
# Rows that tie on a sort key
# ---------------------------------------------------------------------------


def _tied() -> pl.DataFrame:
    return pl.DataFrame({"k": [2, 1, 2, 1, 2], "row": [0, 1, 2, 3, 4]})


def test_tie_fixture_reverses_ties_of_a_sort_that_does_not_pin_them(
    tie_order: Chaos,
) -> None:
    with tie_order() as seen:
        bare = _tied().sort("k")
        by = _tied().select(pl.col("row").sort_by("k"))
        by_expr = _tied().select(pl.col("row").sort_by(pl.col("k"), descending=True))
    assert bare["k"].to_list() == [1, 1, 2, 2, 2]
    assert bare["row"].to_list() == [3, 1, 4, 2, 0]
    assert by["row"].to_list() == [3, 1, 4, 2, 0]
    assert by_expr["row"].to_list() == [4, 2, 0, 3, 1]
    assert seen.reversed_calls == 3


def test_tie_fixture_leaves_a_sort_that_pins_ties(tie_order: Chaos) -> None:
    with tie_order() as seen:
        pinned = _tied().sort("k", maintain_order=True)
        by = _tied().select(pl.col("row").sort_by("k", maintain_order=True))
        unique = _tied().sort("row", descending=True)
    assert pinned["row"].to_list() == [1, 3, 0, 2, 4]
    assert by["row"].to_list() == [1, 3, 0, 2, 4]
    # No two rows tie on ``row``: the perturbation has nothing to move.
    assert unique["row"].to_list() == [4, 3, 2, 1, 0]
    assert seen.reversed_calls == 1


def test_tie_fixture_is_removed_on_exit(tie_order: Chaos) -> None:
    sort, sort_by = pl.DataFrame.sort, pl.Expr.sort_by
    with tie_order():
        assert pl.DataFrame.sort is not sort
    assert pl.DataFrame.sort is sort
    assert pl.Expr.sort_by is sort_by


def _assert_tie_independent(scenario: _corpus.Scenario, chaos: Chaos) -> None:
    plain = _corpus.run_fresh(scenario)
    with chaos():
        perturbed = _corpus.run_fresh(scenario)
    _corpus.assert_exact(perturbed, plain)


def _duplicated_stamps() -> list[datetime.datetime]:
    """Two sessions of 30-minute bars with every timestamp present twice."""
    stamps: list[datetime.datetime] = []
    for day in (2, 3):
        ts = datetime.datetime(2024, 1, day, 9, 30)
        while ts <= datetime.datetime(2024, 1, day, 15, 30):
            stamps.extend([ts, ts])
            ts += datetime.timedelta(minutes=30)
    return stamps


def _row_is(*residues: int) -> Condition:
    """True on the rows whose position modulo 8 is one of *residues*."""
    return Custom((pl.int_range(pl.len()) % 8).is_in(list(residues)))


@dataclasses.dataclass(frozen=True, slots=True)
class _RowStrategy:
    """Hold the row at position ``enter_on`` of every eight, and no other."""

    enter_on: int

    def entry(self) -> Condition:
        return _row_is(self.enter_on)

    def exit(self) -> Condition:
        return _row_is(self.enter_on + 1)


def test_flatten_bar_among_duplicate_timestamps_is_the_last_input_row() -> None:
    """Of the rows sharing a session's last timestamp, the later is flattened."""
    stamps = pl.Series("date", _duplicated_stamps())
    flatten, _ = build_flatten_masks(stamps, get_calendar("XNYS"), FlattenSchedule())
    hits = [i for i, hit in enumerate(flatten.to_list()) if hit]
    assert hits == [len(stamps) // 2 - 1, len(stamps) - 1]


def test_flatten_masks_do_not_depend_on_tie_order(tie_order: Chaos) -> None:
    def scenario(layout: _corpus.Layout) -> object:
        stamps = pl.Series("date", _duplicated_stamps())
        return list(
            build_flatten_masks(
                stamps,
                get_calendar("XNYS"),
                FlattenSchedule(block_entry_minutes_before_close=30),
            )
        )

    _assert_tie_independent(scenario, tie_order)


def test_run_with_duplicate_timestamps_does_not_depend_on_tie_order(
    scan_backend: str, tie_order: Chaos
) -> None:
    def scenario(layout: _corpus.Layout) -> object:
        frame = golden._frame(_duplicated_stamps(), seed=5)
        result = run(
            frame,
            _RowStrategy(0),  # type: ignore[arg-type]
            calendar=get_calendar("XNYS"),
            flatten_eod=True,
        )
        return golden._artifacts(result)

    _assert_tie_independent(scenario, tie_order)


def _tied_dual_trades() -> pl.DataFrame:
    """A long and a short trade on the two rows of one timestamp, repeatedly.

    Rows ``2k`` and ``2k + 1`` share a timestamp.  The short strategy holds
    row 2 and the long strategy row 3 of every eight, so the two trades have
    the same ``entry_date``.
    """
    frame = golden._frame(_duplicated_stamps(), seed=5)
    result = run(
        frame,
        _RowStrategy(3),  # type: ignore[arg-type]
        short_strategy=_RowStrategy(2),  # type: ignore[arg-type]
    )
    return result.trades


def test_dual_trades_can_tie_on_entry_date() -> None:
    trades = _tied_dual_trades()
    tied = trades.filter(pl.col("entry_date").is_duplicated())
    assert tied.height >= 4
    assert set(tied["side"].to_list()) == {1, -1}


def test_dual_trades_tied_on_entry_date_do_not_depend_on_tie_order(
    scan_backend: str, tie_order: Chaos
) -> None:
    _assert_tie_independent(lambda layout: _tied_dual_trades(), tie_order)


def test_dual_trades_tied_on_entry_date_list_the_long_side_first() -> None:
    tied = _tied_dual_trades().filter(pl.col("entry_date").is_duplicated())
    assert tied["side"].to_list() == [1, -1] * (tied.height // 2)


@pytest.mark.parametrize("name", sorted(_corpus.BACKTEST_SCENARIOS))
def test_backtest_does_not_depend_on_tie_order(
    name: str, scan_backend: str, tie_order: Chaos
) -> None:
    _assert_tie_independent(_corpus.BACKTEST_SCENARIOS[name], tie_order)


@pytest.mark.parametrize("name", sorted(_corpus.REPORT_SCENARIOS))
def test_report_does_not_depend_on_tie_order(name: str, tie_order: Chaos) -> None:
    pytest.importorskip("plotly")
    _assert_tie_independent(_corpus.REPORT_SCENARIOS[name], tie_order)
