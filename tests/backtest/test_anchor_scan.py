"""The single-pass scan kernel.

`scan_realized` is a state machine, and its two subtle rules are the ones worth
stating as tests rather than trusting to reading:

* an entry signal **wins** over an exit on a shared bar, so the position stays
  open and the trade keeps its **original** anchor;
* under `flatten_eod` the session-last bar closes regardless, and an entry signal
  on such a bar opens nothing.

The kernel takes plain lists, so these are exact hand-computed cases rather than
whole backtests. Its agreement with the shipped resolver is a separate file;
what is pinned here is the kernel's own contract, including the three
optimizations that are easy to break silently — the `list.index` skip over flat
stretches, the strictness-into-the-latched-threshold fold, and the specialized
take-profit/stop-loss loop.
"""

from __future__ import annotations

import math

import pytest

from mktlib.backtest._scan import ScanLeg, scan_realized

NAN = float("nan")


def _up(value: list[float], level: float, *, strict: bool = True) -> ScanLeg:
    return ScanLeg(
        value=value, threshold=[level] * len(value), direction="above", strict=strict
    )


def _down(value: list[float], level: float, *, strict: bool = True) -> ScanLeg:
    return ScanLeg(
        value=value, threshold=[level] * len(value), direction="below", strict=strict
    )


def _run(entry, legs, **kw):  # noqa: ANN001, ANN003
    return scan_realized(entry=entry, legs=tuple(legs), n=len(entry), **kw)


# --- the basic chain --------------------------------------------------------


def test_one_trade_opens_and_closes() -> None:
    value = [10.0, 10.0, 10.0, 20.0, 10.0]
    entry = [True, False, False, False, False]
    out = _run(entry, [_up(value, 15.0)])

    assert out.realized == [True, False, False, False, False]
    assert out.entries == [0]
    assert out.exits == [3]
    assert out.winners == [0]


def test_next_entry_after_the_exit_reopens() -> None:
    value = [10.0, 20.0, 10.0, 10.0, 20.0]
    entry = [True, False, True, False, False]
    out = _run(entry, [_up(value, 15.0)])

    assert out.entries == [0, 2]
    assert out.exits == [1, 4]


def test_entry_while_held_is_suppressed() -> None:
    """A signal inside an open position opens nothing and moves no anchor."""
    value = [10.0, 11.0, 12.0, 30.0]
    entry = [True, True, True, False]
    out = _run(entry, [_up(value, 25.0)])

    assert out.entries == [0]
    assert out.realized == [True, False, False, False]


def test_position_open_at_the_end_records_no_exit() -> None:
    value = [10.0, 10.0, 10.0]
    entry = [True, False, False]
    out = _run(entry, [_up(value, 99.0)])

    assert out.entries == [0]
    assert out.exits == [None]
    assert out.winners == [-1]


def test_no_entries_at_all() -> None:
    out = _run([False] * 4, [_up([1.0] * 4, 0.5)])
    assert out.entries == []
    assert out.realized == [False] * 4


# --- entry wins on a shared bar ---------------------------------------------


def test_entry_on_the_firing_bar_keeps_the_position_open() -> None:
    """The rule that is easiest to get backwards.

    Bar 2 would close the trade, but an entry signal shares it — so nothing
    closes, and the trade runs on to bar 4 still anchored at bar 0.
    """
    value = [10.0, 10.0, 50.0, 10.0, 50.0]
    entry = [True, False, True, False, False]
    out = _run(entry, [_up(value, 25.0)])

    assert out.entries == [0], "the shared-bar signal must not open a position"
    assert out.exits == [4], "and must not close the one already open"


# --- strictness -------------------------------------------------------------


def test_strict_does_not_fire_on_equality() -> None:
    value = [10.0, 15.0, 20.0]
    entry = [True, False, False]
    assert _run(entry, [_up(value, 15.0, strict=True)]).exits == [2]


def test_non_strict_fires_on_equality() -> None:
    value = [10.0, 15.0, 20.0]
    entry = [True, False, False]
    assert _run(entry, [_up(value, 15.0, strict=False)]).exits == [1]


def test_strict_below_does_not_fire_on_equality() -> None:
    value = [10.0, 5.0, 1.0]
    entry = [True, False, False]
    assert _run(entry, [_down(value, 5.0, strict=True)]).exits == [2]


def test_strictness_survives_the_nextafter_fold() -> None:
    """The fold must not shift the level by anything a price could land on."""
    level = 100.0
    assert math.nextafter(level, float("inf")) > level
    value = [1.0, level, math.nextafter(level, float("inf"))]
    entry = [True, False, False]
    assert _run(entry, [_up(value, level, strict=True)]).exits == [2]


# --- nulls ------------------------------------------------------------------


def test_nan_value_never_fires() -> None:
    value = [10.0, NAN, NAN, 50.0]
    entry = [True, False, False, False]
    assert _run(entry, [_up(value, 25.0)]).exits == [3]


def test_nan_threshold_never_fires() -> None:
    value = [10.0, 50.0, 50.0]
    entry = [True, False, False]
    leg = ScanLeg(value=value, threshold=[NAN] * 3, direction="above", strict=True)
    assert _run(entry, [leg]).exits == [None]


@pytest.mark.parametrize("direction", ["above", "below"])
@pytest.mark.parametrize("strict", [True, False])
def test_nan_is_inert_in_every_configuration(direction: str, strict: bool) -> None:
    value = [1.0, NAN, NAN]
    entry = [True, False, False]
    leg = ScanLeg(
        value=value,
        threshold=[0.5] * 3,
        direction=direction,  # type: ignore[arg-type]
        strict=strict,
    )
    assert _run(entry, [leg]).exits == [None]


# --- two legs ---------------------------------------------------------------


def test_take_profit_and_stop_loss_report_the_winning_leg() -> None:
    value = [10.0, 30.0, 10.0, 1.0]
    entry = [True, False, False, False]
    legs = [_up(value, 25.0), _down(value, 5.0)]
    assert _run(entry, legs).winners == [0]

    entry2 = [False, False, True, False]
    assert _run(entry2, legs).winners == [1]


def test_a_bar_tagging_both_legs_resolves_in_declaration_order() -> None:
    """Whichever leg was declared first wins — in *both* code paths.

    Index 0 is expected for either ordering precisely because index 0 is
    whichever leg came first. The specialized take-profit/stop-loss loop
    originally hardcoded "upward wins", which disagreed with the general loop on
    exactly this bar; a tie must not resolve differently for having taken a
    faster route.
    """
    value = [10.0, 100.0]
    entry = [True, False]
    assert _run(entry, [_up(value, 20.0), _down(value, 200.0)]).winners == [0]
    assert _run(entry, [_down(value, 200.0), _up(value, 20.0)]).winners == [0]


def test_specialized_and_general_paths_agree_on_a_shared_bar() -> None:
    """Passing `fixed` forces the general loop; the answer must not move."""
    value = [10.0, 100.0]
    entry = [True, False]
    for legs in (
        [_up(value, 20.0), _down(value, 200.0)],
        [_down(value, 200.0), _up(value, 20.0)],
    ):
        specialized = _run(entry, legs).winners
        general = _run(entry, legs, fixed=[False, False]).winners
        assert specialized == general


def test_two_legs_over_different_columns() -> None:
    """Falls off the specialized pair loop onto the general one."""
    a = [10.0, 10.0, 10.0]
    b = [0.0, 0.0, 9.0]
    entry = [True, False, False]
    legs = [
        ScanLeg(value=a, threshold=[99.0] * 3, direction="above", strict=True),
        ScanLeg(value=b, threshold=[5.0] * 3, direction="above", strict=True),
    ]
    out = _run(entry, legs)
    assert out.exits == [2]
    assert out.winners == [1]


# --- fixed predicate and session boundaries ---------------------------------


def test_fixed_predicate_closes_the_position() -> None:
    value = [10.0] * 4
    entry = [True, False, False, False]
    out = _run(entry, [_up(value, 99.0)], fixed=[False, False, True, False])
    assert out.exits == [2]
    assert out.winners == [-1], "a fixed close is attributed to no leg"


def test_session_last_forces_a_close() -> None:
    value = [10.0] * 4
    entry = [True, False, False, False]
    out = _run(
        entry, [_up(value, 99.0)], session_last=[False, False, True, False]
    )
    assert out.exits == [2]
    assert out.winners == [-1]


def test_entry_on_a_session_last_bar_opens_nothing() -> None:
    value = [10.0] * 4
    entry = [True, False, False, False]
    out = _run(entry, [_up(value, 99.0)], session_last=[True, False, False, False])
    assert out.entries == []
    assert out.realized == [False] * 4


def test_session_close_beats_a_shared_entry_signal() -> None:
    """`entry` wins over an exit — but not over a forced flatten."""
    value = [10.0] * 5
    entry = [True, False, True, False, False]
    out = _run(
        entry,
        [_up(value, 99.0)],
        session_last=[False, False, True, False, False],
    )
    assert out.entries == [0]
    assert out.exits == [2]


# --- the flat-stretch skip --------------------------------------------------


def test_long_flat_stretch_is_traversed_correctly() -> None:
    """The `list.index` skip must not step over a later entry."""
    n = 500
    value = [10.0] * n
    entry = [False] * n
    entry[10] = True
    entry[400] = True
    value[20] = 99.0
    value[450] = 99.0

    out = _run(entry, [_up(value, 50.0)])
    assert out.entries == [10, 400]
    assert out.exits == [20, 450]
