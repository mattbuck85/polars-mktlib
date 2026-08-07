"""#83 — an offset flatten schedule combined with ``Bracket(anchor="signal")``.

The one configuration where the 0.16.0 flatten/anchor interaction is
**numerically** observable, and the one with no ``run()``-level coverage.

Why the gap mattered. The re-anchor gate keys on the **flatten bar**::

    resignal = _entry & (_pos_d1 == 1) & ~_flatten_bar

Under legacy ``flatten_eod`` the flatten bar *is* the session's last bar, so
every pre-existing anchor x flatten fixture is blind to the distinction: an
implementation that gated on the session's last bar instead would pass the
entire suite. ``FlattenSchedule(minutes_before_close=180)`` separates them —
the flatten bar lands at 13:00 and the session's last bar at 15:45 — and only
then does the gate's choice show up in ``trades`` and ``returns``.

``tests/backtest/test_bracket.py`` does exercise the distinction, but against
``_apply_bracket`` **directly** and by asserting on an internal column that no
numeric output reads. These go through ``run()``.

The load-bearing configuration detail is
``block_entry_minutes_before_close=0``. At its default it equals
*minutes_before_close*, every in-session bar after the flatten bar is blocked,
``_entry`` is zeroed there, and no re-signal survives to reach the gate at all
— see ``test_the_default_block_window_removes_the_resignal_entirely``, which
pins that this fixture is not accidentally testing nothing.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass

import polars as pl
import pytest

from mktlib.backtest import Bracket, Crossover, Crossunder, run
from mktlib.backtest._bracket import Anchor
from mktlib.backtest._flatten import FlattenSchedule
from mktlib.backtest._types import BacktestResult
from mktlib.scheduling import get_calendar

_CAL = get_calendar("XNYS")

#: Two consecutive XNYS sessions of 15-minute bars, 09:30 … 15:45.
_DAYS = (datetime.date(2024, 1, 2), datetime.date(2024, 1, 3))
_BARS_PER_SESSION = 26

#: XNYS closes at 16:00, so ``minutes_before_close=180`` puts the cutoff at
#: 13:00 — bar 14 of each session. The session's LAST bar is 15:45, bar 25.
#: Those two indices being different is the entire point of this module.
_MINUTES_BEFORE_CLOSE = 180
_FLATTEN_BAR = 14
_SESSION_LAST_BAR = 25

#: ``block_entry_minutes_before_close=0`` — nothing is blocked, so a signal
#: after the flatten bar survives. Without this there is no re-signal to gate.
_SCHEDULE = FlattenSchedule(
    minutes_before_close=_MINUTES_BEFORE_CLOSE,
    block_entry_minutes_before_close=0,
)


@dataclass(frozen=True, slots=True)
class _EnterNeverExit:
    """Enter on the crossover; never exit on a signal.

    The exit condition is wired to two constant columns that cannot cross, so
    a position is closed only by the bracket or by the flatten. That removes
    the signal exit as a confounder: any difference between the two anchor
    policies below has to come from where the bracket levels sit.
    """

    def entry(self) -> Crossover:
        return Crossover("fast", "slow")

    def exit(self) -> Crossunder:
        return Crossunder("never_a", "never_b")


def _timestamps() -> list[datetime.datetime]:
    out: list[datetime.datetime] = []
    for day in _DAYS:
        ts = datetime.datetime(day.year, day.month, day.day, 9, 30)
        for _ in range(_BARS_PER_SESSION):
            out.append(ts)
            ts += datetime.timedelta(minutes=15)
    return out


def _mid_hold_entry_bars(signals: pl.DataFrame) -> list[int]:
    """Row indices carrying an entry signal while the position was held.

    The engine's own re-anchor predicate before the gate — ``_entry &
    (_pos_d1 == 1)`` — re-derived from the two public ``signals`` columns.
    """
    marked = signals.with_row_index("_i").with_columns(
        (pl.col("_entry") & (pl.col("_position").shift(1).fill_null(0) == 1))
        .fill_null(False)
        .alias("_mid")
    )
    return marked.filter(pl.col("_mid"))["_i"].to_list()


# ---------------------------------------------------------------------------
# Frame A — the only mid-hold re-signal lands on the SESSION-LAST bar
# ---------------------------------------------------------------------------


def _session_last_resignal_frame() -> pl.DataFrame:
    """One position, held across the session boundary, re-signalled at 15:45.

    Constructed rather than sampled, because what makes this test discriminate
    is a property of the *bar layout* and must not be left to a seed:

    * bar 16 (13:30) — crossover, opens the position; fill at bar 17's open,
      which is 100.0, so a ``take_profit=0.02`` leg anchored to the POSITION
      sits at 102.0.
    * bars 20-24 — ``fast`` drops back below ``slow``. No exit fires (the exit
      condition cannot), so the position simply stays open.
    * bar 25 (15:45) — crossover again. This is the session's last bar and the
      position is held, so it is a re-signal **on a session-last bar**. It is
      the ONLY mid-hold entry signal in the frame.
    * bar 26 — the next session opens at 90.0 after a gap. A re-anchored leg
      moves to 90.0 * 1.02 = 91.8.

    Session 1 highs top out at 100.5 and session 2's at 92.0, so 102.0 is
    never tagged and 91.8 is — at bar 30. The two policies therefore close the
    same trade on different bars, which is what the assertions read.

    A gate keyed on the session's LAST bar would exclude bar 25 and produce
    output identical to ``anchor="position"``. That is precisely the mutation
    #83 reports as passing the whole existing suite.
    """
    ts = _timestamps()
    n = len(ts)
    second = _BARS_PER_SESSION  # first index of session 2

    opens = [100.0 if i < second else 90.0 for i in range(n)]
    closes = list(opens)
    lows = [o - 0.5 for o in opens]
    highs = [o + 0.5 for o in opens]
    highs[second + 4] = 92.0  # bar 30 — tags the re-anchored 91.8 level

    fast = [1.0] * n
    for i in (16, 17, 18, 19):
        fast[i] = 3.0
    for i in range(_SESSION_LAST_BAR, n):
        fast[i] = 3.0

    return pl.DataFrame({
        "date": ts,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "fast": fast,
        "slow": [2.0] * n,
        "never_a": [0.0] * n,
        "never_b": [5.0] * n,
    })


def _run_a(
    anchor: Anchor, *, schedule: FlattenSchedule = _SCHEDULE
) -> BacktestResult:
    return run(
        _session_last_resignal_frame(),
        _EnterNeverExit(),
        calendar=_CAL,
        flatten=schedule,
        bracket=Bracket(take_profit=0.02, anchor=anchor),
    )


class TestSessionLastResignalReAnchors:
    def test_the_flatten_bar_and_the_session_last_bar_are_different_bars(self) -> None:
        """The precondition the whole module rests on.

        Under legacy ``flatten_eod`` these coincide and nothing below can
        discriminate anything. Asserted from the mask the engine actually
        builds, not from the arithmetic in the module docstring.
        """
        from mktlib.backtest._flatten import build_flatten_masks

        df = _session_last_resignal_frame()
        mask = build_flatten_masks(df["date"], _CAL, _SCHEDULE)[0]
        flatten_bars = df.with_row_index("_i").filter(mask)["_i"].to_list()
        assert flatten_bars == [_FLATTEN_BAR, _BARS_PER_SESSION + _FLATTEN_BAR]
        assert _FLATTEN_BAR != _SESSION_LAST_BAR

    def test_the_only_mid_hold_resignal_is_on_the_session_last_bar(self) -> None:
        """Anti-vacuity, and what makes the next test a mutation detector.

        If the frame ever acquired a second mid-hold signal somewhere else,
        ``test_the_two_anchor_policies_differ`` could pass on that one instead
        and would stop saying anything about session-last bars.
        """
        signals = _run_a("signal").signals
        assert _mid_hold_entry_bars(signals) == [_SESSION_LAST_BAR]

    def test_the_two_anchor_policies_differ(self) -> None:
        """The end-to-end assertion #83 asks for, through ``run()``.

        A re-signal on a held session-last bar **does** re-anchor. Gating on
        the session's last bar instead of the flatten bar would make these two
        results identical.
        """
        moved = _run_a("signal")
        held = _run_a("position")
        assert not moved.trades.equals(held.trades)
        assert not moved.returns.equals(held.returns)

    def test_the_re_anchored_level_moves_the_exit_bar(self) -> None:
        """Which bar closed the trade, stated explicitly.

        ``test_the_two_anchor_policies_differ`` says the two disagree; this
        says how, so a change that made them differ for some unrelated reason
        does not quietly satisfy the suite.
        """
        moved = _run_a("signal")
        held = _run_a("position")
        assert moved.trades.height == 1
        assert held.trades.height == 1

        # Re-anchored to 90.0 * 1.02 = 91.8, tagged by bar 30's high of 92.0.
        assert moved.trades["exit_date"][0] == datetime.datetime(2024, 1, 3, 10, 30)
        # Position-anchored at 100.0 * 1.02 = 102.0, never tagged, so the
        # session's 13:00 flatten is what closes it.
        assert held.trades["exit_date"][0] == datetime.datetime(2024, 1, 3, 13, 0)

    def test_the_default_block_window_removes_the_resignal_entirely(self) -> None:
        """Why ``block_entry_minutes_before_close=0`` is load-bearing here.

        At the default the block window equals *minutes_before_close*, so
        every in-session bar from 13:00 is blocked and ``_entry`` is zeroed
        there before the gate ever sees it. The bar-25 re-signal does not
        exist, and the two anchor policies agree — not because the gate is
        right, but because there is nothing to gate.

        Pinned so that a future edit to the fixture's schedule cannot silently
        turn every assertion above into a tautology.
        """
        default_block = FlattenSchedule(minutes_before_close=_MINUTES_BEFORE_CLOSE)
        moved = _run_a("signal", schedule=default_block)
        held = _run_a("position", schedule=default_block)
        assert _mid_hold_entry_bars(moved.signals) == []
        assert moved.trades.equals(held.trades)


# ---------------------------------------------------------------------------
# Frame B — the mid-hold re-signal lands ON the flatten bar
# ---------------------------------------------------------------------------


def _flatten_bar_resignal_frame() -> pl.DataFrame:
    """The other side of the gate: the re-signal is on the flatten bar itself.

    ``_entry`` is **not** cleared on a flatten bar — the deferral ORs the
    signal onto the next bar (``_entry | _suppressed.shift(1)``) and leaves the
    original in place — so ``_entry & (_pos_d1 == 1)`` really is true there and
    the ``& ~_flatten_bar`` term really is evaluated.

    What must happen: on the flatten bar the position is force-closed at that
    bar's own open, so there is nothing left to re-anchor. The deferred signal
    opens a fresh position on the next bar, which latches its own levels like
    any other entry.

    Prices ramp and the second position's bracket genuinely fires, so
    ``signal == position`` below is a statement about two real backtests. An
    earlier version of this fixture held every price at 100.0, which made that
    assertion ``0.0 == 0.0`` — true no matter what the gate did.
    """
    ts = _timestamps()
    n = len(ts)
    opens = [100.0 + 0.2 * i for i in range(n)]
    closes = list(opens)
    highs = [o + 0.3 for o in opens]
    # Tags the SECOND position's take-profit (open[16] * 1.02 = 105.264) and
    # nothing earlier, so the first position is closed by the flatten and the
    # second by the bracket.
    highs[20] = 106.0

    fast = [1.0] * n
    # Crossover at bar 10 opens the position; bar 13 dips back under (no exit
    # fires, so the position simply stays open); bar 14 crosses up again —
    # and bar 14 is the flatten bar.
    for i in (10, 11, 12):
        fast[i] = 3.0
    for i in range(_FLATTEN_BAR, n):
        fast[i] = 3.0

    return pl.DataFrame({
        "date": ts,
        "open": opens,
        "high": highs,
        "low": [o - 0.3 for o in opens],
        "close": closes,
        "fast": fast,
        "slow": [2.0] * n,
        "never_a": [0.0] * n,
        "never_b": [5.0] * n,
    })


def _run_b(anchor: Anchor) -> BacktestResult:
    return run(
        _flatten_bar_resignal_frame(),
        _EnterNeverExit(),
        calendar=_CAL,
        flatten=_SCHEDULE,
        bracket=Bracket(take_profit=0.02, anchor=anchor),
    )


class TestFlattenBarResignalDoesNotReAnchor:
    """The gate's other side — and an honest account of what these pin.

    Removing ``& ~pl.col(FLATTEN_BAR_COLUMN)`` altogether is **not**
    numerically observable, here or in any frame we could construct. That is a
    property of the engine rather than a weakness of this fixture:

    * A ``float`` leg reads ``ANCHOR_FILL_COLUMN``, which an ungated re-signal
      on flatten bar ``k`` would re-latch at ``k + 1``. But ``k + 1`` is the
      bar the deferred signal opens the new position on, so the bracket's
      ``live`` mask (``_pos_d1 == 1``) is false there; at ``k + 2``
      ``is_entry_bar`` re-latches to ``open[k + 2]`` under either gate. The two
      converge before any bar the level is read on.
    * A ``str`` leg's ``coalesce(initial, relatch)`` takes the opening latch
      first, and that bar belongs to the new position — which is what
      ``level_expr``'s own docstring says: "either defence alone resolves the
      collision correctly and this ordering is the second one".

    So the gate is **defence in depth**, and the test that detects its removal
    is ``test_bracket.py::test_the_flatten_bar_is_kept_out_of_the_re_signal_mask``,
    asserting on ``_bracket_resignal`` directly. That is the right layer for
    it. What these tests pin is the observable consequence — a flatten-bar
    signal produces no re-anchor and no changed number — and the mutation this
    module *does* detect is the one that matters numerically: gating on the
    session's last bar instead of the flatten bar. See Frame A.
    """

    def test_the_frame_really_puts_an_entry_signal_on_the_flatten_bar(self) -> None:
        """Anti-vacuity: the gated term must actually be reachable here."""
        signals = _run_b("signal").signals
        assert signals["_entry"][_FLATTEN_BAR] is True
        # Held going into the flatten bar — so without the gate this would be
        # a re-signal.
        assert signals["_position"][_FLATTEN_BAR - 1] == 1

    def test_the_frame_is_not_numerically_degenerate(self) -> None:
        """Prices move and the bracket fires, so the equality below has content.

        Without this the fixture could drift back to a flat tape, where
        ``signal == position`` holds as ``0.0 == 0.0`` regardless of the gate.
        """
        result = _run_b("signal")
        assert result.trades.height >= 2
        assert int((result.returns["return"] != 0.0).sum()) > 0
        assert result.trades.select(pl.col("pnl").abs().max()).item() > 0.0

    def test_a_flatten_bar_signal_does_not_re_anchor(self) -> None:
        """The position is force-closed on that bar; nothing survives to move.

        The accept-twin of Frame A: the gate must exclude the flatten bar and
        *only* the flatten bar.
        """
        moved = _run_b("signal")
        held = _run_b("position")
        assert moved.trades.equals(held.trades)
        assert moved.returns.equals(held.returns)

    def test_the_deferred_signal_still_opens_a_fresh_position(self) -> None:
        """The signal is not lost — it opens a new trade on the next bar.

        Without this, ``test_a_flatten_bar_signal_does_not_re_anchor`` would
        also pass if the flatten bar's signal were silently dropped.
        """
        trades = _run_b("signal").trades
        assert trades.height >= 2
        entries = trades["entry_date"].to_list()
        assert datetime.datetime(2024, 1, 2, 13, 15) in entries


# ---------------------------------------------------------------------------
# The frozen scenario's fixture, checked here rather than only in the baseline
# ---------------------------------------------------------------------------


def test_the_frozen_offset_anchor_scenario_is_not_vacuous() -> None:
    """The golden scenario added alongside these tests must fire for real.

    ``test_golden_baseline.py`` runs its own anti-vacuity gate over
    ``ANCHOR_SCENARIOS``; this states the specific property that made the
    fixture worth freezing — that it reaches a mid-hold re-signal *after* the
    flatten bar, which is the shape no other frozen scenario contains.
    """
    from tests.backtest.test_golden_baseline import (
        _scenario_bracket_anchor_flatten_offset,
    )

    produced = _scenario_bracket_anchor_flatten_offset()
    bars = _mid_hold_entry_bars(produced["signals"])
    assert bars, "the frozen fixture reaches no mid-hold entry signal at all"
    per_session = [b % _BARS_PER_SESSION for b in bars]
    assert any(b > _FLATTEN_BAR for b in per_session), (
        "every mid-hold signal lands at or before the flatten bar, so this "
        "fixture cannot distinguish the flatten bar from the session's last"
    )
    # Stronger, and the condition that actually makes the frozen bytes detect
    # a session-last gate: a re-signal has to land on the session's last bar
    # exactly. A signal merely *after* the flatten bar (say bar 20) is gated
    # identically by both implementations, so `any(b > _FLATTEN_BAR)` alone
    # would stay green while the baseline stopped discriminating.
    assert _SESSION_LAST_BAR in per_session, (
        "no mid-hold re-signal lands on a session-last bar, so the frozen "
        "baseline no longer separates the flatten-bar gate from a "
        "session-last gate"
    )


@pytest.mark.parametrize("anchor", ["signal", "position"])
def test_the_frozen_scenario_runs_under_both_policies(anchor: Anchor) -> None:
    """Both policies produce a usable backtest on the frozen fixture."""
    from tests.backtest.test_golden_baseline import (
        _scenario_bracket_anchor_flatten_offset,
    )

    produced = _scenario_bracket_anchor_flatten_offset(anchor=anchor)
    assert produced["trades"].height > 0
    assert produced["returns"].height > 0
