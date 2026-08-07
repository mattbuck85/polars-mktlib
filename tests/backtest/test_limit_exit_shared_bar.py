"""#78 — a bar carrying both an entry and an exit signal, mid-limit-trade.

``returns`` booked two limit fills where ``trades`` booked one, so
``prod(1 + return)`` over a trade's span did not equal that trade's ``pnl``.
Because :mod:`mktlib.reports` reads ``trades["pnl"]`` while equity curves are
built from ``returns``, one backtest reported two different results depending
on which artifact you read.

The mechanism is an interaction of two rules, only one of which was wrong:

1. The ``_position`` recurrence resolves an entry/exit collision **entry-first**
   — ``when(_entry).then(1).when(_exit).then(0)`` — so a shared bar leaves the
   position open. That is deliberate and stays.
2. ``_is_limit_exit_bar`` was ``_exit & (_pos_d1 == 1)``, which does not know
   the position survived the bar. It therefore held on the shared bar *and* on
   the bar that actually closed the trade.

The resolution is the second of the three candidates in #78 — "the exit is
suppressed while held". It is what ``trades`` already implements, so it makes
``returns`` agree with the existing behaviour rather than inventing a third.

The rule the fix encodes, which generalises past this bug: **every fill
predicate in the return chain must be a function of the ``_position``
transition columns, not of raw ``_exit``.** ``_is_limit_exit_bar`` was the only
one that was not.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass

import polars as pl
import pytest

from mktlib.backtest import (
    Bracket,
    Col,
    Condition,
    Crossover,
    Limit,
    ValueGTE,
    run,
)
from mktlib.backtest._types import BacktestResult
from tests.backtest.test_golden_baseline import (
    _daily_frame,
    _LimitTakeProfitStrategy,
)


@dataclass(frozen=True, slots=True)
class _CrossEntryTakeProfit:
    """Enter on a crossover; exit when ``high`` reaches the ``tp`` column.

    Entry and exit are deliberately **not** complements — that is what makes a
    mid-hold entry signal reachable at all. A ``Crossover``/``Crossunder`` pair
    can never produce one, which is why the pre-existing corpus never hit this.
    """

    def entry(self) -> Crossover:
        return Crossover("fast", "slow")

    def exit(self) -> Condition:
        return Limit(ValueGTE(Col("high"), Col("tp")), price="tp")


def _reconcile(result: BacktestResult) -> list[float]:
    """Per-trade ``prod(1 + return) - 1`` minus ``pnl``, one entry per trade.

    This is the invariant #78 breaks, written exactly as the issue's repro
    writes it: the two artifacts are two views of one run and must agree.
    """
    dates = result.returns["date"].to_list()
    rets = result.returns["return"].to_list()
    out: list[float] = []
    for trade in result.trades.to_dicts():
        lo = dates.index(trade["entry_date"])
        hi = dates.index(trade["exit_date"])
        prod = 1.0
        for value in rets[lo : hi + 1]:
            prod *= 1.0 + (value or 0.0)
        out.append((prod - 1.0) - trade["pnl"])
    return out


# ---------------------------------------------------------------------------
# The hand-built shared-bar frame
# ---------------------------------------------------------------------------

_N = 12
#: Bar whose ``_entry`` and ``_exit`` are both true while the position is held.
_SHARED_BAR = 6
#: Bar on which the trade actually closes — an exit with no entry beside it.
_REAL_EXIT_BAR = 8

_CLOSES = [100.0 + 0.5 * i for i in range(_N)]


def _shared_bar_frame() -> pl.DataFrame:
    """15-minute bars where one mid-hold bar carries an entry *and* an exit.

    ``fast`` crosses up at bar 1, back down at bar 3, and up again at bar 6.
    The second crossover is the mid-hold entry signal. ``tp`` is out of reach
    everywhere except bars 6 and 8, so bar 6 carries both signals and bar 8 is
    where the position actually closes.

    Bar 6 is given a deliberately tall wick so its limit level sits far above
    the bar's own close. That is not decoration: it makes the phantom fill
    worth ~150 bps rather than a rounding difference, so the assertion below
    cannot pass by accident on a change that merely perturbs the arithmetic.
    """
    ts = [
        datetime.datetime(2024, 1, 2, 9, 30) + datetime.timedelta(minutes=15 * i)
        for i in range(_N)
    ]
    fast = [1.0, 3.0, 3.0, 1.0, 1.0, 1.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0]
    highs = [c + 0.4 for c in _CLOSES]
    highs[_SHARED_BAR] = _CLOSES[_SHARED_BAR] + 5.0
    tp = [999.0] * _N
    tp[_SHARED_BAR] = 105.0
    tp[_REAL_EXIT_BAR] = 104.3
    return pl.DataFrame({
        "date": ts,
        "open": [c - 0.2 for c in _CLOSES],
        "high": highs,
        "low": [c - 0.6 for c in _CLOSES],
        "close": _CLOSES,
        "fast": fast,
        "slow": [2.0] * _N,
        "tp": tp,
    })


def _held_return(bar: int) -> float:
    """The plain hold return for *bar* — close-over-previous-close."""
    return _CLOSES[bar] / _CLOSES[bar - 1] - 1.0


def _limit_return(bar: int, level: float) -> float:
    """The limit-fill return for *bar* — level measured from the prior close."""
    return (level - _CLOSES[bar - 1]) / _CLOSES[bar - 1]


class TestSharedBarIsNotAFill:
    def test_the_fixture_really_has_a_shared_mid_hold_bar(self) -> None:
        """Anti-vacuity. Without this the tests below could pin nothing.

        If the fixture ever stopped producing a bar with both signals while
        held, every assertion here would still pass — against a frame that no
        longer exercises the defect. Assert the precondition explicitly.
        """
        signals = run(_shared_bar_frame(), _CrossEntryTakeProfit()).signals
        row = signals.row(_SHARED_BAR, named=True)
        assert row["_entry"] is True
        assert row["_exit"] is True
        # Entry precedence: the position survives the bar. This is the rule
        # the fix keeps, not the one it changes.
        assert row["_position"] == 1
        assert signals["_position"][_SHARED_BAR - 1] == 1

    def test_returns_and_trades_agree_on_the_shared_bar_frame(self) -> None:
        """The #78 invariant, on the frame built to break it.

        Two limit fills are booked into ``returns`` (bar 6 and bar 8) against
        the single exit ``trades`` books, so the two artifacts disagree.
        """
        result = run(_shared_bar_frame(), _CrossEntryTakeProfit())
        assert result.trades.height == 1
        deltas = _reconcile(result)
        assert deltas == pytest.approx([0.0], abs=1e-12)

    def test_the_shared_bar_books_a_hold_not_a_fill(self) -> None:
        """Where the disagreement comes from, asserted directly.

        The reconciliation above says the two views disagree; this says which
        bar is wrong and by how much. Bar 6 must carry the ordinary hold
        return, not the limit level's — those differ by ~195 bps on that bar
        alone, so this also rules out the reconciliation passing through
        compensating errors.
        """
        result = run(_shared_bar_frame(), _CrossEntryTakeProfit())
        rets = result.returns["return"].to_list()
        assert rets[_SHARED_BAR] == pytest.approx(_held_return(_SHARED_BAR), abs=1e-12)
        # And the bar after it is an ordinary hold too — it was being zeroed
        # as "the bar after a limit fill", which is the other half of the bug.
        assert rets[_SHARED_BAR + 1] == pytest.approx(
            _held_return(_SHARED_BAR + 1), abs=1e-12
        )

    def test_the_real_exit_bar_still_fills_at_the_limit(self) -> None:
        """The trade still closes at the limit level on the bar that closed it.

        Suppressing the shared bar must not suppress the fill that actually
        happened one bar later.
        """
        result = run(_shared_bar_frame(), _CrossEntryTakeProfit())
        rets = result.returns["return"].to_list()
        assert rets[_REAL_EXIT_BAR] == pytest.approx(
            _limit_return(_REAL_EXIT_BAR, 104.3), abs=1e-12
        )


class TestGoldenLimitExitReconciles:
    def test_every_limit_exit_trade_reconciles(self) -> None:
        """#78's own repro: the ``limit_exit`` golden scenario, per trade.

        Trade 5 fails today by 12.19 bps — ``CHANGELOG.md`` already names it.
        Asserted over **all** trades rather than trade 5 alone, so a fix that
        repaired one trade by moving the error into another would not pass.
        """
        result = run(_daily_frame(), _LimitTakeProfitStrategy())
        assert result.trades.height > 0
        deltas = _reconcile(result)
        assert deltas == pytest.approx([0.0] * len(deltas), abs=1e-9)


class TestTheGuardDoesNotOverRefuse:
    """The accept-what-it-should half.

    A predicate narrowed from a bug report tends to over-refuse, and here the
    failure would be silent and expensive: every ordinary limit exit would stop
    filling at its level and start filling at the next bar's open. These pin
    the behaviour that must survive the narrowing.
    """

    def test_a_limit_with_no_entry_beside_it_still_fills_same_bar(self) -> None:
        """The ordinary case: one trade, one limit exit, no shared bar at all.

        Deliberately a separate frame from the shared-bar fixture, so this
        cannot be satisfied by whatever that frame happens to do.
        """
        n = 8
        closes = [100.0 + 0.5 * i for i in range(n)]
        tp = [999.0] * n
        tp[5] = 102.4
        df = pl.DataFrame({
            "date": [
                datetime.datetime(2024, 1, 2, 9, 30)
                + datetime.timedelta(minutes=15 * i)
                for i in range(n)
            ],
            "open": [c - 0.2 for c in closes],
            "high": [c + 0.4 for c in closes],
            "low": [c - 0.6 for c in closes],
            "close": closes,
            "fast": [1.0] + [3.0] * (n - 1),
            "slow": [2.0] * n,
            "tp": tp,
        })
        result = run(df, _CrossEntryTakeProfit())

        signals = result.signals
        assert signals["_entry"][5] is False, "this bar must carry no entry signal"
        assert signals["_exit"][5] is True
        assert result.trades.height == 1

        rets = result.returns["return"].to_list()
        expected = (tp[5] - closes[4]) / closes[4]
        assert rets[5] == pytest.approx(expected, abs=1e-12)
        assert _reconcile(result) == pytest.approx([0.0], abs=1e-12)


class TestBracketWithLimitStaysRefused:
    def test_bracket_plus_limit_exit_raises(self) -> None:
        """``_apply_bracket`` rewrites ``_exit_clean`` — which the fix now reads.

        Before #78, ``_is_limit_exit_bar`` read only ``_exit``, a column no
        bracket touches. It now reads ``_exit_clean``, which ``_apply_bracket``
        overwrites at ``_engine.py:423-428``. The combination is refused today,
        so the new read is unreachable — but "unreachable" is a property of
        that guard, not of the predicate, and this repo's characteristic defect
        is a control that nothing ever reaches.

        Pinning the refusal means that if the combination is ever allowed, this
        test is what fails, rather than the bracket silently changing which
        bars a limit fills on.
        """
        with pytest.raises(NotImplementedError, match="not supported together"):
            run(
                _daily_frame(),
                _LimitTakeProfitStrategy(),
                bracket=Bracket(take_profit=0.01),
            )
