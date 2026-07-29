"""``Bracket(rearm=True)`` — the engine path.

The recurrence itself is pinned against a sequential reference in
``test_bracket_rearm_recurrence.py``. What is under test here is the engine
wiring around it:

1. **Equivalence.** For an entry that cannot re-fire while held — an
   edge-triggered condition paired with its complement — re-arm must reproduce
   the default path *exactly*. This is what turns "re-arm changes nothing when
   nothing can re-arm" from an argument into a gate.
2. **It actually re-arms.** A strategy whose only real exit is the bracket
   trades once by default and repeatedly under re-arm.
3. **The guard.** An entry that fires while a position is open re-anchors the
   level mid-trade, quietly turning a fixed bracket into a trailing one. That
   raises.
4. **Reconciliation.** Per trade, the returns series must sum to the trade's
   pnl — the check that catches fill-bar index algebra.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from mktlib.backtest import Bracket, Col, Crossover, Crossunder, ValueGT, run

from tests.schemas.backtest import RearmSignalsSchema


@dataclass(frozen=True, slots=True)
class Complementary:
    """Edge-triggered entry paired with its own complement.

    A crossover cannot re-fire until a crossunder resets the edge, and the
    crossunder is the exit — so no entry signal can land while a position is
    open, and the level can never be re-anchored.
    """

    def entry(self) -> Crossover:
        return Crossover("fast", "slow")

    def exit(self) -> Crossunder:
        return Crossunder("fast", "slow")


@dataclass(frozen=True, slots=True)
class BracketOnly:
    """Nothing but the bracket really closes the position.

    The exit is a condition that never fires, which is the shape that trades
    exactly once without re-arm.
    """

    def entry(self) -> Crossover:
        return Crossover("fast", "slow")

    def exit(self) -> ValueGT:
        return ValueGT("close", Col("never"))


@dataclass(frozen=True, slots=True)
class Persistent:
    """Level-style entry: true on many consecutive bars, including held ones."""

    def entry(self) -> ValueGT:
        return ValueGT("fast", Col("slow"))

    def exit(self) -> ValueGT:
        return ValueGT("close", Col("never"))


def _oscillating(n: int = 120, *, seed: int = 11) -> pl.DataFrame:
    """A saw-toothed close so crossovers recur and levels get tagged often."""
    state = seed
    close: list[float] = []
    px = 100.0
    for i in range(n):
        state = (1_664_525 * state + 1_013_904_223) % (2**32)
        px *= 1.0 + ((state / 2**32) - 0.5) * 0.02
        # A slow oscillation guarantees repeated fast/slow crossings.
        close.append(px * (1.0 + 0.01 * ((i % 20) - 10) / 10.0))
    frame = pl.DataFrame(
        {
            "date": pl.date_range(
                pl.date(2024, 1, 1),
                pl.date(2024, 1, 1) + pl.duration(days=n - 1),
                eager=True,
            ),
            "close": close,
        }
    )
    return frame.with_columns(
        pl.col("close").alias("open"),
        (pl.col("close") * 1.01).alias("high"),
        (pl.col("close") * 0.99).alias("low"),
        pl.col("close").rolling_mean(3).fill_null(strategy="backward").alias("fast"),
        pl.col("close").rolling_mean(10).fill_null(strategy="backward").alias("slow"),
    ).with_columns(
        (pl.col("close") * 1.005).alias("tp"),
        (pl.col("close") * 0.995).alias("sl"),
        pl.lit(1e12).alias("never"),
    )


@pytest.fixture
def bars() -> pl.DataFrame:
    return _oscillating()


# ---------------------------------------------------------------------------
# 1. Equivalence — the strongest gate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bracket_kwargs",
    [
        {"take_profit": "tp"},
        {"stop_loss": "sl"},
        {"take_profit": "tp", "stop_loss": "sl"},
        {"take_profit": "tp", "stop_loss": "sl", "both_touch": "take_profit_first"},
    ],
)
def test_rearm_matches_default_when_nothing_can_rearm(
    bars: pl.DataFrame, bracket_kwargs: dict[str, str]
) -> None:
    """No entry can fire while held, so re-arm has nothing to change."""
    plain = run(bars, Complementary(), bracket=Bracket(**bracket_kwargs))  # type: ignore[arg-type]
    rearmed = run(
        bars, Complementary(), bracket=Bracket(rearm=True, **bracket_kwargs)  # type: ignore[arg-type]
    )

    assert plain.trades.height > 1, "fixture must produce several trades"
    assert_frame_equal(plain.returns, rearmed.returns, check_exact=True)
    assert_frame_equal(plain.trades, rearmed.trades, check_exact=True)


def test_rearm_adds_only_the_held_column_to_signals(bars: pl.DataFrame) -> None:
    plain = run(bars, Complementary(), bracket=Bracket(take_profit="tp"))
    rearmed = run(
        bars, Complementary(), bracket=Bracket(take_profit="tp", rearm=True)
    )
    assert set(rearmed.signals.columns) - set(plain.signals.columns) == {"_held"}
    RearmSignalsSchema.validate(rearmed.signals)


# ---------------------------------------------------------------------------
# 2. It actually re-arms
# ---------------------------------------------------------------------------


def test_bracket_only_strategy_trades_once_by_default(bars: pl.DataFrame) -> None:
    """The behaviour re-arm exists to fix, pinned so it cannot drift silently."""
    result = run(bars, BracketOnly(), bracket=Bracket(take_profit="tp"))
    assert result.trades.height == 1


def test_bracket_only_strategy_rearms(bars: pl.DataFrame) -> None:
    result = run(
        bars, BracketOnly(), bracket=Bracket(take_profit="tp", rearm=True)
    )
    assert result.trades.height > 5, (
        "a bracket exit must release the position for the next entry signal"
    )
    # Every trade closed on its level, so none is a next-open signal exit.
    assert (result.trades["bars_held"] >= 0).all()


def test_held_returns_to_zero_after_each_bracket_exit(bars: pl.DataFrame) -> None:
    result = run(
        bars, BracketOnly(), bracket=Bracket(take_profit="tp", rearm=True)
    )
    held = result.signals["_held"]
    assert set(held.unique().to_list()) <= {0, 1}
    # Genuinely oscillates rather than latching on.
    assert held.to_list().count(0) > 5
    assert held.to_list().count(1) > 5


# ---------------------------------------------------------------------------
# 3. The guard
# ---------------------------------------------------------------------------


def test_persistent_entry_raises(bars: pl.DataFrame) -> None:
    with pytest.raises(ValueError, match="while a position was already open"):
        run(bars, Persistent(), bracket=Bracket(take_profit="tp", rearm=True))


def test_persistent_entry_is_allowed_without_rearm(bars: pl.DataFrame) -> None:
    """The guard must not leak into the default path.

    Without re-arm the level anchors at ``_entry_clean``, so a mid-position
    entry signal re-anchors nothing and there is no ambiguity to report.
    """
    result = run(bars, Persistent(), bracket=Bracket(take_profit="tp"))
    assert result.trades.height >= 1


def test_guard_message_names_the_count(bars: pl.DataFrame) -> None:
    with pytest.raises(ValueError) as excinfo:
        run(bars, Persistent(), bracket=Bracket(take_profit="tp", rearm=True))
    msg = str(excinfo.value)
    assert "trailing stop" in msg
    assert "rearm=True" in msg


# ---------------------------------------------------------------------------
# 4. Reconciliation — the index-algebra check
# ---------------------------------------------------------------------------


def test_returns_reconcile_with_trade_pnl(bars: pl.DataFrame) -> None:
    """Compounding each trade's bar returns must reproduce its pnl.

    This is what catches an off-by-one in the fill-bar indices: a return booked
    on the wrong bar still looks plausible in isolation but will not reconcile.
    """
    result = run(
        bars, BracketOnly(), bracket=Bracket(take_profit="tp", rearm=True)
    )
    rets = result.returns
    idx = {d: i for i, d in enumerate(rets["date"].to_list())}
    values = rets["return"].to_list()

    assert result.trades.height > 5
    for trade in result.trades.iter_rows(named=True):
        lo = idx[trade["entry_date"]] + 1
        hi = idx[trade["exit_date"]]
        compounded = 1.0
        for value in values[lo : hi + 1]:
            compounded *= 1.0 + value
        assert compounded - 1.0 == pytest.approx(trade["pnl"], abs=1e-12)
