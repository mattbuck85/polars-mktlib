"""``EntryRef`` re-anchors mid-trade — pinned as current, documented behaviour.

The docs used to claim the snapshot is forward-filled "through the position's
lifetime". It is not: ``_engine.py`` fills it from **every** raw ``_entry``
signal, including ones the position machinery suppresses because a position is
already open. The reference therefore moves mid-trade, and the exit is measured
against a bar the trade did not begin on.

This is the same latching rule ``Bracket(rearm=True)`` uses — the difference is
that the bracket path *raises* when it detects a mid-position entry signal,
because there the failure silently converts a fixed level into a trailing one.
``EntryRef`` predates that check and stays silent.

These tests pin the behaviour rather than assert it is correct. They exist so
that the eventual fix is a deliberate, visible change with a baseline diff to
review, instead of something that quietly alters everyone's exit timing.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from mktlib.backtest import (
    Col,
    Condition,
    Crossover,
    EntryRef,
    Pct,
    ValueGT,
    ValueLT,
    run,
)


@dataclass(frozen=True, slots=True)
class TargetExit:
    """Crossover entry with a percentage target — a NON-complementary pair.

    Because the exit is not the crossunder, the position can still be open when
    the next crossover arrives, which is what re-anchors the snapshot.
    """

    def entry(self) -> Crossover:
        return Crossover("fast", "slow")

    def exit(self) -> ValueGT:
        return ValueGT("close", Pct(EntryRef("close"), 25.0))


@dataclass(frozen=True, slots=True)
class Complementary:
    """Crossover paired with its own complement.

    The second exit term never fires — it would need a thousand-fold move — and
    exists only so the engine still materializes the ``_entry_close`` snapshot
    this test inspects. The effective exit is the crossunder alone.
    """

    def entry(self) -> Crossover:
        return Crossover("fast", "slow")

    def exit(self) -> Condition:
        return ValueLT("fast", Col("slow")) | ValueGT(
            "close", Pct(EntryRef("close"), 100_000.0)
        )


def _bars(n: int = 90, *, seed: int = 5) -> pl.DataFrame:
    state = seed
    close: list[float] = []
    px = 100.0
    for i in range(n):
        state = (1_664_525 * state + 1_013_904_223) % (2**32)
        px *= 1.0 + ((state / 2**32) - 0.5) * 0.03
        close.append(px * (1.0 + 0.01 * ((i % 14) - 7) / 7.0))
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
        pl.col("close").rolling_mean(12).fill_null(strategy="backward").alias("slow"),
    )


def _snapshot_moves_while_held(result: object) -> int:
    """Bars where ``_entry_close`` changed while a position was open."""
    signals: pl.DataFrame = result.signals  # type: ignore[attr-defined]
    held = pl.col("_position").shift(1) == 1
    return int(
        signals.select(
            (
                held
                & (pl.col("_entry_close") != pl.col("_entry_close").shift(1))
            ).sum()
        ).item()
    )


def test_entry_ref_reanchors_when_entries_can_fire_while_held() -> None:
    """Current behaviour. Not correct — pinned so the fix is deliberate."""
    result = run(_bars(), TargetExit())

    mid_position_entries = int(
        result.signals.select(
            (pl.col("_entry") & (pl.col("_position").shift(1) == 1)).sum()
        ).item()
    )
    assert mid_position_entries > 0, (
        "fixture must produce an entry signal while a position is open, "
        "otherwise this pins nothing"
    )
    assert _snapshot_moves_while_held(result) > 0, (
        "EntryRef's snapshot is expected to move mid-trade today"
    )


def test_entry_ref_is_stable_for_a_complementary_pair() -> None:
    """The case that is exact, and must stay exact through any fix.

    A crossover cannot re-fire until a crossunder resets the edge, and here the
    crossunder is the exit — so no entry signal lands while held and the
    snapshot cannot move.
    """
    result = run(_bars(), Complementary())
    assert result.trades.height > 1, "fixture must trade"
    assert _snapshot_moves_while_held(result) == 0
