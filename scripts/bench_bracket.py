#!/usr/bin/env python3
"""Benchmark: the ``Bracket`` path against an equivalent close-condition exit.

Two questions this answers, neither of which the test suite does:

1. What does ``_apply_bracket`` cost per run at optimizer scale?
2. What does that multiply out to across a campaign?

The second is the one that matters. A campaign is ``combos x folds x symbols``
runs, so a few tens of milliseconds per run is hours of wall clock — enough to
bear on whether a faster engine is needed at all. Reported so a change to
``_apply_bracket`` is judged against a measurement rather than an impression.

Deliberately standalone rather than folded into ``bench_backtest.py``: that
script imports ``polars_talib`` at module scope, which is not a declared
dependency of this package and is absent from the dev environment, so it cannot
run here. This one needs nothing beyond ``polars``.

    python scripts/bench_bracket.py
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass

import polars as pl

from mktlib.backtest import Bracket, run
from mktlib.backtest._conditions import (
    Crossover,
    Crossunder,
    EntryRef,
    Pct,
    PriceIsAbove,
    PriceIsBelow,
)

#: 5 years of RTH 1-minute bars — 390/day x 252 days x 5.
N_ROWS = 491_400
FAST, SLOW = 12, 26

#: The WS-5 campaign anchor: 24,000 combos x 5 folds x 6 symbols.
CAMPAIGN_RUNS = 24_000 * 5 * 6


@dataclass(frozen=True, slots=True)
class SignalExit:
    """Entry + signal exit. The bracket, when passed, supplies TP/SL.

    The signal exit is not optional: a bracket exit does not re-arm the entry,
    so a strategy whose only exit is the bracket trades once and then sits flat
    forever. Both arms therefore carry it, leaving the bracket as the only
    difference between them.
    """

    def entry(self) -> Crossover:
        return Crossover("fast", "slow")

    def exit(self) -> Crossunder:
        return Crossunder("fast", "slow")


@dataclass(frozen=True, slots=True)
class CloseConditionExit:
    """The same strategy with TP/SL expressed as close-price conditions."""

    tp_pct: float
    sl_pct: float

    def entry(self) -> Crossover:
        return Crossover("fast", "slow")

    def exit(self):
        return (
            Crossunder("fast", "slow")
            | PriceIsAbove("close", Pct(EntryRef("close"), self.tp_pct))
            | PriceIsBelow("close", Pct(EntryRef("close"), -self.sl_pct))
        )


def make_bars(n: int = N_ROWS, *, seed: int = 7) -> pl.DataFrame:
    """Deterministic random-walk OHLC with a realistic intrabar range."""
    rng = random.Random(seed)
    px = 100.0
    o: list[float] = []
    h: list[float] = []
    lo: list[float] = []
    c: list[float] = []
    for _ in range(n):
        op = px
        cl = op * (1.0 + rng.gauss(0.0, 0.0010))
        half = abs(rng.gauss(0.0, 0.0010)) * 0.3 * op
        o.append(op)
        h.append(max(op, cl) + half)
        lo.append(min(op, cl) - half)
        c.append(cl)
        px = cl
    return (
        pl.DataFrame({
            "date": pl.datetime_range(
                pl.datetime(2024, 1, 1),
                pl.datetime(2024, 1, 1) + pl.duration(minutes=n),
                interval="1m",
                eager=True,
                closed="left",
            ),
            "open": o, "high": h, "low": lo, "close": c,
        })
        .with_columns(
            pl.col("close").rolling_mean(FAST).alias("fast"),
            pl.col("close").rolling_mean(SLOW).alias("slow"),
            (pl.col("close") * 1.004).alias("tp_level"),
            (pl.col("close") * 0.998).alias("sl_level"),
        )
        .drop_nulls()
    )


def best_of(fn, repeats: int = 7) -> float:
    """Best-of-N wall clock in ms — the machine's floor, not its noise."""
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best * 1e3


def main() -> None:
    df = make_bars()
    bracket = Bracket(take_profit="tp_level", stop_loss="sl_level")

    plain = best_of(lambda: run(df, SignalExit()))
    close = best_of(lambda: run(df, CloseConditionExit(0.4, 0.2)))
    brk = best_of(lambda: run(df, SignalExit(), bracket=bracket))

    print(f"\n{'— Bracket cost —':^62}")
    print(f"rows                        {df.height:>12,}")
    print(f"signal exit only            {plain:>9.1f} ms")
    print(f"close-condition TP/SL       {close:>9.1f} ms")
    print(f"bracket                     {brk:>9.1f} ms   ({brk / close:.2f}x close)")
    print(f"bracket overhead            {brk - close:>9.1f} ms/run")

    print(f"\n{'— Campaign projection —':^62}")
    print(f"runs (24k combos x 5 folds x 6 symbols)  {CAMPAIGN_RUNS:>12,}")
    for label, ms in (("close-condition", close), ("bracket", brk)):
        print(f"  {label:<22}{ms * CAMPAIGN_RUNS / 3.6e6:>8.1f} h")
    print(
        "\nSerial figures; a thread pool scales all rows down together, so the\n"
        "ratio holds even though the absolute hours do not."
    )
    print()


if __name__ == "__main__":
    main()
