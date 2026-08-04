"""Benchmark: EntryRef chain resolution, single pass vs windowed fallback.

Standalone and outside CI, matching the other scripts here. Nothing asserts a
timing; the number is for a PR body, not a gate.

The protocol is deliberate. This repository has been burned twice by casual
benchmarking — an end-to-end ``run()`` comparison of *identical* code once
measured +7.5% median, and a cProfile-guided change measured ~0% unprofiled and
was reverted. So:

* time the two resolvers **in isolation**, never a whole ``run()``, whose other
  ~17 ms would swamp the signal;
* build each frame once, outside the timed region;
* assert the two agree before timing, so a fast-but-wrong path cannot look good;
* let ``timeit`` auto-scale to at least 100 ms per trial and take the median of
  15, reporting the IQR alongside so a noisy machine is visible rather than
  averaged away;
* **report entry density**, because the fallback is O(trades x window) while the
  scan is O(n) — comparing them at different densities measures the fixture.

No cProfile. It is for locating cost, never for reporting it.

Usage::

    python scripts/bench_entryref_resolver.py
"""

from __future__ import annotations

import statistics
import sys
import timeit
from pathlib import Path

# Run from a checkout without installing: `python scripts/...` puts scripts/ on
# sys.path, not the repo root, so an installed mktlib would shadow the working
# tree — which is the opposite of what a benchmark wants. (bench_bracket.py has
# the same problem and no shim; it currently only runs under PYTHONPATH=.)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import polars as pl  # noqa: E402

from mktlib.backtest import EntryRef, Pct, ValueGT, ValueLT  # noqa: E402
from mktlib.backtest._anchor import (  # noqa: E402
    _realized_entries_fast,
    _realized_entries_windowed,
    collect_entry_refs,
    plan_arrays,
    plan_exit,
)

#: The canonical idiom, and the entire measured blast radius of the defect this
#: resolver exists to fix.
EXIT = ValueGT("close", Pct(EntryRef("close"), 5.0)) | ValueLT(
    "close", Pct(EntryRef("close"), -3.0)
)

SIZES = (10_000, 50_000, 100_000, 500_000)
REPEAT = 15
MIN_TRIAL_SECONDS = 0.1


def make_frame(n: int, *, seed: int = 7) -> pl.DataFrame:
    """Deterministic bars. stdlib arithmetic only — no numpy, per repo policy."""
    state = seed
    close: list[float] = []
    price = 100.0
    for _ in range(n):
        state = (1_664_525 * state + 1_013_904_223) % (2**32)
        price *= 1.0 + ((state / 2**32) - 0.5) * 0.02
        close.append(price)

    frame = pl.DataFrame({"close": close}).with_columns(
        pl.col("close").rolling_mean(5).fill_null(strategy="backward").alias("fast"),
        pl.col("close").rolling_mean(20).fill_null(strategy="backward").alias("slow"),
    )
    entry = (
        (pl.col("fast") > pl.col("slow"))
        & (pl.col("fast").shift(1) <= pl.col("slow").shift(1))
    ).fill_null(False)  # noqa: FBT003
    return frame.with_columns(entry.alias("_entry"))


def _auto_number(fn) -> int:  # noqa: ANN001
    """Smallest power-of-ten loop count taking at least MIN_TRIAL_SECONDS."""
    number = 1
    while timeit.timeit(fn, number=number) < MIN_TRIAL_SECONDS:
        number *= 10
        if number > 10_000:
            break
    return number


def _stats(fn) -> tuple[float, float, float]:  # noqa: ANN001
    """(median ms, IQR ms, min ms) per call."""
    number = _auto_number(fn)
    samples = [
        t / number * 1e3 for t in timeit.repeat(fn, repeat=REPEAT, number=number)
    ]
    quartiles = statistics.quantiles(samples, n=4)
    return statistics.median(samples), quartiles[2] - quartiles[0], min(samples)


def main() -> None:
    print(f"{'bars':>9} {'entries':>8} {'density':>8} | "
          f"{'windowed (ms)':>22} | {'single pass (ms)':>22} | {'speedup':>8}")
    print(f"{'':>9} {'':>8} {'':>8} | "
          f"{'median':>8}{'IQR':>7}{'min':>7} | "
          f"{'median':>8}{'IQR':>7}{'min':>7} | {'':>8}")

    for n in SIZES:
        frame = make_frame(n)
        entries = int(frame["_entry"].sum())
        arrays = plan_arrays(frame, plan_exit(EXIT))
        if arrays is None:
            msg = "the benchmark's exit tree must reach the fast path"
            raise SystemExit(msg)
        snapshot_cols = collect_entry_refs(EXIT)

        def fast(arrays=arrays, frame=frame):  # noqa: ANN001, ANN202
            return _realized_entries_fast(
                frame, entry_col="_entry", arrays=arrays, session_last_col=None
            )

        def windowed(frame=frame, cols=snapshot_cols):  # noqa: ANN001, ANN202
            return _realized_entries_windowed(
                frame,
                entry_col="_entry",
                exit_cond=EXIT,
                snapshot_cols=cols,
                session_last_col=None,
            )

        # A fast path that disagrees is not faster, it is wrong.
        if not fast().equals(windowed()):
            msg = f"resolvers disagree at n={n}; refusing to report a timing"
            raise SystemExit(msg)

        w_med, w_iqr, w_min = _stats(windowed)
        f_med, f_iqr, f_min = _stats(fast)
        print(
            f"{n:>9,} {entries:>8,} {entries / n:>7.2%} | "
            f"{w_med:>8.1f}{w_iqr:>7.1f}{w_min:>7.1f} | "
            f"{f_med:>8.1f}{f_iqr:>7.1f}{f_min:>7.1f} | {w_med / f_med:>7.1f}x"
        )

    print(
        "\nEntry density is reported because the fallback is O(trades x window) "
        "and the\nscan is O(n): comparing them at different densities measures "
        "the fixture, not\nthe code."
    )


if __name__ == "__main__":
    main()
