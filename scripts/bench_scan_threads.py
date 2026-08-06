#!/usr/bin/env python3
"""Benchmark: does the mktlib-scan kernel serialize a threaded parameter sweep?

The consumer's real workload is a grid search: many independent ``run()`` calls
over the SAME frame with DIFFERENT strategy parameters, dispatched through a
``ThreadPoolExecutor`` (``tradesignalcore/optimizer/engine.py:195-202``). The
dispatch loop below is a deliberate copy of that shape, down to the
``as_completed`` drain, because the question is about that workload and not
about a microbenchmark of the kernel.

The hypothesis under test
-------------------------
``mktlib_scan``'s ``scan_realized`` never calls ``Python::allow_threads`` —
there is no ``allow_threads`` or ``py.detach`` anywhere in ``py/src/lib.rs``.
Marshalling *and* the kernel scan therefore run with the GIL held. If that
region is a large fraction of ``run()``, N concurrent backtests cannot overlap
it and adding threads buys nothing.

Three arms, because two are not enough
--------------------------------------
* ``native``  — the arm under test.
* ``python``  — negative control. The pure-Python resolver is GIL-held *by
  construction*, so it shows what "cannot scale" looks like on this machine.
* ``polars``  — **positive control**, and the arm that makes a null result
  interpretable. Eager polars compute of comparable duration, which releases
  the GIL. If this one does not scale either, the rig is measuring the machine
  — thermal throttling, SMT, a cgroup quota, an undersized rayon pool — and NO
  conclusion may be drawn about the kernel.

That third arm is not ceremony. It already overturned this script's first
design; see below.

POLARS_MAX_THREADS: pinned to 16, and NOT to 1
----------------------------------------------
It is set at the top of this module, before ``polars`` is imported, because
that is the only moment polars reads it.

The obvious pin is 1 — "stop polars' own parallelism from masking the GIL
effect". That is wrong, and the positive control proved it wrong. Polars owns
**one process-global rayon pool** shared by every Python thread, so
``POLARS_MAX_THREADS=1`` does not mean "one core per worker thread", it means
one core for all polars work in the process. Every arm's GIL-releasing half
then funnels through a single rayon worker. Measured on the reference machine:

    pool=1   positive control  1.03x @ n_jobs=16   (rig cannot express any
                                                    parallelism — uninterpretable)
    pool=4   positive control  1.25x @ n_jobs=16
    pool=16  positive control  5.79x @ n_jobs=16

At pool=1 the whole run is uninterpretable: native looked like 1.5x, but so
did everything, and there was no way to tell serialization from saturation.
Pinning to 16 is what lets the control demonstrate the machine's real ceiling.

Note what that ceiling is: **~5.8x, not 16x.** The reference box is 8 physical
cores with SMT2, and all-core boost drops the clock roughly a third under full
load. Every verdict below is therefore read against the *measured* control
speedup, never against ``n_jobs``. Judging the kernel against an unreachable
16x would manufacture a confirmation on any machine.

Pinning at all still costs something honest: it fixes polars' pool at the
reference machine's core count, so on a box with a different core count the
absolute numbers move. That is acceptable — this script exists to compare one
machine against itself before and after a wheel rebuild, and a pin that drifts
would destroy exactly that comparison.

Reproducibility
---------------
Nothing here is a command-line argument. Frame size, task count, the parameter
grid, the round count and the pool size are module constants, and the frame is
generated from a fixed seed with stdlib arithmetic only. The same file run
against a rebuilt wheel produces the "after" column of the same table.

Honesty about variance
----------------------
Every cell is measured ``ROUNDS`` times, **interleaved** — round by round, arm
inside n_jobs — rather than as sequential batches, so a thermal ramp or a
background process lands on all cells roughly equally instead of on whichever
one happened to run last. Arm order rotates with the round index so no arm
always pays the cache-cold slot. Medians are reported with the IQR and the min
beside them: if the IQR is a large fraction of the median the machine was busy
and the number should not be quoted.

Usage::

    python scripts/bench_scan_threads.py
"""

from __future__ import annotations

import os

# Must precede `import polars`: polars sizes its global rayon pool once, at
# first import, and ignores later changes. `mktlib_scan` statically links its
# own polars-core, whose pool reads the same variable. RAYON_NUM_THREADS is set
# alongside for any rayon pool that does not consult POLARS_MAX_THREADS.
POLARS_THREADS = "16"
os.environ["POLARS_MAX_THREADS"] = POLARS_THREADS
os.environ["RAYON_NUM_THREADS"] = POLARS_THREADS

import random  # noqa: E402
import statistics  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ThreadPoolExecutor, as_completed  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from pathlib import Path  # noqa: E402

# `python scripts/...` puts scripts/ on sys.path, not the repo root, so an
# installed mktlib would shadow the working tree — the opposite of what a
# benchmark wants. Same shim as bench_entryref_resolver.py.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import polars as pl  # noqa: E402

from mktlib.backtest import (  # noqa: E402
    Crossover,
    Crossunder,
    EntryRef,
    Pct,
    ValueGT,
    ValueLT,
    active_scan_backend,
    run,
    set_scan_backend,
)
from mktlib.backtest._anchor import (  # noqa: E402
    _plan_columns,
    collect_entry_refs,
    plan_exit,
)

#: ~2.5 years of RTH 1-minute bars. Big enough that one run is tens of
#: milliseconds, so pool dispatch overhead (tens of microseconds) is noise;
#: small enough that the whole matrix finishes in minutes.
N_ROWS = 250_000

#: Divisible by every entry in N_JOBS, so each configuration runs a whole
#: number of full waves. An odd count would leave the last wave half-empty at
#: n_jobs=16 and charge that arm for a scheduling artifact rather than the GIL.
N_TASKS = 48

N_JOBS = (1, 2, 4, 8, 16)
ROUNDS = 7
SEED = 7

#: Below this, the positive control has failed and the run says nothing about
#: the kernel. Deliberately far under the ~5.8x the reference machine reaches,
#: so only a genuinely broken rig trips it.
CONTROL_FLOOR = 3.0

#: Fraction of the control's demonstrated speedup that the native arm must
#: capture. Under LOW_EFFICIENCY it is serialized; over HIGH_EFFICIENCY it
#: scales. Between them the GIL is a real but partial constraint.
LOW_EFFICIENCY = 0.30
HIGH_EFFICIENCY = 0.75


@dataclass(frozen=True, slots=True)
class MaCrossTpSl:
    """MA crossover in; crossunder, take-profit or stop-loss out.

    The exit tree **must** contain ``EntryRef`` or the resolver never runs:
    ``_engine.py`` only calls ``realized_entries`` when
    ``collect_entry_refs(exit_cond)`` is non-empty, so a bracket-only or plain
    signal exit would measure everything except the thing under test. And the
    tree must additionally be *plannable*, or dispatch takes the windowed
    fallback and still never reaches the kernel. ``main()`` asserts both.

    ``init`` derives the moving averages per parameter set, mirroring a real
    sweep where every combo recomputes its own indicators. Hoisting that out
    would inflate the resolver's share of ``run()`` and overstate the effect.
    """

    fast: int
    slow: int
    tp_pct: float
    sl_pct: float

    def init(self, df: pl.DataFrame) -> pl.DataFrame:
        # backward-fill rather than drop_nulls, so every combo does identical
        # work on an identical row count whatever its window lengths.
        return df.with_columns(
            pl.col("close").rolling_mean(self.fast)
            .fill_null(strategy="backward").alias("fast"),
            pl.col("close").rolling_mean(self.slow)
            .fill_null(strategy="backward").alias("slow"),
        )

    def entry(self) -> Crossover:
        return Crossover("fast", "slow")

    def exit(self):  # noqa: ANN201
        return (
            Crossunder("fast", "slow")
            | ValueGT("close", Pct(EntryRef("close"), self.tp_pct))
            | ValueLT("close", Pct(EntryRef("close"), -self.sl_pct))
        )


def make_bars(n: int = N_ROWS, *, seed: int = SEED) -> pl.DataFrame:
    """Deterministic random-walk OHLC. stdlib arithmetic only, per repo policy."""
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
    return pl.DataFrame({
        "date": pl.datetime_range(
            pl.datetime(2024, 1, 1),
            pl.datetime(2024, 1, 1) + pl.duration(minutes=n),
            interval="1m",
            eager=True,
            closed="left",
        ),
        "open": o,
        "high": h,
        "low": lo,
        "close": c,
    })


def make_grid() -> tuple[MaCrossTpSl, ...]:
    """A deterministic parameter sweep — the consumer's actual unit of work."""
    grid = tuple(
        MaCrossTpSl(fast=f, slow=s, tp_pct=tp, sl_pct=sl)
        for f in (5, 8, 12, 16, 20, 26)
        for s in (30, 50)
        for tp in (0.4, 0.8)
        for sl in (0.2, 0.3)
    )
    if len(grid) != N_TASKS:
        msg = f"grid is {len(grid)} combos, expected N_TASKS={N_TASKS}"
        raise SystemExit(msg)
    return grid


def control_task(close: pl.Series, k: int) -> float:
    """Positive control: CPU-bound polars that releases the GIL.

    Eager Series ops rather than a lazy pipeline, so query planning — which is
    Python and GIL-held — is a negligible share. An earlier lazy version spent
    enough time in the planner that it capped at 2.3x and looked like a
    partial GIL result; this one reaches the machine's real ceiling.

    Single-column on purpose. A multi-column expression parallelizes *inside*
    one task, which consumes the shared rayon pool at n_jobs=1 and leaves no
    headroom to measure. One column per task means one rayon worker per task,
    so n_jobs is the only thing varying.

    ``k`` perturbs the windows so no task is served from a cached result; the
    work is the same for every k.
    """
    total = 0.0
    for _ in range(3):
        total += (
            close.rolling_std(256 + k % 3)
            .rolling_mean(128)
            .rolling_max(64)
            .sum()
            or 0.0
        )
    return total


def backtest_task(df: pl.DataFrame, strategy: MaCrossTpSl, backend: str) -> int:
    """One sweep cell, exactly as the optimizer issues it.

    The backend is re-asserted per task, not just per arm. ``set_scan_backend``
    writes a process global that any code could overwrite, and a benchmark that
    silently measured the Python resolver while reporting "native" would
    produce a confident null result about the wrong thing. The check is a
    global read against a ~30 ms task.
    """
    live = active_scan_backend()
    if live != backend:
        msg = f"backend drifted to {live!r} mid-arm, expected {backend!r}"
        raise RuntimeError(msg)
    return run(df, strategy).trades.height


def dispatch(tasks: list, n_jobs: int) -> float:
    """Wall clock for the whole batch, dispatched like the optimizer does.

    Mirrors ``GridSearch._search_parallel``: one ``ThreadPoolExecutor`` sized
    to ``n_jobs``, every combo submitted up front, drained with
    ``as_completed``.
    """
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n_jobs) as pool:
        futures = {pool.submit(fn): i for i, fn in enumerate(tasks)}
        for fut in as_completed(futures):
            fut.result()
    return time.perf_counter() - t0


def pin(name: str) -> None:
    """Pin the resolver backend and prove it took.

    ``set_scan_backend('native')`` does not itself raise when the accelerator
    is missing — ``active_scan_backend()`` does. Calling it is the difference
    between benchmarking the kernel and benchmarking a silent fallback to the
    Python resolver, which would look like a null result and be one about the
    wrong code.
    """
    set_scan_backend(name)
    live = active_scan_backend()
    if live != name:  # pragma: no cover - defensive
        msg = f"asked for the {name!r} backend, got {live!r}"
        raise SystemExit(msg)


def serial_fraction(speedup: float, achievable: float) -> float:
    """Amdahl inverted, against what this machine can actually deliver.

    ``achievable`` is the positive control's measured speedup at the same
    n_jobs, not n_jobs itself. Using n_jobs would attribute the machine's own
    ceiling — SMT, clock throttling — to the GIL and inflate every number.
    """
    if achievable <= 1.0 or speedup <= 0.0:
        return float("nan")
    return (achievable / speedup - 1.0) / (achievable - 1.0)


def main() -> None:
    df = make_bars()
    grid = make_grid()
    close = df["close"]

    # --- Requirement guard: the resolver must actually run, in the kernel. ---
    probe = grid[0]
    exit_cond = probe.exit()
    if not collect_entry_refs(exit_cond):
        msg = "exit tree has no EntryRef; _engine.py would never call the resolver"
        raise SystemExit(msg)
    if _plan_columns(probe.init(df), plan_exit(exit_cond)) is None:
        msg = "exit tree is unplannable; dispatch takes the windowed fallback"
        raise SystemExit(msg)

    arms: dict[str, tuple] = {
        "native": (
            lambda: pin("native"),
            [lambda s=s: backtest_task(df, s, "native") for s in grid],
        ),
        "python": (
            lambda: pin("python"),
            [lambda s=s: backtest_task(df, s, "python") for s in grid],
        ),
        "polars": (
            lambda: None,
            [lambda k=k: control_task(close, k) for k in range(N_TASKS)],
        ),
    }
    names = tuple(arms)

    pin("native")
    import mktlib_scan

    print(f"\n{'— machine & configuration —':^80}")
    print(f"cpu_count / affinity     {os.cpu_count()} / {len(os.sched_getaffinity(0))}")
    print(f"python                   {sys.version.split()[0]}   "
          f"GIL enabled: {getattr(sys, '_is_gil_enabled', lambda: True)()}")
    print(f"polars                   {pl.__version__}   pool={pl.thread_pool_size()}"
          f"   POLARS_MAX_THREADS={POLARS_THREADS}")
    print(f"mktlib-scan              contract={mktlib_scan.CONTRACT_VERSION}   "
          f"{Path(mktlib_scan.__file__).parent}")
    print(f"rows / tasks / rounds    {N_ROWS:,} / {N_TASKS} / {ROUNDS}")

    # --- Warmup: one full batch per arm, discarded. --------------------------
    for name in names:
        setup, tasks = arms[name]
        setup()
        dispatch(tasks, 1)

    # --- Interleaved rounds. -------------------------------------------------
    samples: dict[tuple[str, int], list[float]] = {
        (n, j): [] for n in names for j in N_JOBS
    }
    for r in range(ROUNDS):
        shift = r % len(names)
        for n_jobs in N_JOBS:
            for name in names[shift:] + names[:shift]:
                setup, tasks = arms[name]
                setup()
                samples[name, n_jobs].append(dispatch(tasks, n_jobs))
        print(f"  round {r + 1}/{ROUNDS}", flush=True)

    def med(name: str, j: int) -> float:
        return statistics.median(samples[name, j])

    def speedup(name: str, j: int) -> float:
        return med(name, 1) / med(name, j)

    # --- Per-arm tables. -----------------------------------------------------
    for name in names:
        print(f"\n{f'— {name} —':^80}")
        print(f"{'n_jobs':>7} {'median s':>10} {'IQR s':>9} {'min s':>9} "
              f"{'ms/task':>9} {'runs/s':>9} {'speedup':>9} {'GIL frac':>9}")
        for n_jobs in N_JOBS:
            xs = samples[name, n_jobs]
            m = statistics.median(xs)
            q = statistics.quantiles(xs, n=4)
            spd = speedup(name, n_jobs)
            frac = (
                "" if n_jobs == 1
                else f"{serial_fraction(spd, speedup('polars', n_jobs)):>9.2f}"
            )
            print(f"{n_jobs:>7} {m:>10.3f} {q[2] - q[0]:>9.3f} {min(xs):>9.3f} "
                  f"{m / N_TASKS * 1e3:>9.1f} {N_TASKS / m:>9.1f} {spd:>8.2f}x"
                  f"{frac}")

    # --- Does the accelerator still pay under concurrency? -------------------
    print(f"\n{'— native vs python resolver, same n_jobs —':^80}")
    print(f"{'n_jobs':>7} {'python s':>10} {'native s':>10} {'native is':>12}")
    for n_jobs in N_JOBS:
        p, n = med("python", n_jobs), med("native", n_jobs)
        print(f"{n_jobs:>7} {p:>10.3f} {n:>10.3f} {p / n:>11.2f}x")

    # --- Verdict. ------------------------------------------------------------
    ctrl = speedup("polars", 16)
    nat = speedup("native", 16)
    pyt = speedup("python", 16)
    eff = nat / ctrl if ctrl > 0 else float("nan")
    print(f"\n{'— verdict —':^80}")
    print(f"positive control (polars)  {ctrl:>6.2f}x @16   "
          f"<- what this machine can deliver")
    print(f"negative control (python)  {pyt:>6.2f}x @16   "
          f"efficiency {pyt / ctrl:>5.0%}")
    print(f"under test      (native)   {nat:>6.2f}x @16   "
          f"efficiency {eff:>5.0%}   "
          f"GIL frac {serial_fraction(nat, ctrl):.2f}")

    if ctrl < CONTROL_FLOOR:
        print(
            f"\nRIG INVALID. The positive control reached only {ctrl:.2f}x, under "
            f"the {CONTROL_FLOOR:.1f}x floor.\nA GIL-releasing workload could not "
            "be parallelized on this machine, so nothing\nhere is evidence about "
            "the kernel. Check background load, CPU quota,\nthermals, and that "
            f"POLARS_MAX_THREADS={POLARS_THREADS} matches the core count.\n"
            "Do not quote any number above."
        )
    elif eff <= LOW_EFFICIENCY:
        print(
            f"\nCONFIRMED. The control scaled {ctrl:.2f}x on the same threads while "
            f"native reached\nonly {nat:.2f}x ({eff:.0%} efficiency). The GIL-held "
            "kernel serializes the sweep."
        )
    elif eff >= HIGH_EFFICIENCY:
        print(
            f"\nREFUTED. Native reached {nat:.2f}x against a control ceiling of "
            f"{ctrl:.2f}x ({eff:.0%}\nefficiency). The GIL-held kernel is not the "
            "binding constraint on this\nworkload, and releasing the GIL around it "
            "would not repay the work."
        )
    else:
        print(
            f"\nPARTIAL. Native reached {nat:.2f}x against a control ceiling of "
            f"{ctrl:.2f}x ({eff:.0%}\nefficiency) — real scaling, well short of the "
            "machine. The GIL-held region\nis a genuine but not total constraint; "
            "read the GIL-frac column to size it."
        )
    print()


if __name__ == "__main__":
    main()
