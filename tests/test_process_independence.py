"""Results that must not depend on the process that computes them.

Each test runs one snippet in several fresh interpreters that differ in
the string hash seed or in the size of the polars thread pool, and
requires one result.  See :mod:`tests._fresh_process`.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

from tests._fresh_process import hash_seeds, run_fresh

# ---------------------------------------------------------------------------
# strategy_artifact
# ---------------------------------------------------------------------------

_HELPER_STRATEGIES = '''
import polars as pl

from mktlib.backtest import Crossover, Crossunder


def _alpha(df):
    return df.with_columns(pl.col("close").rolling_mean(3).alias("fast"))


def _beta(df):
    return df.with_columns(pl.col("close").rolling_mean(8).alias("slow"))


def _gamma(df):
    return df.with_columns((pl.col("fast") - pl.col("slow")).alias("gap"))


def _shared(df):
    return df.with_columns(pl.col("close").diff().alias("step"))


def _left(df):
    return _shared(df).with_columns(pl.col("step").abs().alias("up"))


def _right(df):
    return _shared(df).with_columns((-pl.col("step")).alias("down"))


class _Base:
    def entry(self):
        return Crossover("fast", "slow")

    def exit(self):
        return Crossunder("fast", "slow")


class NoHelper(_Base):
    def init(self, df):
        return df.with_columns(pl.col("close").alias("fast"))


class OneHelper(_Base):
    def init(self, df):
        return _alpha(df)


class ThreeHelpers(_Base):
    def init(self, df):
        return _gamma(_beta(_alpha(df)))


class SharedHelper(_Base):
    def init(self, df):
        return _right(_left(df))
'''

_DIGESTS = """
from mktlib.backtest import strategy_artifact
import helper_strategies as h

for cls in (h.NoHelper, h.OneHelper, h.ThreeHelpers, h.SharedHelper):
    print(cls.__name__, strategy_artifact(cls()))
"""


def _digests(tmp_path: Path, n_seeds: int) -> list[dict[str, str]]:
    module = tmp_path / "helper_strategies.py"
    module.write_text(textwrap.dedent(_HELPER_STRATEGIES))
    outputs = run_fresh(_DIGESTS, hash_seeds(n_seeds), extra_path=[tmp_path])
    return [dict(line.split() for line in out.splitlines()) for out in outputs]


def test_artifact_of_init_helpers_is_one_value_per_hash_seed(
    tmp_path: Path,
) -> None:
    """``init()`` follows three same-module helpers, or two sharing a third.

    The helper sources are part of the digest, so the order they are
    appended in is too.
    """
    per_seed = _digests(tmp_path, 8)
    for name in ("NoHelper", "OneHelper", "ThreeHelpers", "SharedHelper"):
        values = {digests[name] for digests in per_seed}
        assert len(values) == 1, f"{name}: {sorted(values)}"
    assert len(set(per_seed[0].values())) == 4


# ---------------------------------------------------------------------------
# result.signals
# ---------------------------------------------------------------------------

_ENTRY_COLUMNS = """
from mktlib.backtest import Any_, Crossover, EntryRef, ValueGT, ValueLT, run
from tests.backtest import test_golden_baseline as golden


class TwoRefs:
    def entry(self):
        return Crossover("fast", "slow")

    def exit(self):
        return Any_(
            ValueGT("close", EntryRef("high")),
            ValueLT("close", EntryRef("low")),
        )


class FourRefs(TwoRefs):
    def exit(self):
        return Any_(
            Any_(
                ValueGT("close", EntryRef("high")),
                ValueLT("close", EntryRef("low")),
            ),
            Any_(
                ValueGT("open", EntryRef("close")),
                ValueLT("open", EntryRef("open")),
            ),
        )


for cls in (TwoRefs, FourRefs):
    signals = run(golden._daily_frame(seed=7, start=100.0), cls()).signals
    print(",".join(c for c in signals.columns if c.startswith("_entry_")))
"""


def test_signals_entry_columns_are_in_one_order_per_hash_seed() -> None:
    """One ``_entry_<col>`` column per ``EntryRef`` column, in name order."""
    outputs = run_fresh(_ENTRY_COLUMNS, hash_seeds(8))
    assert len(set(outputs)) == 1, sorted(set(outputs))
    two, four = outputs[0].splitlines()
    assert two.split(",") == ["_entry_high", "_entry_low"]
    assert four.split(",") == [
        "_entry_close",
        "_entry_high",
        "_entry_low",
        "_entry_open",
    ]
