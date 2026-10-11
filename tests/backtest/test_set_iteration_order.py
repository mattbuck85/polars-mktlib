"""Results built from a ``set`` of names do not follow its iteration order.

The iteration order of a ``set`` of strings is fixed within a process and
differs between processes (it follows the string hash seed).  Two results
are assembled from such a set: the ``strategy_artifact`` digest, from the
names of the helpers ``init()`` calls, and the ``_entry_<col>`` columns of
``result.signals``, from the columns an exit condition reads through
``EntryRef``.  Each test hands the consumer the same names in several
orders and requires one result.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import polars as pl
import pytest

from mktlib.backtest import (
    Any_,
    Condition,
    Crossover,
    Crossunder,
    EntryRef,
    ValueGT,
    ValueLT,
    _artifact,
    _engine,
    run,
    strategy_artifact,
)
from tests.backtest import test_golden_baseline as golden

Order = Callable[[Iterable[str]], list[str]]


def _ascending(names: Iterable[str]) -> list[str]:
    return sorted(names)


def _descending(names: Iterable[str]) -> list[str]:
    return sorted(names, reverse=True)


def _rotated(names: Iterable[str]) -> list[str]:
    ordered = sorted(names)
    return ordered[1:] + ordered[:1]


def _interleaved(names: Iterable[str]) -> list[str]:
    ordered = sorted(names)
    return ordered[1::2] + ordered[::2]


ORDERS: list[Order] = [_ascending, _descending, _rotated, _interleaved]


# ---------------------------------------------------------------------------
# strategy_artifact
# ---------------------------------------------------------------------------


def _alpha(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(pl.col("close").rolling_mean(3).alias("fast"))


def _beta(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(pl.col("close").rolling_mean(8).alias("slow"))


def _gamma(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns((pl.col("fast") - pl.col("slow")).alias("gap"))


def _shared(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(pl.col("close").diff().alias("step"))


def _left(df: pl.DataFrame) -> pl.DataFrame:
    return _shared(df).with_columns(pl.col("step").abs().alias("up"))


def _right(df: pl.DataFrame) -> pl.DataFrame:
    return _shared(df).with_columns((-pl.col("step")).alias("down"))


class _Cross:
    def entry(self) -> Condition:
        return Crossover("fast", "slow")

    def exit(self) -> Condition:
        return Crossunder("fast", "slow")


class _NoHelper(_Cross):
    def init(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(pl.col("close").alias("fast"))


class _OneHelper(_Cross):
    def init(self, df: pl.DataFrame) -> pl.DataFrame:
        return _alpha(df)


class _ThreeHelpers(_Cross):
    def init(self, df: pl.DataFrame) -> pl.DataFrame:
        return _gamma(_beta(_alpha(df)))


class _SharedHelper(_Cross):
    """Two helpers that both call a third, which is followed once."""

    def init(self, df: pl.DataFrame) -> pl.DataFrame:
        return _right(_left(df))


def _digest_with_names_in(
    order: Order, strategy: object, monkeypatch: pytest.MonkeyPatch
) -> str:
    real = _artifact._get_called_names

    def ordered(source: str) -> list[str]:
        return order(real(source))

    with monkeypatch.context() as patch:
        patch.setattr(_artifact, "_get_called_names", ordered)
        return strategy_artifact(strategy)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "strategy_cls", [_NoHelper, _OneHelper, _ThreeHelpers, _SharedHelper]
)
def test_artifact_does_not_follow_the_order_of_the_called_names(
    strategy_cls: type, monkeypatch: pytest.MonkeyPatch
) -> None:
    strategy = strategy_cls()
    digests = {
        order.__name__: _digest_with_names_in(order, strategy, monkeypatch)
        for order in ORDERS
    }
    assert len(set(digests.values())) == 1, digests
    assert set(digests.values()) == {strategy_artifact(strategy)}


def test_artifact_orders_do_reorder_the_helpers() -> None:
    """The orders above differ on the names ``init()`` hands them."""
    names = _artifact._get_called_names(
        "def init(self, df):\n    return _gamma(_beta(_alpha(df)))\n"
    )
    assert names == {"_alpha", "_beta", "_gamma"}
    assert len({tuple(order(names)) for order in ORDERS}) == len(ORDERS)


def test_artifact_helpers_are_part_of_the_digest() -> None:
    digests = {
        strategy_artifact(cls())  # type: ignore[arg-type]
        for cls in (_NoHelper, _OneHelper, _ThreeHelpers, _SharedHelper)
    }
    assert len(digests) == 4


# ---------------------------------------------------------------------------
# result.signals
# ---------------------------------------------------------------------------


class _TwoRefs(_Cross):
    def exit(self) -> Condition:
        return Any_(
            ValueGT("close", EntryRef("high")),
            ValueLT("close", EntryRef("low")),
        )


class _FourRefs(_Cross):
    def exit(self) -> Condition:
        return Any_(
            _TwoRefs().exit(),
            Any_(
                ValueGT("open", EntryRef("close")),
                ValueLT("open", EntryRef("open")),
            ),
        )


def _signals_with_refs_in(
    order: Order, strategy: object, monkeypatch: pytest.MonkeyPatch
) -> pl.DataFrame:
    real = _engine.collect_entry_refs

    def ordered(cond: Condition) -> list[str]:
        return order(real(cond))

    with monkeypatch.context() as patch:
        patch.setattr(_engine, "collect_entry_refs", ordered)
        result = run(
            golden._daily_frame(seed=7, start=100.0),
            strategy,  # type: ignore[arg-type]
        )
    return result.signals


@pytest.mark.parametrize(
    ("strategy_cls", "expected"),
    [
        (_TwoRefs, ["_entry_high", "_entry_low"]),
        (
            _FourRefs,
            ["_entry_close", "_entry_high", "_entry_low", "_entry_open"],
        ),
    ],
)
def test_signals_entry_columns_are_in_name_order(
    strategy_cls: type,
    expected: list[str],
    scan_backend: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frames = [
        _signals_with_refs_in(order, strategy_cls(), monkeypatch)
        for order in ORDERS
    ]
    for order, signals in zip(ORDERS, frames, strict=True):
        snapshots = [c for c in signals.columns if c in expected]
        assert snapshots == expected, order.__name__
        assert signals.equals(frames[0]), order.__name__
    assert frames[0][expected[0]].drop_nulls().len() > 0
