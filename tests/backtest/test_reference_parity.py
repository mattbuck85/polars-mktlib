"""Bootstrap: the reference backtester must reproduce the frozen baselines.

`_reference.py` is only useful as an oracle if it is independently correct. This
is the evidence for that: it must reproduce the **frozen Parquet artifacts** of
scenarios nobody alleges are broken, before it is allowed to adjudicate the one
that is.

Comparing against the parquets rather than a live `run()` matters — the parquets
were produced by a path that has been in use and reviewed, so agreeing with them
is evidence rather than two implementations sharing a bug.

`entry_ref` is deliberately absent. It is the scenario under adjudication, and
including it here would be circular.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl
import pytest

from tests.backtest._reference import (
    RefBracket,
    RefBars,
    RefConfig,
    run_reference,
)
from tests.backtest.test_golden_baseline import GOLDEN_DIR, _daily_frame

_TOL = 1e-12


def _crossover(a: list[float], b: list[float]) -> list[bool]:
    return [i > 0 and a[i] > b[i] and a[i - 1] <= b[i - 1] for i in range(len(a))]


def _crossunder(a: list[float], b: list[float]) -> list[bool]:
    return [i > 0 and a[i] < b[i] and a[i - 1] >= b[i - 1] for i in range(len(a))]


def _bars(*, exit_signal: list[bool] | None = None, cols: dict | None = None) -> RefBars:
    df = _daily_frame()
    fast, slow = df["fast"].to_list(), df["slow"].to_list()
    return RefBars(
        open=df["open"].to_list(),
        high=df["high"].to_list(),
        low=df["low"].to_list(),
        close=df["close"].to_list(),
        entry=_crossover(fast, slow),
        exit=exit_signal if exit_signal is not None else _crossunder(fast, slow),
        cols=cols or {"tp_level": df["tp_level"].to_list()},
    )


@dataclass(frozen=True, slots=True)
class Case:
    name: str
    cfg: RefConfig
    exit_from_high: bool = False  # limit_exit fires on high >= tp_level
    #: The short-bracket scenario mirrors the strategy too — it ENTERS on the
    #: crossunder and exits on the crossover, rather than reusing the long
    #: signals with a flipped side.
    mirrored_signals: bool = False


CASES: list[Case] = [
    Case("plain", RefConfig(side=1)),
    Case("short", RefConfig(side=-1)),
    Case("bracket_tp_long", RefConfig(side=1, bracket=RefBracket(tp=0.010))),
    Case("bracket_sl_long", RefConfig(side=1, bracket=RefBracket(sl=0.008))),
    Case(
        "bracket_both_stop_first",
        RefConfig(side=1, bracket=RefBracket(tp=0.010, sl=0.008, both_touch="stop_first")),
    ),
    Case(
        "bracket_both_tp_first",
        RefConfig(
            side=1,
            bracket=RefBracket(tp=0.010, sl=0.008, both_touch="take_profit_first"),
        ),
    ),
    Case(
        "bracket_short",
        RefConfig(side=-1, bracket=RefBracket(tp=0.010, sl=0.008)),
        mirrored_signals=True,
    ),
    Case("bracket_col_level", RefConfig(side=1, bracket=RefBracket(tp="tp_level"))),
    Case("bracket_entry_bar_gap", RefConfig(side=1, bracket=RefBracket(sl=0.0005))),
]


def _run_case(case: Case) -> tuple[list, list[float]]:
    if case.exit_from_high:
        df = _daily_frame()
        high, tp = df["high"].to_list(), df["tp_level"].to_list()
        exit_signal = [high[i] >= tp[i] for i in range(len(high))]
        bars = _bars(exit_signal=exit_signal, cols={"_limit_price": tp})
    elif case.mirrored_signals:
        df = _daily_frame()
        fast, slow = df["fast"].to_list(), df["slow"].to_list()
        bars = RefBars(
            open=df["open"].to_list(),
            high=df["high"].to_list(),
            low=df["low"].to_list(),
            close=df["close"].to_list(),
            entry=_crossunder(fast, slow),
            exit=_crossover(fast, slow),
            cols={"tp_level": df["tp_level"].to_list()},
        )
    else:
        bars = _bars()
    result = run_reference(bars, case.cfg)
    return result.trades, result.returns


@pytest.mark.parametrize("case", CASES, ids=[c.name for c in CASES])
def test_reference_reproduces_frozen_trades(case: Case) -> None:
    trades, _ = _run_case(case)
    golden = pl.read_parquet(GOLDEN_DIR / case.name / "trades.parquet")

    assert len(trades) == golden.height, (
        f"{case.name}: {len(trades)} reference trades vs {golden.height} frozen"
    )
    for got, want_pnl, want_held in zip(
        trades, golden["pnl"].to_list(), golden["bars_held"].to_list(), strict=True
    ):
        assert got.pnl == pytest.approx(want_pnl, abs=_TOL), case.name
        assert got.bars_held == want_held, case.name


@pytest.mark.parametrize("case", CASES, ids=[c.name for c in CASES])
def test_reference_reproduces_frozen_returns(case: Case) -> None:
    _, returns = _run_case(case)
    golden = pl.read_parquet(GOLDEN_DIR / case.name / "returns.parquet")

    assert len(returns) == golden.height, case.name
    for i, (got, want) in enumerate(zip(returns, golden["return"].to_list(), strict=True)):
        assert got == pytest.approx(want, abs=_TOL), f"{case.name}: bar {i}"


def test_the_bootstrap_is_not_vacuous() -> None:
    """Every case must actually trade, or agreeing with it proves nothing."""
    for case in CASES:
        trades, _ = _run_case(case)
        assert len(trades) > 0, f"{case.name}: no trades"


def test_reference_is_sensitive_to_a_wrong_fill_rule() -> None:
    """A reference that cannot fail is not evidence.

    Filling the exit one bar early — a plausible off-by-one, and one this
    implementation actually made before the baselines caught it — must break
    parity rather than pass unnoticed.
    """
    from tests.backtest import _reference as ref

    original = ref._resolve_exit

    def early_exit(bars, cfg, fill_bar, end, entry_price, anchor, n):  # noqa: ANN001
        got = original(bars, cfg, fill_bar, end, entry_price, anchor, n)
        if got[0] is None or cfg.bracket is not None:
            return got
        bar = got[0] - 1
        return (bar, bars.open[bar]) if bar > fill_bar else got

    ref._resolve_exit = early_exit  # type: ignore[assignment]
    try:
        trades, _ = _run_case(CASES[0])
        golden = pl.read_parquet(GOLDEN_DIR / "plain" / "trades.parquet")
        mismatch = any(
            got.bars_held != want
            for got, want in zip(trades, golden["bars_held"].to_list(), strict=True)
        )
        assert mismatch, "a one-bar-early exit slipped past the parity check"
    finally:
        ref._resolve_exit = original  # type: ignore[assignment]
