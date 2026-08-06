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

from mktlib.backtest import (
    All,
    Bracket,
    Col,
    Condition,
    Crossover,
    Crossunder,
    TradeSide,
    ValueGTE,
)

from tests.backtest._reference import (
    RefBars,
    RefBracket,
    RefConfig,
    RefLeg,
    run_reference,
)
from tests.backtest.test_golden_baseline import (
    GOLDEN_DIR,
    _daily_dates,
    _daily_frame,
    _frame,
)

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


# --- adjudication -----------------------------------------------------------
#
# Only meaningful because everything above passed first. The bootstrap cases
# establish that the reference reproduces artifacts produced by paths nobody
# alleges are broken; this then uses it to judge the one that was.


def _entry_ref_reference() -> tuple[list, list[float]]:
    df = _daily_frame()
    fast, slow = df["fast"].to_list(), df["slow"].to_list()
    bars = RefBars(
        open=df["open"].to_list(),
        high=df["high"].to_list(),
        low=df["low"].to_list(),
        close=df["close"].to_list(),
        entry=_crossover(fast, slow),
        exit=[False] * len(fast),
        cols={"close": df["close"].to_list()},
    )
    # _EntryRefTargetStrategy: exit at close >= entry_close * 1.0025.
    cfg = RefConfig(
        side=1,
        legs=(RefLeg(source="close", mult=1.0025, direction="above", strict=False),),
    )
    result = run_reference(bars, cfg)
    return result.trades, result.returns


def test_engine_matches_the_reference_on_entry_ref() -> None:
    """The gate that justifies the regenerated `entry_ref` baseline.

    The engine is compared against the oracle directly rather than against a
    frozen file, because the frozen file is the thing being replaced. Agreement
    here — with a reference committed before the fix and never edited for it —
    is what makes the new baseline evidence rather than a restatement.
    """
    from mktlib.backtest import run

    from tests.backtest.test_golden_baseline import _EntryRefTargetStrategy

    engine = run(_daily_frame(), _EntryRefTargetStrategy())
    trades, returns = _entry_ref_reference()

    assert len(trades) == engine.trades.height
    assert len(trades) > 1, "fixture must produce several closed trades"
    for got, want_pnl, want_held in zip(
        trades,
        engine.trades["pnl"].to_list(),
        engine.trades["bars_held"].to_list(),
        strict=True,
    ):
        assert got.pnl == pytest.approx(want_pnl, abs=_TOL)
        assert got.bars_held == want_held
    for i, (got, want) in enumerate(
        zip(returns, engine.returns["return"].to_list(), strict=True)
    ):
        assert got == pytest.approx(want, abs=_TOL), f"bar {i}"


def test_the_anchor_no_longer_moves_mid_trade() -> None:
    """A baseline-free invariant: one anchor per position, by construction."""
    from mktlib.backtest import run

    from tests.backtest.test_golden_baseline import _EntryRefTargetStrategy

    signals = run(_daily_frame(), _EntryRefTargetStrategy()).signals
    # Block ids from the position transitions, so the comparison never straddles
    # two different positions — the anchor is *supposed* to change between them.
    blocks = signals.with_columns(
        ((pl.col("_position") == 1) & (pl.col("_position").shift(1) != 1))
        .cum_sum()
        .alias("_block")
    ).filter(pl.col("_position") == 1)

    assert blocks["_block"].n_unique() > 1, "fixture must open several positions"
    per_block = blocks.group_by("_block").agg(
        pl.col("_entry_close").n_unique().alias("anchors")
    )
    assert per_block.select((pl.col("anchors") == 1).all()).item(), (
        f"an anchor moved mid-position: {per_block.sort('_block').to_dicts()}"
    )


# --- adjudication: Bracket(anchor="signal") ---------------------------------
#
# T1 of the 0.16.0 test ladder, and the fails-first evidence for it.
#
# The frozen corpus cannot reach this feature at all. Every golden fixture
# pairs `Crossover("fast","slow")` with `Crossunder("fast","slow")`, and those
# are exact complements: a second crossover needs a prior bar with
# `fast <= slow`, which is itself a crossunder, so the position is always flat
# again before the next entry signal can fire. `_entry & held` is identically
# false there and `anchor` has nothing to act on. Gating the exit breaks the
# complement — a crossunder that fails the gate leaves the position open, and
# the crossover after it fires while held.
#
# `flatten_eod` is deliberately absent: `_reference.py` puts it out of scope,
# so the oracle cannot adjudicate that corner and must not be asked to.


#: Longer than `_daily_frame`'s 60 bars. At 60 the fixture carries one or two
#: mid-hold re-signals in total and most parametrizations cannot discriminate
#: between the two policies at all.
_ANCHOR_BARS = 400

#: Chosen by measurement, not taste: on each of these the two policies produce
#: different trades for every parametrization below. `test_the_anchor_fixture_
#: discriminates` re-checks that rather than trusting this comment.
_ANCHOR_SEEDS = (7932, 39608, 71284)

#: Bracket half-width. Wide enough that a position survives long enough to see
#: a re-signal — at the golden fixtures' 1%/0.8% the bracket fires within a bar
#: or two of the entry fill and no re-anchor is ever reached.
_ANCHOR_WIDTH = 0.04

_ANCHOR_LEVEL_COLUMNS = ("tp_level", "lvl_up", "lvl_dn")


@dataclass(frozen=True, slots=True)
class _GatedExitLong:
    """Crossover entry, crossunder exit **gated** on a level.

    The gate is what makes a second entry signal reachable while held. It is
    otherwise arbitrary — this is a synthetic discriminator, not a strategy.
    """

    def entry(self) -> Condition:
        return Crossover("fast", "slow")

    def exit(self) -> Condition:
        return All(Crossunder("fast", "slow"), ValueGTE(Col("close"), Col("tp_level")))


@dataclass(frozen=True, slots=True)
class _GatedExitShort:
    """The short mirror. An asymmetry in the re-anchor is the easy bug here."""

    def entry(self) -> Condition:
        return Crossunder("fast", "slow")

    def exit(self) -> Condition:
        return All(Crossover("fast", "slow"), ValueGTE(Col("close"), Col("tp_level")))


def _anchor_frame(seed: int) -> pl.DataFrame:
    """The golden fixture generator, longer, plus two absolute-level columns.

    `lvl_up` / `lvl_dn` exist because the `str` legs need a level on each side
    of the price. The golden frame carries only `tp_level`, which sits *above*
    the bar's open — as a short's take-profit that is tagged on the entry fill
    bar of every single trade, which ends the block before a re-signal can
    happen and would make the short `str` case vacuous.
    """
    return _frame(_daily_dates(_ANCHOR_BARS), seed=seed).with_columns(
        (pl.col("open") * (1.0 + _ANCHOR_WIDTH)).round(6).alias("lvl_up"),
        (pl.col("open") * (1.0 - _ANCHOR_WIDTH)).round(6).alias("lvl_dn"),
    )


def _anchor_bars(df: pl.DataFrame, side: int) -> RefBars:
    """`_GatedExitLong` / `_GatedExitShort`'s signals, re-derived in Python."""
    fast, slow = df["fast"].to_list(), df["slow"].to_list()
    close, tp = df["close"].to_list(), df["tp_level"].to_list()
    up, down = _crossover(fast, slow), _crossunder(fast, slow)
    entry, raw_exit = (up, down) if side == 1 else (down, up)
    gate = [c >= t for c, t in zip(close, tp, strict=True)]
    return RefBars(
        open=df["open"].to_list(),
        high=df["high"].to_list(),
        low=df["low"].to_list(),
        close=close,
        entry=entry,
        exit=[fired and g for fired, g in zip(raw_exit, gate, strict=True)],
        cols={name: df[name].to_list() for name in _ANCHOR_LEVEL_COLUMNS},
    )


def _anchor_specs(kind: str, side: int) -> tuple[float | str, float | str]:
    """``(take_profit, stop_loss)`` for one spec kind, side-appropriate."""
    if kind == "float":
        return (_ANCHOR_WIDTH, _ANCHOR_WIDTH)
    return ("lvl_up", "lvl_dn") if side == 1 else ("lvl_dn", "lvl_up")


_ANCHOR_GRID = pytest.mark.parametrize(
    ("seed", "side", "kind", "both_touch"),
    [
        (seed, side, kind, both_touch)
        for seed in _ANCHOR_SEEDS
        for side in (1, -1)
        for kind in ("float", "col")
        for both_touch in ("stop_first", "take_profit_first")
    ],
    ids=lambda v: str(v),
)


@_ANCHOR_GRID
def test_the_anchor_fixture_discriminates(
    seed: int, side: int, kind: str, both_touch: str
) -> None:
    """Anti-vacuity, judged by the oracle alone — no engine involved.

    A parity assertion against a fixture where the two policies happen to agree
    passes by comparing two identical things. Every parametrization below must
    be one where re-anchoring actually moves a trade.
    """
    bars = _anchor_bars(_anchor_frame(seed), side)
    tp, sl = _anchor_specs(kind, side)
    runs = {
        anchor: run_reference(
            bars,
            RefConfig(
                side=side,
                bracket=RefBracket(tp=tp, sl=sl, both_touch=both_touch, anchor=anchor),
            ),
        )
        for anchor in ("position", "signal")
    }
    held = runs["position"].trades
    moved = runs["signal"].trades
    assert len(held) >= 3, f"fixture produced only {len(held)} trades"
    assert [(t.exit_bar, t.pnl) for t in held] != [(t.exit_bar, t.pnl) for t in moved], (
        "the two anchor policies produce identical trades on this fixture, so "
        "comparing the engine against it would measure nothing"
    )


@pytest.mark.parametrize("anchor", ["position", "signal"])
@_ANCHOR_GRID
def test_engine_matches_the_reference_on_bracket_anchor(
    seed: int, side: int, kind: str, both_touch: str, anchor: str
) -> None:
    """The engine must reproduce the oracle under both anchoring policies.

    ``anchor="position"`` is today's behaviour and is here to show the harness
    is sound: the fixture translation, the signal re-derivation and the oracle's
    per-bar level plumbing all agree with the engine before anything new is
    asked of it. ``anchor="signal"`` is the new semantics.
    """
    from mktlib.backtest import run

    df = _anchor_frame(seed)
    tp, sl = _anchor_specs(kind, side)
    engine = run(
        df,
        _GatedExitLong() if side == 1 else _GatedExitShort(),
        trade_side=TradeSide.LONG if side == 1 else TradeSide.SHORT,
        bracket=Bracket(
            take_profit=tp, stop_loss=sl, both_touch=both_touch, anchor=anchor
        ),
    )
    ref = run_reference(
        _anchor_bars(df, side),
        RefConfig(
            side=side,
            bracket=RefBracket(tp=tp, sl=sl, both_touch=both_touch, anchor=anchor),
        ),
    )

    # `pnl` / `bars_held` / the per-bar returns, the same three the bootstrap
    # cases compare. `exit_date` is deliberately not compared: the engine dates
    # a signal exit on the bar the position closed and fills it at the next
    # open, so it is one bar behind `RefTrade.exit_bar`, which is the fill.
    # The returns series pins exit timing anyway, bar by bar.
    assert len(ref.trades) == engine.trades.height, (
        f"{len(ref.trades)} reference trades vs {engine.trades.height} engine trades"
    )
    for i, (got, want_pnl, want_held) in enumerate(
        zip(
            ref.trades,
            engine.trades["pnl"].to_list(),
            engine.trades["bars_held"].to_list(),
            strict=True,
        )
    ):
        assert got.bars_held == want_held, f"trade {i}: bars_held"
        assert got.pnl == pytest.approx(want_pnl, abs=_TOL), f"trade {i}: pnl"
    for i, (got, want) in enumerate(
        zip(ref.returns, engine.returns["return"].to_list(), strict=True)
    ):
        assert got == pytest.approx(want, abs=_TOL), f"bar {i}"
