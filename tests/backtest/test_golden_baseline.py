"""Golden backward-compatibility baseline for the backtest engine.

This module is the **release gate** for changes to the return-expression
chain in :mod:`mktlib.backtest._engine`.  It pins the exact ``returns``,
``trades`` and ``signals`` frames that :func:`mktlib.backtest.run`
produces for a set of deterministic synthetic fixtures, comparing against
Parquet baselines frozen on disk under ``tests/backtest/data/golden/``.

Why on-disk baselines rather than a same-process before/after comparison:
asserting "new code equals old code" inside one process only works while
both live in the same tree.  A frozen artifact keeps the *pre-change*
numbers around, so an edit that shifts results cannot quietly drag the
expectation along with it.

The fixtures are generated in pure Python (a small LCG plus hand-rolled
rolling means) rather than via Polars kernels, so the only Polars compute
under test is the engine itself.

Scenario coverage maps 1:1 onto the branches of the return-expression
chain, since that is what future work (per-fill costs, bracket exits)
will edit:

===================  ======================================================
scenario             engine path exercised
===================  ======================================================
plain                base entry/middle/exit chain, long
short                same chain with ``effective_side = -1``
entry_ref            ``_collect_entry_refs`` snapshot columns + ``Pct``
limit_exit           ``_is_limit_exit_bar`` / ``_is_post_limit_bar``
flatten_eod          ``_session_last`` overrides + deferred entries
flatten_eod_limit    limit branch *and* session-last branch together
multi                per-instrument partitioning, combined views
multi_weighted       ``MultiBacktestResult._weighted_returns``
dual                 ``_run_dual`` long/short merge
===================  ======================================================

The ``bracket_*`` scenarios pin the bracket path's own output. The base
scenarios above cannot: ``test_golden_baseline_no_bracket`` only pins
``bracket=None`` as a no-op, which proves the non-bracket path is unperturbed
and says nothing about what a firing bracket produces. That distinction matters
whenever ``_apply_bracket`` is *changed* rather than added.

Regenerating the baselines is a deliberate act, never a convenience::

    python tests/backtest/test_golden_baseline.py --regenerate

Only do that when the numbers are *intended* to move (or when a Polars
upgrade is confirmed to be the cause), and review the resulting Parquet
diff as carefully as a code diff.
"""

from __future__ import annotations

import datetime
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from mktlib.backtest import (
    Bracket,
    Col,
    Condition,
    Cost,
    Crossover,
    Crossunder,
    EntryRef,
    Limit,
    Pct,
    TradeSide,
    ValueGTE,
    run,
)
from mktlib.backtest._types import BacktestResult, MultiBacktestResult
from mktlib.scheduling import get_calendar

GOLDEN_DIR = Path(__file__).parent / "data" / "golden"

ARTIFACTS = ("returns", "trades", "signals")


# ---------------------------------------------------------------------------
# Deterministic synthetic data (pure Python — no Polars kernels, no RNG lib)
# ---------------------------------------------------------------------------

# PCG/Knuth 64-bit LCG constants. Pinned here so the fixture is reproducible
# independently of the Python version's `random` implementation.
_LCG_A = 6364136223846793005
_LCG_C = 1442695040888963407
_LCG_M = 1 << 64


def _uniforms(n: int, seed: int) -> list[float]:
    """*n* deterministic floats in ``[0, 1)`` from a fixed 64-bit LCG."""
    state = seed % _LCG_M
    out: list[float] = []
    for _ in range(n):
        state = (_LCG_A * state + _LCG_C) % _LCG_M
        out.append((state >> 11) / float(1 << 53))
    return out


def _ohlc(
    n: int,
    *,
    seed: int,
    start: float = 100.0,
) -> tuple[list[float], list[float], list[float], list[float]]:
    """Random-walk OHLC with ``low <= min(open, close) <= max(...) <= high``."""
    u = _uniforms(3 * n, seed)
    opens: list[float] = []
    highs: list[float] = []
    lows: list[float] = []
    closes: list[float] = []
    prev_close = start
    for i in range(n):
        open_ = round(prev_close * (1.0 + (u[3 * i] - 0.5) * 0.010), 4)
        close = round(prev_close * (1.0 + (u[3 * i + 1] - 0.5) * 0.030), 4)
        wick = u[3 * i + 2] * 0.006
        high = round(max(open_, close) * (1.0 + wick), 4)
        low = round(min(open_, close) * (1.0 - wick), 4)
        opens.append(open_)
        highs.append(high)
        lows.append(low)
        closes.append(close)
        prev_close = close
    return opens, highs, lows, closes


def _rolling_mean(xs: Sequence[float], window: int) -> list[float]:
    """Trailing mean with an expanding warm-up (no nulls), computed in Python."""
    out: list[float] = []
    for i in range(len(xs)):
        chunk = xs[max(0, i - window + 1) : i + 1]
        out.append(sum(chunk) / len(chunk))
    return out


def _frame(
    dates: Sequence[datetime.date] | Sequence[datetime.datetime],
    *,
    seed: int,
    start: float = 100.0,
) -> pl.DataFrame:
    """OHLC frame plus the indicator columns the fixture strategies read."""
    opens, highs, lows, closes = _ohlc(len(dates), seed=seed, start=start)
    return pl.DataFrame(
        {
            "date": list(dates),
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "fast": _rolling_mean(closes, 3),
            "slow": _rolling_mean(closes, 8),
            # Absolute take-profit level, 0.5% above the bar's own open.
            "tp_level": [round(o * 1.005, 6) for o in opens],
        }
    )


_N_DAILY = 60


def _daily_dates(n: int = _N_DAILY) -> list[datetime.date]:
    first = datetime.date(2024, 1, 1)
    return [first + datetime.timedelta(days=i) for i in range(n)]


_SESSION_DAYS = (
    datetime.date(2024, 1, 2),  # Tue
    datetime.date(2024, 1, 3),  # Wed
    datetime.date(2024, 1, 4),  # Thu
)


def _intraday_dates() -> list[datetime.datetime]:
    """15-minute XNYS bars, 09:30 through 15:45, across three sessions."""
    out: list[datetime.datetime] = []
    for day in _SESSION_DAYS:
        ts = datetime.datetime(day.year, day.month, day.day, 9, 30)
        end = datetime.datetime(day.year, day.month, day.day, 15, 45)
        while ts <= end:
            out.append(ts)
            ts += datetime.timedelta(minutes=15)
    return out


def _daily_frame(seed: int = 20260728, start: float = 100.0) -> pl.DataFrame:
    return _frame(_daily_dates(), seed=seed, start=start)


def _intraday_frame(seed: int = 20260729) -> pl.DataFrame:
    return _frame(_intraday_dates(), seed=seed)


# ---------------------------------------------------------------------------
# Fixture strategies (frozen + slots, matching tests/backtest conventions)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _CrossStrategy:
    """Enter on fast/slow crossover, exit on crossunder."""

    def entry(self) -> Crossover:
        return Crossover("fast", "slow")

    def exit(self) -> Crossunder:
        return Crossunder("fast", "slow")


@dataclass(frozen=True, slots=True)
class _ShortCrossStrategy:
    """Mirror of :class:`_CrossStrategy` — used as the short leg in dual mode."""

    def entry(self) -> Crossunder:
        return Crossunder("fast", "slow")

    def exit(self) -> Crossover:
        return Crossover("fast", "slow")


@dataclass(frozen=True, slots=True)
class _EntryRefTargetStrategy:
    """Exit at a fixed percentage above the close snapshotted at entry."""

    # Tuned so the fixture's random walk actually reaches the target and
    # produces multiple closed trades — see test_scenario_is_not_degenerate.
    target_pct: float = 0.25

    def entry(self) -> Crossover:
        return Crossover("fast", "slow")

    def exit(self) -> Condition:
        return ValueGTE(Col("close"), Pct(EntryRef("close"), self.target_pct))


@dataclass(frozen=True, slots=True)
class _LimitTakeProfitStrategy:
    """Same-bar limit exit when the bar's high tags ``tp_level``."""

    def entry(self) -> Crossover:
        return Crossover("fast", "slow")

    def exit(self) -> Condition:
        return Limit(ValueGTE(Col("high"), Col("tp_level")))


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------


def _artifacts(result: BacktestResult | MultiBacktestResult) -> dict[str, pl.DataFrame]:
    return {name: getattr(result, name) for name in ARTIFACTS}


def _scenario_plain(**kw: object) -> dict[str, pl.DataFrame]:
    return _artifacts(run(_daily_frame(), _CrossStrategy(), **kw))  # type: ignore[arg-type]


def _scenario_short(**kw: object) -> dict[str, pl.DataFrame]:
    return _artifacts(
        run(_daily_frame(), _CrossStrategy(), trade_side=TradeSide.SHORT, **kw)  # type: ignore[arg-type]
    )


def _scenario_entry_ref(**kw: object) -> dict[str, pl.DataFrame]:
    return _artifacts(run(_daily_frame(), _EntryRefTargetStrategy(), **kw))  # type: ignore[arg-type]


def _scenario_limit_exit(**kw: object) -> dict[str, pl.DataFrame]:
    return _artifacts(run(_daily_frame(), _LimitTakeProfitStrategy(), **kw))  # type: ignore[arg-type]


def _scenario_flatten_eod(**kw: object) -> dict[str, pl.DataFrame]:
    return _artifacts(
        run(
            _intraday_frame(),
            _CrossStrategy(),
            calendar=get_calendar("XNYS"),
            flatten_eod=True,
            **kw,  # type: ignore[arg-type]
        )
    )


def _scenario_flatten_eod_limit(**kw: object) -> dict[str, pl.DataFrame]:
    return _artifacts(
        run(
            _intraday_frame(),
            _LimitTakeProfitStrategy(),
            calendar=get_calendar("XNYS"),
            flatten_eod=True,
            **kw,  # type: ignore[arg-type]
        )
    )


_MULTI_SEEDS = {"AAA": 11111, "BBB": 22222, "CCC": 33333}


def _multi_frame() -> pl.DataFrame:
    return pl.concat(
        [
            _daily_frame(seed=seed, start=start).with_columns(
                pl.lit(symbol).alias("symbol")
            )
            for start, (symbol, seed) in zip(
                (100.0, 55.0, 240.0), _MULTI_SEEDS.items(), strict=True
            )
        ]
    )


def _scenario_multi(**kw: object) -> dict[str, pl.DataFrame]:
    return _artifacts(
        run(_multi_frame(), _CrossStrategy(), instrument_col="symbol", **kw)  # type: ignore[arg-type]
    )


def _scenario_multi_weighted(**kw: object) -> dict[str, pl.DataFrame]:
    return _artifacts(
        run(
            _multi_frame(),
            _CrossStrategy(),
            instrument_col="symbol",
            instrument_weights={"AAA": 0.5, "BBB": 0.3, "CCC": 0.2},
            **kw,  # type: ignore[arg-type]
        )
    )


def _scenario_dual(**kw: object) -> dict[str, pl.DataFrame]:
    return _artifacts(
        run(
            _daily_frame(),
            _CrossStrategy(),
            short_strategy=_ShortCrossStrategy(),
            **kw,  # type: ignore[arg-type]
        )
    )


# ---------------------------------------------------------------------------
# Bracket scenarios
#
# These pin the OUTPUT OF THE BRACKET PATH ITSELF, which the base scenarios do
# not: ``test_golden_baseline_no_bracket`` only pins ``bracket=None`` as a
# no-op, so before these existed the suite proved the non-bracket path was
# unperturbed and said nothing at all about bracket output. That is a fine gate
# for adding the feature and a useless one for changing its implementation —
# which is exactly what the ``_apply_bracket`` optimisation does.
#
# Each fires a bracket for real. ``test_bracket_scenario_actually_brackets``
# asserts that, so a fixture that quietly stops triggering fails loudly rather
# than pinning a vacuous baseline.
# ---------------------------------------------------------------------------


def _scenario_bracket_tp_long(**kw: object) -> dict[str, pl.DataFrame]:
    """Take-profit leg only — exercises the missing-stop ``pl.lit(False)`` branch."""
    return _artifacts(
        run(_daily_frame(), _CrossStrategy(), bracket=Bracket(take_profit=0.010), **kw)  # type: ignore[arg-type]
    )


def _scenario_bracket_sl_long(**kw: object) -> dict[str, pl.DataFrame]:
    """Stop-loss leg only — the mirror missing-leg branch."""
    return _artifacts(
        run(_daily_frame(), _CrossStrategy(), bracket=Bracket(stop_loss=0.008), **kw)  # type: ignore[arg-type]
    )


def _scenario_bracket_both_stop_first(**kw: object) -> dict[str, pl.DataFrame]:
    """Both legs, conservative tie policy (the default)."""
    return _artifacts(
        run(
            _daily_frame(),
            _CrossStrategy(),
            bracket=Bracket(take_profit=0.010, stop_loss=0.008, both_touch="stop_first"),
            **kw,  # type: ignore[arg-type]
        )
    )


def _scenario_bracket_both_tp_first(**kw: object) -> dict[str, pl.DataFrame]:
    """Both legs, opt-in submission-order tie policy — the other branch."""
    return _artifacts(
        run(
            _daily_frame(),
            _CrossStrategy(),
            bracket=Bracket(
                take_profit=0.010, stop_loss=0.008, both_touch="take_profit_first"
            ),
            **kw,  # type: ignore[arg-type]
        )
    )


def _scenario_bracket_short(**kw: object) -> dict[str, pl.DataFrame]:
    """The short mirror — an asymmetry here is the easy bug to introduce."""
    return _artifacts(
        run(
            _daily_frame(),
            _ShortCrossStrategy(),
            trade_side=TradeSide.SHORT,
            bracket=Bracket(take_profit=0.010, stop_loss=0.008),
            **kw,  # type: ignore[arg-type]
        )
    )


def _scenario_bracket_col_level(**kw: object) -> dict[str, pl.DataFrame]:
    """``str`` spec — absolute levels latched at the entry signal bar."""
    return _artifacts(
        run(
            _daily_frame(),
            _CrossStrategy(),
            bracket=Bracket(take_profit="tp_level"),
            **kw,  # type: ignore[arg-type]
        )
    )


def _scenario_bracket_entry_bar_gap(**kw: object) -> dict[str, pl.DataFrame]:
    """A stop tight enough to fire on the entry fill bar itself.

    That bar owes two fills, so it is the branch where the bracket return and
    ``trades.pnl`` have to agree on leg count.
    """
    return _artifacts(
        run(
            _daily_frame(),
            _CrossStrategy(),
            bracket=Bracket(stop_loss=0.0005),
            **kw,  # type: ignore[arg-type]
        )
    )


def _scenario_bracket_flatten_eod(**kw: object) -> dict[str, pl.DataFrame]:
    """Bracket alongside session-forced exits — the two exit paths interacting."""
    return _artifacts(
        run(
            _intraday_frame(),
            _CrossStrategy(),
            calendar=get_calendar("XNYS"),
            flatten_eod=True,
            bracket=Bracket(take_profit=0.004, stop_loss=0.003),
            **kw,  # type: ignore[arg-type]
        )
    )


def _scenario_bracket_cost(**kw: object) -> dict[str, pl.DataFrame]:
    """Bracket + cost together — pins the both-legs charge on a bracket fill."""
    return _artifacts(
        run(
            _daily_frame(),
            _CrossStrategy(),
            bracket=Bracket(take_profit=0.010, stop_loss=0.008),
            cost=Cost(commission_bps=1.5, slippage_bps=0.75),
            **kw,  # type: ignore[arg-type]
        )
    )


#: The nine original scenarios. These — and only these — feed the three
#: no-op gates, which pass ``cost=``/``bracket=`` themselves and would collide
#: with a scenario that hardcodes those keywords.
BASE_SCENARIOS: dict[str, Callable[..., dict[str, pl.DataFrame]]] = {
    "plain": _scenario_plain,
    "short": _scenario_short,
    "entry_ref": _scenario_entry_ref,
    "limit_exit": _scenario_limit_exit,
    "flatten_eod": _scenario_flatten_eod,
    "flatten_eod_limit": _scenario_flatten_eod_limit,
    "multi": _scenario_multi,
    "multi_weighted": _scenario_multi_weighted,
    "dual": _scenario_dual,
}

#: Scenarios that fire a bracket. Baseline-pinned like the rest, but excluded
#: from the no-op gates.
BRACKET_SCENARIOS: dict[str, Callable[..., dict[str, pl.DataFrame]]] = {
    "bracket_tp_long": _scenario_bracket_tp_long,
    "bracket_sl_long": _scenario_bracket_sl_long,
    "bracket_both_stop_first": _scenario_bracket_both_stop_first,
    "bracket_both_tp_first": _scenario_bracket_both_tp_first,
    "bracket_short": _scenario_bracket_short,
    "bracket_col_level": _scenario_bracket_col_level,
    "bracket_entry_bar_gap": _scenario_bracket_entry_bar_gap,
    "bracket_flatten_eod": _scenario_bracket_flatten_eod,
    "bracket_cost": _scenario_bracket_cost,
}

SCENARIOS: dict[str, Callable[..., dict[str, pl.DataFrame]]] = {
    **BASE_SCENARIOS,
    **BRACKET_SCENARIOS,
}

# Pinned separately so that deleting a scenario is itself a test failure —
# a silently-dropped scenario would silently drop its regression coverage.
EXPECTED_SCENARIOS = frozenset(
    {
        "plain",
        "short",
        "entry_ref",
        "limit_exit",
        "flatten_eod",
        "flatten_eod_limit",
        "multi",
        "multi_weighted",
        "dual",
        "bracket_tp_long",
        "bracket_sl_long",
        "bracket_both_stop_first",
        "bracket_both_tp_first",
        "bracket_short",
        "bracket_col_level",
        "bracket_entry_bar_gap",
        "bracket_flatten_eod",
        "bracket_cost",
    }
)


def _baseline_path(scenario: str, artifact: str) -> Path:
    return GOLDEN_DIR / scenario / f"{artifact}.parquet"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_scenario_set_is_complete() -> None:
    """Scenarios may be added, but never removed without an explicit edit."""
    assert set(SCENARIOS) == set(EXPECTED_SCENARIOS)


@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
def test_scenario_is_not_degenerate(scenario: str) -> None:
    """Guard against a fixture that pins an all-zero, no-trade backtest.

    Without this, a fixture that stopped producing signals would still
    match its baseline and the regression gate would be vacuous.
    """
    produced = SCENARIOS[scenario]()
    assert produced["trades"].height > 0, f"{scenario}: fixture produced no trades"
    returns = produced["returns"]["return"]
    assert returns.len() > 0
    assert (returns != 0.0).sum() > 0, f"{scenario}: fixture produced only zero returns"
    assert returns.is_null().sum() == 0, f"{scenario}: null returns"


@pytest.mark.parametrize("scenario", sorted(BRACKET_SCENARIOS))
def test_bracket_scenario_actually_brackets(scenario: str) -> None:
    """A bracket scenario that stopped firing would pin a vacuous baseline.

    Without this, a change that silently disabled the bracket would still match
    its frozen artifacts — the baseline would faithfully record "nothing
    happened" and the gate would pass while measuring nothing.

    A bracket exit is detectable from the public artifacts alone: the internal
    ``_bracket_*`` columns are dropped before return, but a bracketed trade
    exits on the bar that tagged the level rather than the bar after a signal,
    so its held count differs from the same strategy run without a bracket.
    """
    produced = BRACKET_SCENARIOS[scenario]()
    plain = _scenario_plain()
    assert produced["trades"].height > 0, f"{scenario}: no trades"
    assert not produced["trades"].equals(plain["trades"]), (
        f"{scenario}: trades are identical to the un-bracketed run — "
        f"the bracket never fired, so this baseline pins nothing"
    )


@pytest.mark.parametrize("artifact", ARTIFACTS)
@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
def test_golden_baseline(scenario: str, artifact: str, scan_backend: str) -> None:
    """Engine output must be byte-identical to the frozen baseline.

    A failure here means the return-expression chain, the trade extractor,
    or the signals schema changed.  If the change is intended, regenerate
    with ``python tests/backtest/test_golden_baseline.py --regenerate`` and
    review the Parquet diff.

    Run against **every** resolver backend. These baselines were frozen before
    the compiled resolver existed, so this is the strongest statement available
    that it changed nothing: not "the two backends agree with each other", which
    two equally-wrong implementations would also satisfy, but "each agrees with
    output committed before either of them was written".

    **Never regenerate to make a backend pass.** A moved baseline here means the
    accelerator is wrong.
    """
    path = _baseline_path(scenario, artifact)
    assert path.exists(), (
        f"missing golden baseline {path}. Regenerate with: "
        "python tests/backtest/test_golden_baseline.py --regenerate"
    )
    produced = SCENARIOS[scenario]()[artifact]
    expected = pl.read_parquet(path)
    assert_frame_equal(
        produced,
        expected,
        check_exact=True,
        check_dtypes=True,
        check_column_order=True,
        check_row_order=True,
    )


@pytest.mark.parametrize("artifact", ARTIFACTS)
@pytest.mark.parametrize("scenario", sorted(BASE_SCENARIOS))
def test_golden_baseline_zero_cost(scenario: str, artifact: str) -> None:
    """``cost=Cost()`` is an exact no-op — the 0.13.0 release gate.

    This deliberately exercises the *real* cost arithmetic (a literal
    ``0.0`` bps column is materialized and subtracted) rather than
    short-circuiting on an all-zero model.  If subtracting zero ever
    perturbs a value, this fails and the release is blocked.
    """
    produced = BASE_SCENARIOS[scenario](cost=Cost())[artifact]
    expected = pl.read_parquet(_baseline_path(scenario, artifact))
    assert_frame_equal(
        produced,
        expected,
        check_exact=True,
        check_dtypes=True,
        check_column_order=True,
        check_row_order=True,
    )


@pytest.mark.parametrize("artifact", ARTIFACTS)
@pytest.mark.parametrize("scenario", sorted(BASE_SCENARIOS))
def test_golden_baseline_no_bracket(scenario: str, artifact: str) -> None:
    """``bracket=None`` is an exact no-op — the other half of the 0.13.0 gate.

    Unlike ``Cost()`` there is no all-zero :class:`~mktlib.backtest.Bracket`
    to exercise (an empty one is rejected at construction), so what this
    pins is the *plumbing*: threading a new keyword through ``run`` →
    ``_run_multi`` → ``_run_dual`` → ``_run_core`` must not perturb any of
    the four return-expression chains.
    """
    produced = BASE_SCENARIOS[scenario](bracket=None)[artifact]
    expected = pl.read_parquet(_baseline_path(scenario, artifact))
    assert_frame_equal(
        produced,
        expected,
        check_exact=True,
        check_dtypes=True,
        check_column_order=True,
        check_row_order=True,
    )


@pytest.mark.parametrize("artifact", ARTIFACTS)
@pytest.mark.parametrize("scenario", sorted(BASE_SCENARIOS))
def test_golden_baseline_zero_cost_no_bracket(scenario: str, artifact: str) -> None:
    """Both new 0.13.0 knobs at their no-op settings, together."""
    produced = BASE_SCENARIOS[scenario](cost=Cost(), bracket=None)[artifact]
    expected = pl.read_parquet(_baseline_path(scenario, artifact))
    assert_frame_equal(
        produced,
        expected,
        check_exact=True,
        check_dtypes=True,
        check_column_order=True,
        check_row_order=True,
    )


def test_run_is_deterministic_across_invocations() -> None:
    """Two identical calls agree — a precondition for the pins to mean anything."""
    for scenario in sorted(SCENARIOS):
        first = SCENARIOS[scenario]()
        second = SCENARIOS[scenario]()
        for artifact in ARTIFACTS:
            assert_frame_equal(
                first[artifact], second[artifact], check_exact=True, check_dtypes=True
            )


# ---------------------------------------------------------------------------
# Baseline regeneration (manual, never invoked by the test suite)
# ---------------------------------------------------------------------------


def _regenerate(only: frozenset[str] | None = None) -> None:
    """Rewrite frozen baselines.

    ``only`` restricts the rewrite to named scenarios. Adding a scenario should
    not require rewriting the other twenty-seven artifacts — a bulk rewrite makes
    an unintended numeric change invisible inside a large, expected diff.
    """
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    for scenario, build in SCENARIOS.items():
        if only is not None and scenario not in only:
            continue
        produced = build()
        (GOLDEN_DIR / scenario).mkdir(parents=True, exist_ok=True)
        for artifact in ARTIFACTS:
            path = _baseline_path(scenario, artifact)
            produced[artifact].write_parquet(path)
            print(f"wrote {path} ({produced[artifact].height} rows)")


if __name__ == "__main__":
    _argv = sys.argv[1:]
    if "--regenerate" not in _argv:
        print(__doc__)
        raise SystemExit(
            "refusing to run: pass --regenerate to rewrite the frozen baselines"
        )
    _only: frozenset[str] | None = None
    if "--only" in _argv:
        _names = _argv[_argv.index("--only") + 1 :]
        if not _names:
            raise SystemExit("--only requires at least one scenario name")
        _unknown = set(_names) - set(SCENARIOS)
        if _unknown:
            raise SystemExit(f"unknown scenario(s): {sorted(_unknown)}")
        _only = frozenset(_names)
    _regenerate(_only)
