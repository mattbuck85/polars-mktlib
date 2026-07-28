"""Tests for per-fill transaction costs (:class:`mktlib.backtest.Cost`).

The invariants under test, in order of importance:

1. ``cost=None`` and ``cost=Cost()`` are exact no-ops (also pinned against
   frozen Parquet baselines in ``test_golden_baseline.py``).
2. Cost is charged **at the fill** — the entry bar, the exit bar, the limit
   fill bar and the session-forced-exit bar — and nowhere else. Holding
   bars and flat bars are untouched.
3. Cost is **subtracted**, never multiplied by the trade side: a short pays
   the same haircut a long does.
4. ``trades.pnl`` charges each leg with the same alignment as the price
   that leg pays.
"""

from __future__ import annotations

import datetime
import math
from dataclasses import FrozenInstanceError, dataclass

import polars as pl
import pytest

from mktlib.backtest import (
    Col,
    Condition,
    Cost,
    Crossover,
    Crossunder,
    Limit,
    Lit,
    TradeSide,
    ValueGTE,
    run,
)
from mktlib.scheduling import get_calendar

BPS = 1e-4


# ---------------------------------------------------------------------------
# Fixtures (module-local, matching tests/backtest conventions)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CrossStrategy:
    def entry(self) -> Crossover:
        return Crossover("fast", "slow")

    def exit(self) -> Crossunder:
        return Crossunder("fast", "slow")


@dataclass(frozen=True, slots=True)
class CrossEntryLimitExit:
    """Enter on crossover, exit same-bar at a fixed limit price."""

    limit_price: float = 103.0

    def entry(self) -> Crossover:
        return Crossover("fast", "slow")

    def exit(self) -> Condition:
        return Limit(ValueGTE(Col("high"), Lit(self.limit_price)))


# Same shape as tests/backtest/test_engine.py::ohlcv — position is
# [0,0,1,1,1,0,0,0], so bar 3 is the entry fill bar and bar 6 the exit
# fill bar. Every expectation below is written out from those two bars.
_OPENS = [100.0, 100.5, 101.5, 103.5, 105.5, 103.5, 99.5, 97.5]
_CLOSES = [100.0, 101.0, 103.0, 105.0, 104.0, 100.0, 98.0, 97.0]
_ENTRY_BAR = 3
_EXIT_BAR = 6
_ENTRY_FILL = _OPENS[_ENTRY_BAR]  # 103.5
_EXIT_FILL = _OPENS[_EXIT_BAR]  # 99.5


@pytest.fixture
def daily() -> pl.DataFrame:
    """8 daily bars; fast crosses above slow at bar 2, below at bar 5."""
    return pl.DataFrame(
        {
            "date": pl.date_range(pl.date(2024, 1, 1), pl.date(2024, 1, 8), eager=True),
            "open": _OPENS,
            "close": _CLOSES,
            "fast": [1.0, 1.0, 3.0, 4.0, 3.5, 1.0, 0.5, 0.3],
            "slow": [2.0] * 8,
            # Distinct per-bar slippage so a mis-shifted read is detectable:
            # every bar carries a different value.
            "slip_bps": [float(i) for i in range(8)],
        }
    )


@pytest.fixture
def limit_daily() -> pl.DataFrame:
    """5 bars; crossover at bar 2 → entry fill at open[3]; TP=103 tags bar 3."""
    closes = [100.0, 100.5, 101.0, 102.5, 103.0]
    highs = [100.5, 101.0, 101.5, 103.5, 104.0]
    opens = [round(c - 0.1, 10) for c in closes]
    return pl.DataFrame(
        {
            "date": pl.date_range(pl.date(2024, 1, 1), pl.date(2024, 1, 5), eager=True),
            "open": opens,
            "high": highs,
            "low": [min(o, c) - 0.5 for o, c in zip(opens, closes, strict=True)],
            "close": closes,
            "fast": [99.0, 99.5, 100.5, 101.0, 100.5],
            "slow": [100.0] * 5,
        }
    )


_SESSION_DAYS = (datetime.date(2024, 1, 2), datetime.date(2024, 1, 3))
_BARS_PER_SESSION = 26  # 09:30 … 15:45 inclusive, 15-minute bars


def _intraday_dates() -> list[datetime.datetime]:
    out: list[datetime.datetime] = []
    for day in _SESSION_DAYS:
        ts = datetime.datetime(day.year, day.month, day.day, 9, 30)
        end = datetime.datetime(day.year, day.month, day.day, 15, 45)
        while ts <= end:
            out.append(ts)
            ts += datetime.timedelta(minutes=15)
    return out


def _intraday_frame(entry_signal_bar: int) -> pl.DataFrame:
    """Monotone intraday tape with a single crossover at *entry_signal_bar*."""
    dates = _intraday_dates()
    n = len(dates)
    fast = [1.0] * n
    for i in range(entry_signal_bar, n):
        fast[i] = 3.0
    return pl.DataFrame(
        {
            "date": dates,
            "open": [100.0 + i * 0.1 for i in range(n)],
            "close": [100.05 + i * 0.1 for i in range(n)],
            "fast": fast,
            "slow": [2.0] * n,
        }
    )


def _rets(result_returns: pl.DataFrame) -> list[float]:
    return result_returns["return"].to_list()


# ---------------------------------------------------------------------------
# A) Dataclass contract
# ---------------------------------------------------------------------------


class TestCostDataclass:
    def test_defaults_are_zero(self) -> None:
        cost = Cost()
        assert cost.commission_bps == 0.0
        assert cost.slippage_bps == 0.0
        assert cost.slippage_col is None
        assert cost.flat_bps == 0.0

    def test_flat_bps_sums_the_two_flat_terms(self) -> None:
        assert Cost(commission_bps=1.5, slippage_bps=0.75).flat_bps == 2.25

    def test_is_frozen(self) -> None:
        cost = Cost()
        with pytest.raises(FrozenInstanceError):
            cost.commission_bps = 1.0  # type: ignore[misc]

    def test_is_hashable_and_value_equal(self) -> None:
        """Primitives only — so a consumer's cache key can hold one."""
        assert Cost(1.0, 2.0) == Cost(1.0, 2.0)
        assert hash(Cost(1.0, 2.0)) == hash(Cost(1.0, 2.0))
        assert len({Cost(1.0), Cost(1.0), Cost(2.0)}) == 2

    @pytest.mark.parametrize("field", ["commission_bps", "slippage_bps"])
    def test_rejects_negative(self, field: str) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            Cost(**{field: -0.1})

    @pytest.mark.parametrize("field", ["commission_bps", "slippage_bps"])
    @pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
    def test_rejects_non_finite(self, field: str, value: float) -> None:
        with pytest.raises(ValueError, match="finite"):
            Cost(**{field: value})

    @pytest.mark.parametrize("field", ["commission_bps", "slippage_bps"])
    def test_rejects_non_numeric(self, field: str) -> None:
        with pytest.raises(TypeError, match="real number"):
            Cost(**{field: "1.0"})

    def test_rejects_bool(self) -> None:
        with pytest.raises(TypeError, match="real number"):
            Cost(commission_bps=True)

    def test_rejects_empty_slippage_col(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            Cost(slippage_col="")

    def test_rejects_non_string_slippage_col(self) -> None:
        with pytest.raises(TypeError, match="column name"):
            Cost(slippage_col=3.0)  # type: ignore[arg-type]

    def test_accepts_int_bps(self) -> None:
        assert Cost(commission_bps=2).flat_bps == 2


# ---------------------------------------------------------------------------
# B) Backward compatibility — the release gate
# ---------------------------------------------------------------------------


class TestZeroCostIsANoOp:
    @pytest.mark.parametrize("artifact", ["returns", "trades", "signals"])
    def test_zero_cost_matches_no_cost(self, daily: pl.DataFrame, artifact: str) -> None:
        from polars.testing import assert_frame_equal

        base = run(daily, CrossStrategy())
        zeroed = run(daily, CrossStrategy(), cost=Cost())
        assert_frame_equal(
            getattr(zeroed, artifact),
            getattr(base, artifact),
            check_exact=True,
            check_dtypes=True,
            check_column_order=True,
        )

    def test_cost_column_is_not_leaked_into_signals(self, daily: pl.DataFrame) -> None:
        result = run(daily, CrossStrategy(), cost=Cost(commission_bps=5.0))
        assert not [c for c in result.signals.columns if "cost" in c]

    def test_trades_schema_is_unchanged(self, daily: pl.DataFrame) -> None:
        base = run(daily, CrossStrategy())
        costed = run(daily, CrossStrategy(), cost=Cost(commission_bps=5.0))
        assert costed.trades.columns == base.trades.columns
        assert costed.trades.schema == base.trades.schema


# ---------------------------------------------------------------------------
# C) Charged at the fill, and only at the fill
# ---------------------------------------------------------------------------


class TestCostAtTheFill:
    def test_entry_and_exit_bars_are_charged_long(self, daily: pl.DataFrame) -> None:
        c = 10.0 * BPS  # commission 6 + slippage 4
        rets = _rets(run(daily, CrossStrategy(), cost=Cost(6.0, 4.0)).returns)

        expected_entry = (_CLOSES[3] - _OPENS[3]) / _OPENS[3] - c
        expected_exit = (_OPENS[6] - _CLOSES[5]) / _CLOSES[5] - c
        assert rets[_ENTRY_BAR] == pytest.approx(expected_entry, rel=1e-12)
        assert rets[_EXIT_BAR] == pytest.approx(expected_exit, rel=1e-12)

    def test_holding_and_flat_bars_are_untouched(self, daily: pl.DataFrame) -> None:
        base = _rets(run(daily, CrossStrategy()).returns)
        costed = _rets(run(daily, CrossStrategy(), cost=Cost(50.0)).returns)
        for i in range(len(base)):
            if i in (_ENTRY_BAR, _EXIT_BAR):
                continue
            assert costed[i] == base[i], f"bar {i} was charged but is not a fill bar"

    def test_delta_is_exactly_the_charge_on_fill_bars(self, daily: pl.DataFrame) -> None:
        base = _rets(run(daily, CrossStrategy()).returns)
        costed = _rets(run(daily, CrossStrategy(), cost=Cost(25.0)).returns)
        for bar in (_ENTRY_BAR, _EXIT_BAR):
            assert costed[bar] - base[bar] == pytest.approx(-25.0 * BPS, rel=1e-9)

    def test_short_pays_the_same_haircut_not_the_mirrored_one(
        self, daily: pl.DataFrame
    ) -> None:
        """Cost is subtracted, never multiplied by ``effective_side``."""
        c = 12.0 * BPS
        rets = _rets(
            run(
                daily,
                CrossStrategy(),
                trade_side=TradeSide.SHORT,
                cost=Cost(commission_bps=12.0),
            ).returns
        )
        expected_entry = -((_CLOSES[3] - _OPENS[3]) / _OPENS[3]) - c
        expected_exit = -((_OPENS[6] - _CLOSES[5]) / _CLOSES[5]) - c
        assert rets[_ENTRY_BAR] == pytest.approx(expected_entry, rel=1e-12)
        assert rets[_EXIT_BAR] == pytest.approx(expected_exit, rel=1e-12)

    def test_short_delta_has_the_same_sign_as_long(self, daily: pl.DataFrame) -> None:
        for side in (TradeSide.LONG, TradeSide.SHORT):
            base = _rets(run(daily, CrossStrategy(), trade_side=side).returns)
            costed = _rets(
                run(daily, CrossStrategy(), trade_side=side, cost=Cost(30.0)).returns
            )
            assert costed[_ENTRY_BAR] - base[_ENTRY_BAR] < 0
            assert costed[_EXIT_BAR] - base[_EXIT_BAR] < 0


# ---------------------------------------------------------------------------
# D) trades.pnl
# ---------------------------------------------------------------------------


class TestTradesPnl:
    def test_round_trip_charges_both_legs(self, daily: pl.DataFrame) -> None:
        result = run(daily, CrossStrategy(), cost=Cost(7.0, 3.0))
        assert result.trades.height == 1
        pnl = result.trades["pnl"][0]
        expected = (_EXIT_FILL / _ENTRY_FILL - 1) - 2 * 10.0 * BPS
        assert pnl == pytest.approx(expected, rel=1e-12)

    def test_short_round_trip_charges_both_legs(self, daily: pl.DataFrame) -> None:
        result = run(
            daily, CrossStrategy(), trade_side=TradeSide.SHORT, cost=Cost(10.0)
        )
        pnl = result.trades["pnl"][0]
        expected = -(_EXIT_FILL / _ENTRY_FILL - 1) - 2 * 10.0 * BPS
        assert pnl == pytest.approx(expected, rel=1e-12)

    def test_pnl_delta_is_exactly_two_sides(self, daily: pl.DataFrame) -> None:
        base = run(daily, CrossStrategy()).trades["pnl"][0]
        costed = run(daily, CrossStrategy(), cost=Cost(18.0)).trades["pnl"][0]
        assert costed - base == pytest.approx(-2 * 18.0 * BPS, rel=1e-9)

    def test_entry_leg_reads_the_fill_bar_not_the_signal_bar(
        self, daily: pl.DataFrame
    ) -> None:
        """``slip_bps`` is ``[0,1,2,…]`` so a one-bar misalignment shifts pnl.

        The entry *signal* is bar 2, the entry *fill* is bar 3 — the leg must
        pay ``slip_bps[3] == 3``, not ``slip_bps[2] == 2``.
        """
        result = run(daily, CrossStrategy(), cost=Cost(slippage_col="slip_bps"))
        pnl = result.trades["pnl"][0]
        expected = (_EXIT_FILL / _ENTRY_FILL - 1) - (3.0 + 6.0) * BPS
        assert pnl == pytest.approx(expected, rel=1e-12)

    def test_returns_and_pnl_agree_on_the_per_leg_charge(
        self, daily: pl.DataFrame
    ) -> None:
        """The same two charges appear in the returns series and in ``pnl``.

        Note: compounding the *returns* series does not reproduce ``pnl``
        once costs are on, because ``pnl`` subtracts them arithmetically
        while the returns series compounds them. The reconcilable statement
        is the per-leg delta, asserted here.
        """
        cost = Cost(commission_bps=9.0)
        base, costed = run(daily, CrossStrategy()), run(daily, CrossStrategy(), cost=cost)
        ret_charge = sum(
            _rets(base.returns)[bar] - _rets(costed.returns)[bar]
            for bar in (_ENTRY_BAR, _EXIT_BAR)
        )
        pnl_charge = base.trades["pnl"][0] - costed.trades["pnl"][0]
        assert ret_charge == pytest.approx(pnl_charge, rel=1e-9)
        assert pnl_charge == pytest.approx(2 * 9.0 * BPS, rel=1e-9)

    def test_open_trailing_trade_is_still_dropped(self, daily: pl.DataFrame) -> None:
        """Pre-existing truncation, unchanged by cost.

        ``sum(pnl)`` never reconciles with the full returns series because a
        position still open on the final bar produces no trade row.
        """
        df = daily.head(5)  # entry at bar 2, never exits
        result = run(df, CrossStrategy(), cost=Cost(10.0))
        assert result.trades.height == 0
        assert any(r != 0.0 for r in _rets(result.returns))


# ---------------------------------------------------------------------------
# E) slippage_col
# ---------------------------------------------------------------------------


class TestSlippageColumn:
    def test_per_bar_slippage_stacks_on_the_flat_terms(
        self, daily: pl.DataFrame
    ) -> None:
        base = _rets(run(daily, CrossStrategy()).returns)
        costed = _rets(
            run(
                daily,
                CrossStrategy(),
                cost=Cost(commission_bps=1.0, slippage_bps=2.0, slippage_col="slip_bps"),
            ).returns
        )
        # slip_bps == bar index, read on the fill bar.
        assert costed[_ENTRY_BAR] - base[_ENTRY_BAR] == pytest.approx(
            -(1.0 + 2.0 + 3.0) * BPS, rel=1e-9
        )
        assert costed[_EXIT_BAR] - base[_EXIT_BAR] == pytest.approx(
            -(1.0 + 2.0 + 6.0) * BPS, rel=1e-9
        )

    def test_missing_column_raises(self, daily: pl.DataFrame) -> None:
        with pytest.raises(ValueError, match="slippage_col='nope' not found"):
            run(daily, CrossStrategy(), cost=Cost(slippage_col="nope"))

    def test_all_zero_column_is_a_no_op(self, daily: pl.DataFrame) -> None:
        from polars.testing import assert_frame_equal

        df = daily.with_columns(pl.lit(0.0).alias("zero_bps"))
        base = run(df, CrossStrategy())
        costed = run(df, CrossStrategy(), cost=Cost(slippage_col="zero_bps"))
        assert_frame_equal(costed.returns, base.returns, check_exact=True)


# ---------------------------------------------------------------------------
# F) Limit exits
# ---------------------------------------------------------------------------


class TestLimitExitCost:
    def test_limit_fill_bar_is_charged(self, limit_daily: pl.DataFrame) -> None:
        base = _rets(run(limit_daily, CrossEntryLimitExit()).returns)
        costed = _rets(
            run(limit_daily, CrossEntryLimitExit(), cost=Cost(20.0)).returns
        )
        # bar 3 is the limit fill bar (high 103.5 tags TP 103).
        assert costed[3] - base[3] == pytest.approx(-20.0 * BPS, rel=1e-9)

    def test_post_limit_bar_stays_costless(self, limit_daily: pl.DataFrame) -> None:
        base = _rets(run(limit_daily, CrossEntryLimitExit()).returns)
        costed = _rets(
            run(limit_daily, CrossEntryLimitExit(), cost=Cost(20.0)).returns
        )
        assert base[4] == 0.0
        assert costed[4] == 0.0

    def test_limit_exit_leg_reads_the_limit_bar_in_pnl(
        self, limit_daily: pl.DataFrame
    ) -> None:
        base = run(limit_daily, CrossEntryLimitExit()).trades["pnl"][0]
        costed = run(limit_daily, CrossEntryLimitExit(), cost=Cost(11.0)).trades["pnl"][0]
        assert costed - base == pytest.approx(-2 * 11.0 * BPS, rel=1e-9)

    def test_same_bar_entry_and_limit_exit_charges_one_side_in_returns(
        self, limit_daily: pl.DataFrame
    ) -> None:
        """KNOWN GAP, pinned deliberately — see the module note in the plan.

        When a limit exit fires on the *entry fill bar* itself, two fills
        occur but the return-expression chain takes the ``_limit_ret``
        branch and charges a single side. ``trades.pnl`` charges both. The
        returns series therefore understates cost by one side for this
        (uncommon) overlap. Ratify or fix before relying on cost-adjusted
        returns for TP-on-entry-bar strategies.
        """
        base = run(limit_daily, CrossEntryLimitExit())
        costed = run(limit_daily, CrossEntryLimitExit(), cost=Cost(20.0))
        ret_charge = sum(
            b - c
            for b, c in zip(_rets(base.returns), _rets(costed.returns), strict=True)
        )
        pnl_charge = base.trades["pnl"][0] - costed.trades["pnl"][0]
        assert ret_charge == pytest.approx(20.0 * BPS, rel=1e-9)
        assert pnl_charge == pytest.approx(2 * 20.0 * BPS, rel=1e-9)


# ---------------------------------------------------------------------------
# G) flatten_eod
# ---------------------------------------------------------------------------


class TestFlattenEodCost:
    def test_session_forced_exit_is_charged(self) -> None:
        # Entry signal at bar 20 → fill at bar 21; held to the session-last
        # bar 25, which force-closes at its own open.
        df = _intraday_frame(entry_signal_bar=20)
        cal = get_calendar("XNYS")
        base = run(df, CrossStrategy(), calendar=cal, flatten_eod=True)
        costed = run(
            df, CrossStrategy(), calendar=cal, flatten_eod=True, cost=Cost(15.0)
        )
        b, c = _rets(base.returns), _rets(costed.returns)
        session_last = _BARS_PER_SESSION - 1  # bar 25
        assert c[21] - b[21] == pytest.approx(-15.0 * BPS, rel=1e-9)  # entry fill
        assert c[session_last] - b[session_last] == pytest.approx(
            -15.0 * BPS, rel=1e-9
        )  # forced exit
        for i in (22, 23, 24, session_last + 1):
            assert c[i] == b[i], f"bar {i} charged but no fill occurred"

    def test_session_forced_exit_charges_both_legs_in_pnl(self) -> None:
        df = _intraday_frame(entry_signal_bar=20)
        cal = get_calendar("XNYS")
        base = run(df, CrossStrategy(), calendar=cal, flatten_eod=True)
        costed = run(
            df, CrossStrategy(), calendar=cal, flatten_eod=True, cost=Cost(15.0)
        )
        assert costed.trades["pnl"][0] - base.trades["pnl"][0] == pytest.approx(
            -2 * 15.0 * BPS, rel=1e-9
        )

    def test_suppressed_entry_bar_return_stays_zero(self) -> None:
        """Entry filling *on* the session-last bar returns 0.0, uncharged.

        KNOWN DIVERGENCE, pinned deliberately: the engine models this
        open-and-immediately-flatten as a non-trade in the returns series
        (0.0, no cost), while ``_extract_trades`` still emits a row and
        therefore charges both legs.
        """
        df = _intraday_frame(entry_signal_bar=_BARS_PER_SESSION - 2)  # bar 24
        cal = get_calendar("XNYS")
        costed = run(
            df, CrossStrategy(), calendar=cal, flatten_eod=True, cost=Cost(15.0)
        )
        rets = _rets(costed.returns)
        assert all(r == 0.0 for r in rets)
        assert costed.trades.height == 1
        assert costed.trades["pnl"][0] == pytest.approx(-2 * 15.0 * BPS, rel=1e-9)


# ---------------------------------------------------------------------------
# H) Threading through the multi-instrument and dual-strategy paths
# ---------------------------------------------------------------------------


class TestCostThreading:
    def test_multi_instrument_charges_every_symbol(self, daily: pl.DataFrame) -> None:
        multi = pl.concat(
            [
                daily.with_columns(pl.lit(sym).alias("symbol"))
                for sym in ("AAA", "BBB")
            ]
        )
        base = run(multi, CrossStrategy(), instrument_col="symbol")
        costed = run(
            multi, CrossStrategy(), instrument_col="symbol", cost=Cost(14.0)
        )
        for sym in ("AAA", "BBB"):
            assert costed[sym].trades["pnl"][0] - base[sym].trades["pnl"][0] == (
                pytest.approx(-2 * 14.0 * BPS, rel=1e-9)
            )

    def test_multi_weighted_returns_reflect_cost(self, daily: pl.DataFrame) -> None:
        multi = pl.concat(
            [
                daily.with_columns(pl.lit(sym).alias("symbol"))
                for sym in ("AAA", "BBB")
            ]
        )
        weights = {"AAA": 0.5, "BBB": 0.5}
        base = run(
            multi, CrossStrategy(), instrument_col="symbol", instrument_weights=weights
        )
        costed = run(
            multi,
            CrossStrategy(),
            instrument_col="symbol",
            instrument_weights=weights,
            cost=Cost(14.0),
        )
        b, c = _rets(base.returns), _rets(costed.returns)
        assert c[_ENTRY_BAR] - b[_ENTRY_BAR] == pytest.approx(-14.0 * BPS, rel=1e-9)

    def test_dual_strategy_charges_both_legs(self, daily: pl.DataFrame) -> None:
        @dataclass(frozen=True, slots=True)
        class ShortLeg:
            def entry(self) -> Crossunder:
                return Crossunder("fast", "slow")

            def exit(self) -> Crossover:
                return Crossover("fast", "slow")

        base = run(daily, CrossStrategy(), short_strategy=ShortLeg())
        costed = run(
            daily, CrossStrategy(), short_strategy=ShortLeg(), cost=Cost(13.0)
        )
        b, c = _rets(base.returns), _rets(costed.returns)
        charged = [i for i in range(len(b)) if c[i] != b[i]]
        assert _ENTRY_BAR in charged
        assert all(c[i] - b[i] < 0 for i in charged)
