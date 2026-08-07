"""Tests for same-bar limit-fill exits (``Limit`` wrapper)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import polars as pl
import pytest

from mktlib.backtest import (
    BacktestResult,
    Col,
    Condition,
    Crossover,
    Crossunder,
    Limit,
    Lit,
    TradeSide,
    ValueGTE,
    ValueLTE,
    run,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _d(day: int) -> date:
    return date(2024, 1, day)


@dataclass(frozen=True, slots=True)
class _CrossEntryLimitExit:
    """Enter on fast crossover; exit via Limit wrapper supplied by the fixture."""
    exit_cond: Condition

    def entry(self) -> Crossover:
        return Crossover("fast", "slow")

    def exit(self) -> Condition:
        return self.exit_cond


def _base_ohlcv(*, closes, opens=None, highs=None, lows=None) -> pl.DataFrame:
    """5-bar OHLCV with a crossover at bar 2 — position becomes 1 from bar 3."""
    n = len(closes)
    opens = opens if opens is not None else [c - 0.1 for c in closes]
    highs = highs if highs is not None else [max(o, c) + 0.5 for o, c in zip(opens, closes)]
    lows = lows if lows is not None else [min(o, c) - 0.5 for o, c in zip(opens, closes)]
    # fast crosses above slow at bar 2
    slow = [100.0] * n
    fast = [99.0, 99.5, 100.5, 101.0, 100.5][:n] + [100.5] * max(0, n - 5)
    return pl.DataFrame({
        "date": [_d(i + 1) for i in range(n)],
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "fast": fast,
        "slow": slow,
    })


# ---------------------------------------------------------------------------
# A) Core behavior
# ---------------------------------------------------------------------------


class TestLimitCoreBehavior:
    def test_tp_hit_long_fills_at_limit_same_bar(self):
        """Crossover at d(3) (entry fill at open[d(4)]=102.4); TP at 103 fires
        on d(4) (high=103.5).

        The limit fires on the bar the entry filled, so the return is measured
        from the entry fill — the position was never held across close[d(3)].
        This assertion previously used close[d(3)] as the base while the
        ``pnl`` assertion below used the entry fill, so the same test asserted
        both a correct and an incorrect view of one trade and passed, because
        the engine disagreed with itself the same way.
        """
        closes = [100.0, 100.5, 101.0, 102.5, 103.0]
        highs = [100.5, 101.0, 101.5, 103.5, 104.0]
        df = _base_ohlcv(closes=closes, highs=highs)

        tp = 103.0
        strat = _CrossEntryLimitExit(Limit(ValueGTE(Col("high"), Lit(tp))))
        result = run(df, strat)

        trades = result.trades
        assert trades.height == 1
        row = trades.row(0, named=True)
        assert row["exit_date"] == _d(4)
        # Entry fills at open[d(4)]=102.4; exit at TP=103 → pnl = 103/102.4 - 1
        assert row["pnl"] == pytest.approx(103.0 / 102.4 - 1, rel=1e-9)

        ret = result.returns
        r4 = ret.filter(pl.col("date") == _d(4)).row(0, named=True)["return"]
        # Return at d(4) = (limit - entry fill) / entry fill — the same base
        # ``pnl`` uses above. The two views of one trade must agree.
        entry_fill = df.filter(pl.col("date") == _d(4)).row(0, named=True)["open"]
        assert r4 == pytest.approx((tp - entry_fill) / entry_fill, rel=1e-9)
        assert r4 == pytest.approx(row["pnl"], rel=1e-9)
        # Return on d(5) = 0 (flat post-limit)
        r5 = ret.filter(pl.col("date") == _d(5)).row(0, named=True)["return"]
        assert r5 == pytest.approx(0.0, abs=1e-12)

    def test_sl_hit_long_fills_at_limit_same_bar(self):
        """Long position; SL at 99 fires on d(4) (low=98.5).

        The limit fires on the bar the entry filled, so the return is measured
        from the entry fill, not from close[d(3)].
        """
        closes = [100.0, 100.5, 101.0, 99.5, 99.0]
        lows = [99.5, 100.0, 100.5, 98.5, 98.5]
        df = _base_ohlcv(closes=closes, lows=lows)

        sl = 99.0
        strat = _CrossEntryLimitExit(Limit(ValueLTE(Col("low"), Lit(sl))))
        result = run(df, strat)

        trades = result.trades
        assert trades.height == 1
        assert trades.row(0, named=True)["exit_date"] == _d(4)

        ret = result.returns
        r4 = ret.filter(pl.col("date") == _d(4)).row(0, named=True)["return"]
        # (SL - entry fill) / entry fill, the same base ``pnl`` uses.
        entry_fill = df.filter(pl.col("date") == _d(4)).row(0, named=True)["open"]
        assert r4 == pytest.approx((sl - entry_fill) / entry_fill, rel=1e-9)
        assert r4 == pytest.approx(trades.row(0, named=True)["pnl"], rel=1e-9)
        r5 = ret.filter(pl.col("date") == _d(5)).row(0, named=True)["return"]
        assert r5 == pytest.approx(0.0, abs=1e-12)

    def test_limit_never_hit_stays_in_trade(self):
        """When the limit never fires, there's no trade exit."""
        closes = [100.0, 100.5, 101.0, 101.2, 101.4]
        highs = [100.5, 101.0, 101.5, 101.7, 101.9]
        df = _base_ohlcv(closes=closes, highs=highs)

        tp = 110.0  # never hit
        strat = _CrossEntryLimitExit(Limit(ValueGTE(Col("high"), Lit(tp))))
        result = run(df, strat)

        assert result.trades.height == 0  # open trade not logged

    def test_short_side_tp_via_value_lte(self):
        """Short TP: price drops through a lower level.

        The limit fires on the bar the entry filled, so the return is measured
        from the short's entry fill — ``open[d(4)] = 98.4`` — and covering at
        98.0 is a gain of 0.4 on a short.

        The fixture previously closed d(4) at 98.0, putting the entry fill at
        97.9 — *below* the 98.0 "take profit", so covering there was a loss.
        It only read as a gain because the return was measured from
        ``close[d(3)] = 99.5``, crediting the gap down to the entry that the
        position never held. That is the defect this file's sibling
        ``test_limit_exit_pricing.py`` pins; the fixture is corrected here so
        the test still asserts what its name says.
        """
        closes = [100.0, 100.5, 99.5, 98.5, 97.5]
        lows = [99.5, 100.0, 99.0, 97.5, 97.0]
        df = _base_ohlcv(closes=closes, lows=lows)

        tp = 98.0
        strat = _CrossEntryLimitExit(Limit(ValueLTE(Col("low"), Lit(tp))))
        result = run(df, strat, trade_side=TradeSide.SHORT)

        assert result.trades.height == 1
        ret = result.returns
        r4 = ret.filter(pl.col("date") == _d(4)).row(0, named=True)["return"]
        entry_fill = df.filter(pl.col("date") == _d(4)).row(0, named=True)["open"]
        assert entry_fill > tp, "fixture: a short must cover BELOW its entry to profit"
        # Short: (limit - entry fill) / entry fill * -1
        assert r4 == pytest.approx((tp - entry_fill) / entry_fill * -1.0, rel=1e-9)
        assert r4 == pytest.approx(result.trades.row(0, named=True)["pnl"], rel=1e-9)
        assert r4 > 0  # favorable short move


# ---------------------------------------------------------------------------
# B) Explicit-price decoupling (trailing stop)
# ---------------------------------------------------------------------------


class TestLimitExplicitPrice:
    def test_trailing_stop_uses_price_column(self):
        """price= argument lets fill price come from a per-bar column."""
        closes = [100.0, 100.5, 101.0, 101.5, 98.5]
        lows = [99.5, 100.0, 100.5, 101.0, 98.0]
        trails = [99.0, 99.5, 99.8, 99.8, 99.8]
        df = _base_ohlcv(closes=closes, lows=lows).with_columns(
            pl.Series("trail", trails),
        )

        strat = _CrossEntryLimitExit(
            Limit(ValueLTE(Col("low"), Col("trail")), price=Col("trail")),
        )
        result = run(df, strat)

        ret = result.returns
        r5 = ret.filter(pl.col("date") == _d(5)).row(0, named=True)["return"]
        # low[4]=98 <= trail[4]=99.8 → fill at 99.8 on bar 5, prev close = close[3] = 101.5
        assert r5 == pytest.approx((99.8 - 101.5) / 101.5, rel=1e-9)

    def test_auto_extract_fails_on_non_value_condition(self):
        """Limit(Crossunder(...)) without explicit price raises at resolve."""
        lim = Limit(Crossunder("fast", "slow"))
        with pytest.raises(ValueError, match="ValueGT/GTE/LT/LTE"):
            lim.resolve_price()


# ---------------------------------------------------------------------------
# C) Re-entry on next bar
# ---------------------------------------------------------------------------


class TestLimitReentry:
    def test_re_entry_after_limit_exit(self):
        """A crossover-entry after a limit exit produces a second trade."""
        # Bars 1-5: first TP hit. Bars 6-9: fast crosses slow again and stays.
        closes = [100.0, 100.5, 101.0, 103.0, 102.0, 101.5, 102.0, 103.0, 104.0]
        highs = [100.5, 101.0, 101.5, 103.5, 102.5, 102.0, 102.5, 103.5, 104.5]
        n = len(closes)
        df = pl.DataFrame({
            "date": [_d(i + 1) for i in range(n)],
            "open": [c - 0.1 for c in closes],
            "high": highs,
            "low": [c - 0.5 for c in closes],
            "close": closes,
            # fast crosses slow at bars 2 and 6
            "fast": [99.0, 99.5, 100.5, 101.5, 99.8, 99.5, 100.5, 101.5, 102.5],
            "slow": [100.0] * n,
        })

        strat = _CrossEntryLimitExit(Limit(ValueGTE(Col("high"), Lit(103.0))))
        result = run(df, strat)

        # First trade: entry ~bar 3, exit bar 4 at TP.
        # Second trade: entry ~bar 7, exit bar 8 at TP.
        assert result.trades.height == 2
        exits = result.trades["exit_date"].to_list()
        assert _d(4) in exits and _d(8) in exits


# ---------------------------------------------------------------------------
# D) Backward compatibility
# ---------------------------------------------------------------------------


class TestLimitBackCompat:
    def test_non_limit_exit_unchanged(self):
        """Non-limit exit condition produces the pre-feature result byte-for-byte."""
        closes = [100.0, 100.5, 101.0, 101.5, 100.5, 99.5]
        df = _base_ohlcv(closes=closes)

        # Same strategy shape but without the Limit wrapper
        @dataclass(frozen=True, slots=True)
        class _Plain:
            def entry(self) -> Crossover:
                return Crossover("fast", "slow")

            def exit(self) -> Crossunder:
                return Crossunder("fast", "slow")

        result = run(df, _Plain())
        assert isinstance(result, BacktestResult)
        # Sanity — returns and trades are present and have the right shape
        assert set(result.returns.columns) == {"date", "return"}
        assert set(result.trades.columns) == {
            "entry_date", "exit_date", "side", "pnl", "bars_held",
        }

    def test_limit_does_not_pollute_signals_columns(self):
        """_limit_price is dropped from the returned signals DataFrame."""
        closes = [100.0, 100.5, 101.0, 103.5, 102.0]
        highs = [100.5, 101.0, 101.5, 104.0, 102.5]
        df = _base_ohlcv(closes=closes, highs=highs)

        strat = _CrossEntryLimitExit(Limit(ValueGTE(Col("high"), Lit(103.0))))
        result = run(df, strat)
        assert "_limit_price" not in result.signals.columns


# ---------------------------------------------------------------------------
# E) Entry-side Limit is rejected (v1 guardrail)
# ---------------------------------------------------------------------------


class TestLimitEntryRejected:
    def test_limit_on_entry_raises_not_implemented(self):
        """Entry-side Limit is explicitly unsupported in v1 — fail fast."""
        @dataclass(frozen=True, slots=True)
        class _LimitEntry:
            def entry(self) -> Limit:
                return Limit(ValueGTE(Col("high"), Lit(105.0)))

            def exit(self) -> Crossunder:
                return Crossunder("fast", "slow")

        closes = [100.0, 100.5, 101.0, 102.5, 103.0]
        df = _base_ohlcv(closes=closes)

        with pytest.raises(NotImplementedError, match="Entry-side limit"):
            run(df, _LimitEntry())
