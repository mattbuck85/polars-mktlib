from __future__ import annotations

import polars as pl
import pytest

from mktlib.backtest._conditions import (
    All,
    Any_,
    Crossover,
    Crossunder,
    Custom,
    IsFalling,
    IsRising,
    Not,
    PriceIsAbove,
    PriceIsBelow,
)
from mktlib.backtest._types import TradeSide


@pytest.fixture
def df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "a": [1.0, 3.0, 5.0, 4.0, 2.0],
            "b": [2.0, 2.0, 2.0, 5.0, 5.0],
        }
    )


class TestCrossover:
    def test_column_cross(self, df: pl.DataFrame) -> None:
        # a crosses above b at index 1 (a goes from 1<2 to 3>2)
        result = df.select(Crossover("a", "b").resolve())
        flags = result.to_series().to_list()
        # index 0: a(1)<b(2) so False & null = False
        assert flags == [False, True, False, False, False]

    def test_constant_cross(self) -> None:
        df = pl.DataFrame({"x": [1.0, 3.0, 5.0]})
        result = df.select(Crossover("x", 2.5).resolve()).to_series().to_list()
        # index 0: x(1)<2.5 → False; index 1: x(3)>2.5 & x.shift(1)=1<=2.5 → True
        assert result == [False, True, False]


class TestCrossunder:
    def test_column_cross(self, df: pl.DataFrame) -> None:
        # a crosses below b at index 3 (a goes from 5>2 to 4<5)
        result = df.select(Crossunder("a", "b").resolve())
        flags = result.to_series().to_list()
        assert flags == [None, False, False, True, False]


class TestPriceIsAbove:
    def test_column(self, df: pl.DataFrame) -> None:
        result = df.select(PriceIsAbove("a", "b").resolve()).to_series().to_list()
        assert result == [False, True, True, False, False]

    def test_constant(self, df: pl.DataFrame) -> None:
        result = df.select(PriceIsAbove("a", 3.5).resolve()).to_series().to_list()
        assert result == [False, False, True, True, False]


class TestPriceIsBelow:
    def test_column(self, df: pl.DataFrame) -> None:
        result = df.select(PriceIsBelow("a", "b").resolve()).to_series().to_list()
        assert result == [True, False, False, True, True]


class TestIsRising:
    def test_default_period(self, df: pl.DataFrame) -> None:
        result = df.select(IsRising("a").resolve()).to_series().to_list()
        assert result == [None, True, True, False, False]

    def test_period_2(self, df: pl.DataFrame) -> None:
        result = df.select(IsRising("a", period=2).resolve()).to_series().to_list()
        assert result == [None, None, True, True, False]


class TestIsFalling:
    def test_default_period(self, df: pl.DataFrame) -> None:
        result = df.select(IsFalling("a").resolve()).to_series().to_list()
        assert result == [None, False, False, True, True]


class TestCombinators:
    def test_and(self, df: pl.DataFrame) -> None:
        cond = PriceIsAbove("a", "b") & IsRising("a")
        assert isinstance(cond, All)
        result = df.select(cond.resolve()).to_series().to_list()
        # above: [F,T,T,F,F] & rising: [null,T,T,F,F]
        assert result == [False, True, True, False, False]

    def test_or(self, df: pl.DataFrame) -> None:
        cond = PriceIsAbove("a", 4.5) | PriceIsBelow("a", 1.5)
        assert isinstance(cond, Any_)
        result = df.select(cond.resolve()).to_series().to_list()
        # above 4.5: [F,F,T,F,F] | below 1.5: [T,F,F,F,F]
        assert result == [True, False, True, False, False]

    def test_not(self, df: pl.DataFrame) -> None:
        cond = ~PriceIsAbove("a", "b")
        assert isinstance(cond, Not)
        result = df.select(cond.resolve()).to_series().to_list()
        assert result == [True, False, False, True, True]

    def test_chained(self, df: pl.DataFrame) -> None:
        cond = (PriceIsAbove("a", "b") & IsRising("a")) | PriceIsBelow("a", 1.5)
        result = df.select(cond.resolve()).to_series().to_list()
        # (above & rising): [N,T,T,F,F] | below 1.5: [T,F,F,F,F]
        assert result == [True, True, True, False, False]


class TestTradeSide:
    def test_default_is_none(self) -> None:
        assert Crossover("a", "b").trade_side is None
        assert Crossunder("a", "b").trade_side is None
        assert PriceIsAbove("a", "b").trade_side is None
        assert PriceIsBelow("a", "b").trade_side is None
        assert IsRising("a").trade_side is None
        assert IsFalling("a").trade_side is None
        assert Custom(pl.lit(True)).trade_side is None

    def test_set_via_constructor(self) -> None:
        cond = Crossover("a", "b", trade_side=TradeSide.SHORT)
        assert cond.trade_side is TradeSide.SHORT

    def test_combinators_default_none(self) -> None:
        left = Crossover("a", "b")
        right = Crossunder("a", "b")
        assert All(left, right).trade_side is None
        assert Any_(left, right).trade_side is None
        assert Not(left).trade_side is None

    def test_combinators_accept_trade_side(self) -> None:
        left = Crossover("a", "b")
        right = Crossunder("a", "b")
        assert All(left, right, trade_side=TradeSide.LONG).trade_side is TradeSide.LONG


class TestCustom:
    def test_basic_expr(self, df: pl.DataFrame) -> None:
        # Custom(a > b) should match PriceIsAbove("a", "b")
        custom = Custom(pl.col("a") > pl.col("b"))
        expected = PriceIsAbove("a", "b")
        result = df.select(custom.resolve()).to_series().to_list()
        assert result == df.select(expected.resolve()).to_series().to_list()

    def test_trade_side(self) -> None:
        cond = Custom(pl.lit(True), trade_side=TradeSide.SHORT)
        assert cond.trade_side is TradeSide.SHORT

    def test_composable(self, df: pl.DataFrame) -> None:
        cond = Custom(pl.col("a") > pl.col("b")) & IsRising("a")
        assert isinstance(cond, All)
        result = df.select(cond.resolve()).to_series().to_list()
        # above: [F,T,T,F,F] & rising: [null,T,T,F,F]
        assert result == [False, True, True, False, False]


class TestBareExprEngine:
    """Strategy returning bare pl.Expr works via auto-wrapping."""

    def test_bare_expr_runs(self) -> None:
        from mktlib.backtest import run

        df = pl.DataFrame(
            {
                "date": pl.date_range(
                    pl.date(2024, 1, 1), pl.date(2024, 1, 5), eager=True
                ),
                "open": [100.0, 101.0, 102.0, 103.0, 104.0],
                "close": [101.0, 102.0, 103.0, 104.0, 105.0],
                "rsi": [25.0, 35.0, 75.0, 80.0, 20.0],
            }
        )

        class RsiStrategy:
            def entry(self) -> pl.Expr:
                return pl.col("rsi") < 30

            def exit(self) -> pl.Expr:
                return pl.col("rsi") > 70

        result = run(df, RsiStrategy())  # type: ignore[arg-type]
        assert result.signals.height == 5
        assert "_entry" in result.signals.columns
        assert "_exit" in result.signals.columns
