from __future__ import annotations

import polars as pl
import pytest

from mktlib.backtest._conditions import (
    All,
    Any_,
    Crossover,
    Crossunder,
    IsFalling,
    IsRising,
    Not,
    PriceIsAbove,
    PriceIsBelow,
)


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
