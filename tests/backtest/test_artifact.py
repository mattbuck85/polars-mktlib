"""Tests for strategy artifact hashing."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from mktlib.backtest import (
    Col,
    Crossover,
    Crossunder,
    Custom,
    EntryRef,
    IsFalling,
    Pct,
    PriceIsAbove,
    PriceIsBelow,
    ValueGT,
    ValueGTE,
    ValueLT,
    ValueLTE,
    register_alias,
    strategy_artifact,
)


# ---------------------------------------------------------------------------
# Test strategies
# ---------------------------------------------------------------------------


def _helper_init(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(pl.col("close").rolling_mean(10).alias("sma"))


@dataclass(frozen=True, slots=True)
class SimpleLong:
    threshold: float = 100.0

    def entry(self):
        return Crossover("fast", "slow")

    def exit(self):
        return Crossunder("fast", "slow")


@dataclass(frozen=True, slots=True)
class SimpleLongWithInit:
    threshold: float = 100.0

    def init(self, df: pl.DataFrame) -> pl.DataFrame:
        return _helper_init(df)

    def entry(self):
        return Crossover("fast", "slow")

    def exit(self):
        return Crossunder("fast", "slow")


@dataclass(frozen=True, slots=True)
class TPSLStrategy:
    tp_pct: float = 5.0
    sl_pct: float = 3.0

    def entry(self):
        return Crossover("fast", "slow") & ValueGT("signal_strength", 10)

    def exit(self):
        return (
            Crossunder("fast", "slow")
            | ValueGT("close", Pct(EntryRef("close"), self.tp_pct))
            | ValueLT("close", Pct(EntryRef("close"), -self.sl_pct))
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestArtifactIdentity:
    def test_same_instance_same_hash(self):
        s = SimpleLong()
        assert strategy_artifact(s) == strategy_artifact(s)

    def test_equal_instances_same_hash(self):
        assert strategy_artifact(SimpleLong()) == strategy_artifact(SimpleLong())

    def test_different_param_different_hash(self):
        """Changing a param that affects conditions changes the hash."""
        assert strategy_artifact(TPSLStrategy(tp_pct=5.0)) != strategy_artifact(TPSLStrategy(tp_pct=10.0))

    def test_hash_is_16_hex_chars(self):
        h = strategy_artifact(SimpleLong())
        assert len(h) == 16
        int(h, 16)  # valid hex


class TestArtifactAliases:
    def test_price_is_above_equals_value_gt(self):
        """PriceIsAbove is an alias for ValueGT — same class, same hash."""

        @dataclass(frozen=True, slots=True)
        class UsingOld:
            def entry(self):
                return PriceIsAbove("close", 100)
            def exit(self):
                return PriceIsBelow("close", 50)

        @dataclass(frozen=True, slots=True)
        class UsingNew:
            def entry(self):
                return ValueGT("close", 100)
            def exit(self):
                return ValueLT("close", 50)

        # Different class names → different hash prefix, but condition tree is same
        # Actually the class name IS part of the hash, so let's test the condition hashing directly
        from mktlib.backtest._artifact import _hash_condition
        assert _hash_condition(PriceIsAbove("close", 100)) == _hash_condition(ValueGT("close", 100))
        assert _hash_condition(PriceIsBelow("close", 50)) == _hash_condition(ValueLT("close", 50))

    def test_col_gt_equals_value_gt(self):
        """Col('a') > 100 produces ValueGT(Col('a'), Lit(100.0)) — same hash as ValueGT('a', 100)."""
        from mktlib.backtest._artifact import _hash_condition
        cond_via_operator = Col("a") > 100
        cond_via_class = ValueGT("a", 100)
        assert _hash_condition(cond_via_operator) == _hash_condition(cond_via_class)

    def test_col_lt_equals_value_lt(self):
        from mktlib.backtest._artifact import _hash_condition
        assert _hash_condition(Col("a") < 100) == _hash_condition(ValueLT("a", 100))

    def test_col_ge_equals_value_gte(self):
        from mktlib.backtest._artifact import _hash_condition
        assert _hash_condition(Col("a") >= 100) == _hash_condition(ValueGTE("a", 100))

    def test_col_le_equals_value_lte(self):
        from mktlib.backtest._artifact import _hash_condition
        assert _hash_condition(Col("a") <= 100) == _hash_condition(ValueLTE("a", 100))

    def test_register_alias(self):
        """User-defined alias class hashes with canonical name."""
        from mktlib.backtest._artifact import _hash_condition

        @dataclass(frozen=True, slots=True)
        class MyCustomGT:
            a: str
            b: float
            trade_side: None = None

        register_alias(MyCustomGT, "ValueGT")

        assert _hash_condition(MyCustomGT("close", 100.0)) == _hash_condition(ValueGT("close", 100.0))  # type: ignore[arg-type]


class TestArtifactCustomExpr:
    def test_custom_expr_same_hash(self):
        """Identical pl.Expr built twice → same hash."""
        from mktlib.backtest._artifact import _hash_condition
        c1 = Custom(pl.col("a") > 0)
        c2 = Custom(pl.col("a") > 0)
        assert _hash_condition(c1) == _hash_condition(c2)

    def test_custom_expr_different_hash(self):
        from mktlib.backtest._artifact import _hash_condition
        c1 = Custom(pl.col("a") > 0)
        c2 = Custom(pl.col("b") > 0)
        assert _hash_condition(c1) != _hash_condition(c2)


class TestArtifactInit:
    def test_init_changes_hash(self):
        """Strategy with init() has different hash from without."""
        assert strategy_artifact(SimpleLong()) != strategy_artifact(SimpleLongWithInit())

    def test_init_follows_same_module_refs(self):
        """init() that calls a module-level helper includes the helper source."""
        from mktlib.backtest._artifact import _hash_init
        h = _hash_init(SimpleLongWithInit())
        # Should contain both the init method source and _helper_init source
        assert b"_helper_init" in h
        assert b"rolling_mean" in h


class TestArtifactComposition:
    def test_and_or_order_matters(self):
        """(A & B) | C has different hash from A & (B | C)."""
        from mktlib.backtest._artifact import _hash_condition
        a = Crossover("fast", "slow")
        b = ValueGT("rsi", 30)
        c = IsFalling("vol", 3)

        h1 = _hash_condition((a & b) | c)
        h2 = _hash_condition(a & (b | c))
        assert h1 != h2

    def test_entry_ref_arithmetic_stable(self):
        """EntryRef('close') + EntryRef('atr') * 2.0 → stable hash."""
        from mktlib.backtest._artifact import _hash_condition
        cond1 = ValueGT("close", EntryRef("close") + EntryRef("atr") * 2.0)
        cond2 = ValueGT("close", EntryRef("close") + EntryRef("atr") * 2.0)
        assert _hash_condition(cond1) == _hash_condition(cond2)

    def test_tpsl_strategy_stable(self):
        """Complex TPSL strategy produces stable hash."""
        h1 = strategy_artifact(TPSLStrategy())
        h2 = strategy_artifact(TPSLStrategy())
        assert h1 == h2

    def test_tpsl_param_change(self):
        """Changing tp_pct changes the hash."""
        assert strategy_artifact(TPSLStrategy(tp_pct=5.0)) != strategy_artifact(TPSLStrategy(tp_pct=10.0))


class TestArtifactConditionCoverage:
    """Exercise every condition arm in _hash_condition."""

    def test_crossover_crossunder(self):
        from mktlib.backtest._artifact import _hash_condition
        h1 = _hash_condition(Crossover("a", "b"))
        h2 = _hash_condition(Crossunder("a", "b"))
        assert h1 != h2
        assert b"Crossover" in h1
        assert b"Crossunder" in h2

    def test_value_gte_lte(self):
        from mktlib.backtest._artifact import _hash_condition
        h1 = _hash_condition(ValueGTE("a", 10))
        h2 = _hash_condition(ValueLTE("a", 10))
        assert h1 != h2
        assert b"ValueGTE" in h1
        assert b"ValueLTE" in h2

    def test_is_rising_falling(self):
        from mktlib.backtest._artifact import _hash_condition
        h1 = _hash_condition(IsFalling("vol", 3))
        assert b"IsFalling" in h1

    def test_not_condition(self):
        from mktlib.backtest._artifact import _hash_condition
        h = _hash_condition(~ValueGT("a", 100))
        assert b"Not" in h

    def test_bare_expr_strategy(self):
        """Strategy returning bare pl.Expr gets normalized to Custom."""

        @dataclass(frozen=True, slots=True)
        class BareExprStrategy:
            def entry(self):
                return pl.col("fast") > pl.col("slow")
            def exit(self):
                return pl.col("fast") < pl.col("slow")

        h = strategy_artifact(BareExprStrategy())
        assert len(h) == 16

    def test_pct_col_expr_hash(self):
        """Pct and BinOp nodes in ColExpr tree produce stable hashes."""
        from mktlib.backtest._artifact import _hash_col_expr
        pe = Pct(EntryRef("close"), 5.0)
        h = _hash_col_expr(pe)
        assert b"Pct" in h
        assert b"EntryRef" in h
