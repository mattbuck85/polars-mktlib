"""Tests for portfolio-weights input normalization + weighted aggregation.

Covers:
- :func:`mktlib.backtest.to_portfolio_weights_df` schema + invariant validation
  (both dict and DataFrame input forms).
- :attr:`MultiBacktestResult.returns` weighted aggregation correctness,
  proportional-vs-normalized parity, and dynamic-denominator renormalization
  when a symbol is absent on a given date.
- ``run(instrument_weights=...)`` wiring: cross-validation, error paths,
  back-compat when weights are omitted.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import polars as pl
import pytest

from mktlib.backtest import (
    BacktestResult,
    Crossover,
    Crossunder,
    InvalidPortfolioWeights,
    MultiBacktestResult,
    PORTFOLIO_WEIGHTS_COLUMNS,
    run,
    to_portfolio_weights_df,
)

from tests.schemas.backtest import PortfolioWeightsSchema, WeightedReturnsSchema


# ---------------------------------------------------------------------------
# A) to_portfolio_weights_df — input normalization + validation
# ---------------------------------------------------------------------------


class TestToPortfolioWeightsDfFromMapping:
    def test_happy_path(self):
        df = to_portfolio_weights_df({"A": 0.5, "B": 0.5})
        assert df.columns == list(PORTFOLIO_WEIGHTS_COLUMNS)
        assert df.schema == {"instrument": pl.Utf8, "weight": pl.Float64}
        assert df.height == 2
        assert sorted(df["instrument"].to_list()) == ["A", "B"]
        PortfolioWeightsSchema.validate(df)

    def test_proportional_weights_allowed(self):
        """Weights need not sum to 1 — renormalization happens at aggregation."""
        df = to_portfolio_weights_df({"A": 5, "B": 2})
        assert df["weight"].sum() == pytest.approx(7.0)
        PortfolioWeightsSchema.validate(df)

    def test_integer_values_coerced(self):
        df = to_portfolio_weights_df({"A": 1, "B": 3})
        assert df.schema["weight"] == pl.Float64

    def test_many_symbols(self):
        weights = {f"SYM{i}": 1.0 for i in range(100)}
        df = to_portfolio_weights_df(weights)
        assert df.height == 100
        PortfolioWeightsSchema.validate(df)

    def test_empty_mapping_rejected(self):
        with pytest.raises(InvalidPortfolioWeights, match="empty"):
            to_portfolio_weights_df({})

    def test_negative_weight_rejected(self):
        with pytest.raises(InvalidPortfolioWeights, match="negative"):
            to_portfolio_weights_df({"A": 0.5, "B": -0.3})

    def test_nan_weight_rejected(self):
        with pytest.raises(InvalidPortfolioWeights, match="NaN"):
            to_portfolio_weights_df({"A": 0.5, "B": float("nan")})

    def test_none_value_rejected(self):
        # Dict path: polars may coerce None to null.
        with pytest.raises(InvalidPortfolioWeights):
            to_portfolio_weights_df({"A": 0.5, "B": None})  # type: ignore[dict-item]

    def test_zero_sum_rejected(self):
        with pytest.raises(InvalidPortfolioWeights, match="sum"):
            to_portfolio_weights_df({"A": 0.0, "B": 0.0})

    def test_non_numeric_rejected(self):
        with pytest.raises((InvalidPortfolioWeights, TypeError)):
            to_portfolio_weights_df({"A": "0.5"})  # type: ignore[dict-item]

    def test_non_mapping_non_dataframe_rejected(self):
        with pytest.raises(TypeError, match="Mapping.*or pl.DataFrame"):
            to_portfolio_weights_df([("A", 0.5), ("B", 0.5)])  # type: ignore[arg-type]


class TestToPortfolioWeightsDfFromDataFrame:
    def test_happy_path(self):
        src = pl.DataFrame(
            {"instrument": ["A", "B"], "weight": [0.6, 0.4]},
            schema={"instrument": pl.Utf8, "weight": pl.Float64},
        )
        df = to_portfolio_weights_df(src)
        assert df.columns == list(PORTFOLIO_WEIGHTS_COLUMNS)
        assert df.schema == {"instrument": pl.Utf8, "weight": pl.Float64}
        PortfolioWeightsSchema.validate(df)

    def test_weight_coerced_to_float64(self):
        """Int weight columns are coerced so downstream schema is identical."""
        src = pl.DataFrame({"instrument": ["A", "B"], "weight": [3, 7]})
        df = to_portfolio_weights_df(src)
        assert df.schema["weight"] == pl.Float64

    def test_extra_columns_dropped(self):
        """Canonical schema is two columns — extras are trimmed."""
        src = pl.DataFrame(
            {"instrument": ["A"], "weight": [1.0], "note": ["top holding"]},
        )
        df = to_portfolio_weights_df(src)
        assert df.columns == list(PORTFOLIO_WEIGHTS_COLUMNS)

    def test_missing_instrument_col(self):
        src = pl.DataFrame({"ticker": ["A"], "weight": [1.0]})
        with pytest.raises(InvalidPortfolioWeights, match="instrument"):
            to_portfolio_weights_df(src)

    def test_missing_weight_col(self):
        src = pl.DataFrame({"instrument": ["A"], "w": [1.0]})
        with pytest.raises(InvalidPortfolioWeights, match="weight"):
            to_portfolio_weights_df(src)

    def test_non_string_instrument_rejected(self):
        src = pl.DataFrame({"instrument": [1, 2], "weight": [0.5, 0.5]})
        with pytest.raises(InvalidPortfolioWeights, match="Utf8"):
            to_portfolio_weights_df(src)

    def test_non_numeric_weight_rejected(self):
        src = pl.DataFrame({"instrument": ["A"], "weight": ["0.5"]})
        with pytest.raises(InvalidPortfolioWeights, match="numeric"):
            to_portfolio_weights_df(src)

    def test_null_weight_rejected(self):
        src = pl.DataFrame({"instrument": ["A", "B"], "weight": [0.5, None]})
        with pytest.raises(InvalidPortfolioWeights, match="null"):
            to_portfolio_weights_df(src)

    def test_duplicate_instrument_rejected(self):
        src = pl.DataFrame({"instrument": ["A", "A"], "weight": [0.5, 0.5]})
        with pytest.raises(InvalidPortfolioWeights, match="duplicate"):
            to_portfolio_weights_df(src)

    def test_empty_dataframe_rejected(self):
        src = pl.DataFrame(
            {"instrument": [], "weight": []},
            schema={"instrument": pl.Utf8, "weight": pl.Float64},
        )
        with pytest.raises(InvalidPortfolioWeights, match="empty"):
            to_portfolio_weights_df(src)


# ---------------------------------------------------------------------------
# B) Weighted aggregation — MultiBacktestResult.returns
# ---------------------------------------------------------------------------


def _make_result(
    per_symbol_returns: dict[str, list[tuple[date, float]]],
    weights: pl.DataFrame | None,
) -> MultiBacktestResult:
    """Construct a MultiBacktestResult from per-symbol (date, return) tuples."""
    empty_trades = pl.DataFrame(
        schema={
            "entry_date": pl.Date, "exit_date": pl.Date,
            "side": pl.Int8, "pnl": pl.Float64, "bars_held": pl.Int64,
        },
    )
    empty_signals = pl.DataFrame(
        schema={
            "_entry": pl.Boolean, "_exit": pl.Boolean,
            "_position": pl.Int32, "_side": pl.Int8,
        },
    )
    by_inst = {}
    for sym, rows in per_symbol_returns.items():
        returns = pl.DataFrame(
            {"date": [r[0] for r in rows], "return": [r[1] for r in rows]},
            schema={"date": pl.Date, "return": pl.Float64},
        )
        by_inst[sym] = BacktestResult(
            returns=returns, trades=empty_trades, signals=empty_signals,
        )
    return MultiBacktestResult(by_inst, instrument_col="symbol", weights=weights)


class TestWeightedAggregation:
    def test_two_symbol_weighted_returns(self):
        """Known values: 0.7*A + 0.3*B per date."""
        d1 = date(2024, 1, 2)
        d2 = date(2024, 1, 3)
        weights = to_portfolio_weights_df({"A": 0.7, "B": 0.3})
        result = _make_result(
            {"A": [(d1, 0.10), (d2, -0.05)], "B": [(d1, -0.02), (d2, 0.04)]},
            weights=weights,
        )
        df = result.returns
        WeightedReturnsSchema.validate(df)
        by_date = {r["date"]: r["return"] for r in df.iter_rows(named=True)}
        assert by_date[d1] == pytest.approx(0.7 * 0.10 + 0.3 * (-0.02))
        assert by_date[d2] == pytest.approx(0.7 * (-0.05) + 0.3 * 0.04)

    def test_proportional_weights_match_normalized(self):
        """{A:7, B:3} produces identical returns to {A:0.7, B:0.3}."""
        d1 = date(2024, 1, 2)
        rows = {"A": [(d1, 0.10)], "B": [(d1, -0.02)]}
        norm = _make_result(rows, weights=to_portfolio_weights_df({"A": 0.7, "B": 0.3}))
        prop = _make_result(rows, weights=to_portfolio_weights_df({"A": 7, "B": 3}))
        assert norm.returns["return"].to_list() == pytest.approx(
            prop.returns["return"].to_list(),
        )

    def test_dynamic_denominator_when_symbol_missing(self):
        """Missing symbol on a date drops from that date's denominator."""
        d1 = date(2024, 1, 2)
        d2 = date(2024, 1, 3)
        weights = to_portfolio_weights_df({"A": 0.5, "B": 0.5})
        # B is absent on d2 — portfolio return on d2 should equal A's return
        # (renormalized by A's weight alone).
        result = _make_result(
            {"A": [(d1, 0.10), (d2, 0.04)], "B": [(d1, -0.02)]},
            weights=weights,
        )
        by_date = {r["date"]: r["return"] for r in result.returns.iter_rows(named=True)}
        assert by_date[d1] == pytest.approx(0.5 * 0.10 + 0.5 * (-0.02))
        assert by_date[d2] == pytest.approx(0.04)  # A alone, weight renormalized

    def test_equal_proportional_matches_mean(self):
        """{A:1, B:1, C:1} == simple .mean() aggregation."""
        d1 = date(2024, 1, 2)
        weights = to_portfolio_weights_df({"A": 1, "B": 1, "C": 1})
        result = _make_result(
            {"A": [(d1, 0.03)], "B": [(d1, 0.06)], "C": [(d1, 0.09)]},
            weights=weights,
        )
        by_date = {r["date"]: r["return"] for r in result.returns.iter_rows(named=True)}
        assert by_date[d1] == pytest.approx((0.03 + 0.06 + 0.09) / 3)

    def test_weights_none_preserves_concat_behavior(self):
        """Without weights, .returns is the per-symbol concatenation."""
        d1 = date(2024, 1, 2)
        result = _make_result(
            {"A": [(d1, 0.10)], "B": [(d1, -0.02)]}, weights=None,
        )
        df = result.returns
        assert set(df.columns) == {"symbol", "date", "return"}
        assert df.height == 2


# ---------------------------------------------------------------------------
# C) run() wiring — cross-validation + back-compat
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _SimpleStrat:
    def entry(self) -> Crossover:
        return Crossover("fast", "slow")

    def exit(self) -> Crossunder:
        return Crossunder("fast", "slow")


def _multi_ohlcv() -> pl.DataFrame:
    """2-symbol synthetic OHLCV with a crossover on both symbols."""
    dates = pl.date_range(date(2024, 1, 1), date(2024, 1, 8), eager=True)
    base = {
        "date": list(dates),
        "open": [100.0, 100.5, 101.5, 103.5, 105.5, 103.5, 99.5, 97.5],
        "high": [101.0, 101.5, 103.0, 105.0, 106.0, 104.0, 100.0, 98.0],
        "low": [99.0, 100.0, 101.0, 103.0, 104.0, 99.0, 98.5, 97.0],
        "close": [100.5, 101.5, 102.5, 104.5, 103.0, 99.0, 98.0, 97.5],
        "fast": [100.0, 100.5, 101.5, 103.0, 104.0, 102.0, 99.0, 97.5],
        "slow": [100.0, 100.0, 100.5, 101.0, 102.0, 103.0, 102.0, 100.0],
    }
    a = pl.DataFrame({**base, "symbol": ["A"] * 8})
    b = pl.DataFrame({**base, "symbol": ["B"] * 8})
    return pl.concat([a, b])


class TestRunWiring:
    def test_weights_without_instrument_col_defaults_to_instrument(self):
        """If weights are passed but no instrument_col, default to ``instrument``."""
        df = _multi_ohlcv().rename({"symbol": "instrument"})
        result = run(df, _SimpleStrat(), instrument_weights={"A": 0.5, "B": 0.5})
        assert isinstance(result, MultiBacktestResult)
        assert result._instrument_col == "instrument"  # type: ignore[attr-defined]

    def test_weights_without_instrument_col_no_canonical_column_errors(self):
        """Default 'instrument' fails cleanly if data doesn't have that column."""
        df = _multi_ohlcv()  # has 'symbol', not 'instrument'
        with pytest.raises(ValueError, match="instrument"):
            run(df, _SimpleStrat(), instrument_weights={"A": 0.5, "B": 0.5})

    def test_weighted_run_produces_weighted_returns_shape(self):
        result = run(
            _multi_ohlcv(),
            _SimpleStrat(),
            instrument_col="symbol",
            instrument_weights={"A": 0.5, "B": 0.5},
        )
        assert isinstance(result, MultiBacktestResult)
        df = result.returns
        assert set(df.columns) == {"date", "return"}
        WeightedReturnsSchema.validate(df)

    def test_unweighted_multi_run_unchanged(self):
        """Back-compat: no weights → concatenated per-symbol returns."""
        result = run(
            _multi_ohlcv(), _SimpleStrat(), instrument_col="symbol",
        )
        assert isinstance(result, MultiBacktestResult)
        assert set(result.returns.columns) == {"symbol", "date", "return"}

    def test_dataframe_weights_input(self):
        weights_df = pl.DataFrame(
            {"instrument": ["A", "B"], "weight": [0.3, 0.7]},
            schema={"instrument": pl.Utf8, "weight": pl.Float64},
        )
        result = run(
            _multi_ohlcv(),
            _SimpleStrat(),
            instrument_col="symbol",
            instrument_weights=weights_df,
        )
        assert isinstance(result, MultiBacktestResult)
        assert set(result.returns.columns) == {"date", "return"}

    def test_missing_symbol_in_weights_raises(self):
        """Data has B but weights only covers A → ambiguous, raise."""
        with pytest.raises(InvalidPortfolioWeights, match="backtested but not"):
            run(
                _multi_ohlcv(),
                _SimpleStrat(),
                instrument_col="symbol",
                instrument_weights={"A": 1.0},
            )

    def test_extra_symbol_in_weights_warns(self, caplog):
        """Weights lists C which isn't in the data → warn, continue."""
        import logging as _logging
        caplog.set_level(_logging.WARNING, logger="mktlib.backtest._engine")
        result = run(
            _multi_ohlcv(),
            _SimpleStrat(),
            instrument_col="symbol",
            instrument_weights={"A": 0.3, "B": 0.5, "C": 0.2},
        )
        assert isinstance(result, MultiBacktestResult)
        assert "C" in caplog.text
