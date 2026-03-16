"""pandera schemas for mktlib.rates output DataFrames."""

from __future__ import annotations

import pandera.polars as pa
import polars as pl


class SingleRateSchema(pa.DataFrameModel):
    """get_treasury_rates(instrument=single) output: (date, rate)."""

    date: pl.Date
    rate: float = pa.Field(ge=0)


class SpreadSchema(pa.DataFrameModel):
    """get_treasury_spread() output: (date, spread)."""

    date: pl.Date
    spread: float


def validate_multi_rate_df(
    df: pl.DataFrame,
    expected_cols: list[str] | None = None,
) -> None:
    """Validate a wide-format multi-instrument rates DataFrame.

    Dynamic column names (one per TreasuryRate member) prevent use of a
    static DataFrameModel.  This helper checks structural invariants:
    first column is ``date``/Date, remaining columns are Float64.
    """
    assert df.columns[0] == "date"
    assert df["date"].dtype == pl.Date
    for col in df.columns[1:]:
        assert df[col].dtype == pl.Float64, f"{col}: expected Float64, got {df[col].dtype}"
    if expected_cols is not None:
        assert df.columns[1:] == expected_cols
