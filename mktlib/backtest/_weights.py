"""Portfolio weights for multi-instrument backtests.

Canonical schema::

    instrument : pl.Utf8     -- matches run()'s instrument_col at join time
    weight     : pl.Float64  -- non-negative; sum > 0

Input forms
-----------
Callers of :func:`mktlib.backtest.run` may pass ``instrument_weights`` as
either a plain ``Mapping[str, float]`` (e.g. ``{"TQQQ": 0.5, "AAPL": 0.1}``)
or a pre-built ``pl.DataFrame`` with the canonical columns above. Both
paths are normalized through :func:`to_portfolio_weights_df` into the same
validated DataFrame before mktlib uses them internally.

Renormalization
---------------
Weights do not need to sum to 1 — they are divided by their sum at
aggregation time. ``{"A": 5, "B": 2}`` and ``{"A": 0.714, "B": 0.286}``
produce identical portfolio returns.

Dynamic denominator
-------------------
When a symbol is absent on a given date (e.g. no return row because it
was not yet listed, suspended, or missing data), its weight is excluded
from that date's denominator. The portfolio return for that date is then
the weighted mean over only the symbols that reported a return. This
keeps the series continuous across alignment gaps.
"""

from __future__ import annotations

from collections.abc import Mapping

import polars as pl

__all__ = [
    "INSTRUMENT_COLUMN",
    "InvalidPortfolioWeights",
    "PORTFOLIO_WEIGHTS_COLUMNS",
    "WEIGHT_COLUMN",
    "to_portfolio_weights_df",
]


INSTRUMENT_COLUMN = "instrument"
WEIGHT_COLUMN = "weight"
PORTFOLIO_WEIGHTS_COLUMNS: tuple[str, str] = (INSTRUMENT_COLUMN, WEIGHT_COLUMN)


class InvalidPortfolioWeights(ValueError):
    """Portfolio weights input failed schema or invariant validation."""


def to_portfolio_weights_df(
    weights: Mapping[str, float] | pl.DataFrame,
) -> pl.DataFrame:
    """Normalize a dict or DataFrame into the canonical portfolio-weights schema.

    Accepts either a ``Mapping[str, float]`` or a ``pl.DataFrame`` with
    columns ``(instrument, weight)``. Returns a DataFrame with those exact
    columns and dtypes (``Utf8``, ``Float64``), validated against:

    - non-empty
    - no nulls in either column
    - no NaN in ``weight``
    - all weights ``>= 0``
    - no duplicate ``instrument`` values
    - ``sum(weight) > 0``

    Raises
    ------
    InvalidPortfolioWeights
        On any schema or invariant violation.
    TypeError
        When *weights* is neither a ``Mapping`` nor a ``pl.DataFrame``.
    """
    if isinstance(weights, pl.DataFrame):
        df = _normalize_dataframe_input(weights)
    elif isinstance(weights, Mapping):
        df = _normalize_mapping_input(weights)
    else:  # type: ignore[unreachable]
        # Runtime guard — type system narrows this away, but dynamic
        # callers may pass e.g. a list or str.
        msg = (
            "instrument_weights must be a Mapping[str, float] or pl.DataFrame, "
            f"got {type(weights).__name__}"
        )
        raise TypeError(msg)

    _validate_invariants(df)
    return df


def _normalize_mapping_input(weights: Mapping[str, float]) -> pl.DataFrame:
    if not weights:
        msg = "instrument_weights mapping is empty"
        raise InvalidPortfolioWeights(msg)
    try:
        return pl.DataFrame(
            {
                INSTRUMENT_COLUMN: list(weights.keys()),
                WEIGHT_COLUMN: list(weights.values()),
            },
            schema={INSTRUMENT_COLUMN: pl.Utf8, WEIGHT_COLUMN: pl.Float64},
        )
    except (TypeError, ValueError) as e:
        msg = f"instrument_weights could not be coerced to (Utf8, Float64): {e}"
        raise InvalidPortfolioWeights(msg) from e


def _normalize_dataframe_input(df: pl.DataFrame) -> pl.DataFrame:
    for col in PORTFOLIO_WEIGHTS_COLUMNS:
        if col not in df.columns:
            msg = (
                f"instrument_weights DataFrame missing column {col!r}. "
                f"Required columns: {list(PORTFOLIO_WEIGHTS_COLUMNS)}"
            )
            raise InvalidPortfolioWeights(msg)

    inst_dtype = df.schema[INSTRUMENT_COLUMN]
    if inst_dtype != pl.Utf8:
        msg = f"{INSTRUMENT_COLUMN!r} column must be Utf8, got {inst_dtype}"
        raise InvalidPortfolioWeights(msg)

    weight_dtype = df.schema[WEIGHT_COLUMN]
    if not weight_dtype.is_numeric():
        msg = f"{WEIGHT_COLUMN!r} column must be numeric, got {weight_dtype}"
        raise InvalidPortfolioWeights(msg)

    # Select only the canonical columns and coerce weight to Float64 so the
    # downstream schema is identical regardless of input source.
    return df.select(
        pl.col(INSTRUMENT_COLUMN),
        pl.col(WEIGHT_COLUMN).cast(pl.Float64),
    )


def _validate_invariants(df: pl.DataFrame) -> None:
    if df.height == 0:
        msg = "instrument_weights is empty"
        raise InvalidPortfolioWeights(msg)

    if df[INSTRUMENT_COLUMN].null_count() > 0:
        msg = f"{INSTRUMENT_COLUMN!r} column has null values"
        raise InvalidPortfolioWeights(msg)

    w = df[WEIGHT_COLUMN]
    if w.null_count() > 0:
        msg = f"{WEIGHT_COLUMN!r} column has null values"
        raise InvalidPortfolioWeights(msg)
    if w.is_nan().any():
        msg = f"{WEIGHT_COLUMN!r} column has NaN values"
        raise InvalidPortfolioWeights(msg)

    if (w < 0).any():
        negatives = df.filter(pl.col(WEIGHT_COLUMN) < 0).to_dicts()
        msg = f"negative weight(s) found: {negatives}"
        raise InvalidPortfolioWeights(msg)

    if df[INSTRUMENT_COLUMN].n_unique() != df.height:
        msg = "duplicate instrument(s) in weights"
        raise InvalidPortfolioWeights(msg)

    total = w.sum()
    if total is None or total <= 0:
        msg = f"{WEIGHT_COLUMN!r} sum must be > 0 (got {total})"
        raise InvalidPortfolioWeights(msg)
