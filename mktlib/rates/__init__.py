"""Treasury yield curve rates for risk-free rate estimation."""
from __future__ import annotations

from datetime import date
from enum import StrEnum

__all__ = ["TreasuryRate", "get_risk_free_rate"]


class TreasuryRate(StrEnum):
    """Treasury yield curve instruments available from Treasury.gov."""

    THREE_MONTH = "BC_3MONTH"
    SIX_MONTH = "BC_6MONTH"
    ONE_YEAR = "BC_1YEAR"
    TWO_YEAR = "BC_2YEAR"
    FIVE_YEAR = "BC_5YEAR"
    TEN_YEAR = "BC_10YEAR"
    THIRTY_YEAR = "BC_30YEAR"


def get_risk_free_rate(
    start: date | str,
    end: date | str,
    instrument: TreasuryRate = TreasuryRate.THREE_MONTH,
) -> float:
    """Fetch the average annualised risk-free rate for a date range.

    Returns the arithmetic mean of daily Treasury yields as a decimal
    (e.g., 0.0436 for 4.36%).

    Parameters
    ----------
    start, end
        Date range (inclusive). Accepts ``date`` objects or ISO strings
        (``"2024-01-01"``).
    instrument
        Which Treasury yield to use. Defaults to the 3-month T-bill,
        the standard academic proxy for the risk-free rate.
    """
    from . import _treasury

    if isinstance(start, str):
        start = date.fromisoformat(start)
    if isinstance(end, str):
        end = date.fromisoformat(end)

    return _treasury.fetch_average_rate(start, end, instrument.value)
