"""Tests for mktlib.rates._bundled — CSV loader for bundled data."""
from __future__ import annotations

from datetime import date

import pytest

from mktlib.rates._bundled import load_year


class TestLoadYear:
    def test_loads_real_bundled_2024(self):
        rows = load_year(2024)
        assert len(rows) > 200
        # Check structure
        row_date, rates = rows[0]
        assert isinstance(row_date, date)
        assert "BC_3MONTH" in rates
        assert isinstance(rates["BC_3MONTH"], float)

    def test_percentage_to_decimal(self):
        """Values in CSV are percentages; load_year divides by 100."""
        rows = load_year(2024)
        _, rates = rows[0]
        # Treasury 3-month rates are typically 0.01–0.10 as decimals
        assert 0.0 < rates["BC_3MONTH"] < 0.20

    def test_missing_year_returns_empty(self):
        assert load_year(1900) == []

    def test_empty_cells_skipped(self):
        """Rows with empty rate cells should not include those keys."""
        rows = load_year(2024)
        for _, rates in rows:
            for val in rates.values():
                assert isinstance(val, float)

    def test_dates_are_sorted(self):
        rows = load_year(2024)
        dates = [r[0] for r in rows]
        assert dates == sorted(dates)

    def test_multiple_instruments_present(self):
        rows = load_year(2024)
        all_keys: set[str] = set()
        for _, rates in rows:
            all_keys.update(rates.keys())
        assert "BC_3MONTH" in all_keys
        assert "BC_10YEAR" in all_keys
        assert "BC_30YEAR" in all_keys
