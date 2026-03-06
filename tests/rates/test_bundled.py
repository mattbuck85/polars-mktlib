"""Tests for mktlib.rates._bundled — CSV loader for bundled data."""

from __future__ import annotations

from datetime import date

import pytest

from mktlib.rates._bundled import YEAR_RANGE, bundled_path, load_year


class TestLoadYear:
    def test_loads_real_bundled_2024(self):
        rows = load_year(2024)
        assert len(rows) > 200
        # Check structure
        assert isinstance(rows[0]["date"], date)
        assert "BC_3MONTH" in rows[0]
        assert isinstance(rows[0]["BC_3MONTH"], float)

    def test_percentage_to_decimal(self):
        """Values in CSV are percentages; load_year divides by 100."""
        rows = load_year(2024)
        # Treasury 3-month rates are typically 0.01–0.10 as decimals
        assert 0.0 < rows[0]["BC_3MONTH"] < 0.20

    def test_missing_year_returns_empty(self):
        assert load_year(1900) == []

    def test_empty_cells_skipped(self):
        """Rows with empty rate cells should not include those keys."""
        rows = load_year(2024)
        for row in rows:
            for key, val in row.items():
                if key != "date":
                    assert isinstance(val, float)

    def test_dates_are_sorted(self):
        rows = load_year(2024)
        dates = [r["date"] for r in rows]
        assert dates == sorted(dates)

    def test_multiple_instruments_present(self):
        rows = load_year(2024)
        all_keys: set[str] = set()
        for row in rows:
            all_keys.update(k for k in row if k != "date")
        assert "BC_3MONTH" in all_keys
        assert "BC_10YEAR" in all_keys
        assert "BC_30YEAR" in all_keys


class TestBundledPath:
    def test_returns_path_for_known_year(self):
        p = bundled_path(2024)
        assert p is not None
        assert p.exists()
        assert p.name == "2024.csv"

    def test_returns_none_for_missing_year(self):
        assert bundled_path(1900) is None

    def test_year_range_covers_bundled_files(self):
        """Every year in YEAR_RANGE has a bundled CSV."""
        for year in YEAR_RANGE:
            p = bundled_path(year)
            assert p is not None, f"Missing bundled CSV for {year}"
