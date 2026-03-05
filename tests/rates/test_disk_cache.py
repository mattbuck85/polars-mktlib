"""Tests for mktlib.rates._disk_cache — persistent disk cache."""

from __future__ import annotations

import os
import time
from datetime import date
from unittest.mock import patch

import pytest

from mktlib.rates import _disk_cache

_SAMPLE_ROWS: list[dict[str, date | float]] = [
    {"date": date(2024, 1, 2), "BC_3MONTH": 0.054, "BC_10YEAR": 0.0388},
    {"date": date(2024, 1, 3), "BC_3MONTH": 0.0538, "BC_10YEAR": 0.0392},
]


@pytest.fixture(autouse=True)
def _use_tmp_cache(tmp_path):
    """Redirect disk cache to a temp directory for every test."""
    with patch.object(_disk_cache, "_CACHE_DIR", tmp_path):
        yield


class TestSaveAndLoad:
    def test_roundtrip(self, tmp_path):
        _disk_cache.save_year(2024, _SAMPLE_ROWS)
        result = _disk_cache.load_year(2024)

        assert result is not None
        assert len(result) == 2
        assert result[0]["date"] == date(2024, 1, 2)
        assert result[0]["BC_3MONTH"] == pytest.approx(0.054)
        assert result[0]["BC_10YEAR"] == pytest.approx(0.0388)
        assert result[1]["date"] == date(2024, 1, 3)
        assert result[1]["BC_3MONTH"] == pytest.approx(0.0538)

    def test_missing_file_returns_none(self):
        assert _disk_cache.load_year(1999) is None


class TestStaleness:
    def test_past_year_never_stale(self, tmp_path):
        """Past-year cache is valid regardless of mtime."""
        _disk_cache.save_year(2023, _SAMPLE_ROWS)
        # Set mtime to 30 days ago
        path = tmp_path / "2023.csv"
        old_time = time.time() - 30 * 86400
        os.utime(path, (old_time, old_time))

        result = _disk_cache.load_year(2023)
        assert result is not None
        assert len(result) == 2

    def test_current_year_stale_after_7_days(self):
        """Current-year cache older than 7 days returns None."""
        year = date.today().year
        _disk_cache.save_year(year, _SAMPLE_ROWS)
        # Set mtime to 8 days ago
        path = _disk_cache._CACHE_DIR / f"{year}.csv"
        old_time = time.time() - 8 * 86400
        os.utime(path, (old_time, old_time))

        assert _disk_cache.load_year(year) is None

    def test_current_year_fresh_within_7_days(self):
        """Current-year cache within 7 days is valid."""
        year = date.today().year
        _disk_cache.save_year(year, _SAMPLE_ROWS)

        result = _disk_cache.load_year(year)
        assert result is not None
        assert len(result) == 2


class TestIgnoreStale:
    def test_load_ignore_stale(self):
        """ignore_stale=True returns data even when mtime is old."""
        year = date.today().year
        _disk_cache.save_year(year, _SAMPLE_ROWS)
        # Set mtime to 30 days ago — well past the 7-day threshold
        path = _disk_cache._CACHE_DIR / f"{year}.csv"
        old_time = time.time() - 30 * 86400
        os.utime(path, (old_time, old_time))

        # Normal load returns None (stale)
        assert _disk_cache.load_year(year) is None
        # ignore_stale bypasses staleness check
        result = _disk_cache.load_year(year, ignore_stale=True)
        assert result is not None
        assert len(result) == 2

    def test_ignore_stale_still_none_when_missing(self):
        """ignore_stale=True still returns None when file doesn't exist."""
        assert _disk_cache.load_year(1999, ignore_stale=True) is None


class TestClear:
    def test_clear_deletes_files(self, tmp_path):
        _disk_cache.save_year(2023, _SAMPLE_ROWS)
        _disk_cache.save_year(2024, _SAMPLE_ROWS)
        assert len(list(tmp_path.glob("*.csv"))) == 2

        _disk_cache.clear()
        assert len(list(tmp_path.glob("*.csv"))) == 0

    def test_clear_on_empty_dir(self, tmp_path):
        """clear() on nonexistent/empty dir doesn't raise."""
        _disk_cache.clear()  # no files to delete — should not raise
