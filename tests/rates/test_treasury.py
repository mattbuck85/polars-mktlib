"""Tests for mktlib.rates — Treasury yield curve fetcher."""

from __future__ import annotations

import io
import math
import warnings
import xml.etree.ElementTree as ET
from datetime import date
from unittest.mock import patch

import pytest

from mktlib.rates import (
    MeanMethod,
    TreasuryRate,
    get_mean_treasury_rate,
    get_risk_free_rate,
    get_treasury_rates,
    get_treasury_spread,
    get_treasury_spread_matrix,
)
from mktlib.rates._treasury import (
    clear_cache,
    fetch_average_rate,
    fetch_mean_rate,
    fetch_year,
)

# Save real disk cache functions before any fixture patches them
from mktlib.rates import _disk_cache as _dc_mod

_orig_dc_load = _dc_mod.load_year
_orig_dc_save = _dc_mod.save_year

# ---------------------------------------------------------------------------
# XML fixture — mimics Treasury.gov Atom/OData feed
# ---------------------------------------------------------------------------

_XML_2024 = """\
<?xml version="1.0" encoding="utf-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:m="http://schemas.microsoft.com/ado/2007/08/dataservices/metadata"
      xmlns:d="http://schemas.microsoft.com/ado/2007/08/dataservices">
  <entry>
    <content type="application/xml">
      <m:properties>
        <d:NEW_DATE>2024-01-02T00:00:00</d:NEW_DATE>
        <d:BC_3MONTH>5.40</d:BC_3MONTH>
        <d:BC_6MONTH>5.26</d:BC_6MONTH>
        <d:BC_2YEAR>4.32</d:BC_2YEAR>
        <d:BC_10YEAR>3.88</d:BC_10YEAR>
      </m:properties>
    </content>
  </entry>
  <entry>
    <content type="application/xml">
      <m:properties>
        <d:NEW_DATE>2024-01-03T00:00:00</d:NEW_DATE>
        <d:BC_3MONTH>5.38</d:BC_3MONTH>
        <d:BC_6MONTH>5.24</d:BC_6MONTH>
        <d:BC_2YEAR>4.34</d:BC_2YEAR>
        <d:BC_10YEAR>3.92</d:BC_10YEAR>
      </m:properties>
    </content>
  </entry>
  <entry>
    <content type="application/xml">
      <m:properties>
        <d:NEW_DATE>2024-01-04T00:00:00</d:NEW_DATE>
        <d:BC_3MONTH>5.36</d:BC_3MONTH>
        <d:BC_2YEAR>4.36</d:BC_2YEAR>
        <d:BC_10YEAR>3.95</d:BC_10YEAR>
      </m:properties>
    </content>
  </entry>
</feed>
"""

_XML_2025 = """\
<?xml version="1.0" encoding="utf-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:m="http://schemas.microsoft.com/ado/2007/08/dataservices/metadata"
      xmlns:d="http://schemas.microsoft.com/ado/2007/08/dataservices">
  <entry>
    <content type="application/xml">
      <m:properties>
        <d:NEW_DATE>2025-01-02T00:00:00</d:NEW_DATE>
        <d:BC_3MONTH>4.32</d:BC_3MONTH>
        <d:BC_10YEAR>4.55</d:BC_10YEAR>
      </m:properties>
    </content>
  </entry>
</feed>
"""

# XML with a missing BC_3MONTH field on one entry
_XML_MISSING = """\
<?xml version="1.0" encoding="utf-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:m="http://schemas.microsoft.com/ado/2007/08/dataservices/metadata"
      xmlns:d="http://schemas.microsoft.com/ado/2007/08/dataservices">
  <entry>
    <content type="application/xml">
      <m:properties>
        <d:NEW_DATE>2024-06-03T00:00:00</d:NEW_DATE>
        <d:BC_3MONTH>5.50</d:BC_3MONTH>
      </m:properties>
    </content>
  </entry>
  <entry>
    <content type="application/xml">
      <m:properties>
        <d:NEW_DATE>2024-06-04T00:00:00</d:NEW_DATE>
        <d:BC_10YEAR>4.30</d:BC_10YEAR>
      </m:properties>
    </content>
  </entry>
</feed>
"""


def _mock_urlopen(xml_by_year: dict[int, str]):
    """Return a context-manager mock for urlopen keyed by year in the URL."""

    def _urlopen(url, *, timeout=30):
        for year, xml in xml_by_year.items():
            if str(year) in url:
                cm = io.BytesIO(xml.encode())
                cm.status = 200  # type: ignore[attr-defined]
                cm.__enter__ = lambda s: s  # type: ignore[attr-defined]
                cm.__exit__ = lambda s, *a: None  # type: ignore[attr-defined]
                return cm
        raise OSError(f"No fixture for URL: {url}")

    return _urlopen


@pytest.fixture(autouse=True)
def _clear_treasury_cache(tmp_path):
    """Ensure each test starts with a clean cache; isolate disk cache.

    Redirects _CACHE_DIR to a temp directory so ``load_years`` can write
    and scan real CSVs, while the dict-based ``load_year`` is still
    mocked to None (preventing ``fetch_year`` from short-circuiting).
    """
    clear_cache()
    with (
        patch.object(_dc_mod, "_CACHE_DIR", tmp_path),
        patch("mktlib.rates._disk_cache.load_year", return_value=None),
        patch("mktlib.rates._disk_cache.save_year", wraps=_orig_dc_save),
        patch("mktlib.rates._treasury._bundled.load_year", return_value=[]),
        patch(
            "mktlib.rates._treasury._bundled.bundled_path", return_value=None
        ),
    ):
        yield
    clear_cache()


# ---------------------------------------------------------------------------
# _treasury.py unit tests
# ---------------------------------------------------------------------------


class TestFetchAverageRate:
    def test_average(self):
        with patch(
            "mktlib.rates._treasury.urlopen", _mock_urlopen({2024: _XML_2024})
        ):
            avg = fetch_average_rate(date(2024, 1, 1), date(2024, 1, 31))

        expected = (0.054 + 0.0538 + 0.0536) / 3
        assert avg == pytest.approx(expected)

    def test_empty_range_falls_back_to_last_available(self):
        """No data in range but year has earlier data → last available rate."""
        with patch(
            "mktlib.rates._treasury.urlopen", _mock_urlopen({2024: _XML_2024})
        ):
            # July has no data, but Jan 4 has BC_3MONTH=0.0536
            avg = fetch_average_rate(date(2024, 7, 1), date(2024, 7, 31))

        assert avg == pytest.approx(0.0536)

    def test_recent_range_no_data_uses_last_available(self):
        """Date range has no trading days but year has earlier data → last available rate."""
        with patch(
            "mktlib.rates._treasury.urlopen", _mock_urlopen({2024: _XML_2024})
        ):
            # Jan 5-6 is a weekend — no data, but Jan 4 has BC_3MONTH=0.0536
            avg = fetch_average_rate(date(2024, 1, 5), date(2024, 1, 6))

        assert avg == pytest.approx(0.0536)

    def test_recent_range_no_data_in_year_returns_zero(self):
        """Year has no data at all → still returns 0.0."""
        # Empty XML feed for 2024
        empty_xml = (
            '<?xml version="1.0" encoding="utf-8"?>'
            '<feed xmlns="http://www.w3.org/2005/Atom"'
            ' xmlns:m="http://schemas.microsoft.com/ado/2007/08/dataservices/metadata"'
            ' xmlns:d="http://schemas.microsoft.com/ado/2007/08/dataservices">'
            "</feed>"
        )
        with patch(
            "mktlib.rates._treasury.urlopen",
            _mock_urlopen({2023: empty_xml, 2024: empty_xml}),
        ):
            avg = fetch_average_rate(date(2024, 1, 5), date(2024, 1, 6))

        assert avg == 0.0

    def test_partial_range_uses_exact_matches_first(self):
        """Some days in range have data → returns their mean (no fallback)."""
        with patch(
            "mktlib.rates._treasury.urlopen", _mock_urlopen({2024: _XML_2024})
        ):
            # Jan 2 and Jan 3 have data; fallback should NOT be used
            avg = fetch_average_rate(date(2024, 1, 2), date(2024, 1, 3))

        expected = (0.054 + 0.0538) / 2
        assert avg == pytest.approx(expected)

    def test_january_rollover_uses_previous_year(self):
        """Early Jan with no current-year data falls back to last rate from previous Dec."""
        # 2025 has no data; 2024 has data through Jan 4
        empty_2025 = (
            '<?xml version="1.0" encoding="utf-8"?>'
            '<feed xmlns="http://www.w3.org/2005/Atom"'
            ' xmlns:m="http://schemas.microsoft.com/ado/2007/08/dataservices/metadata"'
            ' xmlns:d="http://schemas.microsoft.com/ado/2007/08/dataservices">'
            "</feed>"
        )
        with patch(
            "mktlib.rates._treasury.urlopen",
            _mock_urlopen({2024: _XML_2024, 2025: empty_2025}),
        ):
            avg = fetch_average_rate(date(2025, 1, 1), date(2025, 1, 3))

        # Should fall back to last 2024 rate: Jan 4 BC_3MONTH = 0.0536
        assert avg == pytest.approx(0.0536)

    def test_network_error_no_bundled(self):
        """Network failure with no bundled data re-raises original exception."""

        def _fail(url, *, timeout=30):
            raise OSError("network down")

        with (
            patch("mktlib.rates._treasury.urlopen", _fail),
            patch(
                "mktlib.rates._treasury._bundled.load_year", return_value=[]
            ),
        ):
            with pytest.raises(OSError, match="Failed to fetch"):
                fetch_average_rate(date(2024, 1, 1), date(2024, 1, 31))


# ---------------------------------------------------------------------------
# Public API tests
# ---------------------------------------------------------------------------


class TestGetRiskFreeRate:
    def test_with_date_objects(self):
        with patch(
            "mktlib.rates._treasury.urlopen", _mock_urlopen({2024: _XML_2024})
        ):
            rate = get_risk_free_rate(date(2024, 1, 1), date(2024, 1, 31))

        assert rate == pytest.approx((0.054 + 0.0538 + 0.0536) / 3)

    def test_with_string_dates(self):
        with patch(
            "mktlib.rates._treasury.urlopen", _mock_urlopen({2024: _XML_2024})
        ):
            rate = get_risk_free_rate("2024-01-01", "2024-01-31")

        assert rate == pytest.approx((0.054 + 0.0538 + 0.0536) / 3)

    def test_with_instrument_enum(self):
        with patch(
            "mktlib.rates._treasury.urlopen", _mock_urlopen({2024: _XML_2024})
        ):
            rate = get_risk_free_rate(
                "2024-01-02", "2024-01-04", instrument=TreasuryRate.TEN_YEAR
            )

        expected = (0.0388 + 0.0392 + 0.0395) / 3
        assert rate == pytest.approx(expected)


# ---------------------------------------------------------------------------
# fetch_mean_rate tests
# ---------------------------------------------------------------------------


class TestFetchMeanRate:
    def test_arithmetic_matches_fetch_average_rate(self):
        with patch(
            "mktlib.rates._treasury.urlopen", _mock_urlopen({2024: _XML_2024})
        ):
            arith = fetch_mean_rate(
                date(2024, 1, 1), date(2024, 1, 31), method="arithmetic"
            )
            avg = fetch_average_rate(date(2024, 1, 1), date(2024, 1, 31))

        assert arith == pytest.approx(avg)

    def test_geometric_hand_calculated(self):
        """Geometric mean of 5.40%, 5.38%, 5.36% (as decimals)."""
        import math

        r1, r2, r3 = 0.054, 0.0538, 0.0536
        expected = (
            math.exp(
                (math.log(1 + r1) + math.log(1 + r2) + math.log(1 + r3)) / 3
            )
            - 1
        )

        with patch(
            "mktlib.rates._treasury.urlopen", _mock_urlopen({2024: _XML_2024})
        ):
            geo = fetch_mean_rate(
                date(2024, 1, 1), date(2024, 1, 31), method="geometric"
            )

        assert geo == pytest.approx(expected)

    def test_geometric_less_than_arithmetic(self):
        """AM-GM inequality: geometric mean < arithmetic mean for non-uniform rates."""
        with patch(
            "mktlib.rates._treasury.urlopen", _mock_urlopen({2024: _XML_2024})
        ):
            arith = fetch_mean_rate(
                date(2024, 1, 1), date(2024, 1, 31), method="arithmetic"
            )
            geo = fetch_mean_rate(
                date(2024, 1, 1), date(2024, 1, 31), method="geometric"
            )

        assert geo < arith

    def test_empty_range_arithmetic(self):
        """No data in range → falls back to last available rate."""
        with patch(
            "mktlib.rates._treasury.urlopen", _mock_urlopen({2024: _XML_2024})
        ):
            assert fetch_mean_rate(
                date(2024, 7, 1), date(2024, 7, 31), method="arithmetic"
            ) == pytest.approx(0.0536)

    def test_empty_range_geometric(self):
        """No data in range → falls back to last available rate."""
        with patch(
            "mktlib.rates._treasury.urlopen", _mock_urlopen({2024: _XML_2024})
        ):
            assert fetch_mean_rate(
                date(2024, 7, 1), date(2024, 7, 31), method="geometric"
            ) == pytest.approx(0.0536)


# ---------------------------------------------------------------------------
# get_mean_treasury_rate public API tests
# ---------------------------------------------------------------------------


class TestGetMeanTreasuryRate:
    def test_arithmetic_enum(self):
        with patch(
            "mktlib.rates._treasury.urlopen", _mock_urlopen({2024: _XML_2024})
        ):
            rate = get_mean_treasury_rate(
                date(2024, 1, 1),
                date(2024, 1, 31),
                method=MeanMethod.ARITHMETIC,
            )

        expected = (0.054 + 0.0538 + 0.0536) / 3
        assert rate == pytest.approx(expected)

    def test_geometric_enum(self):
        import math

        r1, r2, r3 = 0.054, 0.0538, 0.0536
        expected = (
            math.exp(
                (math.log(1 + r1) + math.log(1 + r2) + math.log(1 + r3)) / 3
            )
            - 1
        )

        with patch(
            "mktlib.rates._treasury.urlopen", _mock_urlopen({2024: _XML_2024})
        ):
            rate = get_mean_treasury_rate(
                date(2024, 1, 1),
                date(2024, 1, 31),
                method=MeanMethod.GEOMETRIC,
            )

        assert rate == pytest.approx(expected)

    def test_string_dates(self):
        with patch(
            "mktlib.rates._treasury.urlopen", _mock_urlopen({2024: _XML_2024})
        ):
            rate = get_mean_treasury_rate("2024-01-01", "2024-01-31")

        expected = (0.054 + 0.0538 + 0.0536) / 3
        assert rate == pytest.approx(expected)

    def test_instrument_enum(self):
        with patch(
            "mktlib.rates._treasury.urlopen", _mock_urlopen({2024: _XML_2024})
        ):
            rate = get_mean_treasury_rate(
                "2024-01-02",
                "2024-01-04",
                instrument=TreasuryRate.TEN_YEAR,
            )

        expected = (0.0388 + 0.0392 + 0.0395) / 3
        assert rate == pytest.approx(expected)


# ---------------------------------------------------------------------------
# reports integration
# ---------------------------------------------------------------------------


class TestReportsRfAuto:
    def test_metrics_rf_auto(self):
        import polars as pl

        from mktlib.reports import metrics

        dates = pl.date_range(date(2024, 1, 2), date(2024, 1, 4), eager=True)
        ret_df = pl.DataFrame(
            {"date": dates, "return": [0.001, 0.002, -0.001]}
        )

        with patch(
            "mktlib.rates._treasury.urlopen", _mock_urlopen({2024: _XML_2024})
        ):
            m = metrics(ret_df, rf="auto")

        # rf should have been resolved to a float, verify Sharpe uses it
        assert isinstance(m.sharpe, float)

    def test_metrics_rf_explicit_unchanged(self):
        """Passing a float still works as before."""
        import polars as pl

        from mktlib.reports import metrics

        dates = pl.date_range(date(2024, 1, 2), date(2024, 1, 4), eager=True)
        ret_df = pl.DataFrame(
            {"date": dates, "return": [0.001, 0.002, -0.001]}
        )

        m = metrics(ret_df, rf=0.05)
        assert isinstance(m.sharpe, float)


# ---------------------------------------------------------------------------
# Bundled fallback tests
# ---------------------------------------------------------------------------

_BUNDLED_2024 = [
    {"date": date(2024, 1, 2), "BC_3MONTH": 0.054, "BC_10YEAR": 0.0388},
    {"date": date(2024, 1, 3), "BC_3MONTH": 0.0538, "BC_10YEAR": 0.0392},
]

# Current-year fixtures — seed block is skipped for current year, so these
# let us test the network-error fallback path in isolation.
_THIS_YEAR = date.today().year
_XML_THIS_YEAR = _XML_2024.replace("2024", str(_THIS_YEAR))
_BUNDLED_THIS_YEAR = [
    {"date": date(_THIS_YEAR, 1, 2), "BC_3MONTH": 0.054, "BC_10YEAR": 0.0388},
    {"date": date(_THIS_YEAR, 1, 3), "BC_3MONTH": 0.0538, "BC_10YEAR": 0.0392},
]


class TestBundledFallback:
    def test_fallback_on_network_error(self):
        """Network error with bundled data available: warn and return data."""

        def _fail(url, *, timeout=30):
            raise OSError("network down")

        with (
            patch("mktlib.rates._treasury.urlopen", _fail),
            patch(
                "mktlib.rates._treasury._bundled.load_year",
                return_value=_BUNDLED_THIS_YEAR,
            ),
            warnings.catch_warnings(record=True) as w,
        ):
            warnings.simplefilter("always")
            rates = fetch_year(_THIS_YEAR)

        assert len(rates) == 2
        assert rates[0]["date"] == date(_THIS_YEAR, 1, 2)
        assert rates[0]["BC_3MONTH"] == pytest.approx(0.054)
        assert len(w) == 1
        assert "bundled" in str(w[0].message).lower()

    def test_fallback_data_is_cached(self):
        """After fallback, subsequent calls use the cache (no re-fetch)."""

        def _fail(url, *, timeout=30):
            raise OSError("network down")

        with (
            patch("mktlib.rates._treasury.urlopen", _fail),
            patch(
                "mktlib.rates._treasury._bundled.load_year",
                return_value=_BUNDLED_2024,
            ) as mock_load,
        ):
            warnings.simplefilter("always")
            fetch_year(2024)
            # Second call should hit cache, not bundled loader
            fetch_year(2024)

        assert mock_load.call_count == 1

    def test_network_success_skips_bundled(self):
        """When network works, bundled data is not consulted."""
        with (
            patch(
                "mktlib.rates._treasury.urlopen",
                _mock_urlopen({_THIS_YEAR: _XML_THIS_YEAR}),
            ),
            patch("mktlib.rates._treasury._bundled.load_year") as mock_load,
        ):
            rates = fetch_year(_THIS_YEAR)

        assert len(rates) == 3
        mock_load.assert_not_called()

    def test_stale_disk_cache_preferred_over_bundled(self):
        """Network fails, stale disk cache has more rows → prefer disk cache."""
        stale_rows = [
            {"date": date(_THIS_YEAR, 1, 2), "BC_3MONTH": 0.054},
            {"date": date(_THIS_YEAR, 1, 3), "BC_3MONTH": 0.0538},
            {"date": date(_THIS_YEAR, 1, 4), "BC_3MONTH": 0.0536},
            {"date": date(_THIS_YEAR, 1, 5), "BC_3MONTH": 0.0534},
            {"date": date(_THIS_YEAR, 1, 8), "BC_3MONTH": 0.0532},
        ]
        bundled_rows = [
            {"date": date(_THIS_YEAR, 1, 2), "BC_3MONTH": 0.054},
            {"date": date(_THIS_YEAR, 1, 3), "BC_3MONTH": 0.0538},
        ]

        def _fail(url, *, timeout=30):
            raise OSError("network down")

        with (
            patch("mktlib.rates._treasury.urlopen", _fail),
            patch(
                "mktlib.rates._treasury._disk_cache.load_year",
                side_effect=lambda year, **kw: (
                    stale_rows if kw.get("ignore_stale") else None
                ),
            ),
            patch(
                "mktlib.rates._treasury._bundled.load_year",
                return_value=bundled_rows,
            ),
            warnings.catch_warnings(record=True) as w,
        ):
            warnings.simplefilter("always")
            rates = fetch_year(_THIS_YEAR)

        assert len(rates) == 5
        assert len(w) == 1
        assert "disk cache" in str(w[0].message).lower()

    def test_bundled_preferred_when_no_disk_cache(self):
        """Network fails, no disk cache → fall back to bundled data."""

        def _fail(url, *, timeout=30):
            raise OSError("network down")

        with (
            patch("mktlib.rates._treasury.urlopen", _fail),
            patch(
                "mktlib.rates._treasury._disk_cache.load_year",
                return_value=None,
            ),
            patch(
                "mktlib.rates._treasury._bundled.load_year",
                return_value=_BUNDLED_THIS_YEAR,
            ),
            warnings.catch_warnings(record=True) as w,
        ):
            warnings.simplefilter("always")
            rates = fetch_year(_THIS_YEAR)

        assert len(rates) == 2
        assert len(w) == 1
        assert "bundled data" in str(w[0].message).lower()


# ---------------------------------------------------------------------------
# Disk cache integration tests
# ---------------------------------------------------------------------------


class TestDiskCacheIntegration:
    """Verify fetch_year writes/reads disk cache end-to-end.

    These tests override the autouse fixture's disk-cache patches
    to exercise the real disk cache against a tmp directory.
    """

    def test_network_fetch_writes_disk_cache(self, tmp_path):
        """After a network fetch, data is persisted to disk."""
        with (
            patch.object(_dc_mod, "_CACHE_DIR", tmp_path),
            patch("mktlib.rates._disk_cache.load_year", wraps=_orig_dc_load),
            patch("mktlib.rates._disk_cache.save_year", wraps=_orig_dc_save),
            patch(
                "mktlib.rates._treasury.urlopen",
                _mock_urlopen({2024: _XML_2024}),
            ),
        ):
            clear_cache()
            fetch_year(2024)

        assert (tmp_path / "2024.csv").exists()

    def test_disk_cache_avoids_network_on_cold_start(self, tmp_path):
        """With warm disk cache, network is not hit."""
        # Seed disk cache via a real network-mocked fetch
        with (
            patch.object(_dc_mod, "_CACHE_DIR", tmp_path),
            patch("mktlib.rates._disk_cache.load_year", wraps=_orig_dc_load),
            patch("mktlib.rates._disk_cache.save_year", wraps=_orig_dc_save),
            patch(
                "mktlib.rates._treasury.urlopen",
                _mock_urlopen({2024: _XML_2024}),
            ),
        ):
            clear_cache()
            fetch_year(2024)

        # Now clear in-memory cache and fetch again — should come from disk
        clear_cache()
        with (
            patch.object(_dc_mod, "_CACHE_DIR", tmp_path),
            patch("mktlib.rates._disk_cache.load_year", wraps=_orig_dc_load),
            patch("mktlib.rates._disk_cache.save_year", wraps=_orig_dc_save),
            patch(
                "mktlib.rates._treasury.urlopen",
                side_effect=AssertionError("should not fetch from network"),
            ),
        ):
            rates = fetch_year(2024)

        assert len(rates) == 3
        assert rates[0]["date"] == date(2024, 1, 2)
        assert rates[0]["BC_3MONTH"] == pytest.approx(0.054)


# ---------------------------------------------------------------------------
# Bundled → disk cache seeding for past years
# ---------------------------------------------------------------------------


class TestBundledDiskSeed:
    def test_past_year_seeded_from_bundled(self, tmp_path):
        """Past year with no disk cache seeds from bundled, skips network."""
        with (
            patch.object(_dc_mod, "_CACHE_DIR", tmp_path),
            patch("mktlib.rates._disk_cache.load_year", wraps=_orig_dc_load),
            patch("mktlib.rates._disk_cache.save_year", wraps=_orig_dc_save),
            patch(
                "mktlib.rates._treasury._bundled.load_year",
                return_value=_BUNDLED_2024,
            ),
            patch(
                "mktlib.rates._treasury.urlopen",
                side_effect=AssertionError("should not fetch from network"),
            ),
        ):
            clear_cache()
            rates = fetch_year(2024)

        assert len(rates) == 2
        assert rates[0]["date"] == date(2024, 1, 2)
        assert rates[0]["BC_3MONTH"] == pytest.approx(0.054)
        assert (tmp_path / "2024.csv").exists()

    def test_current_year_not_seeded(self):
        """Current year still hits network even when bundled data exists."""
        with (
            patch(
                "mktlib.rates._treasury._bundled.load_year",
                return_value=_BUNDLED_THIS_YEAR,
            ) as mock_bundled,
            patch(
                "mktlib.rates._treasury.urlopen",
                _mock_urlopen({_THIS_YEAR: _XML_THIS_YEAR}),
            ),
        ):
            rates = fetch_year(_THIS_YEAR)

        # Bundled should not be consulted before network for current year
        mock_bundled.assert_not_called()
        assert len(rates) == 3


# ---------------------------------------------------------------------------
# TreasuryRate enum tests
# ---------------------------------------------------------------------------


class TestTreasuryRateEnum:
    def test_all_15_members(self):
        assert len(TreasuryRate) == 15

    def test_values_match_schema_fields(self):
        from mktlib.rates._schema import all_fields

        enum_values = sorted(m.value for m in TreasuryRate)
        assert enum_values == sorted(all_fields())

    def test_original_members_unchanged(self):
        """The 7 original members keep their names and values."""
        assert TreasuryRate.THREE_MONTH == "BC_3MONTH"
        assert TreasuryRate.SIX_MONTH == "BC_6MONTH"
        assert TreasuryRate.ONE_YEAR == "BC_1YEAR"
        assert TreasuryRate.TWO_YEAR == "BC_2YEAR"
        assert TreasuryRate.FIVE_YEAR == "BC_5YEAR"
        assert TreasuryRate.TEN_YEAR == "BC_10YEAR"
        assert TreasuryRate.THIRTY_YEAR == "BC_30YEAR"


# ---------------------------------------------------------------------------
# get_treasury_rates tests
# ---------------------------------------------------------------------------


class TestGetTreasuryRates:
    def test_single_instrument(self):
        import polars as pl

        with patch(
            "mktlib.rates._treasury.urlopen", _mock_urlopen({2024: _XML_2024})
        ):
            df = get_treasury_rates(
                date(2024, 1, 1), date(2024, 1, 31), TreasuryRate.THREE_MONTH
            )

        assert df.shape == (3, 2)
        assert df["rate"][0] == pytest.approx(0.054)

    def test_multi_instrument(self):
        with patch(
            "mktlib.rates._treasury.urlopen", _mock_urlopen({2024: _XML_2024})
        ):
            df = get_treasury_rates(
                date(2024, 1, 1),
                date(2024, 1, 31),
                [TreasuryRate.THREE_MONTH, TreasuryRate.TEN_YEAR],
            )

        assert "date" in df.columns
        assert "three_month" in df.columns
        assert "ten_year" in df.columns
        assert df.shape[0] == 3

    def test_none_returns_all_columns(self):
        with patch(
            "mktlib.rates._treasury.urlopen", _mock_urlopen({2024: _XML_2024})
        ):
            df = get_treasury_rates(date(2024, 1, 1), date(2024, 1, 31), None)

        # 1 date column + 15 instrument columns
        assert len(df.columns) == 16
        assert df.columns[0] == "date"

    def test_string_dates(self):
        with patch(
            "mktlib.rates._treasury.urlopen", _mock_urlopen({2024: _XML_2024})
        ):
            df = get_treasury_rates(
                "2024-01-01", "2024-01-31", TreasuryRate.TEN_YEAR
            )

        assert df.shape == (3, 2)
        assert df["rate"][0] == pytest.approx(0.0388)

    def test_empty_range(self):
        import polars as pl

        with patch(
            "mktlib.rates._treasury.urlopen",
            _mock_urlopen({2024: _XML_2024}),
        ):
            df = get_treasury_rates(
                date(2024, 7, 1),
                date(2024, 7, 31),
                TreasuryRate.THREE_MONTH,
            )

        assert df.shape == (0, 2)

    def test_single_instrument_missing_column(self):
        """Request an instrument not present in data → empty 2-col df."""
        import polars as pl

        with patch(
            "mktlib.rates._treasury.urlopen",
            _mock_urlopen({2024: _XML_2024}),
        ):
            df = get_treasury_rates(
                date(2024, 1, 1),
                date(2024, 1, 31),
                TreasuryRate.THIRTY_YEAR_DISPLAY,
            )

        assert df.shape == (0, 2)

    def test_multi_instrument_empty_range(self):
        """Multi-instrument with no data in range → empty wide df."""
        import polars as pl

        with patch(
            "mktlib.rates._treasury.urlopen",
            _mock_urlopen({2024: _XML_2024}),
        ):
            df = get_treasury_rates(
                date(2024, 7, 1),
                date(2024, 7, 31),
                [TreasuryRate.THREE_MONTH, TreasuryRate.TEN_YEAR],
            )

        assert df.shape == (0, 3)
        assert df.columns == ["date", "three_month", "ten_year"]


# ---------------------------------------------------------------------------
# get_treasury_spread tests
# ---------------------------------------------------------------------------


class TestGetTreasurySpread:
    def test_default_10y_2y(self):
        with patch(
            "mktlib.rates._treasury.urlopen", _mock_urlopen({2024: _XML_2024})
        ):
            df = get_treasury_spread(date(2024, 1, 1), date(2024, 1, 31))

        assert df.shape[0] == 3
        # Jan 2: 10Y=0.0388, 2Y=0.0432 → spread = -0.0044
        assert df["spread"][0] == pytest.approx(0.0388 - 0.0432)

    def test_custom_long_short(self):
        with patch(
            "mktlib.rates._treasury.urlopen", _mock_urlopen({2024: _XML_2024})
        ):
            df = get_treasury_spread(
                date(2024, 1, 1),
                date(2024, 1, 31),
                long=TreasuryRate.SIX_MONTH,
                short=TreasuryRate.THREE_MONTH,
            )

        # Jan 4 has no BC_6MONTH, so only 2 rows
        assert df.shape[0] == 2
        # Jan 2: 6M=0.0526, 3M=0.054 → spread = -0.0014
        assert df["spread"][0] == pytest.approx(0.0526 - 0.054)

    def test_missing_one_instrument_excluded(self):
        """Days where one instrument is missing should be excluded."""
        with patch(
            "mktlib.rates._treasury.urlopen",
            _mock_urlopen({2024: _XML_MISSING}),
        ):
            df = get_treasury_spread(
                date(2024, 6, 1),
                date(2024, 6, 30),
                long=TreasuryRate.TEN_YEAR,
                short=TreasuryRate.THREE_MONTH,
            )

        # Jun 3 only has 3M, Jun 4 only has 10Y — no day has both
        assert df.shape[0] == 0


# ---------------------------------------------------------------------------
# get_treasury_spread_matrix tests
# ---------------------------------------------------------------------------


class TestGetTreasurySpreadMatrix:
    def test_subset_columns_and_values(self):
        instruments = [
            TreasuryRate.THREE_MONTH,
            TreasuryRate.SIX_MONTH,
            TreasuryRate.TWO_YEAR,
            TreasuryRate.TEN_YEAR,
        ]
        with patch(
            "mktlib.rates._treasury.urlopen", _mock_urlopen({2024: _XML_2024})
        ):
            df = get_treasury_spread_matrix(
                date(2024, 1, 1), date(2024, 1, 31), instruments
            )

        # C(4, 2) = 6 pair columns, plus date. 3 rows (Jan 2, 3, 4).
        assert df.shape == (3, 7)
        assert df.columns == [
            "date",
            "spread_six_month_three_month",
            "spread_two_year_three_month",
            "spread_ten_year_three_month",
            "spread_two_year_six_month",
            "spread_ten_year_six_month",
            "spread_ten_year_two_year",
        ]
        # Jan 2: 10Y=0.0388, 2Y=0.0432 -> spread = -0.0044
        assert df["spread_ten_year_two_year"][0] == pytest.approx(
            0.0388 - 0.0432
        )
        # Jan 2: 2Y=0.0432, 3M=0.0540
        assert df["spread_two_year_three_month"][0] == pytest.approx(
            0.0432 - 0.0540
        )

    def test_argument_order_does_not_flip_sign(self):
        """Long-minus-short holds regardless of caller argument order."""
        with patch(
            "mktlib.rates._treasury.urlopen", _mock_urlopen({2024: _XML_2024})
        ):
            df = get_treasury_spread_matrix(
                date(2024, 1, 1),
                date(2024, 1, 31),
                # Deliberately long-first / out of maturity order.
                [TreasuryRate.TEN_YEAR, TreasuryRate.TWO_YEAR],
            )

        assert df.columns == ["date", "spread_ten_year_two_year"]
        assert df["spread_ten_year_two_year"][0] == pytest.approx(
            0.0388 - 0.0432
        )

    def test_missing_leg_nulls_column_without_dropping_rows(self):
        """Jan 4 lacks BC_6MONTH: six-month spreads are null, row kept."""
        with patch(
            "mktlib.rates._treasury.urlopen", _mock_urlopen({2024: _XML_2024})
        ):
            df = get_treasury_spread_matrix(
                date(2024, 1, 1),
                date(2024, 1, 31),
                [TreasuryRate.SIX_MONTH, TreasuryRate.TEN_YEAR],
            )

        # Row for Jan 4 is retained even though its six-month spread is null.
        assert df.shape[0] == 3
        assert df["spread_ten_year_six_month"][2] is None
        assert df["spread_ten_year_six_month"][0] == pytest.approx(
            0.0388 - 0.0526
        )

    def test_instrument_absent_from_data_is_all_null_column(self):
        """A tenor with no data (e.g. a newly-listed bill not yet in the
        historical feed) yields present, all-null spread columns of the right
        dtype; rows are kept and pairs that do have data are unaffected."""
        import polars as pl

        with patch(
            "mktlib.rates._treasury.urlopen", _mock_urlopen({2024: _XML_2024})
        ):
            # _XML_2024 has no BC_30YEAR, so THIRTY_YEAR is entirely absent.
            df = get_treasury_spread_matrix(
                date(2024, 1, 1),
                date(2024, 1, 31),
                [
                    TreasuryRate.TWO_YEAR,
                    TreasuryRate.TEN_YEAR,
                    TreasuryRate.THIRTY_YEAR,
                ],
            )

        # All three pair columns are present regardless of data availability.
        assert df.columns == [
            "date",
            "spread_ten_year_two_year",
            "spread_thirty_year_two_year",
            "spread_thirty_year_ten_year",
        ]
        # Rows are not dropped just because a pair is entirely null.
        assert df.shape[0] == 3
        # Spreads touching the absent tenor are all-null but typed Float64.
        for col in (
            "spread_thirty_year_two_year",
            "spread_thirty_year_ten_year",
        ):
            assert df[col].null_count() == df.shape[0]
            assert df[col].dtype == pl.Float64
        # The pair with data is unaffected.
        assert df["spread_ten_year_two_year"][0] == pytest.approx(
            0.0388 - 0.0432
        )

    def test_default_all_tenors_excludes_display_duplicate(self):
        with patch(
            "mktlib.rates._treasury.urlopen", _mock_urlopen({2024: _XML_2024})
        ):
            df = get_treasury_spread_matrix(
                date(2024, 1, 1), date(2024, 1, 31)
            )

        # Default = every tenor except THIRTY_YEAR_DISPLAY. Derived from the
        # enum so a newly-added Treasury bill flows through automatically
        # rather than spuriously failing this count.
        expected_pairs = math.comb(len(TreasuryRate) - 1, 2)
        assert len(df.columns) == 1 + expected_pairs
        assert all("thirty_year_display" not in c for c in df.columns)

    def test_empty_range_returns_typed_empty_frame(self):
        import polars as pl

        with patch(
            "mktlib.rates._treasury.urlopen", _mock_urlopen({2024: _XML_2024})
        ):
            df = get_treasury_spread_matrix(
                date(2024, 3, 1),
                date(2024, 3, 31),
                [TreasuryRate.TWO_YEAR, TreasuryRate.TEN_YEAR],
            )

        assert df.shape[0] == 0
        assert df.columns == ["date", "spread_ten_year_two_year"]
        assert df["date"].dtype == pl.Date
        assert df["spread_ten_year_two_year"].dtype == pl.Float64

    def test_longs_shorts_cross_product(self):
        """Explicit long/short leg sets → only that block of the matrix."""
        with patch(
            "mktlib.rates._treasury.urlopen", _mock_urlopen({2024: _XML_2024})
        ):
            df = get_treasury_spread_matrix(
                date(2024, 1, 1),
                date(2024, 1, 31),
                longs=[TreasuryRate.TEN_YEAR],
                shorts=[TreasuryRate.THREE_MONTH, TreasuryRate.TWO_YEAR],
            )

        assert df.columns == [
            "date",
            "spread_ten_year_three_month",
            "spread_ten_year_two_year",
        ]
        # Jan 2: 10Y=0.0388, 3M=0.0540 / 2Y=0.0432
        assert df["spread_ten_year_three_month"][0] == pytest.approx(
            0.0388 - 0.0540
        )
        assert df["spread_ten_year_two_year"][0] == pytest.approx(
            0.0388 - 0.0432
        )

    def test_longs_only_expands_short_leg(self):
        """longs set, shorts=None → every shorter tenor as the short leg."""
        with patch(
            "mktlib.rates._treasury.urlopen", _mock_urlopen({2024: _XML_2024})
        ):
            df = get_treasury_spread_matrix(
                date(2024, 1, 1),
                date(2024, 1, 31),
                longs=[TreasuryRate.TEN_YEAR],
            )

        spreads = df.columns[1:]
        # Every tenor with maturity below 10Y (11 of them: 1M … 7Y).
        assert len(spreads) == 11
        assert all(c.startswith("spread_ten_year_") for c in spreads)

    def test_self_and_inverted_pairs_skipped(self):
        """A leg appearing in both sets never spreads against itself."""
        with patch(
            "mktlib.rates._treasury.urlopen", _mock_urlopen({2024: _XML_2024})
        ):
            df = get_treasury_spread_matrix(
                date(2024, 1, 1),
                date(2024, 1, 31),
                longs=[TreasuryRate.TEN_YEAR],
                shorts=[TreasuryRate.TWO_YEAR, TreasuryRate.TEN_YEAR],
            )

        # (10Y, 10Y) self-pair dropped; only the valid 10y-2y remains.
        assert df.columns == ["date", "spread_ten_year_two_year"]

    def test_instruments_universe_with_leg_override(self):
        """instruments sets the universe; longs refines the long leg."""
        with patch(
            "mktlib.rates._treasury.urlopen", _mock_urlopen({2024: _XML_2024})
        ):
            df = get_treasury_spread_matrix(
                date(2024, 1, 1),
                date(2024, 1, 31),
                [
                    TreasuryRate.THREE_MONTH,
                    TreasuryRate.TWO_YEAR,
                    TreasuryRate.TEN_YEAR,
                ],
                longs=[TreasuryRate.TEN_YEAR],
            )

        assert df.columns == [
            "date",
            "spread_ten_year_three_month",
            "spread_ten_year_two_year",
        ]

    def test_inverted_leg_sets_yield_date_only(self):
        """Nothing in longs out-ranks shorts → no columns (date-only), no error."""
        with patch(
            "mktlib.rates._treasury.urlopen", _mock_urlopen({2024: _XML_2024})
        ):
            df = get_treasury_spread_matrix(
                date(2024, 1, 1),
                date(2024, 1, 31),
                longs=[TreasuryRate.TWO_YEAR],
                shorts=[TreasuryRate.TEN_YEAR],
            )

        assert df.columns == ["date"]
        assert df.shape[0] == 3


# ---------------------------------------------------------------------------
# Robustness / hardening tests
# ---------------------------------------------------------------------------

_TRUNCATED_XML = """\
<?xml version="1.0" encoding="utf-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:m="http://schemas.microsoft.com/ado/2007/08/dataservices/metadata"
      xmlns:d="http://schemas.microsoft.com/ado/2007/08/dataservices">
  <entry>
    <content type="application/xml">
      <m:properties>
        <d:NEW_DATE>2024-01-02T00:00:00</d:NEW_DATE>
        <d:BC_3MONTH>5.40</d:BC_3MONTH>
"""


class TestFetchYearHardening:
    def test_truncated_xml_falls_back_to_stale_cache(self):
        """ParseError from truncated XML triggers fallback to stale disk cache."""
        stale = [{"date": date(_THIS_YEAR, 1, 2), "BC_3MONTH": 0.054}]

        def _urlopen_truncated(url, *, timeout=30):
            cm = io.BytesIO(_TRUNCATED_XML.encode())
            cm.status = 200  # type: ignore[attr-defined]
            cm.__enter__ = lambda s: s  # type: ignore[attr-defined]
            cm.__exit__ = lambda s, *a: None  # type: ignore[attr-defined]
            return cm

        with (
            patch("mktlib.rates._treasury.urlopen", _urlopen_truncated),
            patch(
                "mktlib.rates._treasury._disk_cache.load_year",
                side_effect=lambda year, **kw: (
                    stale if kw.get("ignore_stale") else None
                ),
            ),
            patch(
                "mktlib.rates._treasury._bundled.load_year", return_value=[]
            ),
            warnings.catch_warnings(record=True) as w,
        ):
            warnings.simplefilter("always")
            rates = fetch_year(_THIS_YEAR)

        assert len(rates) == 1
        assert len(w) == 1
        assert "disk cache" in str(w[0].message).lower()

    def test_truncated_xml_falls_back_to_bundled(self):
        """ParseError with no disk cache falls back to bundled data."""

        def _urlopen_truncated(url, *, timeout=30):
            cm = io.BytesIO(_TRUNCATED_XML.encode())
            cm.status = 200  # type: ignore[attr-defined]
            cm.__enter__ = lambda s: s  # type: ignore[attr-defined]
            cm.__exit__ = lambda s, *a: None  # type: ignore[attr-defined]
            return cm

        with (
            patch("mktlib.rates._treasury.urlopen", _urlopen_truncated),
            patch(
                "mktlib.rates._treasury._disk_cache.load_year",
                return_value=None,
            ),
            patch(
                "mktlib.rates._treasury._bundled.load_year",
                return_value=_BUNDLED_THIS_YEAR,
            ),
            warnings.catch_warnings(record=True) as w,
        ):
            warnings.simplefilter("always")
            rates = fetch_year(_THIS_YEAR)

        assert len(rates) == 2
        assert len(w) == 1
        assert "bundled" in str(w[0].message).lower()

    def test_truncated_xml_no_fallback_raises(self):
        """ParseError with no fallback data re-raises as ParseError."""

        def _urlopen_truncated(url, *, timeout=30):
            cm = io.BytesIO(_TRUNCATED_XML.encode())
            cm.status = 200  # type: ignore[attr-defined]
            cm.__enter__ = lambda s: s  # type: ignore[attr-defined]
            cm.__exit__ = lambda s, *a: None  # type: ignore[attr-defined]
            return cm

        with (
            patch("mktlib.rates._treasury.urlopen", _urlopen_truncated),
            patch(
                "mktlib.rates._treasury._disk_cache.load_year",
                return_value=None,
            ),
            patch(
                "mktlib.rates._treasury._bundled.load_year", return_value=[]
            ),
        ):
            with pytest.raises(ET.ParseError, match="Failed to fetch"):
                fetch_year(_THIS_YEAR)

    def test_http_error_falls_back(self):
        """Non-200 HTTP status triggers the fallback path."""

        def _urlopen_500(url, *, timeout=30):
            cm = io.BytesIO(b"<html>Internal Server Error</html>")
            cm.status = 500  # type: ignore[attr-defined]
            cm.__enter__ = lambda s: s  # type: ignore[attr-defined]
            cm.__exit__ = lambda s, *a: None  # type: ignore[attr-defined]
            return cm

        with (
            patch("mktlib.rates._treasury.urlopen", _urlopen_500),
            patch(
                "mktlib.rates._treasury._bundled.load_year",
                return_value=_BUNDLED_THIS_YEAR,
            ),
            warnings.catch_warnings(record=True) as w,
        ):
            warnings.simplefilter("always")
            rates = fetch_year(_THIS_YEAR)

        assert len(rates) == 2
        assert len(w) == 1

    def test_partial_fetch_does_not_overwrite_fuller_cache(self):
        """A fetch returning fewer rows than existing disk cache does not overwrite it."""
        fuller_cache = [
            {"date": date(_THIS_YEAR, 1, 2), "BC_3MONTH": 0.054},
            {"date": date(_THIS_YEAR, 1, 3), "BC_3MONTH": 0.0538},
            {"date": date(_THIS_YEAR, 1, 4), "BC_3MONTH": 0.0536},
        ]

        # XML with only 1 entry
        _XML_PARTIAL = _XML_2025.replace("2025", str(_THIS_YEAR))
        # _XML_2025 has 1 entry

        save_mock = patch("mktlib.rates._treasury._disk_cache.save_year")
        with (
            patch(
                "mktlib.rates._treasury.urlopen",
                _mock_urlopen({_THIS_YEAR: _XML_PARTIAL}),
            ),
            patch(
                "mktlib.rates._treasury._disk_cache.load_year",
                side_effect=lambda year, **kw: (
                    fuller_cache if kw.get("ignore_stale") else None
                ),
            ),
            save_mock as mock_save,
        ):
            rates = fetch_year(_THIS_YEAR)

        # Function returns the new (partial) data for in-memory use
        assert len(rates) == 1
        # save_year should NOT have been called — partial data is smaller
        mock_save.assert_not_called()

    def test_equal_or_larger_fetch_overwrites_cache(self):
        """A fetch with >= rows than existing disk cache does overwrite it."""
        existing_cache = [
            {"date": date(_THIS_YEAR, 1, 2), "BC_3MONTH": 0.054},
        ]

        with (
            patch(
                "mktlib.rates._treasury.urlopen",
                _mock_urlopen({_THIS_YEAR: _XML_THIS_YEAR}),
            ),
            patch(
                "mktlib.rates._treasury._disk_cache.load_year",
                side_effect=lambda year, **kw: (
                    existing_cache if kw.get("ignore_stale") else None
                ),
            ),
            patch("mktlib.rates._treasury._disk_cache.save_year") as mock_save,
        ):
            rates = fetch_year(_THIS_YEAR)

        assert len(rates) == 3
        mock_save.assert_called_once()
