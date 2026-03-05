"""Tests for mktlib.rates — Treasury yield curve fetcher."""
from __future__ import annotations

import io
import warnings
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
)
from mktlib.rates._treasury import (
    clear_cache,
    fetch_average_rate,
    fetch_daily_rates,
    fetch_daily_rates_multi,
    fetch_mean_rate,
    fetch_spread,
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
                cm.__enter__ = lambda s: s  # type: ignore[attr-defined]
                cm.__exit__ = lambda s, *a: None  # type: ignore[attr-defined]
                return cm
        raise OSError(f"No fixture for URL: {url}")

    return _urlopen


@pytest.fixture(autouse=True)
def _clear_treasury_cache():
    """Ensure each test starts with a clean cache; isolate disk cache."""
    clear_cache()
    with (
        patch("mktlib.rates._disk_cache.load_year", return_value=None),
        patch("mktlib.rates._disk_cache.save_year"),
        patch("mktlib.rates._treasury._bundled.load_year", return_value=[]),
    ):
        yield
    clear_cache()


# ---------------------------------------------------------------------------
# _treasury.py unit tests
# ---------------------------------------------------------------------------


class TestFetchDailyRates:
    def test_basic_fetch(self):
        with patch("mktlib.rates._treasury.urlopen", _mock_urlopen({2024: _XML_2024})):
            rates = fetch_daily_rates(date(2024, 1, 1), date(2024, 1, 31))

        assert len(rates) == 3
        assert rates[0] == (date(2024, 1, 2), pytest.approx(0.054))
        assert rates[1] == (date(2024, 1, 3), pytest.approx(0.0538))
        assert rates[2] == (date(2024, 1, 4), pytest.approx(0.0536))

    def test_date_range_filtering(self):
        """Only return rates within the requested range."""
        with patch("mktlib.rates._treasury.urlopen", _mock_urlopen({2024: _XML_2024})):
            rates = fetch_daily_rates(date(2024, 1, 3), date(2024, 1, 3))

        assert len(rates) == 1
        assert rates[0][0] == date(2024, 1, 3)

    def test_percentage_to_decimal(self):
        """5.40% should become 0.054."""
        with patch("mktlib.rates._treasury.urlopen", _mock_urlopen({2024: _XML_2024})):
            rates = fetch_daily_rates(date(2024, 1, 2), date(2024, 1, 2))

        assert rates[0][1] == pytest.approx(0.054)

    def test_alternate_instrument(self):
        with patch("mktlib.rates._treasury.urlopen", _mock_urlopen({2024: _XML_2024})):
            rates = fetch_daily_rates(
                date(2024, 1, 2), date(2024, 1, 4), instrument="BC_10YEAR"
            )

        assert len(rates) == 3
        assert rates[0][1] == pytest.approx(0.0388)

    def test_missing_field_skipped(self):
        """When a row lacks the requested field, it's skipped."""
        with patch("mktlib.rates._treasury.urlopen", _mock_urlopen({2024: _XML_MISSING})):
            rates = fetch_daily_rates(date(2024, 6, 1), date(2024, 6, 30))

        # Only the first entry has BC_3MONTH
        assert len(rates) == 1
        assert rates[0][1] == pytest.approx(0.055)

    def test_multi_year_span(self):
        """Spanning two years fetches data from both."""
        with patch(
            "mktlib.rates._treasury.urlopen",
            _mock_urlopen({2024: _XML_2024, 2025: _XML_2025}),
        ):
            rates = fetch_daily_rates(date(2024, 1, 1), date(2025, 12, 31))

        assert len(rates) == 4  # 3 from 2024 + 1 from 2025
        assert rates[-1][0] == date(2025, 1, 2)


class TestFetchAverageRate:
    def test_average(self):
        with patch("mktlib.rates._treasury.urlopen", _mock_urlopen({2024: _XML_2024})):
            avg = fetch_average_rate(date(2024, 1, 1), date(2024, 1, 31))

        expected = (0.054 + 0.0538 + 0.0536) / 3
        assert avg == pytest.approx(expected)

    def test_empty_range_returns_zero(self):
        """No data in range → return 0.0."""
        with patch("mktlib.rates._treasury.urlopen", _mock_urlopen({2024: _XML_2024})):
            avg = fetch_average_rate(date(2024, 7, 1), date(2024, 7, 31))

        assert avg == 0.0

    def test_network_error_no_bundled(self):
        """Network failure with no bundled data raises ConnectionError."""
        def _fail(url, *, timeout=30):
            raise OSError("network down")

        with (
            patch("mktlib.rates._treasury.urlopen", _fail),
            patch("mktlib.rates._treasury._bundled.load_year", return_value=[]),
        ):
            with pytest.raises(ConnectionError, match="Failed to fetch"):
                fetch_average_rate(date(2024, 1, 1), date(2024, 1, 31))


# ---------------------------------------------------------------------------
# Public API tests
# ---------------------------------------------------------------------------


class TestGetRiskFreeRate:
    def test_with_date_objects(self):
        with patch("mktlib.rates._treasury.urlopen", _mock_urlopen({2024: _XML_2024})):
            rate = get_risk_free_rate(date(2024, 1, 1), date(2024, 1, 31))

        assert rate == pytest.approx((0.054 + 0.0538 + 0.0536) / 3)

    def test_with_string_dates(self):
        with patch("mktlib.rates._treasury.urlopen", _mock_urlopen({2024: _XML_2024})):
            rate = get_risk_free_rate("2024-01-01", "2024-01-31")

        assert rate == pytest.approx((0.054 + 0.0538 + 0.0536) / 3)

    def test_with_instrument_enum(self):
        with patch("mktlib.rates._treasury.urlopen", _mock_urlopen({2024: _XML_2024})):
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
        with patch("mktlib.rates._treasury.urlopen", _mock_urlopen({2024: _XML_2024})):
            arith = fetch_mean_rate(date(2024, 1, 1), date(2024, 1, 31), method="arithmetic")
            avg = fetch_average_rate(date(2024, 1, 1), date(2024, 1, 31))

        assert arith == pytest.approx(avg)

    def test_geometric_hand_calculated(self):
        """Geometric mean of 5.40%, 5.38%, 5.36% (as decimals)."""
        import math

        r1, r2, r3 = 0.054, 0.0538, 0.0536
        expected = math.exp((math.log(1 + r1) + math.log(1 + r2) + math.log(1 + r3)) / 3) - 1

        with patch("mktlib.rates._treasury.urlopen", _mock_urlopen({2024: _XML_2024})):
            geo = fetch_mean_rate(date(2024, 1, 1), date(2024, 1, 31), method="geometric")

        assert geo == pytest.approx(expected)

    def test_geometric_less_than_arithmetic(self):
        """AM-GM inequality: geometric mean < arithmetic mean for non-uniform rates."""
        with patch("mktlib.rates._treasury.urlopen", _mock_urlopen({2024: _XML_2024})):
            arith = fetch_mean_rate(date(2024, 1, 1), date(2024, 1, 31), method="arithmetic")
            geo = fetch_mean_rate(date(2024, 1, 1), date(2024, 1, 31), method="geometric")

        assert geo < arith

    def test_empty_range_arithmetic(self):
        with patch("mktlib.rates._treasury.urlopen", _mock_urlopen({2024: _XML_2024})):
            assert fetch_mean_rate(date(2024, 7, 1), date(2024, 7, 31), method="arithmetic") == 0.0

    def test_empty_range_geometric(self):
        with patch("mktlib.rates._treasury.urlopen", _mock_urlopen({2024: _XML_2024})):
            assert fetch_mean_rate(date(2024, 7, 1), date(2024, 7, 31), method="geometric") == 0.0


# ---------------------------------------------------------------------------
# get_mean_treasury_rate public API tests
# ---------------------------------------------------------------------------


class TestGetMeanTreasuryRate:
    def test_arithmetic_enum(self):
        with patch("mktlib.rates._treasury.urlopen", _mock_urlopen({2024: _XML_2024})):
            rate = get_mean_treasury_rate(
                date(2024, 1, 1), date(2024, 1, 31), method=MeanMethod.ARITHMETIC,
            )

        expected = (0.054 + 0.0538 + 0.0536) / 3
        assert rate == pytest.approx(expected)

    def test_geometric_enum(self):
        import math

        r1, r2, r3 = 0.054, 0.0538, 0.0536
        expected = math.exp((math.log(1 + r1) + math.log(1 + r2) + math.log(1 + r3)) / 3) - 1

        with patch("mktlib.rates._treasury.urlopen", _mock_urlopen({2024: _XML_2024})):
            rate = get_mean_treasury_rate(
                date(2024, 1, 1), date(2024, 1, 31), method=MeanMethod.GEOMETRIC,
            )

        assert rate == pytest.approx(expected)

    def test_string_dates(self):
        with patch("mktlib.rates._treasury.urlopen", _mock_urlopen({2024: _XML_2024})):
            rate = get_mean_treasury_rate("2024-01-01", "2024-01-31")

        expected = (0.054 + 0.0538 + 0.0536) / 3
        assert rate == pytest.approx(expected)

    def test_instrument_enum(self):
        with patch("mktlib.rates._treasury.urlopen", _mock_urlopen({2024: _XML_2024})):
            rate = get_mean_treasury_rate(
                "2024-01-02", "2024-01-04", instrument=TreasuryRate.TEN_YEAR,
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
        ret_df = pl.DataFrame({"date": dates, "return": [0.001, 0.002, -0.001]})

        with patch("mktlib.rates._treasury.urlopen", _mock_urlopen({2024: _XML_2024})):
            m = metrics(ret_df, rf="auto")

        # rf should have been resolved to a float, verify Sharpe uses it
        assert isinstance(m.sharpe, float)

    def test_metrics_rf_explicit_unchanged(self):
        """Passing a float still works as before."""
        import polars as pl

        from mktlib.reports import metrics

        dates = pl.date_range(date(2024, 1, 2), date(2024, 1, 4), eager=True)
        ret_df = pl.DataFrame({"date": dates, "return": [0.001, 0.002, -0.001]})

        m = metrics(ret_df, rf=0.05)
        assert isinstance(m.sharpe, float)


# ---------------------------------------------------------------------------
# Bundled fallback tests
# ---------------------------------------------------------------------------

_BUNDLED_2024 = [
    (date(2024, 1, 2), {"BC_3MONTH": 0.054, "BC_10YEAR": 0.0388}),
    (date(2024, 1, 3), {"BC_3MONTH": 0.0538, "BC_10YEAR": 0.0392}),
]

# Current-year fixtures — seed block is skipped for current year, so these
# let us test the network-error fallback path in isolation.
_THIS_YEAR = date.today().year
_XML_THIS_YEAR = _XML_2024.replace("2024", str(_THIS_YEAR))
_BUNDLED_THIS_YEAR = [
    (date(_THIS_YEAR, 1, 2), {"BC_3MONTH": 0.054, "BC_10YEAR": 0.0388}),
    (date(_THIS_YEAR, 1, 3), {"BC_3MONTH": 0.0538, "BC_10YEAR": 0.0392}),
]


class TestBundledFallback:
    def test_fallback_on_network_error(self):
        """Network error with bundled data available: warn and return data."""
        def _fail(url, *, timeout=30):
            raise OSError("network down")

        with (
            patch("mktlib.rates._treasury.urlopen", _fail),
            patch("mktlib.rates._treasury._bundled.load_year", return_value=_BUNDLED_THIS_YEAR),
            warnings.catch_warnings(record=True) as w,
        ):
            warnings.simplefilter("always")
            rates = fetch_daily_rates(date(_THIS_YEAR, 1, 1), date(_THIS_YEAR, 1, 31))

        assert len(rates) == 2
        assert rates[0] == (date(_THIS_YEAR, 1, 2), pytest.approx(0.054))
        assert len(w) == 1
        assert "bundled" in str(w[0].message).lower()

    def test_fallback_data_is_cached(self):
        """After fallback, subsequent calls use the cache (no re-fetch)."""
        def _fail(url, *, timeout=30):
            raise OSError("network down")

        with (
            patch("mktlib.rates._treasury.urlopen", _fail),
            patch("mktlib.rates._treasury._bundled.load_year", return_value=_BUNDLED_2024) as mock_load,
        ):
            warnings.simplefilter("always")
            fetch_daily_rates(date(2024, 1, 1), date(2024, 1, 31))
            # Second call should hit cache, not bundled loader
            fetch_daily_rates(date(2024, 1, 1), date(2024, 1, 31))

        assert mock_load.call_count == 1

    def test_network_success_skips_bundled(self):
        """When network works, bundled data is not consulted."""
        with (
            patch("mktlib.rates._treasury.urlopen", _mock_urlopen({_THIS_YEAR: _XML_THIS_YEAR})),
            patch("mktlib.rates._treasury._bundled.load_year") as mock_load,
        ):
            rates = fetch_daily_rates(date(_THIS_YEAR, 1, 1), date(_THIS_YEAR, 1, 31))

        assert len(rates) == 3
        mock_load.assert_not_called()

    def test_stale_disk_cache_preferred_over_bundled(self):
        """Network fails, stale disk cache has more rows → prefer disk cache."""
        stale_rows = [
            (date(_THIS_YEAR, 1, 2), {"BC_3MONTH": 0.054}),
            (date(_THIS_YEAR, 1, 3), {"BC_3MONTH": 0.0538}),
            (date(_THIS_YEAR, 1, 4), {"BC_3MONTH": 0.0536}),
            (date(_THIS_YEAR, 1, 5), {"BC_3MONTH": 0.0534}),
            (date(_THIS_YEAR, 1, 8), {"BC_3MONTH": 0.0532}),
        ]
        bundled_rows = [
            (date(_THIS_YEAR, 1, 2), {"BC_3MONTH": 0.054}),
            (date(_THIS_YEAR, 1, 3), {"BC_3MONTH": 0.0538}),
        ]

        def _fail(url, *, timeout=30):
            raise OSError("network down")

        with (
            patch("mktlib.rates._treasury.urlopen", _fail),
            patch(
                "mktlib.rates._treasury._disk_cache.load_year",
                side_effect=lambda year, **kw: stale_rows if kw.get("ignore_stale") else None,
            ),
            patch("mktlib.rates._treasury._bundled.load_year", return_value=bundled_rows),
            warnings.catch_warnings(record=True) as w,
        ):
            warnings.simplefilter("always")
            rates = fetch_daily_rates(date(_THIS_YEAR, 1, 1), date(_THIS_YEAR, 1, 31))

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
            patch("mktlib.rates._treasury._bundled.load_year", return_value=_BUNDLED_THIS_YEAR),
            warnings.catch_warnings(record=True) as w,
        ):
            warnings.simplefilter("always")
            rates = fetch_daily_rates(date(_THIS_YEAR, 1, 1), date(_THIS_YEAR, 1, 31))

        assert len(rates) == 2
        assert len(w) == 1
        assert "bundled data" in str(w[0].message).lower()


# ---------------------------------------------------------------------------
# Disk cache integration tests
# ---------------------------------------------------------------------------


class TestDiskCacheIntegration:
    """Verify _fetch_year writes/reads disk cache end-to-end.

    These tests override the autouse fixture's disk-cache patches
    to exercise the real disk cache against a tmp directory.
    """

    def test_network_fetch_writes_disk_cache(self, tmp_path):
        """After a network fetch, data is persisted to disk."""
        with (
            patch.object(_dc_mod, "_CACHE_DIR", tmp_path),
            patch("mktlib.rates._disk_cache.load_year", wraps=_orig_dc_load),
            patch("mktlib.rates._disk_cache.save_year", wraps=_orig_dc_save),
            patch("mktlib.rates._treasury.urlopen", _mock_urlopen({2024: _XML_2024})),
        ):
            clear_cache()
            fetch_daily_rates(date(2024, 1, 1), date(2024, 1, 31))

        assert (tmp_path / "2024.csv").exists()

    def test_disk_cache_avoids_network_on_cold_start(self, tmp_path):
        """With warm disk cache, network is not hit."""
        # Seed disk cache via a real network-mocked fetch
        with (
            patch.object(_dc_mod, "_CACHE_DIR", tmp_path),
            patch("mktlib.rates._disk_cache.load_year", wraps=_orig_dc_load),
            patch("mktlib.rates._disk_cache.save_year", wraps=_orig_dc_save),
            patch("mktlib.rates._treasury.urlopen", _mock_urlopen({2024: _XML_2024})),
        ):
            clear_cache()
            fetch_daily_rates(date(2024, 1, 1), date(2024, 1, 31))

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
            rates = fetch_daily_rates(date(2024, 1, 1), date(2024, 1, 31))

        assert len(rates) == 3
        assert rates[0] == (date(2024, 1, 2), pytest.approx(0.054))


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
            patch("mktlib.rates._treasury._bundled.load_year", return_value=_BUNDLED_2024),
            patch(
                "mktlib.rates._treasury.urlopen",
                side_effect=AssertionError("should not fetch from network"),
            ),
        ):
            clear_cache()
            rates = fetch_daily_rates(date(2024, 1, 1), date(2024, 1, 31))

        assert len(rates) == 2
        assert rates[0] == (date(2024, 1, 2), pytest.approx(0.054))
        assert (tmp_path / "2024.csv").exists()

    def test_current_year_not_seeded(self):
        """Current year still hits network even when bundled data exists."""
        with (
            patch("mktlib.rates._treasury._bundled.load_year", return_value=_BUNDLED_THIS_YEAR) as mock_bundled,
            patch("mktlib.rates._treasury.urlopen", _mock_urlopen({_THIS_YEAR: _XML_THIS_YEAR})),
        ):
            rates = fetch_daily_rates(date(_THIS_YEAR, 1, 1), date(_THIS_YEAR, 1, 31))

        # Bundled should not be consulted before network for current year
        mock_bundled.assert_not_called()
        assert len(rates) == 3


# ---------------------------------------------------------------------------
# TreasuryRate enum tests
# ---------------------------------------------------------------------------


class TestTreasuryRateEnum:
    def test_all_14_members(self):
        assert len(TreasuryRate) == 14

    def test_values_match_disk_cache_fields(self):
        from mktlib.rates._disk_cache import _FIELDS

        enum_values = sorted(m.value for m in TreasuryRate)
        assert enum_values == sorted(_FIELDS)

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
# fetch_daily_rates_multi tests
# ---------------------------------------------------------------------------


class TestFetchDailyRatesMulti:
    def test_filter_by_instrument_list(self):
        with patch("mktlib.rates._treasury.urlopen", _mock_urlopen({2024: _XML_2024})):
            rows = fetch_daily_rates_multi(
                date(2024, 1, 1), date(2024, 1, 31), ["BC_3MONTH", "BC_10YEAR"]
            )

        assert len(rows) == 3
        for _, rates in rows:
            assert set(rates.keys()) <= {"BC_3MONTH", "BC_10YEAR"}

    def test_none_returns_all_fields(self):
        with patch("mktlib.rates._treasury.urlopen", _mock_urlopen({2024: _XML_2024})):
            rows = fetch_daily_rates_multi(date(2024, 1, 1), date(2024, 1, 31), None)

        assert len(rows) == 3
        # First entry has 4 fields (3MONTH, 6MONTH, 2YEAR, 10YEAR)
        assert len(rows[0][1]) == 4

    def test_date_range_filtering(self):
        with patch("mktlib.rates._treasury.urlopen", _mock_urlopen({2024: _XML_2024})):
            rows = fetch_daily_rates_multi(date(2024, 1, 3), date(2024, 1, 3))

        assert len(rows) == 1
        assert rows[0][0] == date(2024, 1, 3)

    def test_skips_days_without_requested_instruments(self):
        """Entry on Jan 4 has no BC_6MONTH — should still appear if it has BC_3MONTH."""
        with patch("mktlib.rates._treasury.urlopen", _mock_urlopen({2024: _XML_2024})):
            rows = fetch_daily_rates_multi(
                date(2024, 1, 1), date(2024, 1, 31), ["BC_6MONTH"]
            )

        # Only Jan 2 and Jan 3 have BC_6MONTH
        assert len(rows) == 2


# ---------------------------------------------------------------------------
# get_treasury_rates tests
# ---------------------------------------------------------------------------


class TestGetTreasuryRates:
    def test_single_instrument(self):
        import polars as pl

        with patch("mktlib.rates._treasury.urlopen", _mock_urlopen({2024: _XML_2024})):
            df = get_treasury_rates(date(2024, 1, 1), date(2024, 1, 31), TreasuryRate.THREE_MONTH)

        assert df.columns == ["date", "rate"]
        assert df.shape == (3, 2)
        assert df.dtypes == [pl.Date, pl.Float64]
        assert df["rate"][0] == pytest.approx(0.054)

    def test_multi_instrument(self):
        with patch("mktlib.rates._treasury.urlopen", _mock_urlopen({2024: _XML_2024})):
            df = get_treasury_rates(
                date(2024, 1, 1), date(2024, 1, 31),
                [TreasuryRate.THREE_MONTH, TreasuryRate.TEN_YEAR],
            )

        assert "date" in df.columns
        assert "three_month" in df.columns
        assert "ten_year" in df.columns
        assert df.shape[0] == 3

    def test_none_returns_all_columns(self):
        with patch("mktlib.rates._treasury.urlopen", _mock_urlopen({2024: _XML_2024})):
            df = get_treasury_rates(date(2024, 1, 1), date(2024, 1, 31), None)

        # 1 date column + 14 instrument columns
        assert len(df.columns) == 15
        assert df.columns[0] == "date"

    def test_string_dates(self):
        with patch("mktlib.rates._treasury.urlopen", _mock_urlopen({2024: _XML_2024})):
            df = get_treasury_rates("2024-01-01", "2024-01-31", TreasuryRate.TEN_YEAR)

        assert df.shape == (3, 2)
        assert df["rate"][0] == pytest.approx(0.0388)

    def test_empty_range(self):
        import polars as pl

        with patch("mktlib.rates._treasury.urlopen", _mock_urlopen({2024: _XML_2024})):
            df = get_treasury_rates(date(2024, 7, 1), date(2024, 7, 31), TreasuryRate.THREE_MONTH)

        assert df.shape == (0, 2)
        assert df.dtypes == [pl.Date, pl.Float64]


# ---------------------------------------------------------------------------
# get_treasury_spread tests
# ---------------------------------------------------------------------------


class TestGetTreasurySpread:
    def test_default_10y_2y(self):
        with patch("mktlib.rates._treasury.urlopen", _mock_urlopen({2024: _XML_2024})):
            df = get_treasury_spread(date(2024, 1, 1), date(2024, 1, 31))

        assert df.columns == ["date", "spread"]
        assert df.shape[0] == 3
        # Jan 2: 10Y=0.0388, 2Y=0.0432 → spread = -0.0044
        assert df["spread"][0] == pytest.approx(0.0388 - 0.0432)

    def test_custom_long_short(self):
        with patch("mktlib.rates._treasury.urlopen", _mock_urlopen({2024: _XML_2024})):
            df = get_treasury_spread(
                date(2024, 1, 1), date(2024, 1, 31),
                long=TreasuryRate.SIX_MONTH,
                short=TreasuryRate.THREE_MONTH,
            )

        # Jan 4 has no BC_6MONTH, so only 2 rows
        assert df.shape[0] == 2
        # Jan 2: 6M=0.0526, 3M=0.054 → spread = -0.0014
        assert df["spread"][0] == pytest.approx(0.0526 - 0.054)

    def test_missing_one_instrument_excluded(self):
        """Days where one instrument is missing should be excluded."""
        with patch("mktlib.rates._treasury.urlopen", _mock_urlopen({2024: _XML_MISSING})):
            df = get_treasury_spread(
                date(2024, 6, 1), date(2024, 6, 30),
                long=TreasuryRate.TEN_YEAR,
                short=TreasuryRate.THREE_MONTH,
            )

        # Jun 3 only has 3M, Jun 4 only has 10Y — no day has both
        assert df.shape[0] == 0
