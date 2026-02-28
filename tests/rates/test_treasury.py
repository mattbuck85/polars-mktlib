"""Tests for mktlib.rates — Treasury yield curve fetcher."""
from __future__ import annotations

import io
import warnings
from datetime import date
from unittest.mock import patch

import pytest

from mktlib.rates import TreasuryRate, get_risk_free_rate
from mktlib.rates._treasury import (
    clear_cache,
    fetch_average_rate,
    fetch_daily_rates,
)

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
        <d:BC_10YEAR>3.92</d:BC_10YEAR>
      </m:properties>
    </content>
  </entry>
  <entry>
    <content type="application/xml">
      <m:properties>
        <d:NEW_DATE>2024-01-04T00:00:00</d:NEW_DATE>
        <d:BC_3MONTH>5.36</d:BC_3MONTH>
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
    """Ensure each test starts with a clean cache."""
    clear_cache()
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


class TestBundledFallback:
    def test_fallback_on_network_error(self):
        """Network error with bundled data available: warn and return data."""
        def _fail(url, *, timeout=30):
            raise OSError("network down")

        with (
            patch("mktlib.rates._treasury.urlopen", _fail),
            patch("mktlib.rates._treasury._bundled.load_year", return_value=_BUNDLED_2024),
            warnings.catch_warnings(record=True) as w,
        ):
            warnings.simplefilter("always")
            rates = fetch_daily_rates(date(2024, 1, 1), date(2024, 1, 31))

        assert len(rates) == 2
        assert rates[0] == (date(2024, 1, 2), pytest.approx(0.054))
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
            patch("mktlib.rates._treasury.urlopen", _mock_urlopen({2024: _XML_2024})),
            patch("mktlib.rates._treasury._bundled.load_year") as mock_load,
        ):
            rates = fetch_daily_rates(date(2024, 1, 1), date(2024, 1, 31))

        assert len(rates) == 3
        mock_load.assert_not_called()
