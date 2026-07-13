"""Schema validation tests for mktlib.rates output DataFrames."""

from __future__ import annotations

import io
import math
from datetime import date
from unittest.mock import patch

import pytest

from mktlib.rates import (
    TreasuryRate,
    get_treasury_rates,
    get_treasury_spread,
    get_treasury_spread_matrix,
)
from mktlib.rates import _disk_cache as _dc_mod
from mktlib.rates._treasury import clear_cache

from tests.schemas.rates import (
    SingleRateSchema,
    SpreadSchema,
    validate_multi_rate_df,
    validate_spread_matrix_df,
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


def _mock_urlopen(xml_by_year: dict[int, str]):
    """Return a context-manager mock for urlopen keyed by year in the URL."""

    def _urlopen(url, *, timeout=30):  # noqa: ARG001
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
    """Isolate each test from in-memory and disk caches."""
    clear_cache()
    with (
        patch.object(_dc_mod, "_CACHE_DIR", tmp_path),
        patch("mktlib.rates._disk_cache.load_year", return_value=None),
        patch("mktlib.rates._disk_cache.save_year"),
        patch("mktlib.rates._treasury._bundled.load_year", return_value=[]),
        patch(
            "mktlib.rates._treasury._bundled.bundled_path", return_value=None
        ),
    ):
        yield
    clear_cache()


class TestSingleRateSchema:
    def test_basic(self):
        with patch(
            "mktlib.rates._treasury.urlopen",
            _mock_urlopen({2024: _XML_2024}),
        ):
            df = get_treasury_rates(
                date(2024, 1, 1),
                date(2024, 1, 31),
                TreasuryRate.THREE_MONTH,
            )
        SingleRateSchema.validate(df)

    def test_different_instrument(self):
        with patch(
            "mktlib.rates._treasury.urlopen",
            _mock_urlopen({2024: _XML_2024}),
        ):
            df = get_treasury_rates(
                date(2024, 1, 1),
                date(2024, 1, 31),
                TreasuryRate.TEN_YEAR,
            )
        SingleRateSchema.validate(df)


class TestSpreadSchema:
    def test_default_10y_2y(self):
        with patch(
            "mktlib.rates._treasury.urlopen",
            _mock_urlopen({2024: _XML_2024}),
        ):
            df = get_treasury_spread(date(2024, 1, 1), date(2024, 1, 31))
        SpreadSchema.validate(df)


class TestSpreadMatrixSchema:
    def test_subset(self):
        with patch(
            "mktlib.rates._treasury.urlopen",
            _mock_urlopen({2024: _XML_2024}),
        ):
            df = get_treasury_spread_matrix(
                date(2024, 1, 1),
                date(2024, 1, 31),
                [TreasuryRate.TWO_YEAR, TreasuryRate.TEN_YEAR],
            )
        validate_spread_matrix_df(
            df, expected_cols=["spread_ten_year_two_year"]
        )

    def test_all_instruments(self):
        with patch(
            "mktlib.rates._treasury.urlopen",
            _mock_urlopen({2024: _XML_2024}),
        ):
            df = get_treasury_spread_matrix(
                date(2024, 1, 1), date(2024, 1, 31)
            )
        validate_spread_matrix_df(df)
        # date + C(n-1, 2) pairs (all tenors except THIRTY_YEAR_DISPLAY).
        expected_pairs = math.comb(len(TreasuryRate) - 1, 2)
        assert len(df.columns) == 1 + expected_pairs


class TestMultiRateSchema:
    def test_multi_instrument(self):
        with patch(
            "mktlib.rates._treasury.urlopen",
            _mock_urlopen({2024: _XML_2024}),
        ):
            df = get_treasury_rates(
                date(2024, 1, 1),
                date(2024, 1, 31),
                [TreasuryRate.THREE_MONTH, TreasuryRate.TEN_YEAR],
            )
        validate_multi_rate_df(df, expected_cols=["three_month", "ten_year"])

    def test_all_instruments(self):
        with patch(
            "mktlib.rates._treasury.urlopen",
            _mock_urlopen({2024: _XML_2024}),
        ):
            df = get_treasury_rates(date(2024, 1, 1), date(2024, 1, 31), None)
        validate_multi_rate_df(df)
        assert df.columns[0] == "date"
        assert len(df.columns) == 16  # date + 15 instruments
