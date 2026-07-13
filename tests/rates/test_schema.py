"""Tests for mktlib.rates._schema — schema.csv reader."""

from __future__ import annotations

import csv
from importlib.resources import files

from mktlib.rates._schema import all_fields, load_schema


class TestAllFields:
    def test_returns_bc_columns(self):
        fields = all_fields()
        assert len(fields) > 0
        assert all(f.startswith("BC_") for f in fields)

    def test_count_matches_treasury_rate_enum(self):
        from mktlib.rates import TreasuryRate

        assert len(all_fields()) == len(TreasuryRate)

    def test_data_fields_are_all_known_to_enum(self):
        """Guard against a newly-listed Treasury instrument slipping in.

        Every BC_* field in the bundled data must map to a TreasuryRate
        member. When the monthly refresh pulls a tenor Treasury.gov just
        added, schema.csv gains a field the enum doesn't know — and this test
        fails on that refresh PR, blocking auto-merge until a maintainer adds
        the member to TreasuryRate (mktlib/rates/__init__.py). Without the
        enum entry, get_treasury_rates / get_treasury_spread_matrix would
        silently drop the new instrument.
        """
        from mktlib.rates import TreasuryRate

        known = {m.value for m in TreasuryRate}
        unknown = sorted(set(all_fields()) - known)
        assert not unknown, (
            f"Bundled Treasury data has fields not in TreasuryRate: {unknown}. "
            "Add each as a TreasuryRate member in mktlib/rates/__init__.py "
            '(FRIENDLY_NAME = "BC_..."), then re-run.'
        )


class TestLoadSchema:
    def test_covers_bundled_years(self):
        schema = load_schema()
        for year in range(2000, 2027):
            assert year in schema, f"year {year} missing from schema"

    def test_fields_are_subset_of_all_fields(self):
        schema = load_schema()
        universe = set(all_fields())
        for year, fields in schema.items():
            assert set(fields) <= universe, f"year {year} has unknown fields"

    def test_schema_fields_match_bundled_csv_headers(self):
        """For each year, bundled CSV headers are a subset of all_fields()."""
        universe = set(all_fields())
        data_dir = files("mktlib.rates") / "_data"
        for year in range(2000, 2027):
            text = (data_dir / f"{year}.csv").read_text(encoding="utf-8")
            reader = csv.DictReader(text.splitlines())
            csv_fields = {
                f for f in (reader.fieldnames or []) if f.startswith("BC_")
            }
            assert (
                csv_fields <= universe
            ), f"year {year} CSV has fields not in schema: {csv_fields - universe}"
