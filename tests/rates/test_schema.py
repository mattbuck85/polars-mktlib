"""Tests for mktlib.rates._schema — schema.csv reader."""

from __future__ import annotations

import csv
from importlib.resources import files

from mktlib.rates._schema import all_fields, load_schema

# ---------------------------------------------------------------------------
# Deterministic Treasury BC_* tenor helpers.
#
# Used by the guard tests below to (1) assert the TreasuryRate enum is declared
# in ascending-maturity order and (2) emit a copy-paste-ready fix — including
# the correct sorted insertion point — when a newly-listed instrument shows up
# in the refreshed data. Insertion ORDER matters: get_treasury_spread_matrix
# pairs tenors by enum declaration order, so a member added out of place
# silently mislabels/sign-flips spread columns.
# ---------------------------------------------------------------------------

_ONES = (
    "",
    "ONE",
    "TWO",
    "THREE",
    "FOUR",
    "FIVE",
    "SIX",
    "SEVEN",
    "EIGHT",
    "NINE",
    "TEN",
    "ELEVEN",
    "TWELVE",
    "THIRTEEN",
    "FOURTEEN",
    "FIFTEEN",
    "SIXTEEN",
    "SEVENTEEN",
    "EIGHTEEN",
    "NINETEEN",
)
_TENS = (
    "",
    "",
    "TWENTY",
    "THIRTY",
    "FORTY",
    "FIFTY",
    "SIXTY",
    "SEVENTY",
    "EIGHTY",
    "NINETY",
)


def _tenor_months(bc_field: str) -> float:
    """Maturity of a Treasury ``BC_*`` field in months, for ordering.

    ``BC_3MONTH`` -> 3, ``BC_1_5MONTH`` -> 1.5, ``BC_2YEAR`` -> 24,
    ``BC_30YEARDISPLAY`` -> 360 (a display duplicate of ``BC_30YEAR``).
    """
    body = bc_field.removeprefix("BC_").removesuffix("DISPLAY")
    if body.endswith("MONTH"):
        return float(body.removesuffix("MONTH").replace("_", "."))
    if body.endswith("YEAR"):
        return float(body.removesuffix("YEAR").replace("_", ".")) * 12
    raise ValueError(f"cannot parse tenor from {bc_field!r}")


def _number_word(n: int) -> str | None:
    """English SCREAMING_SNAKE word for a small whole number, else None."""
    if not 0 < n < 100:
        return None
    if n < 20:
        return _ONES[n]
    tens, ones = divmod(n, 10)
    return _TENS[tens] + (f"_{_ONES[ones]}" if ones else "")


def _suggested_member_line(bc_field: str) -> str:
    """Suggested ``NAME = "BC_..."`` enum line for a BC_* field.

    Derives the friendly name for the standard ``<int><UNIT>`` pattern; falls
    back to a ``<NAME>`` placeholder for non-standard (fractional / display)
    fields, which need a human-chosen name.
    """
    body = bc_field.removeprefix("BC_")
    for unit in ("MONTH", "YEAR"):
        stem = body.removesuffix(unit)
        if body.endswith(unit) and stem.isdigit():
            word = _number_word(int(stem))
            if word:
                return f'{word}_{unit} = "{bc_field}"'
    return f'<NAME> = "{bc_field}"'


def _placement_hint(bc_field: str) -> str:
    """`field (N months): add <line> after X and before Y` for a new tenor."""
    from mktlib.rates import TreasuryRate

    months = _tenor_months(bc_field)
    ordered = sorted(TreasuryRate, key=lambda m: _tenor_months(m.value))
    lower = [m for m in ordered if _tenor_months(m.value) <= months]
    higher = [m for m in ordered if _tenor_months(m.value) > months]
    insert_after = lower[-1].name if lower else "(start of enum)"
    insert_before = higher[0].name if higher else "(end of enum)"
    return (
        f"{bc_field} ({months:g} months): add `{_suggested_member_line(bc_field)}` "
        f"after {insert_after} and before {insert_before}"
    )


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
        hints = "\n  ".join(_placement_hint(f) for f in unknown)
        assert not unknown, (
            "Bundled Treasury data has BC_* fields not in the TreasuryRate "
            "enum (mktlib/rates/__init__.py). Add each member IN MATURITY "
            "ORDER (get_treasury_spread_matrix pairs tenors by declaration "
            f"order, so position matters):\n  {hints}"
        )


class TestTreasuryRateOrdering:
    def test_declared_in_ascending_maturity(self):
        """A misplaced enum insert silently mislabels spread-matrix columns
        (pairing keys off declaration order), so pin the ordering invariant."""
        from mktlib.rates import TreasuryRate

        months = [_tenor_months(m.value) for m in TreasuryRate]
        assert months == sorted(months), (
            "TreasuryRate must be declared in ascending maturity order. Got: "
            + ", ".join(
                f"{m.name}={mo:g}" for m, mo in zip(TreasuryRate, months)
            )
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
