"""Frame comparison helpers shared across the test suite."""

from __future__ import annotations

import polars as pl
from polars.testing import assert_frame_equal


def assert_frames_close(
    left: pl.DataFrame,
    right: pl.DataFrame,
    *,
    rel_tol: float = 1e-12,
    abs_tol: float = 1e-15,
) -> None:
    """Assert two frames match, with a tolerance on float columns only.

    Dtypes, column order and row order are compared exactly.  Float columns
    tolerate ulp-level arithmetic drift across polars releases (#93: polars
    1.44.2 moved one ``return`` cell by 1 ulp, trades identical); anything
    larger than ``rel_tol`` relative is a real change and still fails.

    The tolerances are passed as ``rel_tol`` / ``abs_tol``, the spelling
    every supported polars version accepts.
    """
    assert_frame_equal(
        left,
        right,
        check_exact=False,
        rel_tol=rel_tol,
        abs_tol=abs_tol,
        check_dtypes=True,
        check_column_order=True,
        check_row_order=True,
    )
