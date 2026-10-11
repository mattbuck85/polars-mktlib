"""Fixtures for the engine-parity tests."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager

import pytest

from tests._scan_backend import scan_backend
from tests.engine_parity._chaos import (
    Perturbation,
    perturb_tie_order,
    perturb_unspecified_order,
)

__all__ = ["scan_backend", "tie_order", "unspecified_order"]


@pytest.fixture
def unspecified_order() -> Callable[[], AbstractContextManager[Perturbation]]:
    """A context manager that reverses every row order polars does not promise.

    Inside ``with unspecified_order():`` a ``DataFrame.join`` without
    ``maintain_order`` and a ``group_by`` without ``maintain_order=True``
    return their rows reversed.  See :mod:`tests.engine_parity._chaos`.
    """
    return perturb_unspecified_order


@pytest.fixture
def tie_order() -> Callable[[], AbstractContextManager[Perturbation]]:
    """A context manager that reverses rows tied on a sort key.

    Inside ``with tie_order():`` a ``DataFrame.sort`` or ``Expr.sort_by``
    without ``maintain_order=True`` returns rows that are equal on the sort
    key in the reverse of their input order.
    """
    return perturb_tie_order
