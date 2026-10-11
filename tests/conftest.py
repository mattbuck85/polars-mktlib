"""Suite-wide test configuration."""

from __future__ import annotations

import warnings
from collections.abc import Generator
from pathlib import Path

import pytest

import mktlib

# A DeprecationWarning attributed to a file under one of these directories
# fails the test that triggered it.
_OWN_ROOTS = (
    Path(mktlib.__file__).resolve().parent,
    Path(__file__).resolve().parent,
)


def _is_own(filename: str) -> bool:
    try:
        path = Path(filename).resolve()
    except (OSError, ValueError):
        return False
    return any(path.is_relative_to(root) for root in _OWN_ROOTS)


def _fail_on_own_deprecations() -> Generator[None, object, object]:
    """Record the warnings of one test phase; fail on an own deprecation.

    The supported polars range spans a major version: a spelling deprecated
    on one end is removed on the other.  The warnings are recorded and
    judged after the phase rather than escalated with an ``error`` filter,
    because polars raises some deprecations while it evaluates an
    expression in native code, and a warning turned into an exception there
    is discarded: the call returns normally and nothing is reported.

    The filters in force are left as they are, so ``-W``, the
    ``filterwarnings`` mark, ``pytest.warns`` and ``warnings.catch_warnings``
    in a test body keep their meaning: a warning they ignore or capture
    never reaches this recorder.  Every other recorded warning is emitted
    again so that it still reaches the warnings summary.
    """
    with warnings.catch_warnings(record=True) as recorded:
        try:
            result = yield
        finally:
            caught = list(recorded)
    own: list[str] = []
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        for w in caught:
            if issubclass(w.category, DeprecationWarning) and _is_own(
                w.filename
            ):
                own.append(f"{w.filename}:{w.lineno}: {w.message}")
            else:
                warnings.warn_explicit(
                    w.message, w.category, w.filename, w.lineno,
                    source=w.source,
                )
    if own:
        pytest.fail(
            "DeprecationWarning attributed to mktlib or its tests:\n"
            + "\n".join(own),
            pytrace=False,
        )
    return result


@pytest.hookimpl(wrapper=True)
def pytest_runtest_setup() -> Generator[None, object, object]:
    return (yield from _fail_on_own_deprecations())


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call() -> Generator[None, object, object]:
    return (yield from _fail_on_own_deprecations())


@pytest.hookimpl(wrapper=True)
def pytest_runtest_teardown() -> Generator[None, object, object]:
    return (yield from _fail_on_own_deprecations())
