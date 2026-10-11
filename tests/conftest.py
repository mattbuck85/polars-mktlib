"""Suite-wide test configuration.

A ``DeprecationWarning`` attributed to a file under ``mktlib/`` or
``tests/`` fails the run.  The supported polars range spans a major
version: a spelling deprecated on one end is removed on the other.

The failure is a session-level one, not a per-test one.  This module
installs no warnings filter and no recorder; it reads what pytest itself
records (``pytest_warning_recorded``) during configuration, collection and
every test.  The outcome of each test, ``recwarn``, ``pytest.warns`` and
the warnings summary are therefore the ones pytest produces without this
file.  When at least one such deprecation was recorded, a section at the
end of the report lists each of them, and a run that would have exited
with status 0 exits with status 1 instead.

The warnings are not escalated with an ``error`` filter because polars
raises some deprecations while it evaluates an expression in native code,
and a warning turned into an exception there is discarded: the call
returns normally and nothing is reported.

A warning that a test ignores or captures itself (``-W ignore``, the
``filterwarnings`` mark, ``pytest.warns``, ``warnings.catch_warnings``) is
not recorded by pytest and so is not counted.  Neither is a warning whose
filename is not an absolute path (``<string>``, ``<frozen ...>``).
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import NamedTuple

import pytest

import mktlib

# Importing ``mktlib`` here is covered too: pytest records a warning raised
# while it loads this file and reports it to the hook below with
# ``when="config"``.
_OWN_ROOTS = (
    Path(mktlib.__file__).resolve().parent,
    Path(__file__).resolve().parent,
)


class _OwnDeprecation(NamedTuple):
    when: str
    nodeid: str
    filename: str
    lineno: int
    message: str


_own_deprecations: list[_OwnDeprecation] = []


def _is_own(filename: str) -> bool:
    if not os.path.isabs(filename):
        return False
    try:
        path = Path(filename).resolve()
    except (OSError, ValueError):
        return False
    return any(path.is_relative_to(root) for root in _OWN_ROOTS)


def pytest_warning_recorded(
    warning_message: warnings.WarningMessage, when: str, nodeid: str
) -> None:
    if not issubclass(warning_message.category, DeprecationWarning):
        return
    if not _is_own(warning_message.filename):
        return
    _own_deprecations.append(
        _OwnDeprecation(
            when,
            nodeid,
            warning_message.filename,
            warning_message.lineno,
            str(warning_message.message),
        )
    )


def pytest_terminal_summary(
    terminalreporter: pytest.TerminalReporter,
) -> None:
    if not _own_deprecations:
        return
    terminalreporter.section(
        "DeprecationWarning attributed to mktlib or its tests",
        red=True,
        bold=True,
    )
    for d in _own_deprecations:
        where = d.nodeid or "-"
        terminalreporter.write_line(
            f"[{d.when}] {where}: {d.filename}:{d.lineno}: {d.message}"
        )
    terminalreporter.write_line(
        f"{len(_own_deprecations)} own deprecation(s): the run fails."
    )


def pytest_sessionfinish(session: pytest.Session) -> None:
    if _own_deprecations and session.exitstatus == pytest.ExitCode.OK:
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
