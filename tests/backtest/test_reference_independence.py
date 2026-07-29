"""The reference backtester must not import the code it is checking.

An oracle that shares an implementation with the thing under test proves
nothing — it inherits the same bug and agrees enthusiastically. That property
is worth enforcing mechanically rather than by review, because the failure mode
is one convenient import added months from now to avoid re-deriving a rule.

Parsed with ``ast`` rather than by executing the module, so a module that is
merely *importable* cannot slip anything past this.
"""

from __future__ import annotations

import ast
from pathlib import Path

_REFERENCE = Path(__file__).parent / "_reference.py"
_FORBIDDEN_PREFIX = "mktlib.backtest"


def _imported_modules(tree: ast.AST) -> set[str]:
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
    return found


def test_reference_does_not_import_the_engine() -> None:
    tree = ast.parse(_REFERENCE.read_text())
    offenders = sorted(
        name
        for name in _imported_modules(tree)
        if name == _FORBIDDEN_PREFIX or name.startswith(_FORBIDDEN_PREFIX + ".")
    )
    assert not offenders, (
        f"_reference.py imports {offenders} — it is the oracle for that code and "
        "must be an independent implementation. Re-derive the rule from the docs "
        "instead of importing it."
    )


def test_reference_does_not_import_polars() -> None:
    """Data in, no expressions in.

    The reference takes plain Python lists so it can be read as an executable
    statement of the documented fill semantics. A polars import would let
    expression logic — the very thing under test — leak back in.
    """
    tree = ast.parse(_REFERENCE.read_text())
    assert "polars" not in _imported_modules(tree)


def test_the_lint_would_catch_an_offending_import() -> None:
    """Non-vacuity: the check must fail on a module that does import the engine."""
    tree = ast.parse(
        "from mktlib.backtest import run\nimport polars as pl\n"
    )
    modules = _imported_modules(tree)
    assert any(m.startswith(_FORBIDDEN_PREFIX) for m in modules)
    assert "polars" in modules
