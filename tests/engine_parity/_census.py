"""What the library's source calls, read from its syntax tree.

The order fixtures of :mod:`tests.engine_parity._chaos` perturb calls of
the eager polars API as they run.  They say nothing about a call no test
reaches, and nothing about a spelling they do not patch.  The functions
here enumerate the calls in ``mktlib/`` from the source, so that a test
can compare what is written with what was reached.

A site is ``(file, line, kind)``.  ``line`` is the line of the method
name, which is the line a frame reports while that call runs.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import mktlib
from tests.engine_parity._chaos import GROUP_BY_TERMINALS

LIBRARY_ROOT = Path(mktlib.__file__).resolve().parent


@dataclass(frozen=True, order=True)
class Call:
    """One method call in the library's source."""

    file: str
    line: int
    kind: str
    symbol: str

    @property
    def site(self) -> tuple[str, int, str]:
        return (self.file, self.line, self.kind)

    @property
    def where(self) -> str:
        return f"{self.file}::{self.symbol}"


def _modules() -> Iterator[tuple[str, ast.Module]]:
    for path in sorted(LIBRARY_ROOT.rglob("*.py")):
        module = path.relative_to(LIBRARY_ROOT.parent).as_posix()
        yield module, ast.parse(path.read_text(encoding="utf-8"))


def _method_calls(
    node: ast.AST, symbol: str = "<module>"
) -> Iterator[tuple[str, ast.Call, ast.Attribute]]:
    """Every ``<expr>.<name>(...)`` under *node*, with its enclosing symbol."""
    for child in ast.iter_child_nodes(node):
        inner = symbol
        if isinstance(
            child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            inner = child.name
            if symbol != "<module>":
                inner = f"{symbol}.{child.name}"
        if isinstance(child, ast.Call) and isinstance(
            child.func, ast.Attribute
        ):
            yield inner, child, child.func
        yield from _method_calls(child, inner)


def _keyword(call: ast.Call, name: str) -> ast.expr | None:
    for keyword in call.keywords:
        if keyword.arg == name:
            return keyword.value
    return None


def _line(func: ast.Attribute) -> int:
    assert func.end_lineno is not None
    return func.end_lineno


def _is_frame_join(call: ast.Call) -> bool:
    """A frame ``.join(...)``: a join keyword, or the key given positionally.

    ``str.join`` takes one argument and none of these keywords.
    """
    return len(call.args) >= 2 or any(
        _keyword(call, name) is not None
        for name in ("on", "how", "left_on", "right_on", "maintain_order")
    )


def _pins_join_order(call: ast.Call) -> bool:
    value = _keyword(call, "maintain_order")
    if value is None:
        return False
    unpinned = isinstance(value, ast.Constant) and value.value in (None, "none")
    return not unpinned


def _pins_true(call: ast.Call) -> bool:
    value = _keyword(call, "maintain_order")
    return isinstance(value, ast.Constant) and value.value is True


def unpinned_order_calls() -> set[Call]:
    """Every join and ``group_by`` terminal that does not pin its row order.

    A ``group_by`` is recorded at its terminal (``agg``, ``len``, ...),
    which is where the perturbation acts.  A ``group_by`` whose result is
    not consumed by a terminal in the same expression cannot be placed and
    is an error.
    """
    found: set[Call] = set()
    for module, tree in _modules():
        consumed: set[int] = set()
        group_bys: dict[int, tuple[str, ast.Attribute]] = {}
        for symbol, call, func in _method_calls(tree):
            if func.attr == "join" and _is_frame_join(call):
                if not _pins_join_order(call):
                    found.add(Call(module, _line(func), "join", symbol))
            elif func.attr == "group_by":
                group_bys[id(call)] = (symbol, func)
            elif func.attr in GROUP_BY_TERMINALS:
                receiver = func.value
                if (
                    isinstance(receiver, ast.Call)
                    and isinstance(receiver.func, ast.Attribute)
                    and receiver.func.attr == "group_by"
                ):
                    consumed.add(id(receiver))
                    if not _pins_true(receiver):
                        kind = f"group_by.{func.attr}"
                        found.add(Call(module, _line(func), kind, symbol))
        loose = sorted(
            f"{module}:{_line(func)} ({symbol})"
            for key, (symbol, func) in group_bys.items()
            if key not in consumed
        )
        assert not loose, (
            f"a group_by with no terminal in the same expression: {loose}; "
            "the census cannot tell where its rows are produced"
        )
    return found


def sort_calls() -> set[Call]:
    """Every ``.sort(...)`` and ``.sort_by(...)`` call, pinned or not."""
    return {
        Call(module, _line(func), func.attr, symbol)
        for module, tree in _modules()
        for symbol, _call, func in _method_calls(tree)
        if func.attr in ("sort", "sort_by")
    }


#: Method and function names that build or run a lazy query.
_LAZY_ATTRIBUTES = frozenset(
    {
        "lazy",
        "collect",
        "collect_all",
        "collect_all_async",
        "collect_async",
        "collect_batches",
        "collect_schema",
        "sink_batches",
        "sink_csv",
        "sink_ipc",
        "sink_ndjson",
        "sink_parquet",
    }
)


def _lazy_kind(name: str) -> str | None:
    if name in _LAZY_ATTRIBUTES or name == "LazyFrame":
        return name
    if name.startswith("scan_"):
        return name
    return None


def lazy_calls() -> set[Call]:
    """Every call that builds, scans into or collects a ``LazyFrame``.

    By name: ``.lazy()``, ``.collect*()``, ``.sink_*()``, ``scan_*()``,
    ``collect_all()`` and ``LazyFrame(...)``, as an attribute or as a bare
    imported name.
    """
    found: set[Call] = set()
    for module, tree in _modules():
        for symbol, call, func in _method_calls(tree):
            kind = _lazy_kind(func.attr)
            if kind is not None:
                found.add(Call(module, _line(func), kind, symbol))
        for symbol, call in _name_calls(tree):
            assert isinstance(call.func, ast.Name)
            kind = _lazy_kind(call.func.id)
            if kind is not None:
                found.add(Call(module, call.lineno, kind, symbol))
    return found


def _name_calls(
    node: ast.AST, symbol: str = "<module>"
) -> Iterator[tuple[str, ast.Call]]:
    """Every ``<name>(...)`` under *node*, with its enclosing symbol."""
    for child in ast.iter_child_nodes(node):
        inner = symbol
        if isinstance(
            child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            inner = child.name
            if symbol != "<module>":
                inner = f"{symbol}.{child.name}"
        if isinstance(child, ast.Call) and isinstance(child.func, ast.Name):
            yield inner, child
        yield from _name_calls(child, inner)
