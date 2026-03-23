"""Strategy artifact — deterministic fingerprint for mktlib strategies."""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import inspect
import textwrap
import types
from collections.abc import Callable
from typing import Any

from mktlib.backtest._types import InitStrategy, Strategy

from mktlib.backtest._conditions import (
    All,
    Any_,
    ColExpr,
    Condition,
    Crossover,
    Crossunder,
    Custom,
    EntryRef,
    IsFalling,
    IsRising,
    Not,
    Pct,
    ValueGT,
    ValueGTE,
    ValueLT,
    ValueLTE,
    Col,
    Lit,
    _BinOp,
)

# ---------------------------------------------------------------------------
# Alias registry — for user-defined wrapper classes
# ---------------------------------------------------------------------------

_ALIASES: dict[type, str] = {}


def register_alias(alias_cls: type, canonical_name: str) -> None:
    """Register a class as an alias for a canonical condition name.

    Use this when you create a wrapper class that should hash identically
    to its canonical condition.  Built-in aliases (``PriceIsAbove = ValueGT``,
    etc.) are handled automatically since they're the same class.
    """
    _ALIASES[alias_cls] = canonical_name


# ---------------------------------------------------------------------------
# Value normalization
# ---------------------------------------------------------------------------


def _hash_val(v: str | float | int | ColExpr) -> bytes:
    """Normalize a condition field value to canonical bytes.

    ``str`` -> ``Col(s)``, ``float/int`` -> ``Lit(v)``, ``ColExpr`` -> recurse.
    """
    if isinstance(v, ColExpr):
        return _hash_col_expr(v)
    if isinstance(v, str):
        return _hash_col_expr(Col(v))
    if isinstance(v, (int, float)):
        return _hash_col_expr(Lit(float(v)))
    return str(v).encode()


# ---------------------------------------------------------------------------
# ColExpr tree hashing
# ---------------------------------------------------------------------------


def _hash_col_expr(pe: ColExpr) -> bytes:
    """Recursively hash a ColExpr tree to canonical bytes."""
    match pe:
        case Col(name):
            return b"Col(" + name.encode() + b")"
        case Lit(value):
            return b"Lit(" + str(value).encode() + b")"
        case _BinOp(left, right, op):
            return b"BinOp(" + _hash_col_expr(left) + op.encode() + _hash_col_expr(right) + b")"
        case Pct(base, pct):
            return b"Pct(" + _hash_val(base) + b"," + str(pct).encode() + b")"
        case EntryRef(col):
            return b"EntryRef(" + col.encode() + b")"
        case _:
            return b"ColExpr(?)"


# ---------------------------------------------------------------------------
# Condition tree hashing
# ---------------------------------------------------------------------------


def _hash_condition(cond: Condition) -> bytes:
    """Recursively hash a Condition tree to canonical bytes.

    Uses canonical class names (not Python class names) so aliases
    hash identically.  ``trade_side`` is excluded — it's an execution
    config concern, not part of the signal logic.
    """
    # Check alias registry first
    canonical = _ALIASES.get(type(cond))
    if canonical is not None:
        # Alias: use canonical name, hash fields generically.
        # Registered aliases are expected to be dataclasses.
        fields = [
            f.name for f in dataclasses.fields(cond)  # type: ignore[arg-type]
            if f.name != "trade_side"
        ]
        parts = [canonical.encode(), b"("]
        for i, fname in enumerate(fields):
            if i > 0:
                parts.append(b",")
            parts.append(_hash_val(getattr(cond, fname)))
        parts.append(b")")
        return b"".join(parts)

    match cond:
        case All(left, right, _):
            return b"All(" + _hash_condition(left) + b"," + _hash_condition(right) + b")"
        case Any_(left, right, _):
            return b"Any(" + _hash_condition(left) + b"," + _hash_condition(right) + b")"
        case Not(inner, _):
            return b"Not(" + _hash_condition(inner) + b")"
        case ValueGT(a, b, _):
            return b"ValueGT(" + _hash_val(a) + b"," + _hash_val(b) + b")"
        case ValueGTE(a, b, _):
            return b"ValueGTE(" + _hash_val(a) + b"," + _hash_val(b) + b")"
        case ValueLT(a, b, _):
            return b"ValueLT(" + _hash_val(a) + b"," + _hash_val(b) + b")"
        case ValueLTE(a, b, _):
            return b"ValueLTE(" + _hash_val(a) + b"," + _hash_val(b) + b")"
        case Crossover(a, b, _):
            return b"Crossover(" + _hash_val(a) + b"," + _hash_val(b) + b")"
        case Crossunder(a, b, _):
            return b"Crossunder(" + _hash_val(a) + b"," + _hash_val(b) + b")"
        case IsRising(col, period, _):
            return b"IsRising(" + col.encode() + b"," + str(period).encode() + b")"
        case IsFalling(col, period, _):
            return b"IsFalling(" + col.encode() + b"," + str(period).encode() + b")"
        case Custom(expr, _):
            return b"Custom(" + expr.meta.serialize() + b")"
        case _:
            return b"Unknown(?)"


# ---------------------------------------------------------------------------
# Init hashing with same-module reference following
# ---------------------------------------------------------------------------


def _get_called_names(source: str) -> set[str]:
    """Extract function names called in *source* via AST walking."""
    try:
        tree = ast.parse(textwrap.dedent(source))
    except SyntaxError:
        return set()
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            names.add(node.func.id)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            # self.method() — skip, method is on the instance
            pass
    return names


def _get_source_with_refs(
    fn: Callable[..., Any],
    module: types.ModuleType | None,
    visited: set[str],
) -> bytes:
    """Get dedented source of *fn* and recursively follow same-module references."""
    try:
        source = textwrap.dedent(inspect.getsource(fn))
    except (OSError, TypeError):
        return b""

    fn_name: str = getattr(fn, "__name__", getattr(fn, "__func__", fn).__name__)
    if fn_name in visited:
        return b""
    visited.add(fn_name)

    parts = [source.encode()]

    # Follow function references within the same module
    if module is not None:
        for name in _get_called_names(source):
            ref = getattr(module, name, None)
            if ref is None or not callable(ref):
                continue
            # Only follow if defined in the same module
            ref_module = inspect.getmodule(ref)
            if ref_module is not module:
                continue
            parts.append(_get_source_with_refs(ref, module, visited))

    return b"".join(parts)


def _hash_init(strategy: Strategy) -> bytes:
    """Hash the init() method source + same-module helper references."""
    if not isinstance(strategy, InitStrategy):
        return b""
    init_fn = strategy.init
    module = inspect.getmodule(type(strategy))
    visited: set[str] = set()
    return _get_source_with_refs(init_fn, module, visited)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def strategy_artifact(strategy: Strategy) -> str:
    """Return a deterministic 16-char hex digest for a strategy instance.

    The hash is derived from:
    1. The strategy class name
    2. The entry/exit condition trees (recursive walk)
    3. The ``init()`` method source + same-module helper functions

    Aliases (e.g. ``PriceIsAbove = ValueGT``) hash identically since
    they're the same class.  ``Col("a") > 100`` and ``ValueGT("a", 100)``
    also hash identically via value normalization.
    """
    entry = strategy.entry()
    exit_ = strategy.exit()

    # Normalize bare pl.Expr to Custom
    if not isinstance(entry, Condition):
        entry = Custom(entry)
    if not isinstance(exit_, Condition):
        exit_ = Custom(exit_)

    parts = [
        type(strategy).__name__.encode(),
        _hash_condition(entry),
        _hash_condition(exit_),
        _hash_init(strategy),
    ]
    return hashlib.sha256(b"|".join(parts)).hexdigest()[:16]
