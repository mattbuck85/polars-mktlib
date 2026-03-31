"""Strategy artifact — deterministic fingerprint for mktlib strategies."""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import inspect
import logging
import textwrap
import types
from collections.abc import Callable
from typing import Any

from mktlib.backtest._types import InitStrategy, Strategy

logger = logging.getLogger(__name__)

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


def _hash_val(v: str | float | int | ColExpr, *, flat: bool = False) -> bytes:
    """Normalize a condition field value to canonical bytes.

    ``str`` -> ``Col(s)``, ``float/int`` -> ``Lit(v)``, ``ColExpr`` -> recurse.
    When *flat* is True, all numeric literals hash as ``0``.
    """
    if isinstance(v, ColExpr):
        return _hash_col_expr(v, flat=flat)
    if isinstance(v, str):
        return _hash_col_expr(Col(v))
    if isinstance(v, (int, float)):
        return _hash_col_expr(Lit(float(v)), flat=flat)
    return str(v).encode()


# ---------------------------------------------------------------------------
# ColExpr tree hashing
# ---------------------------------------------------------------------------


def _hash_col_expr(pe: ColExpr, *, flat: bool = False) -> bytes:
    """Recursively hash a ColExpr tree to canonical bytes.

    When *flat* is True, ``Lit`` values and ``Pct.pct`` hash as ``"0"``.
    """
    match pe:
        case Col(name):
            return b"Col(" + name.encode() + b")"
        case Lit(value):
            v = "0" if flat else str(value)
            return b"Lit(" + v.encode() + b")"
        case _BinOp(left, right, op):
            return b"BinOp(" + _hash_col_expr(left, flat=flat) + op.encode() + _hash_col_expr(right, flat=flat) + b")"
        case Pct(base, pct):
            p = "0" if flat else str(pct)
            return b"Pct(" + _hash_val(base, flat=flat) + b"," + p.encode() + b")"
        case EntryRef(col):
            return b"EntryRef(" + col.encode() + b")"
        case _:
            return b"ColExpr(?)"


# ---------------------------------------------------------------------------
# Condition tree hashing
# ---------------------------------------------------------------------------


def _hash_condition(cond: Condition, *, flat: bool = False) -> bytes:
    """Recursively hash a Condition tree to canonical bytes.

    Uses canonical class names (not Python class names) so aliases
    hash identically.  ``trade_side`` is excluded — it's an execution
    config concern, not part of the signal logic.

    When *flat* is True, all numeric values (``Lit``, ``Pct.pct``,
    ``IsRising``/``IsFalling`` period) hash as ``"0"``, making the
    digest parameter-insensitive.
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
            parts.append(_hash_val(getattr(cond, fname), flat=flat))
        parts.append(b")")
        return b"".join(parts)

    match cond:
        case All(left, right, _):
            return b"All(" + _hash_condition(left, flat=flat) + b"," + _hash_condition(right, flat=flat) + b")"
        case Any_(left, right, _):
            return b"Any(" + _hash_condition(left, flat=flat) + b"," + _hash_condition(right, flat=flat) + b")"
        case Not(inner, _):
            return b"Not(" + _hash_condition(inner, flat=flat) + b")"
        case ValueGT(a, b, _):
            return b"ValueGT(" + _hash_val(a, flat=flat) + b"," + _hash_val(b, flat=flat) + b")"
        case ValueGTE(a, b, _):
            return b"ValueGTE(" + _hash_val(a, flat=flat) + b"," + _hash_val(b, flat=flat) + b")"
        case ValueLT(a, b, _):
            return b"ValueLT(" + _hash_val(a, flat=flat) + b"," + _hash_val(b, flat=flat) + b")"
        case ValueLTE(a, b, _):
            return b"ValueLTE(" + _hash_val(a, flat=flat) + b"," + _hash_val(b, flat=flat) + b")"
        case Crossover(a, b, _):
            return b"Crossover(" + _hash_val(a, flat=flat) + b"," + _hash_val(b, flat=flat) + b")"
        case Crossunder(a, b, _):
            return b"Crossunder(" + _hash_val(a, flat=flat) + b"," + _hash_val(b, flat=flat) + b")"
        case IsRising(col, period, _):
            p = "0" if flat else str(period)
            return b"IsRising(" + col.encode() + b"," + p.encode() + b")"
        case IsFalling(col, period, _):
            p = "0" if flat else str(period)
            return b"IsFalling(" + col.encode() + b"," + p.encode() + b")"
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


def _normalize_source(source: str) -> str:
    """Return a canonical, formatting-independent representation of *source*.

    Parses through ``ast`` and unparses back, stripping comments,
    normalizing whitespace/quotes, and producing a deterministic string
    across formatters and CPython versions.
    """
    try:
        return ast.unparse(ast.parse(textwrap.dedent(source)))
    except SyntaxError:
        return source


def _get_source_with_refs(
    fn: Callable[..., Any],
    module: types.ModuleType | None,
    visited: set[str],
) -> bytes:
    """Get normalized source of *fn* and recursively follow same-module references."""
    try:
        source = textwrap.dedent(inspect.getsource(fn))
    except (OSError, TypeError):
        return b""

    fn_name: str = getattr(fn, "__name__", getattr(fn, "__func__", fn).__name__)
    if fn_name in visited:
        return b""
    visited.add(fn_name)

    parts = [_normalize_source(source).encode()]

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


def combined_strategy_artifact(
    long_strategy: Strategy,
    short_strategy: Strategy,
) -> str:
    """Return a deterministic 16-char hex digest for a long+short strategy pair.

    Delegates to ``strategy_artifact`` for each side and hashes the
    pair. Order matters — swapping long/short changes the hash.
    """
    long_art = strategy_artifact(long_strategy)
    short_art = strategy_artifact(short_strategy)
    return hashlib.sha256(f"{long_art}|{short_art}".encode()).hexdigest()[:16]


def _warn_nondeterministic_branches(strategy_cls: type) -> None:
    """Warn if ``entry()``/``exit()`` contain ``if self.*`` branches.

    Conditional branches on instance attributes produce different condition
    trees for different parameter values, making the artifact hash
    parameter-dependent.  The optimizer cache key uses the artifact, so
    parameter combos that change the tree structure will collide with
    those that don't.

    Fix: replace ``if self.flag: cond = cond & X`` with an always-present
    condition using a no-op threshold (e.g., ``float('inf')``).
    """
    for method_name in ("entry", "exit"):
        method = getattr(strategy_cls, method_name, None)
        if method is None:
            continue
        try:
            source = textwrap.dedent(inspect.getsource(method))
            tree = ast.parse(source)
        except (OSError, TypeError, SyntaxError):
            continue

        for node in ast.walk(tree):
            if not isinstance(node, ast.If):
                continue
            for sub in ast.walk(node.test):
                if (
                    isinstance(sub, ast.Attribute)
                    and isinstance(sub.value, ast.Name)
                    and sub.value.id == "self"
                ):
                    logger.warning(
                        "Strategy %s.%s() contains 'if self.%s' branch. "
                        "This makes the artifact hash parameter-dependent — "
                        "different parameter values may produce different condition "
                        "trees, causing cache key collisions. Use a no-op threshold "
                        "(e.g., float('inf')) instead of conditional branching.",
                        strategy_cls.__name__,
                        method_name,
                        sub.attr,
                    )
                    return  # one warning per strategy is enough


def strategy_artifact(strategy: Strategy) -> str:
    """Return a deterministic 16-char hex digest for a strategy instance.

    The hash is derived from:
    1. The strategy class name
    2. The evaluated condition trees from ``entry()`` / ``exit()``,
       with all ``Lit`` values and numeric fields flattened to ``0``
    3. The ``init()`` method source + same-module helper functions

    Flattening literals makes the digest **parameter-insensitive** —
    stable across different parameter values for the same strategy class.
    """
    _warn_nondeterministic_branches(type(strategy))
    entry = strategy.entry()
    exit_ = strategy.exit()
    if not isinstance(entry, Condition):
        entry = Custom(entry)
    if not isinstance(exit_, Condition):
        exit_ = Custom(exit_)
    parts = [
        type(strategy).__name__.encode(),
        _hash_condition(entry, flat=True),
        _hash_condition(exit_, flat=True),
        _hash_init(strategy),
    ]
    return hashlib.sha256(b"|".join(parts)).hexdigest()[:16]
