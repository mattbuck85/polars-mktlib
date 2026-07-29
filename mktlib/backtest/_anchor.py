"""Entry-bar anchoring: which entry signals actually open a position.

``EntryRef`` snapshots a column "at the entry bar" and forward-fills it. The
subtlety, and the reason this module exists, is *which* entry bar.

An entry **signal** is not the same thing as an entry. The position machinery
suppresses a signal that fires while a position is already open — it cannot open
a second one — but a naive ``forward_fill`` from every raw signal does not, so
the anchor moves mid-trade and the exit ends up measured against a bar the trade
did not begin on. For a take-profit / stop-loss written against ``EntryRef`` that
silently converts a fixed bracket into a ratcheting trailing one.

Resolving it looks circular: the anchor decides the exit, the exit decides the
position, and the position decides which signals are real entries. The
circularity is broken by evaluating *candidates independently*:

    For a candidate entry bar ``j`` the anchor would be ``col[j]`` — known, with
    no position state involved. So the exit bar

        e(j) = min{ t > j : the exit fires with the anchor pinned at bar j }

    is computable for every ``j`` at once. The realized entries are then the
    orbit of a strictly increasing jump function: the first signal opens, the
    next real entry is the first signal at or after ``e(j)``, and so on.

Nothing here evaluates a user's exit condition twice against the same state, and
nothing needs the position column.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import polars as pl

from mktlib.backtest._conditions import (
    All,
    Any_,
    Col,
    ColExpr,
    Condition,
    Custom,
    EntryRef,
    Limit,
    Lit,
    Not,
    Pct,
    ValueGT,
    ValueGTE,
    ValueLT,
    ValueLTE,
    _BinOp,
)

#: Which side of a pinned level a crossing is looking for.
Direction = Literal["above", "below"]

#: Realized entries — the subset of ``_entry`` that actually opens a position.
#: Internal; dropped before the ``BacktestResult`` is handed back.
ANCHOR_ENTRY_COLUMN = "_entry_realized"


def collect_entry_refs(cond: Condition) -> set[str]:
    """Return all column names referenced by ``EntryRef`` nodes in *cond*."""
    cols: set[str] = set()
    _walk_cond(cond, cols)
    return cols


def _walk_cond(cond: Condition, cols: set[str]) -> None:
    match cond:
        case All(left, right, _) | Any_(left, right, _):
            _walk_cond(left, cols)
            _walk_cond(right, cols)
        case Not(inner, _):
            _walk_cond(inner, cols)
        case ValueGT(a, b, _) | ValueGTE(a, b, _) | ValueLT(a, b, _) | ValueLTE(a, b, _):
            _walk_expr(a, cols)
            _walk_expr(b, cols)
        case _:
            pass


def _walk_expr(node: str | float | ColExpr, cols: set[str]) -> None:
    match node:
        case EntryRef(col):
            cols.add(col)
        case Pct(base, _):
            _walk_expr(base, cols)
        case _BinOp(left, right, _):
            _walk_expr(left, cols)
            _walk_expr(right, cols)
        case _:
            pass


@dataclass(frozen=True, slots=True)
class AnchoredLeg:
    """One exit term of the form "series crosses a level pinned at entry".

    ``value`` is read at the bar being tested; ``threshold`` is read once, at
    the candidate entry bar. The split is the whole point — it is what makes
    the crossing computable for every candidate independently.

    ``threshold`` resolves ``EntryRef("close")`` to the **raw** ``close``
    column, not the ``_entry_close`` snapshot, because it is evaluated at the
    candidate bar rather than forward-filled from one.
    """

    value: pl.Expr
    threshold: pl.Expr
    direction: Direction
    strict: bool


@dataclass(frozen=True, slots=True)
class ExitPlan:
    """How (and whether) an exit condition can be resolved per candidate."""

    legs: tuple[AnchoredLeg, ...]
    fixed: pl.Expr | None
    eligible: bool
    reason: str | None = None


def _flatten_or(cond: Condition) -> list[Condition]:
    """Split a tree on ``Any_`` only. ``All``/``Not`` stay whole."""
    if isinstance(cond, Any_):
        return _flatten_or(cond.left) + _flatten_or(cond.right)
    return [cond]


def _leaf_kinds(node: str | float | int | ColExpr) -> tuple[bool, bool]:
    """``(references an EntryRef, reads the bar under test)``.

    A node that only ever reads snapshots and literals is constant for a given
    candidate — which is the sole property the crossing search needs. A node
    containing a plain column reference moves as the bar advances and cannot be
    a per-candidate threshold, however much arithmetic surrounds it.
    """
    match node:
        case EntryRef():
            return (True, False)
        case Col() | str():
            return (False, True)
        case Lit() | float() | int():
            return (False, False)
        case Pct(base, _):
            return _leaf_kinds(base)
        case _BinOp(left, right, _):
            a, b = _leaf_kinds(left), _leaf_kinds(right)
            return (a[0] or b[0], a[1] or b[1])
        case _:
            # Unknown ColExpr subclass — assume the worst.
            return (False, True)


def _resolve_raw(node: str | float | int | ColExpr) -> pl.Expr:
    """Resolve with ``EntryRef(c)`` mapped to the RAW column ``c``.

    The threshold is evaluated at the candidate bar, so it must read ``close``,
    not the ``_entry_close`` forward-fill that only exists once entries are
    known.
    """
    match node:
        case EntryRef(col):
            return pl.col(col)
        case Col(name):
            return pl.col(name)
        case str():
            return pl.col(node)
        case Lit(value):
            return pl.lit(value)
        case float() | int():
            return pl.lit(node)
        case Pct(base, pct):
            return _resolve_raw(base) * (1.0 + pct / 100.0)
        case _BinOp(left, right, op):
            lhs, rhs = _resolve_raw(left), _resolve_raw(right)
            match op:
                case "+":
                    return lhs + rhs
                case "-":
                    return lhs - rhs
                case "*":
                    return lhs * rhs
                case "/":
                    return lhs / rhs
            msg = f"unsupported operator {op!r}"
            raise ValueError(msg)
        case _:
            msg = f"cannot resolve {node!r} at the candidate bar"
            raise TypeError(msg)


_COMPARISONS: dict[type, tuple[Direction, bool]] = {
    ValueGT: ("above", True),
    ValueGTE: ("above", False),
    ValueLT: ("below", True),
    ValueLTE: ("below", False),
}

_FLIP: dict[Direction, Direction] = {"above": "below", "below": "above"}


def _touches_snapshot(expr: pl.Expr) -> bool:
    """Does a raw polars expression read a ``_entry_*`` snapshot column?

    ``Custom`` bypasses the typed AST entirely, so the only way to see an
    anchored reference inside one is to inspect the expression's roots.
    """
    try:
        roots = expr.meta.root_names()
    except Exception:  # noqa: BLE001 - meta is best-effort on exotic exprs
        return True
    return any(name.startswith("_entry_") for name in roots)


def _plan_term(cond: Condition) -> tuple[AnchoredLeg | None, pl.Expr | None, str | None]:
    """Classify one OR-term: anchored leg, fixed predicate, or a refusal."""
    kind = _COMPARISONS.get(type(cond))
    if kind is not None:
        direction, strict = kind
        a_ref, a_moving = _leaf_kinds(cond.a)  # type: ignore[attr-defined]
        b_ref, b_moving = _leaf_kinds(cond.b)  # type: ignore[attr-defined]

        if not a_ref and not b_ref:
            return (None, cond.resolve(), None)

        if b_ref and not b_moving and not a_ref:
            # value <op> threshold — the common orientation.
            return (
                AnchoredLeg(
                    value=_resolve_raw(cond.a),  # type: ignore[attr-defined]
                    threshold=_resolve_raw(cond.b),  # type: ignore[attr-defined]
                    direction=direction,
                    strict=strict,
                ),
                None,
                None,
            )
        if a_ref and not a_moving and not b_ref:
            # threshold <op> value — same crossing, read the other way round.
            return (
                AnchoredLeg(
                    value=_resolve_raw(cond.b),  # type: ignore[attr-defined]
                    threshold=_resolve_raw(cond.a),  # type: ignore[attr-defined]
                    direction=_FLIP[direction],
                    strict=strict,
                ),
                None,
                None,
            )
        return (
            None,
            None,
            "an EntryRef term whose threshold also reads the current bar "
            "(e.g. EntryRef('close') - Col('atr') * 2) is a moving barrier, "
            "not a level pinned at entry",
        )

    if isinstance(cond, Custom):
        if _touches_snapshot(cond.resolve()):
            return (None, None, "a Custom expression reads an _entry_* snapshot")
        return (None, cond.resolve(), None)

    if collect_entry_refs(cond):
        return (
            None,
            None,
            f"{type(cond).__name__} combines an EntryRef with other terms; only "
            "a top-level OR of threshold crossings can be resolved per candidate",
        )
    return (None, cond.resolve(), None)


def plan_exit(cond: Condition) -> ExitPlan:
    """Decompose an exit condition into per-candidate crossings + a fixed part.

    Eligible shapes are a top-level OR of terms, where each term either pins its
    threshold at the entry bar (any arithmetic over ``EntryRef`` and literals —
    ``EntryRef('close') + EntryRef('atr') * 2`` qualifies) or does not mention
    ``EntryRef`` at all.

    Ineligibility is never an error. It selects a slower resolver that evaluates
    the user's own expression, so every shape stays correct.
    """
    inner = cond.inner if isinstance(cond, Limit) else cond

    legs: list[AnchoredLeg] = []
    fixed: list[pl.Expr] = []
    for term in _flatten_or(inner):
        leg, fixed_expr, reason = _plan_term(term)
        if reason is not None:
            return ExitPlan(legs=(), fixed=None, eligible=False, reason=reason)
        if leg is not None:
            legs.append(leg)
        if fixed_expr is not None:
            fixed.append(fixed_expr)

    if not legs:
        return ExitPlan(
            legs=(),
            fixed=None,
            eligible=False,
            reason="no EntryRef-anchored term to resolve",
        )

    combined: pl.Expr | None = None
    for extra in fixed:
        combined = extra if combined is None else combined | extra
    return ExitPlan(legs=tuple(legs), fixed=combined, eligible=True)
