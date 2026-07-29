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

from mktlib.backtest._conditions import (
    All,
    Any_,
    ColExpr,
    Condition,
    EntryRef,
    Not,
    Pct,
    ValueGT,
    ValueGTE,
    ValueLT,
    ValueLTE,
    _BinOp,
)

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
