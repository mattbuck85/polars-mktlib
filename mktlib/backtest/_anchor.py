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

from mktlib.backtest._backend import active_scan_backend, native_module
from mktlib.backtest._scan import ScanLeg, scan_realized
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
        case Limit():
            # Without this the engine creates no snapshot column for a
            # Limit-wrapped EntryRef exit, and resolving it raises
            # ColumnNotFoundError on `_entry_*`. Both are documented public API
            # and the combination had no test.
            _walk_cond(cond.inner, cols)
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


#: Starting look-ahead when searching for a candidate's exit bar. Doubled until
#: the exit is found, so a long-held position costs log(span) evaluations rather
#: than a full-frame scan per candidate.
_WINDOW = 256


def _first_exit_after(
    frame: pl.DataFrame,
    candidate: int,
    *,
    exit_expr: pl.Expr,
    snapshot_cols: set[str],
    entry: list[bool],
    session_last: list[bool] | None,
) -> int | None:
    """First bar after *candidate* where the exit fires and takes effect.

    "Takes effect" is doing real work: the position recurrence gives ``entry``
    priority, so an exit sharing a bar with an entry signal does not close the
    position. Such a bar is skipped, and the trade continues on the same anchor.

    The anchor is pinned by substituting each snapshot column with the constant
    the candidate bar would have latched — which is what makes this evaluable
    without knowing the position.
    """
    n = frame.height
    anchors = {col: frame[col][candidate] for col in snapshot_cols}
    start = candidate + 1
    width = _WINDOW
    while start < n:
        stop = min(start + width, n)
        window = frame.slice(start, stop - start).with_columns(
            [
                pl.lit(value).alias(f"_entry_{col}")
                for col, value in anchors.items()
            ]
        )
        fired = window.select(exit_expr.alias("_f"))["_f"].fill_null(False).to_list()
        for offset, hit in enumerate(fired):
            bar = start + offset
            forced = session_last is not None and session_last[bar]
            opens = entry[bar] and not forced
            if (hit or forced) and not opens:
                return bar
        if stop >= n:
            return None
        width *= 2
    return None


def _realized_entries_windowed(
    frame: pl.DataFrame,
    *,
    entry_col: str,
    exit_cond: Condition,
    snapshot_cols: set[str],
    session_last_col: str | None = None,
) -> pl.Series:
    """Which ``_entry`` signals actually open a position — the general path.

    The orbit of the jump function described in the module docstring: the first
    signal opens, its exit is resolved with the anchor pinned at that bar, and
    the next signal after that exit is the next real entry.

    Correct for every exit tree, including ones :func:`plan_exit` refuses. It
    evaluates the user's own expression, so it cannot disagree with the engine
    about what the condition means — which is exactly why it stays as the
    fallback rather than being replaced.
    """
    n = frame.height
    entry = frame[entry_col].fill_null(False).to_list()  # noqa: FBT003
    session_last: list[bool] | None = (
        frame[session_last_col].to_list() if session_last_col else None
    )
    realized = [False] * n
    if not any(entry) or not snapshot_cols:
        return pl.Series(ANCHOR_ENTRY_COLUMN, realized, dtype=pl.Boolean)

    inner = exit_cond.inner if isinstance(exit_cond, Limit) else exit_cond
    exit_expr = inner.resolve()
    candidates = [i for i, fired in enumerate(entry) if fired]

    cursor = 0
    for candidate in candidates:
        if candidate < cursor:
            continue  # suppressed: a position is still open here
        if session_last is not None and session_last[candidate]:
            # Under flatten_eod a session-last entry opens nothing — it is
            # deferred to the next session, where its own signal is waiting.
            continue
        realized[candidate] = True
        closed = _first_exit_after(
            frame,
            candidate,
            exit_expr=exit_expr,
            snapshot_cols=snapshot_cols,
            entry=entry,
            session_last=session_last,
        )
        if closed is None:
            break  # open at the end of the frame; nothing after it can realize
        cursor = closed + 1

    return pl.Series(ANCHOR_ENTRY_COLUMN, realized, dtype=pl.Boolean)


#: Nulls become NaN on the way into the scan kernel: every NaN comparison is
#: false, which is exactly "this bar never fires", with no extra branch.
_NAN = float("nan")


@dataclass(frozen=True, slots=True)
class PlannedColumns:
    """An :class:`ExitPlan` evaluated over the frame, still as polars Series.

    The split from :func:`plan_arrays` exists because **eligibility and
    materialization are different jobs**, and only the second one is expensive.
    Everything that can refuse the fast path happens here; converting to Python
    lists happens afterwards and cannot change the answer. A resolver that walks
    the Series directly therefore accepts exactly the trees the list-walking one
    does, which is what ``test_plan_columns_decides_eligibility_alone`` pins.

    ``value_ids[i]`` identifies which value column leg *i* reads. Legs sharing an
    id read the same column — a take-profit and a stop-loss are usually both
    written against ``close`` — and the kernel's specialized pair loop uses that
    to halve its per-bar reads. In :func:`plan_arrays` the sharing is carried by
    Python object identity (one list handed to both legs); an id is the same fact
    stated explicitly, so it survives being handed to a resolver that cannot see
    object identity.
    """

    values: tuple[pl.Series, ...]
    thresholds: tuple[pl.Series, ...]
    value_ids: tuple[int, ...]
    directions: tuple[Literal["above", "below"], ...]
    stricts: tuple[bool, ...]
    fixed: pl.Series | None


def _plan_columns(frame: pl.DataFrame, plan: ExitPlan) -> PlannedColumns | None:
    """Evaluate an :class:`ExitPlan` over *frame*, or refuse the fast path.

    **This function alone decides eligibility.** Every expression is evaluated
    once over the whole frame, which is sound because a leg's threshold reads
    only snapshots and literals, so its value at row *j* is exactly the constant
    the windowed resolver pins for a candidate entering at *j*.

    Nulls and NaNs are normalized to NaN here rather than at the point of use.
    NaN is the "never fires" sentinel: every comparison against it is false in
    both directions and at both strictness settings, with no per-bar branch.
    (Infinity would not do — ``+inf >= +inf`` is true, so a real infinity in the
    data would tie against the sentinel.)
    """
    if not plan.eligible or not plan.legs:
        return None

    exprs: list[pl.Expr] = []
    for i, leg in enumerate(plan.legs):
        exprs.append(leg.value.alias(f"__scan_v{i}"))
        exprs.append(leg.threshold.alias(f"__scan_t{i}"))
    if plan.fixed is not None:
        exprs.append(plan.fixed.alias("__scan_fixed"))

    try:
        out = frame.select(exprs)
    except pl.exceptions.PolarsError:
        return None

    # Integers above 2**53 do not survive the cast to float, so refuse rather
    # than resolve a level that is quietly off by one.
    for i in range(len(plan.legs)):
        for prefix in ("__scan_v", "__scan_t"):
            if not out[f"{prefix}{i}"].dtype.is_float():
                return None

    # Legs reading the same value column get the same id. Keyed on the rendered
    # expression, which is what makes two legs written against `close` share.
    seen: dict[str, int] = {}
    value_ids: list[int] = []
    values: list[pl.Series] = []
    for i, leg in enumerate(plan.legs):
        key = str(leg.value)
        index = seen.get(key)
        if index is None:
            index = len(values)
            seen[key] = index
            values.append(
                out[f"__scan_v{i}"].fill_null(_NAN).fill_nan(_NAN)
            )
        value_ids.append(index)

    return PlannedColumns(
        values=tuple(values),
        thresholds=tuple(
            out[f"__scan_t{i}"].fill_null(_NAN).fill_nan(_NAN)
            for i in range(len(plan.legs))
        ),
        value_ids=tuple(value_ids),
        directions=tuple(leg.direction for leg in plan.legs),
        stricts=tuple(leg.strict for leg in plan.legs),
        fixed=(
            out["__scan_fixed"].fill_null(False)  # noqa: FBT003
            if plan.fixed is not None
            else None
        ),
    )


def plan_arrays(
    frame: pl.DataFrame, plan: ExitPlan
) -> tuple[tuple[ScanLeg, ...], list[bool] | None] | None:
    """Materialize an :class:`ExitPlan` into arrays the scan kernel can walk.

    Returns ``None`` when the fast path does not apply — which is also the
    predicate the equivalence tests partition on, so production dispatch and the
    test corpus cannot drift apart. The refusal is entirely
    :func:`_plan_columns`'s; this function only converts.

    Legs that read the same column share **one list object**, because the
    kernel's specialized pair loop keys on that identity to halve its per-bar
    reads — handing it two equal-but-distinct lists would silently cost the
    optimization. ``PlannedColumns.value_ids`` is the same fact stated without
    relying on identity.
    """
    columns = _plan_columns(frame, plan)
    if columns is None:
        return None
    return _arrays_from_columns(columns)


def _arrays_from_columns(
    columns: PlannedColumns,
) -> tuple[tuple[ScanLeg, ...], list[bool] | None]:
    """Box every column into Python lists for the pure-Python kernel.

    This is the expensive half of the old ``plan_arrays`` — measured at 71.0 ms
    of a 108.9 ms resolver at 500k bars, against 0.4 ms for the polars evaluation
    that produced the columns. The native backend skips this entirely and hands
    the Series across the FFI instead, which is where nearly all of its advantage
    comes from.
    """
    shared = [series.to_list() for series in columns.values]
    legs = tuple(
        ScanLeg(
            value=shared[columns.value_ids[i]],
            threshold=columns.thresholds[i].to_list(),
            direction=columns.directions[i],
            strict=columns.stricts[i],
        )
        for i in range(len(columns.thresholds))
    )
    fixed = columns.fixed.to_list() if columns.fixed is not None else None
    return legs, fixed


def _realized_entries_fast(
    frame: pl.DataFrame,
    *,
    entry_col: str,
    arrays: tuple[tuple[ScanLeg, ...], list[bool] | None],
    session_last_col: str | None,
) -> pl.Series:
    """Resolve the chain with one forward pass. See :mod:`._scan`."""
    legs, fixed = arrays
    entry = frame[entry_col].fill_null(False).to_list()  # noqa: FBT003
    # `fill_null` here matches `entry` and `fixed`. Without it a null arrives as
    # `None`, which Python happens to treat as falsy — the right answer, reached
    # by accident. Stated explicitly, the "null means not a session boundary"
    # rule holds for any resolver, including one that cannot rely on Python
    # truthiness.
    session_last = (
        frame[session_last_col].fill_null(False).to_list()  # noqa: FBT003
        if session_last_col
        else None
    )
    result = scan_realized(
        entry=entry,
        legs=legs,
        fixed=fixed,
        session_last=session_last,
        n=frame.height,
    )
    return pl.Series(ANCHOR_ENTRY_COLUMN, result.realized, dtype=pl.Boolean)


def _realized_entries_native(
    frame: pl.DataFrame,
    *,
    entry_col: str,
    columns: PlannedColumns,
    session_last_col: str | None,
) -> pl.Series:
    """Resolve the chain in the compiled kernel, with no Python list of floats.

    Deliberately the same shape as :func:`_realized_entries_fast` minus the
    boxing: the Series go straight across the FFI, which is where the speedup
    lives. ``value_ids`` carries the column sharing that the pure-Python kernel
    expresses as object identity and an FFI cannot.

    Output is required to be **bit-identical** to the Python path, not merely
    close — see ``tests/backtest/test_anchor_equivalence.py``, which runs the
    whole corpus and every frozen golden baseline against both.
    """
    module = native_module()
    realized, _entries, _exits, _winners = module.scan_realized(
        frame[entry_col].fill_null(False),  # noqa: FBT003
        list(columns.values),
        list(columns.thresholds),
        list(columns.value_ids),
        [d == "above" for d in columns.directions],
        list(columns.stricts),
        columns.fixed,
        frame[session_last_col].fill_null(False)  # noqa: FBT003
        if session_last_col
        else None,
        frame.height,
    )
    return pl.Series(ANCHOR_ENTRY_COLUMN, realized, dtype=pl.Boolean)


def realized_entries(
    frame: pl.DataFrame,
    *,
    entry_col: str,
    exit_cond: Condition,
    snapshot_cols: set[str],
    session_last_col: str | None = None,
) -> pl.Series:
    """Which ``_entry`` signals actually open a position.

    Two resolvers, one answer. When :func:`plan_exit` recognizes the exit tree
    as a set of levels pinned at entry, the chain resolves in a single forward
    pass; otherwise the user's own expression is evaluated over a doubling
    window, which is slower but correct for anything.

    The two paths are held to agreement by
    ``tests/backtest/test_anchor_equivalence.py``, which partitions its corpus
    using :func:`_plan_columns` — the same call dispatched on here — so widening
    eligibility later cannot quietly move results.

    When ``mktlib-scan`` is installed the single pass runs in a compiled kernel
    instead, on the same eligibility decision and with output required to be
    bit-identical. Pin either implementation with
    :func:`~mktlib.backtest.set_scan_backend`.
    """
    columns = _plan_columns(frame, plan_exit(exit_cond))
    if columns is not None:
        if active_scan_backend() == "native":
            return _realized_entries_native(
                frame,
                entry_col=entry_col,
                columns=columns,
                session_last_col=session_last_col,
            )
        return _realized_entries_fast(
            frame,
            entry_col=entry_col,
            arrays=_arrays_from_columns(columns),
            session_last_col=session_last_col,
        )
    return _realized_entries_windowed(
        frame,
        entry_col=entry_col,
        exit_cond=exit_cond,
        snapshot_cols=snapshot_cols,
        session_last_col=session_last_col,
    )
