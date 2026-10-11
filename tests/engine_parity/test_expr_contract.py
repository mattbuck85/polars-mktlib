"""Expression builders give the same column on every polars engine.

The library hands polars expressions to its own eager frames, and a caller
may place the same expressions in a lazy query of their own.  Each builder
is evaluated here through ``LazyFrame.with_columns(...).collect()`` on the
in-memory engine and on the streaming engine, over a frame held in one
chunk and over the same frame in 7-row chunks.  The four columns must be
identical, and identical to the eager one.

**Census.**  The builders are enumerated from the source, not listed by
hand: every function, method or nested function under ``mktlib/`` whose
return annotation mentions ``Expr`` in any spelling (``pl.Expr``,
``pl.Expr | None``, ``list[pl.Expr]``, a tuple holding one, a string
annotation).  Each is either in :data:`CASES`, which evaluates the
expressions it returns, or in :data:`EXEMPT` with the reason it cannot be
evaluated on its own.  A builder in neither fails
``test_every_builder_has_a_case``.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Callable
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

import mktlib
from mktlib.backtest import _anchor, _bracket, _conditions, _cost
from mktlib.backtest._conditions import (
    All,
    Any_,
    Col,
    ColExpr,
    Condition,
    Crossover,
    Crossunder,
    Custom,
    EntryRef,
    IsFalling,
    IsRising,
    Limit,
    Lit,
    Not,
    Pct,
    ValueGT,
    ValueGTE,
    ValueLT,
    ValueLTE,
)
from mktlib.data import _gbm, _ornstein_uhlenbeck, _random_walk
from tests.backtest import test_golden_baseline as golden
from tests.engine_parity import _corpus

_ROOT = Path(mktlib.__file__).resolve().parent

#: The number of builders the census found when this file was last edited.
#: The census may grow; it may not silently shrink.
CENSUS_FLOOR = 36

N_ROWS = 2000


# ---------------------------------------------------------------------------
# Census
# ---------------------------------------------------------------------------


_EXPR = re.compile(r"\bExpr\b")


def _annotation_text(annotation: ast.expr) -> str:
    """The annotation as written, with every string inside it spelled out."""
    strings = [
        node.value
        for node in ast.walk(annotation)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]
    return " ".join([ast.unparse(annotation), *strings])


def _returns_expr(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Does the return annotation mention ``Expr``, in any spelling?"""
    if node.returns is None:
        return False
    return _EXPR.search(_annotation_text(node.returns)) is not None


def _builders(node: ast.AST, prefix: str = "") -> set[str]:
    """Qualified names of the builders under *node*, at any depth."""
    found: set[str] = set()
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            name = prefix + child.name
            if _returns_expr(child):
                found.add(name)
            found |= _builders(child, f"{name}.<locals>.")
        elif isinstance(child, ast.ClassDef):
            found |= _builders(child, f"{prefix}{child.name}.")
        else:
            found |= _builders(child, prefix)
    return found


def census_of(source: str, module: str) -> set[str]:
    """``<module path>::<qualified name>`` of the builders in *source*."""
    return {f"{module}::{name}" for name in _builders(ast.parse(source))}


def builder_census() -> set[str]:
    """``<module path>::<qualified name>`` of every expression builder."""
    found: set[str] = set()
    for path in sorted(_ROOT.rglob("*.py")):
        module = path.relative_to(_ROOT.parent).as_posix()
        found |= census_of(path.read_text(encoding="utf-8"), module)
    return found


# ---------------------------------------------------------------------------
# The frame the expressions are evaluated on
# ---------------------------------------------------------------------------


def _frame() -> pl.DataFrame:
    n = N_ROWS
    opens, highs, lows, closes = golden._ohlc(n, seed=20260901)
    u = golden._uniforms(4 * n, 77)
    cov_row = _random_walk._build_covariance_row(n // 2, 0.7)
    assert len(cov_row) == n
    return pl.DataFrame(
        {
            "step": pl.int_range(n, eager=True),
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "fast": golden._rolling_mean(closes, 3),
            "slow": golden._rolling_mean(closes, 8),
            # Alternately above and below the open, so that a bar's range
            # reaches it on some rows and not on others.
            "level": [
                round(o * (1.004 if i % 2 else 0.996), 6) for i, o in enumerate(opens)
            ],
            "_entry_close": [closes[i - i % 50] for i in range(n)],
            "entry_fill": [opens[i - i % 50] for i in range(n)],
            "entry_clean": [i % 50 == 3 for i in range(n)],
            "resignal": [i % 50 == 20 for i in range(n)],
            "slip": [x * 2.0 for x in u[:n]],
            "z": [x - 0.5 for x in u[n : 2 * n]],
            "z_re": [x - 0.5 for x in u[2 * n : 3 * n]],
            "z_im": [x - 0.5 for x in u[3 * n :]],
            "cov_row": cov_row,
        }
    )


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------

_GT = ValueGT("close", "slow")
_LT = ValueLT(Col("close"), Pct(EntryRef("close"), -0.5))

_ARITHMETIC: list[ColExpr] = [
    Col("close") + Col("open"),
    Col("close") - 1.5,
    2.0 * Col("close"),
    Col("high") / Col("low"),
    Col("close") % 7,
    -Col("close"),
    100.0 - Col("close"),
    1.0 / Col("close"),
    0.5 + Col("close"),
    250 % Col("close"),
]


def _conditions_of(*conds: Condition) -> list[pl.Expr]:
    return [c.resolve() for c in conds]


def _bracket_levels() -> list[pl.Expr]:
    out: list[pl.Expr] = []
    for leg in (_bracket.TAKE_PROFIT, _bracket.STOP_LOSS):
        for is_long in (True, False):
            common = dict(
                leg=leg,
                is_long=is_long,
                entry_clean_col="entry_clean",
                entry_fill_col="entry_fill",
            )
            out.append(_bracket.level_expr(0.02, **common))  # type: ignore[arg-type]
            out.append(_bracket.level_expr("level", **common))  # type: ignore[arg-type]
            out.append(
                _bracket.level_expr("level", resignal_col="resignal", **common)  # type: ignore[arg-type]
            )
    return out


def _per_leg(builder: Callable[..., pl.Expr]) -> list[pl.Expr]:
    return [
        builder(leg=leg, is_long=is_long, level_col="level")
        for leg in (_bracket.TAKE_PROFIT, _bracket.STOP_LOSS)
        for is_long in (True, False)
    ]


def _plan_term_exprs() -> list[pl.Expr]:
    """The expressions ``_plan_term`` hands back, for each kind of term."""
    out: list[pl.Expr] = []
    kinds: set[str] = set()
    for cond in (
        _GT,
        ValueLT("close", EntryRef("close")),
        ValueGTE(EntryRef("close"), Pct("open", 0.1)),
        ValueLTE("low", Pct(EntryRef("close"), -1.0)),
        Custom(pl.col("close").rolling_mean(5) > pl.col("slow")),
    ):
        leg, fixed, refusal = _anchor._plan_term(cond)
        assert refusal is None, refusal
        if leg is not None:
            kinds.add("leg")
            out.extend([leg.value, leg.threshold])
        if fixed is not None:
            kinds.add("fixed")
            out.append(fixed)
    assert kinds == {"leg", "fixed"}
    return out


_C = "mktlib/backtest/_conditions.py::"

#: Census key -> the expressions that exercise that builder.  An empty list
#: is a builder with nothing to evaluate (an abstract method).
CASES: dict[str, Callable[[], list[pl.Expr]]] = {
    _C + "_ref": lambda: [
        _conditions._ref("close"),
        _conditions._ref(1.25),
        _conditions._ref(Pct("close", 1.0)),
    ],
    _C + "ColExpr.resolve": lambda: [],
    _C + "Condition.resolve": lambda: [],
    _C + "Col.resolve": lambda: [Col("close").resolve()],
    _C + "Lit.resolve": lambda: [Lit(3.5).resolve()],
    _C + "_BinOp.resolve": lambda: [e.resolve() for e in _ARITHMETIC],
    _C + "Pct.resolve": lambda: [
        Pct("close", 1.0).resolve(),
        Pct(Col("open"), -0.5).resolve(),
        Pct(100.0, 2.0).resolve(),
        Pct(EntryRef("close"), 0.75).resolve(),
    ],
    _C + "EntryRef.resolve": lambda: [EntryRef("close").resolve()],
    _C + "Crossover.resolve": lambda: _conditions_of(
        Crossover("fast", "slow"), Crossover("close", 100.0)
    ),
    _C + "Crossunder.resolve": lambda: _conditions_of(
        Crossunder("fast", "slow"), Crossunder("close", 100.0)
    ),
    _C + "ValueGT.resolve": lambda: _conditions_of(
        _GT, ValueGT(Col("high"), Pct("open", 0.2)), Col("close") > 100
    ),
    _C + "ValueGTE.resolve": lambda: _conditions_of(
        ValueGTE("high", "level"), Col("close") >= Col("open")
    ),
    _C + "ValueLT.resolve": lambda: _conditions_of(_LT, Col("close") < "slow"),
    _C + "ValueLTE.resolve": lambda: _conditions_of(
        ValueLTE("low", 99.0), Col("low") <= Col("level") * 0.99
    ),
    _C + "IsRising.resolve": lambda: _conditions_of(
        IsRising("close"), IsRising("close", 5)
    ),
    _C + "IsFalling.resolve": lambda: _conditions_of(
        IsFalling("close"), IsFalling("close", 5)
    ),
    _C + "Custom.resolve": lambda: _conditions_of(
        Custom(pl.col("close").rolling_mean(5) > pl.col("slow")),
        Custom(pl.col("close").diff().fill_null(0.0).cum_sum() > 0),
    ),
    _C + "All.resolve": lambda: _conditions_of(All(_GT, _LT), _GT & IsRising("fast")),
    _C + "Any_.resolve": lambda: _conditions_of(Any_(_GT, _LT), _GT | _LT),
    _C + "Not.resolve": lambda: _conditions_of(Not(_GT), ~Crossover("fast", "slow")),
    _C + "Limit.resolve": lambda: _conditions_of(
        Limit(ValueGTE("high", "level")), Limit(_LT, price="open")
    ),
    _C + "Limit.resolve_price": lambda: [
        Limit(ValueGTE("high", "level")).resolve_price(),
        Limit(ValueGTE("high", Pct(EntryRef("close"), 1.0))).resolve_price(),
        Limit(_GT, price="open").resolve_price(),
        Limit(_GT, price=Col("open") * 1.001).resolve_price(),
        Limit(_GT, price=101.5).resolve_price(),
    ],
    "mktlib/backtest/_anchor.py::_resolve_raw": lambda: [
        _anchor._resolve_raw(EntryRef("close")),
        _anchor._resolve_raw("close"),
        _anchor._resolve_raw(2.5),
        _anchor._resolve_raw(Pct(EntryRef("close"), 1.0)),
        *[_anchor._resolve_raw(e) for e in _ARITHMETIC if "%" not in repr(e)],
    ],
    "mktlib/backtest/_anchor.py::_plan_term": lambda: _plan_term_exprs(),
    "mktlib/backtest/_cost.py::cost_bps_expr": lambda: [
        _cost.cost_bps_expr(_cost.Cost()),
        _cost.cost_bps_expr(_cost.Cost(commission_bps=0.7, slippage_bps=0.3)),
        _cost.cost_bps_expr(_cost.Cost(commission_bps=0.7, slippage_col="slip")),
    ],
    "mktlib/backtest/_bracket.py::level_expr": _bracket_levels,
    "mktlib/backtest/_bracket.py::trigger_expr": lambda: _per_leg(
        _bracket.trigger_expr
    ),
    "mktlib/backtest/_bracket.py::fill_expr": lambda: _per_leg(_bracket.fill_expr),
    "mktlib/data/_gbm.py::_gbm_price_expr": lambda: [
        _gbm._gbm_price_expr(4.6, 0.0002, 0.012)
    ],
    "mktlib/data/_ornstein_uhlenbeck.py::_ou_value_expr": lambda: [
        _ornstein_uhlenbeck._ou_value_expr(100.0, 0.9972, 0.28, 0.063)
    ],
    "mktlib/data/_random_walk.py::_sqrt_eigenvalue_expr": lambda: [
        _random_walk._sqrt_eigenvalue_expr()
    ],
    "mktlib/data/_random_walk.py::_frw_increments_expr": lambda: [
        _random_walk._sqrt_eigenvalue_expr(),
        _random_walk._frw_increments_expr(),
    ],
}

#: Builders that are abstract: the base-class method raises.
ABSTRACT = {_C + "ColExpr.resolve": ColExpr, _C + "Condition.resolve": Condition}


_RUN_CORE = "mktlib/backtest/_engine.py::_run_core.<locals>."
_STRATEGY = "mktlib/backtest/_types.py::Strategy."

#: Builders with no case, each with the reason.
EXEMPT: dict[str, str] = {
    _RUN_CORE + "_charge": (
        "a closure over the cost and columns of one run; reached through "
        "run() by every backtest scenario"
    ),
    _RUN_CORE + "_with_bracket": (
        "a closure over the bracket columns of one run; reached through "
        "run() by the bracket golden scenarios"
    ),
    _STRATEGY + "entry": "a Protocol method with no body",
    _STRATEGY + "exit": "a Protocol method with no body",
}


def test_every_builder_has_a_case() -> None:
    census = builder_census()
    covered = set(CASES) | set(EXEMPT)
    assert census - covered == set(), (
        "functions whose return annotation mentions Expr and that have "
        "neither a case in CASES nor a reason in EXEMPT"
    )
    assert covered - census == set(), "cases or exemptions for builders gone"
    assert set(CASES) & set(EXEMPT) == set()
    assert all(EXEMPT.values())
    assert len(census) >= CENSUS_FLOOR


_SPELLINGS = '''
def plain() -> pl.Expr: ...
def optional() -> pl.Expr | None: ...
def listed() -> list[pl.Expr]: ...
def quoted() -> "pl.Expr": ...
def quoted_inside() -> tuple[int, "polars.Expr | None"]: ...
def bare() -> Expr: ...
def outer() -> int:
    def inner() -> pl.Expr: ...
    return 0
class Holder:
    def method(self) -> dict[str, pl.Expr]: ...
    async def later(self) -> pl.Expr: ...
def frame() -> pl.DataFrame: ...
def node() -> ColExpr: ...
def unannotated(): ...
'''


def test_census_reads_every_spelling_of_the_annotation() -> None:
    assert census_of(_SPELLINGS, "m.py") == {
        "m.py::plain",
        "m.py::optional",
        "m.py::listed",
        "m.py::quoted",
        "m.py::quoted_inside",
        "m.py::bare",
        "m.py::outer.<locals>.inner",
        "m.py::Holder.method",
        "m.py::Holder.later",
    }


def test_census_finds_the_public_condition_classes() -> None:
    """Every ``ColExpr`` / ``Condition`` subclass is in the census by name."""

    def subclasses(cls: type) -> set[type]:
        direct = set(cls.__subclasses__())
        return direct | {s for d in direct for s in subclasses(d)}

    classes = {
        c
        for base in (ColExpr, Condition)
        for c in subclasses(base)
        if c.__module__ == _conditions.__name__
    }
    assert len(classes) >= 18
    missing = {c.__name__ for c in classes if f"{_C}{c.__name__}.resolve" not in CASES}
    assert missing == set()


@pytest.mark.parametrize("key", sorted(ABSTRACT))
def test_abstract_builder_raises(key: str) -> None:
    assert CASES[key]() == []
    with pytest.raises(NotImplementedError):
        ABSTRACT[key]().resolve()


def test_arithmetic_cases_cover_every_operator() -> None:
    ops = {e.op for e in _ARITHMETIC}  # type: ignore[attr-defined]
    assert ops == {"+", "-", "*", "/", "%"}


#: Output names a later expression of the same case reads.
_KEPT_NAMES = {"sqrt_eig"}


def _named(exprs: list[pl.Expr]) -> list[pl.Expr]:
    """Each expression under a name of its own, so none overwrites another."""
    out: list[pl.Expr] = []
    for i, expr in enumerate(exprs):
        name = expr.meta.output_name(raise_if_undetermined=False)
        out.append(expr if name in _KEPT_NAMES else expr.alias(f"out_{i}"))
    return out


def _collect(frame: pl.DataFrame, exprs: list[pl.Expr], engine: str) -> pl.DataFrame:
    lazy = frame.lazy()
    # One with_columns per expression: a later builder may read the column
    # an earlier one made, as the library's own pipeline does.
    for expr in _named(exprs):
        lazy = lazy.with_columns(expr)
    return lazy.collect(engine=engine)  # type: ignore[call-overload]


def _eager(frame: pl.DataFrame, exprs: list[pl.Expr]) -> pl.DataFrame:
    out = frame
    for expr in _named(exprs):
        out = out.with_columns(expr)
    return out


@pytest.mark.parametrize("key", sorted(set(CASES) - set(ABSTRACT)))
def test_builder_gives_the_same_column_on_every_engine(key: str) -> None:
    exprs = CASES[key]()
    assert exprs, f"{key}: no expression to evaluate"
    frame = _frame()
    assert frame.height >= 500
    expected = _eager(frame, exprs)
    assert expected.width == frame.width + len(exprs)
    for layout in (_corpus.as_built, _corpus.in_chunks):
        for engine in ("in-memory", "streaming"):
            produced = _collect(layout(frame), exprs, engine)
            try:
                assert_frame_equal(
                    produced,
                    expected,
                    check_exact=True,
                    check_dtypes=True,
                    check_column_order=True,
                    check_row_order=True,
                )
            except AssertionError as exc:
                raise AssertionError(
                    f"{key} [{engine}, {layout.__name__}]: {exc}"
                ) from None
