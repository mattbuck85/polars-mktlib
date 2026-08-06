from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING, overload

import polars as pl

logger = logging.getLogger(__name__)

from mktlib.backtest._bracket import (
    ANCHOR_FILL_COLUMN,
    BLOCK_COLUMN,
    BRACKET_CANDIDATE_COLUMN,
    BRACKET_COLUMNS,
    BRACKET_EXIT_COLUMN,
    BRACKET_KIND_COLUMN,
    BRACKET_LEVEL_COLUMN,
    BRACKET_POST_COLUMN,
    BRACKET_SEEN_COLUMN,
    ENTRY_FILL_COLUMN,
    RESIGNAL_COLUMN,
    SL_LEVEL_COLUMN,
    STOP_LOSS,
    TAKE_PROFIT,
    TP_LEVEL_COLUMN,
    Bracket,
    fill_expr,
    level_expr,
    trigger_expr,
)
from mktlib.backtest._anchor import (
    ANCHOR_ENTRY_COLUMN,
    collect_entry_refs,
    realized_entries,
)
from mktlib.backtest._conditions import (
    Condition,
    Custom,
    Limit,
)
from mktlib.backtest._cost import (
    COST_COLUMN,
    Cost,
    cost_bps_expr,
    validate_slippage_col,
)
from mktlib.backtest._flatten import (
    FLATTEN_BAR_COLUMN,
    Flatten,
    FlattenSchedule,
    build_flatten_masks,
    resolve_flatten,
)
from mktlib.backtest._types import BacktestResult, MultiBacktestResult, Strategy, TradeSide
from mktlib.backtest._weights import (
    INSTRUMENT_COLUMN,
    InvalidPortfolioWeights,
    to_portfolio_weights_df,
)

if TYPE_CHECKING:
    from mktlib.scheduling import ExchangeCalendar


# ---------------------------------------------------------------------------
# Bracket exits — protective TP/SL resting against every open position
# ---------------------------------------------------------------------------

_DUAL_BRACKET_MSG = (
    "bracket=... is not supported together with short_strategy=.... The dual "
    "path merges only _position and _side from the two legs, so a bracket "
    "would silently evaluate the long leg's levels against the short leg's "
    "position. Run the two sides as separate single-side backtests, each with "
    "its own bracket, and combine the returns yourself."
)


def _apply_bracket(
    signals: pl.DataFrame,
    bracket: Bracket,
    *,
    is_long: bool,
    flatten_active: bool,
) -> pl.DataFrame:
    """Materialize the bracket columns and truncate bracketed position blocks.

    Consumes ``_pos_d1`` / ``_pos_d2`` / ``_entry_clean`` / ``_exit_clean``
    (all already materialized by :func:`_run_core`) and adds:

    ``_bracket_level``
        Realized fill price on the bar the bracket closed the position,
        null everywhere else. Doubles as the discriminator that lets
        :func:`_extract_trades` recognize a same-bar bracket fill.
    ``_bracket_kind``
        ``"take_profit"`` / ``"stop_loss"`` on that same bar — which leg
        fired, which ``_bracket_level`` alone cannot say.
    ``_bracket_exit_bar`` / ``_bracket_post``
        The exit bar itself, and the bars after it that the stale
        ``_position`` still marks as held and whose returns must be zeroed.

    ``_position`` is deliberately **left inconsistent** — the same trick the
    same-bar ``Limit`` exit uses. The return expression and the trade
    extractor carry the truth; recomputing ``_position`` would require a
    sequential scan for no gain, since a bracketed block cannot re-enter.
    """
    missing_ohlc = [col for col in ("high", "low") if col not in signals.columns]
    if missing_ohlc:
        msg = (
            f"Bracket requires {missing_ohlc} column(s) in the DataFrame: a "
            "take-profit or stop-loss can only be evaluated against the bar's "
            "full range."
        )
        raise ValueError(msg)

    for col in bracket.level_columns:
        if col not in signals.columns:
            msg = (
                f"Bracket level column {col!r} not found in DataFrame columns: "
                f"{sorted(signals.columns)}"
            )
            raise ValueError(msg)
        # Projected rather than ``filter(...)[col]``: filtering materialises
        # every column of the frame in order to null-check one of them, which at
        # 491k rows x ~15 columns is most of a copy for a single boolean.
        stale = signals.select(
            (pl.col(col).is_null() & pl.col("_entry_clean")).any()
        ).item()
        if stale:
            msg = (
                f"Bracket level column {col!r} is null on at least one entry "
                "signal bar. The level is latched at entry and forward-filled, "
                "so a null there would silently carry the previous trade's "
                "level into this one."
            )
            raise ValueError(msg)

    is_entry_bar = (pl.col("_pos_d1") == 1) & (pl.col("_pos_d2") == 0)

    # Under anchor="signal" the levels also move on entry signals that fire
    # while the position is held. NOTHING below is emitted under the default
    # anchor="position", so that path's expression tree is untouched and its
    # output is byte-identical by construction rather than by assertion.
    re_anchor = bracket.anchor == "signal"

    # Latch the entry fill price (the entry bar's own open) and give every
    # position block an id, so "first trigger wins" becomes a cumulative count
    # that resets per block.
    _block_exprs: list[pl.Expr] = [
        pl.when(is_entry_bar)
        .then(pl.col("open"))
        .otherwise(None)
        .forward_fill()
        .alias(ENTRY_FILL_COLUMN),
        is_entry_bar.cast(pl.UInt32).cum_sum().alias(BLOCK_COLUMN),
    ]
    if re_anchor:
        # A re-signal is an entry signal on a bar the position was ALREADY
        # held on (``_pos_d1 == 1``), so it opens nothing and the position
        # machinery discards it — which is precisely what this policy exists
        # to stop discarding.
        #
        # ``fill_null(False)`` keeps this a non-nullable Boolean. The null is
        # real and reachable: ``_entry`` is null wherever the strategy's
        # indicators are still warming up or its expression admits nulls, and
        # ``null & true`` is null under Kleene logic, so on a held bar the
        # ``&`` above propagates it.
        #
        # Measured, not assumed: with the three consumers this column has
        # today the fill changes nothing — ``when(null)`` takes its otherwise
        # branch, the anchor-fill column re-fills after its own ``shift(1)``,
        # and the null guard's ``any()`` ignores nulls. It is kept because a
        # nullable mask is a trap for the NEXT consumer, which may combine it
        # with ``&``/``|`` outside a ``when``; 0.15.0 removed exactly this
        # class of accidental truthiness from ``session_last``.
        resignal = pl.col("_entry") & (pl.col("_pos_d1") == 1)
        if flatten_eod:
            # A session-last signal fires against a position that is
            # flattened at that same bar's open, so there is nothing left to
            # re-anchor. The signal itself is not lost: it was deferred to
            # the next session's first bar, where it opens a fresh position
            # and latches that position's own levels.
            resignal = resignal & ~pl.col("_session_last")
        # Materialized rather than inlined: it is read by the anchor-fill
        # column, by each str leg, and by the post-hoc null guard, and a
        # ``with_columns`` does not common-subexpression-eliminate.
        _block_exprs.append(resignal.fill_null(False).alias(RESIGNAL_COLUMN))
    signals = signals.with_columns(_block_exprs)

    if re_anchor:
        # The price a float leg is measured from. It starts as the entry fill
        # (the entry bar's own open, same as ENTRY_FILL_COLUMN) and moves to
        # ``open[k + 1]`` after a re-signal on bar ``k`` — the price a fresh
        # entry on that signal would itself have filled at, so no second
        # price model is introduced. ENTRY_FILL_COLUMN is left alone because
        # trade P&L must keep measuring from the true entry fill.
        signals = signals.with_columns(
            pl.when(
                is_entry_bar | pl.col(RESIGNAL_COLUMN).shift(1).fill_null(False)
            )
            .then(pl.col("open"))
            .otherwise(None)
            .forward_fill()
            .alias(ANCHOR_FILL_COLUMN),
        )

    # Which column each spec kind reads is decided HERE rather than inside
    # ``level_expr``: the float branch needs a different anchor price, the str
    # branch needs an extra latch bar, and only the caller knows both.
    _fill_col = ANCHOR_FILL_COLUMN if re_anchor else ENTRY_FILL_COLUMN
    _resignal_col = RESIGNAL_COLUMN if re_anchor else None

    legs: list[tuple[str, str]] = []
    level_exprs: list[pl.Expr] = []
    if bracket.take_profit is not None:
        legs.append((TAKE_PROFIT, TP_LEVEL_COLUMN))
        level_exprs.append(
            level_expr(
                bracket.take_profit,
                leg=TAKE_PROFIT,
                is_long=is_long,
                entry_clean_col="_entry_clean",
                entry_fill_col=_fill_col,
                resignal_col=_resignal_col,
            ).alias(TP_LEVEL_COLUMN)
        )
    if bracket.stop_loss is not None:
        legs.append((STOP_LOSS, SL_LEVEL_COLUMN))
        level_exprs.append(
            level_expr(
                bracket.stop_loss,
                leg=STOP_LOSS,
                is_long=is_long,
                entry_clean_col="_entry_clean",
                entry_fill_col=_fill_col,
                resignal_col=_resignal_col,
            ).alias(SL_LEVEL_COLUMN)
        )
    signals = signals.with_columns(level_exprs)

    # The bracket rests from the entry fill bar (``_pos_d1 == 1``) until the
    # position closes. Under flatten_eod the session-last bar is exempt: the
    # engine flattens at that bar's *open*, so the position is already gone
    # before any intra-bar level could be tagged.
    live = pl.col("_pos_d1") == 1
    if flatten_active:
        live = live & ~pl.col(FLATTEN_BAR_COLUMN)

    # Each leg's trigger is MATERIALIZED as a boolean column. It is read
    # twice — once to build the candidate, once to pick the fill price — and a
    # ``with_columns`` does not common-subexpression-eliminate a repeated
    # expression, so leaving it inline evaluates the comparison chain twice.
    _hit_cols = {leg: f"__bracket_hit_{leg}" for leg, _ in legs}
    signals = signals.with_columns(
        [
            (
                live
                & trigger_expr(leg=leg, is_long=is_long, level_col=level_col)
            )
            .fill_null(False)
            .alias(_hit_cols[leg])
            for leg, level_col in legs
        ]
    )

    # Resolution order for a bar that tags BOTH levels. The candidate string
    # and the fill price are both built from this one list, so the
    # ``both_touch`` policy is expressed exactly once rather than mirrored in
    # two places that could drift apart.
    _priority = (
        (STOP_LOSS, TAKE_PROFIT)
        if bracket.both_touch == "stop_first"
        else (TAKE_PROFIT, STOP_LOSS)
    )
    ordered: list[tuple[str, str]] = [
        (leg, level_col)
        for leg in _priority
        for present, level_col in legs
        if present == leg
    ]

    # Built back to front so the first entry in ``ordered`` ends up outermost
    # and therefore wins.
    candidate = pl.lit(None, dtype=pl.String)
    for leg, _ in reversed(ordered):
        candidate = (
            pl.when(pl.col(_hit_cols[leg]))
            .then(pl.lit(leg))
            .otherwise(candidate)
        )
    signals = signals.with_columns(candidate.alias(BRACKET_CANDIDATE_COLUMN))

    # Only the first trigger in a block closes the position; later triggers
    # are noise against a position that is already flat.
    #
    # ONE count serves all three columns. ``seen`` was previously a second
    # ``cum_sum().over(BLOCK)`` taken over ``exit_bar``, but the two are
    # algebraically identical: ``exit_bar`` is true exactly once per block, on
    # the first trigger, so ``cum_sum(exit_bar) >= 1`` and ``cum_sum(triggered)
    # >= 1`` both flip true on that same bar and stay true.
    #
    # That count is computed WITHOUT a window. ``BLOCK_COLUMN`` is a
    # ``cum_sum`` of a non-negative cast, so it is contiguous and
    # monotonically non-decreasing — and for that shape a per-block
    # cumulative count is just the GLOBAL cumulative count minus its value on
    # the row before the block began. The base is recovered by forward-filling
    # from each block-start row, which costs a shift and a forward_fill in
    # place of a ``.over()``.
    #
    # The count is MATERIALIZED, then read three times as a plain column.
    # Inlining the expression into the three consumers instead is measurably
    # slower (41.7ms vs 38.6ms per run at 491k rows): a ``with_columns`` does
    # not common-subexpression-eliminate a repeated expression, so it
    # evaluates it once per consumer.
    _cum = "__bracket_trigger_cum"
    _base = "__bracket_block_base"
    _count = "__bracket_trigger_count"
    triggered = pl.col(BRACKET_CANDIDATE_COLUMN).is_not_null()
    signals = signals.with_columns(
        triggered.cast(pl.UInt32).cum_sum().alias(_cum),
    )
    # Row 0: ``shift(1)`` is null, so the ``!=`` is null, so ``when`` is false
    # and the ``fill_null(0)`` supplies the base — which is correct, because
    # the first block's base genuinely is 0. ``_base <= _cum`` everywhere by
    # construction (it is an earlier value of a non-decreasing series), so the
    # UInt32 subtraction below cannot underflow.
    signals = signals.with_columns(
        pl.when(pl.col(BLOCK_COLUMN) != pl.col(BLOCK_COLUMN).shift(1))
        .then(pl.col(_cum).shift(1).fill_null(0))
        .otherwise(None)
        .forward_fill()
        .fill_null(0)
        .alias(_base),
    )
    signals = signals.with_columns(
        (pl.col(_cum) - pl.col(_base)).alias(_count),
    )
    # ``seen`` used to be a SECOND window over ``exit_bar``. It is algebraically
    # the same series: ``exit_bar`` is true exactly once per block, on the first
    # trigger, so ``cum_sum(exit_bar) >= 1`` and ``cum_sum(triggered) >= 1`` flip
    # true on that same bar and stay true.
    is_first_trigger = triggered & (pl.col(_count) == 1)
    signals = signals.with_columns(
        is_first_trigger.alias(BRACKET_EXIT_COLUMN),
        pl.when(is_first_trigger)
        .then(pl.col(BRACKET_CANDIDATE_COLUMN))
        .otherwise(None)
        .alias(BRACKET_KIND_COLUMN),
        (pl.col(_count) >= 1).alias(BRACKET_SEEN_COLUMN),
    ).drop(_cum, _base, _count)

    if re_anchor and bracket.level_columns:
        # The entry-signal null guard above cannot cover re-latch bars: it runs
        # before BLOCK_COLUMN and BRACKET_SEEN_COLUMN exist, and without them
        # there is no way to tell a re-latch that matters from one in a block
        # the bracket already closed. So this is a SECOND, deliberately
        # POST-HOC guard — by the time it runs the levels have been computed
        # and the stale value has already been used. It is a diagnostic that
        # refuses to return the result, not a precondition.
        #
        # It is still exact rather than approximate. A null on a re-latch bar
        # in a block that has NOT yet fired gives ``seen == False``, and the
        # forward-fill would carry the pre-re-signal level onward as though
        # the re-signal never happened — raise. A null at or after the block's
        # first trigger gives ``seen == True``, and every level in that block
        # from the trigger bar on is dead: the position closed there and later
        # triggers lose the ``_count == 1`` gate.
        for col in bracket.level_columns:
            stale_relatch = signals.select(
                (
                    pl.col(col).is_null()
                    & pl.col(RESIGNAL_COLUMN)
                    & ~pl.col(BRACKET_SEEN_COLUMN)
                ).any()
            ).item()
            if stale_relatch:
                msg = (
                    f"Bracket level column {col!r} is null on at least one "
                    "re-anchoring entry signal bar. Under anchor='signal' the "
                    "level is re-latched on every entry signal that fires "
                    "while the position is held and forward-filled, so a null "
                    "there would silently hold the level the re-signal was "
                    "meant to move."
                )
                raise ValueError(msg)

    # Which leg fired is already known as a pair of booleans, so the fill price
    # is selected from those rather than by comparing the ``_bracket_kind``
    # String against a leg name once per leg. Same ``ordered`` priority, so
    # this picks exactly the leg ``_bracket_kind`` names; gating on the
    # exit-bar mask reproduces its null-off-the-exit-bar pattern.
    fill_price = pl.lit(None, dtype=pl.Float64)
    for leg, level_col in reversed(ordered):
        fill_price = (
            pl.when(pl.col(BRACKET_EXIT_COLUMN) & pl.col(_hit_cols[leg]))
            .then(fill_expr(leg=leg, is_long=is_long, level_col=level_col))
            .otherwise(fill_price)
        )
    signals = signals.with_columns(
        fill_price.cast(pl.Float64).alias(BRACKET_LEVEL_COLUMN),
        (pl.col(BRACKET_SEEN_COLUMN) & ~pl.col(BRACKET_EXIT_COLUMN))
        .alias(BRACKET_POST_COLUMN),
    )

    # Redirect the trade extractor: the bracket bar is this block's exit, and
    # the signal exit that would have followed it never happens.
    return signals.with_columns(
        (
            pl.col(BRACKET_EXIT_COLUMN)
            | (pl.col("_exit_clean") & ~pl.col(BRACKET_SEEN_COLUMN))
        ).alias("_exit_clean"),
    ).drop(_hit_cols.values())


def _run_core(
    df: pl.DataFrame,
    strategy: Strategy,
    *,
    trade_side: TradeSide,
    calendar: ExchangeCalendar | None,
    flatten: FlattenSchedule | None,
    cost: Cost | None = None,
    bracket: Bracket | None = None,
) -> BacktestResult:
    """Inner engine: assumes *df* is already calendar-filtered."""
    # Let strategy enrich the DataFrame with indicator columns
    _init = getattr(strategy, "init", None)
    if _init is not None:
        df = _init(df)

    entry_raw = strategy.entry()
    exit_raw = strategy.exit()
    entry_cond = entry_raw if isinstance(entry_raw, Condition) else Custom(entry_raw)
    exit_cond = exit_raw if isinstance(exit_raw, Condition) else Custom(exit_raw)

    # Entry-side limits are not supported yet (v1 scope: exit-only). The
    # wrapper would resolve silently to the inner boolean with the
    # ``price`` kwarg ignored — a footgun. Fail fast with a clear message
    # so callers don't get subtle incorrect fill semantics.
    if isinstance(entry_cond, Limit):
        msg = (
            "Limit(...) is only supported on exit conditions in this "
            "release. Entry-side limit fills are planned for a future "
            "version — for now, use the inner condition directly for "
            "entry and rely on fill-at-next-open semantics."
        )
        raise NotImplementedError(msg)

    entry_expr = entry_cond.resolve()

    # Limit exit: same-bar fill at a specified price. v1 scope is a
    # top-level ``Limit`` wrapper on the exit tree only — nested limit
    # usage is not recognized and behaves as a plain boolean condition.
    is_limit_exit = isinstance(exit_cond, Limit)
    limit_price_expr: pl.Expr | None = (
        exit_cond.resolve_price() if is_limit_exit else None  # type: ignore[attr-defined]
    )

    if bracket is not None and is_limit_exit:
        # Both want to own the same-bar fill. Which one wins when a Limit
        # exit and a bracket leg tag the same bar is exactly the within-bar
        # ordering question OHLC cannot answer, and picking silently would
        # bury an assumption in the engine. Refuse instead of guessing.
        msg = (
            "bracket=... is not supported together with a Limit(...) exit "
            "condition. Both fill inside the same bar, and OHLC cannot say "
            "which came first. Express the target as Bracket(take_profit=...) "
            "and use a plain (non-Limit) exit condition."
        )
        raise NotImplementedError(msg)

    effective_side = int(entry_cond.trade_side or trade_side)

    # Pass 1: compute _entry
    signals = df.with_columns(entry_expr.alias("_entry"))

    # Flattening rewrites _entry (deferring a flatten-bar signal to the next
    # bar) and that rewrite must land BEFORE anything reads _entry. It used to
    # run after the EntryRef snapshot, which latched the anchor on the bar the
    # deferral then moved away from — the trade opened on one bar and was
    # measured against another.
    if flatten is not None:
        if calendar is None:
            # run() refuses this combination up front; this guard covers the
            # private entry points, where a caller could otherwise get a
            # schedule silently ignored rather than an error.
            msg = "a flatten schedule requires a calendar"
            raise ValueError(msg)
        _flatten_mask, _blocked_mask = build_flatten_masks(
            signals["date"], calendar, flatten
        )
        signals = signals.with_columns(_flatten_mask.alias(FLATTEN_BAR_COLUMN))
        # Defer entry SIGNALS on flatten bars to the next bar — the signal
        # moves, not the fill. At the default offset the flatten bar is the
        # session's last, so a crossover on 15:59 rewrites ``_entry`` onto the
        # next session's 09:30 bar; the recurrence below then opens the
        # position there, and the usual fill-at-next-open rule puts the entry
        # fill on the bar AFTER 09:30. (``is_entry_bar`` in ``_apply_bracket``
        # is ``_pos_d1 == 1 & _pos_d2 == 0``, one bar later than
        # ``_entry_clean``.) Under an offset schedule the flatten bar is
        # in-session, so "the next bar" is in-session too.
        #
        # A flatten bar that is ITSELF inside the block window does not defer,
        # it drops. Deferring it would carry the signal past the block — the
        # next bar is often the next session's open, which no window covers —
        # and land a fill hours after the signal, which is exactly what
        # blocking is documented to prevent. With block == 0 nothing is
        # blocked, so this is the unchanged legacy deferral.
        _suppressed = pl.col("_entry") & pl.col(FLATTEN_BAR_COLUMN)
        if flatten.block_entry_minutes_before_close:
            _suppressed = _suppressed & ~pl.lit(_blocked_mask)
        signals = signals.with_columns(
            (pl.col("_entry") | _suppressed.shift(1).fill_null(False)).alias("_entry"),
        )
        # The block window has the LAST say, deliberately after the deferral.
        #
        # The flatten bar is not necessarily inside the block window: with a
        # bar grid that does not divide the two cutoffs evenly, the flatten bar
        # can start strictly before `close - block_minutes` while the next bar
        # starts after it. Blocking first would then let the deferral write an
        # entry into the very window it had just cleared. Blocking last cannot,
        # so no precedence rule is needed between the two.
        #
        # Gated so that block == 0 does not even build the expression: the
        # default path stays byte-identical by construction, not by argument.
        if flatten.block_entry_minutes_before_close:
            signals = signals.with_columns(
                (pl.col("_entry") & ~pl.lit(_blocked_mask)).alias("_entry"),
            )

    # Create snapshot columns for any EntryRef nodes in exit condition.
    #
    # These latch on the entries that actually OPEN a position, not on every
    # raw signal. A signal that fires while a position is already open is
    # suppressed by the position machinery, and forward-filling from it would
    # move the anchor mid-trade — measured at +26.9 bps/trade of inflation on
    # the canonical take-profit / stop-loss idiom, worst in trending regimes,
    # because a re-anchored level ratchets the target away and the stop up.
    entry_refs = collect_entry_refs(exit_cond)
    if entry_refs:
        signals = signals.with_columns(
            realized_entries(
                signals,
                entry_col="_entry",
                exit_cond=exit_cond,
                snapshot_cols=entry_refs,
                session_last_col=(
                    FLATTEN_BAR_COLUMN if flatten is not None else None
                ),
            ),
        )
        signals = signals.with_columns(
            pl.when(pl.col(ANCHOR_ENTRY_COLUMN)).then(pl.col(col)).otherwise(None)
            .forward_fill().alias(f"_entry_{col}")
            for col in entry_refs
        )

    # Pass 2: compute _exit (snapshot columns now exist for EntryRef.resolve())
    exit_expr = exit_cond.resolve()
    signals = signals.with_columns(exit_expr.alias("_exit"))

    # For limit exits, also materialize the fill price. Only consumed on
    # bars where the exit fires while we're holding; values elsewhere are
    # harmless.
    if is_limit_exit:
        signals = signals.with_columns(
            limit_price_expr.alias("_limit_price"),  # type: ignore[union-attr]
        )

    # Position tracking: 1 on entry, 0 on exit, forward-fill
    if flatten is not None:
        # Suppress entries on session-last bars (position opens and immediately
        # force-closes in the same bar — not a valid trade).
        signals = signals.with_columns(
            pl.when(pl.col("_entry") & ~pl.col(FLATTEN_BAR_COLUMN))
            .then(pl.lit(1))
            .when(pl.col("_exit") | pl.col(FLATTEN_BAR_COLUMN))
            .then(pl.lit(0))
            .otherwise(pl.lit(None))
            .forward_fill()
            .fill_null(0)
            .alias("_position"),
        )
    else:
        signals = signals.with_columns(
            pl.when(pl.col("_entry"))
            .then(pl.lit(1))
            .when(pl.col("_exit"))
            .then(pl.lit(0))
            .otherwise(pl.lit(None))
            .forward_fill()
            .fill_null(0)
            .alias("_position"),
        )

    # Materialize shared shifted expressions once
    signals = signals.with_columns(
        pl.col("_position").shift(1).fill_null(0).alias("_pos_d1"),
        pl.col("_position").shift(2).fill_null(0).alias("_pos_d2"),
        pl.col("close").shift(1).alias("_close_prev"),
    )
    # Transition detection (uses materialized _pos_d1)
    signals = signals.with_columns(
        ((pl.col("_position") == 1) & (pl.col("_pos_d1") == 0)).alias("_entry_clean"),
        ((pl.col("_position") == 0) & (pl.col("_pos_d1") == 1)).alias("_exit_clean"),
    )

    # _side = position * effective_side (scalar multiply, no forward-fill)
    signals = signals.with_columns(
        (pl.col("_position") * effective_side).cast(pl.Int8).alias("_side"),
    )

    # Bracket exits: resolved before costs so the bracket bar can be charged
    # like any other fill bar.
    if bracket is not None:
        signals = _apply_bracket(
            signals,
            bracket,
            is_long=effective_side == 1,
            flatten_active=flatten is not None,
        )

    # Detect transition bars (after the 1-bar delay for fill). Defined here,
    # ahead of the cost column, because the fill-bar mask that validates a
    # per-bar slippage column is built out of exactly these predicates.
    _is_entry_bar = (pl.col("_pos_d1") == 1) & (pl.col("_pos_d2") == 0)
    _is_exit_bar = (pl.col("_pos_d1") == 0) & (pl.col("_pos_d2") == 1)

    # Limit-exit bars fill *inside* the bar the inner condition fires while
    # we were holding; the bar after is stale and contributes nothing.
    _is_limit_exit_bar: pl.Expr = pl.lit(False)
    _is_post_limit_bar: pl.Expr = pl.lit(False)
    if is_limit_exit:
        _is_limit_exit_bar = pl.col("_exit") & (pl.col("_pos_d1") == 1)
        _is_post_limit_bar = _is_limit_exit_bar.shift(1).fill_null(False)
        # Narrow ``_limit_price`` to the bars the limit actually FILLS on.
        #
        # It is materialized unconditionally further up, because the level is
        # an expression over the whole frame and ``_pos_d1`` does not exist
        # yet. That left a non-null level on every bar, and
        # :func:`_extract_trades` reads it through
        # ``_limit_price.is_not_null()`` — so any exit the *session flatten*
        # forced was booked at a take-profit the tape never traded, while
        # ``returns`` correctly used the flatten bar's open. The two views of
        # one run disagreed, and because ``mktlib.reports`` reads
        # ``trades["pnl"]``, a losing strategy could report a 100% win rate.
        #
        # Narrowing here, from the same predicate the return expression uses,
        # makes ``is_not_null()`` mean exactly "the limit filled on this bar"
        # for every downstream reader, rather than each having to re-derive it.
        signals = signals.with_columns(
            pl.when(_is_limit_exit_bar)
            .then(pl.col("_limit_price"))
            .otherwise(None)
            .alias("_limit_price"),
        )

    # Transaction costs: one per-bar basis-point column, materialized once and
    # read *unshifted on the fill bar* by every branch that pays a fill. It is
    # dropped again before the result is handed back.
    if cost is not None:
        # Every bar on which some branch below reads the cost column. A
        # per-bar slippage column is only ever consumed here, so this is
        # also exactly the set of bars it has to be well-formed on.
        _fill_bar = _is_entry_bar | _is_exit_bar
        if is_limit_exit:
            _fill_bar = _fill_bar | _is_limit_exit_bar
        if bracket is not None:
            _fill_bar = _fill_bar | pl.col(BRACKET_EXIT_COLUMN)
        if flatten is not None:
            _fill_bar = _fill_bar | (
                pl.col(FLATTEN_BAR_COLUMN) & (pl.col("_pos_d1") == 1)
            )
        validate_slippage_col(signals, cost, fill_bar=_fill_bar.fill_null(False))
        signals = signals.with_columns(cost_bps_expr(cost).alias(COST_COLUMN))

    def _charge(expr: pl.Expr, *, sides: int = 1) -> pl.Expr:
        """Deduct the fill-bar cost from a fill-bar return expression.

        Subtracted, never multiplied by ``effective_side``: a cost reduces
        the realized return on both sides of the market. *sides* is the
        number of fills the bar pays for — two when a bracket closes the
        position on the very bar the entry filled.
        """
        if cost is None:
            return expr
        return expr - pl.col(COST_COLUMN) * sides / 1e4

    # Per-bar returns with fill-at-open adjustment. Entry/exit/limit bars are
    # fill bars and therefore pay cost; middle bars hold and do not.
    _entry_ret = _charge(
        ((pl.col("close") - pl.col("open")) / pl.col("open")) * effective_side
    )
    _normal_ret = (pl.col("close") / pl.col("_close_prev") - 1) * effective_side
    _exit_ret = _charge(
        ((pl.col("open") - pl.col("_close_prev")) / pl.col("_close_prev"))
        * effective_side
    )

    # Limit-exit branches: fill at _limit_price on the same bar the
    # inner condition fires while we were holding. Suppress the normal
    # next-bar exit-open fill that would otherwise apply.
    #
    # Two shapes, differing only in how many fills the bar pays for: when
    # the limit fires on the bar the entry itself filled, the entry and the
    # exit both land inside that one bar, so it owes *two* sides — the same
    # accounting ``_bracket_entry_ret`` does, and the same two legs
    # ``trades.pnl`` charges for that trade.
    # The two shapes also differ in the BASE the return is measured against,
    # which is the part that was wrong. A limit firing on a bar the position
    # was already holding is measured from the previous close. A limit firing
    # on the bar the entry itself filled was never held across that close —
    # the position opened at this bar's ``open`` — so measuring it from
    # ``_close_prev`` credits a gap that was never carried. On a bar that gaps
    # 100.0 -> 110.0 and fills a 115.0 limit, that is +15.0% booked against a
    # position that earned +4.55%. ``_bracket_entry_ret`` below already gets
    # this right and uses ``open``; these two are the same accounting and must
    # agree.
    _limit_ret: pl.Expr = pl.lit(0.0)
    _limit_entry_ret: pl.Expr = pl.lit(0.0)
    if is_limit_exit:
        _limit_held_raw = (
            (pl.col("_limit_price") - pl.col("_close_prev")) / pl.col("_close_prev")
        ) * effective_side
        _limit_opened_raw = (
            (pl.col("_limit_price") - pl.col("open")) / pl.col("open")
        ) * effective_side
        _limit_ret = _charge(_limit_held_raw)
        _limit_entry_ret = _charge(_limit_opened_raw, sides=2)

    # Under flatten_eod an entry fill can land on the session-last bar: the
    # position opens at that bar's open and is force-flattened at the same
    # open. The price return is exactly zero, but two fills happened, so the
    # bar owes two sides — ``trades.pnl`` already charges both.
    _flat_entry_ret: pl.Expr = _charge(pl.lit(0.0), sides=2)

    # Bracket branches: the level is tagged *inside* the bar, so the fill is
    # same-bar. Two shapes, because the base the return is measured against
    # differs — the entry fill price when the bracket closes the position on
    # the bar it opened, the previous close otherwise. The first case pays
    # two fills, matching what ``trades.pnl`` charges for that trade.
    _bracket_ret: pl.Expr = pl.lit(0.0)
    _bracket_entry_ret: pl.Expr = pl.lit(0.0)
    if bracket is not None:
        _bracket_ret = _charge(
            (
                (pl.col(BRACKET_LEVEL_COLUMN) - pl.col("_close_prev"))
                / pl.col("_close_prev")
            )
            * effective_side
        )
        _bracket_entry_ret = _charge(
            ((pl.col(BRACKET_LEVEL_COLUMN) - pl.col("open")) / pl.col("open"))
            * effective_side,
            sides=2,
        )

    def _with_bracket(base: pl.Expr) -> pl.Expr:
        """Give the bracket first refusal on every bar of a bracketed block."""
        if bracket is None:
            return base
        return (
            pl.when(pl.col(BRACKET_POST_COLUMN)).then(0.0)
            .when(pl.col(BRACKET_EXIT_COLUMN) & _is_entry_bar).then(_bracket_entry_ret)
            .when(pl.col(BRACKET_EXIT_COLUMN)).then(_bracket_ret)
            .otherwise(base)
            .fill_null(0.0)
        )

    # Compute returns + flatten_eod overrides in minimal with_columns calls
    if flatten is not None:
        # Base returns + session-last override in one pass. When limits
        # are active they win against session-last on the same bar
        # (intra-bar fill precedes the session-close flatten).
        if is_limit_exit:
            ret_expr = (
                pl.when(_is_limit_exit_bar & _is_entry_bar).then(_limit_entry_ret)
                .when(_is_limit_exit_bar).then(_limit_ret)
                .when(_is_post_limit_bar).then(0.0)
                .when(pl.col(FLATTEN_BAR_COLUMN) & _is_entry_bar).then(_flat_entry_ret)
                .when(pl.col(FLATTEN_BAR_COLUMN) & (pl.col("_pos_d1") == 1)).then(_exit_ret)
                .when(_is_entry_bar).then(_entry_ret)
                .when(_is_exit_bar).then(_exit_ret)
                .when(pl.col("_pos_d1") == 1).then(_normal_ret)
                .otherwise(0.0)
                .fill_null(0.0)
            )
        else:
            ret_expr = (
                pl.when(pl.col(FLATTEN_BAR_COLUMN) & _is_entry_bar).then(_flat_entry_ret)
                .when(pl.col(FLATTEN_BAR_COLUMN) & (pl.col("_pos_d1") == 1)).then(_exit_ret)
                .when(_is_entry_bar).then(_entry_ret)
                .when(_is_exit_bar).then(_exit_ret)
                .when(pl.col("_pos_d1") == 1).then(_normal_ret)
                .otherwise(0.0)
                .fill_null(0.0)
            )
        signals = signals.with_columns(_with_bracket(ret_expr).alias("return"))
        # Zero the bar *after* a flatten bar. The flatten filled at the flatten
        # bar's own open, so the default next-open exit branch would otherwise
        # pay a second, phantom exit return here.
        #
        # This is unconditional on purpose. ``_position`` is structurally 0 on
        # every flatten bar (see the ``_position`` expression above: the entry
        # branch is gated on ``~flatten_bar``, and the exit branch is satisfied
        # by ``flatten_bar`` alone), so at flatten+1 ``_pos_d1 == 0`` and
        # therefore ``_is_entry_bar`` — which requires ``_pos_d1 == 1`` —
        # cannot be true. A ``& ~_is_entry_bar`` guard stood here for exactly
        # that reason and was a tautology; it never once changed the result.
        # It is gone rather than tested around, because a guard that cannot
        # fire reads as protection that is not there.
        # ``test_position_is_always_zero_on_the_flatten_bar`` pins the
        # invariant the deletion rests on.
        signals = signals.with_columns(
            pl.when(pl.col(FLATTEN_BAR_COLUMN).shift(1).fill_null(False))
            .then(0.0)
            .otherwise(pl.col("return"))
            .alias("return"),
        )
    else:
        if is_limit_exit:
            ret_expr = (
                pl.when(_is_limit_exit_bar & _is_entry_bar).then(_limit_entry_ret)
                .when(_is_limit_exit_bar).then(_limit_ret)
                .when(_is_post_limit_bar).then(0.0)
                .when(_is_entry_bar).then(_entry_ret)
                .when(_is_exit_bar).then(_exit_ret)
                .when(pl.col("_pos_d1") == 1).then(_normal_ret)
                .otherwise(0.0)
                .fill_null(0.0)
            )
        else:
            ret_expr = (
                pl.when(_is_entry_bar).then(_entry_ret)
                .when(_is_exit_bar).then(_exit_ret)
                .when(pl.col("_pos_d1") == 1).then(_normal_ret)
                .otherwise(0.0)
                .fill_null(0.0)
            )
        signals = signals.with_columns(_with_bracket(ret_expr).alias("return"))

    returns = signals.select("date", "return")

    # Build trade log from entry/exit transitions (before dropping internal cols)
    trades = _extract_trades(signals, flatten_active=flatten is not None)

    # Drop internal columns before return
    _drop_cols = ["_pos_d1", "_pos_d2", "_close_prev", "_entry_clean", "_exit_clean"]
    if ANCHOR_ENTRY_COLUMN in signals.columns:
        _drop_cols.append(ANCHOR_ENTRY_COLUMN)
    if flatten is not None:
        _drop_cols.append(FLATTEN_BAR_COLUMN)
    if is_limit_exit:
        _drop_cols.append("_limit_price")
    if bracket is not None:
        # Only one of the two level columns exists for a single-leg bracket.
        _drop_cols.extend(col for col in BRACKET_COLUMNS if col in signals.columns)
    if cost is not None:
        _drop_cols.append(COST_COLUMN)
    signals = signals.drop(_drop_cols)

    return BacktestResult(returns=returns, trades=trades, signals=signals)


def _run_dual(
    df: pl.DataFrame,
    long_strategy: Strategy,
    short_strategy: Strategy,
    *,
    calendar: ExchangeCalendar | None,
    flatten: FlattenSchedule | None,
    cost: Cost | None = None,
    bracket: Bracket | None = None,
) -> BacktestResult:
    """Run long and short strategies independently, validate, merge.

    Both sides run concurrently via threads — Polars releases the GIL
    during computation so this gives real parallelism.  Merged signals
    use the long strategy's indicator columns as base.
    """
    if bracket is not None:
        raise NotImplementedError(_DUAL_BRACKET_MSG)

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=2) as pool:
        long_future = pool.submit(
            _run_core, df, long_strategy, trade_side=TradeSide.LONG,
            calendar=calendar, flatten=flatten, cost=cost,
        )
        short_future = pool.submit(
            _run_core, df, short_strategy, trade_side=TradeSide.SHORT,
            calendar=calendar, flatten=flatten, cost=cost,
        )
        long = long_future.result()
        short = short_future.result()

    # Validate mutual exclusivity
    overlap = (long.signals["_position"] == 1) & (short.signals["_position"] == 1)
    if overlap.any():
        msg = "Long and short strategies have overlapping positions"
        raise ValueError(msg)

    # Merge returns (additive — flat bars contribute 0)
    returns = pl.DataFrame({
        "date": long.returns["date"],
        "return": long.returns["return"] + short.returns["return"],
    })

    # Merge trades (concat + sort by entry_date)
    trades = pl.concat([long.trades, short.trades]).sort("entry_date")

    # Merge signals: long signals as base, overlay combined columns
    signals = long.signals.with_columns(
        (long.signals["_entry"] | short.signals["_entry"]).alias("_entry"),
        (long.signals["_exit"] | short.signals["_exit"]).alias("_exit"),
        (long.signals["_position"] + short.signals["_position"]).alias("_position"),
        (long.signals["_side"] + short.signals["_side"]).alias("_side"),
    )

    return BacktestResult(returns=returns, trades=trades, signals=signals)


def _run_multi(
    df: pl.DataFrame,
    strategy: Strategy,
    *,
    short_strategy: Strategy | None = None,
    instrument_col: str,
    trade_side: TradeSide,
    calendar: ExchangeCalendar | None,
    flatten: FlattenSchedule | None,
    weights: pl.DataFrame | None = None,
    cost: Cost | None = None,
    bracket: Bracket | None = None,
) -> MultiBacktestResult:
    """Run independent backtests per instrument and combine results."""
    if instrument_col not in df.columns:
        msg = f"instrument_col={instrument_col!r} not found in DataFrame columns"
        raise ValueError(msg)

    # Calendar filter once on full df
    if calendar is not None:
        df = calendar.filter_market_hours(df, "date")

    by_instrument: dict[str, BacktestResult] = {}
    for inst_df in df.partition_by(instrument_col, maintain_order=True):
        instrument = inst_df[instrument_col][0]
        inst_data = inst_df.drop(instrument_col)
        if short_strategy is not None:
            by_instrument[instrument] = _run_dual(
                inst_data,
                strategy,
                short_strategy,
                calendar=calendar,
                flatten=flatten,
                cost=cost,
                bracket=bracket,
            )
        else:
            by_instrument[instrument] = _run_core(
                inst_data,
                strategy,
                trade_side=trade_side,
                calendar=calendar,
                flatten=flatten,
                cost=cost,
                bracket=bracket,
            )

    if weights is not None:
        _cross_validate_weights(by_instrument, weights)

    return MultiBacktestResult(
        by_instrument, instrument_col=instrument_col, weights=weights,
    )


def _cross_validate_weights(
    by_instrument: dict[str, BacktestResult],
    weights: pl.DataFrame,
) -> None:
    """Check that every backtested symbol has a weight; warn on extra weights.

    Symbols in the data but not in *weights* are ambiguous — raise. Symbols
    in *weights* but not in the data are harmless (master-config pattern) —
    log a warning and let the downstream inner-join drop them.
    """
    data_symbols = set(by_instrument)
    weight_symbols = set(weights[INSTRUMENT_COLUMN].to_list())

    missing = data_symbols - weight_symbols
    if missing:
        msg = (
            f"symbols backtested but not in instrument_weights: {sorted(missing)}. "
            "Either add them to the weights dict or filter them out of the input "
            "DataFrame before calling run()."
        )
        raise InvalidPortfolioWeights(msg)

    extras = weight_symbols - data_symbols
    if extras:
        logger.warning(
            "instrument_weights contains symbols not present in the data; "
            "these will be ignored: %s",
            sorted(extras),
        )


@overload
def run(
    df: pl.DataFrame,
    strategy: Strategy,
    *,
    short_strategy: Strategy,
    calendar: ExchangeCalendar | None = ...,
    flatten: Flatten = ...,
    flatten_eod: bool = ...,
    cost: Cost | None = ...,
    bracket: Bracket | None = ...,
    instrument_col: str,
    instrument_weights: Mapping[str, float] | pl.DataFrame | None = ...,
) -> MultiBacktestResult: ...


@overload
def run(
    df: pl.DataFrame,
    strategy: Strategy,
    *,
    short_strategy: Strategy,
    calendar: ExchangeCalendar | None = ...,
    flatten: Flatten = ...,
    flatten_eod: bool = ...,
    cost: Cost | None = ...,
    bracket: Bracket | None = ...,
    instrument_col: None = ...,
) -> BacktestResult: ...


@overload
def run(
    df: pl.DataFrame,
    strategy: Strategy,
    *,
    trade_side: TradeSide = ...,
    calendar: ExchangeCalendar | None = ...,
    flatten: Flatten = ...,
    flatten_eod: bool = ...,
    cost: Cost | None = ...,
    bracket: Bracket | None = ...,
    instrument_col: str,
    instrument_weights: Mapping[str, float] | pl.DataFrame | None = ...,
) -> MultiBacktestResult: ...


@overload
def run(
    df: pl.DataFrame,
    strategy: Strategy,
    *,
    trade_side: TradeSide = ...,
    calendar: ExchangeCalendar | None = ...,
    flatten: Flatten = ...,
    flatten_eod: bool = ...,
    cost: Cost | None = ...,
    bracket: Bracket | None = ...,
    instrument_col: None = ...,
) -> BacktestResult: ...


@overload
def run(
    df: pl.DataFrame,
    strategy: Strategy,
    *,
    trade_side: TradeSide = ...,
    calendar: ExchangeCalendar | None = ...,
    flatten: Flatten = ...,
    flatten_eod: bool = ...,
    cost: Cost | None = ...,
    bracket: Bracket | None = ...,
    instrument_col: None = ...,
    instrument_weights: Mapping[str, float] | pl.DataFrame,
) -> MultiBacktestResult: ...


def run(
    df: pl.DataFrame,
    strategy: Strategy,
    *,
    short_strategy: Strategy | None = None,
    trade_side: TradeSide = TradeSide.LONG,
    calendar: ExchangeCalendar | None = None,
    flatten: Flatten = None,
    flatten_eod: bool = False,
    cost: Cost | None = None,
    bracket: Bracket | None = None,
    instrument_col: str | None = None,
    instrument_weights: Mapping[str, float] | pl.DataFrame | None = None,
) -> BacktestResult | MultiBacktestResult:
    """Run a vectorized backtest with fill-at-next-open semantics.

    Parameters
    ----------
    df
        Must contain ``date``, ``open``, ``close``, and any indicator
        columns referenced by the strategy.
    strategy
        Object with ``entry()`` and ``exit()`` returning Conditions.
        May optionally define ``init(df) -> pl.DataFrame`` to enrich the
        DataFrame with indicator columns before signal evaluation.
    short_strategy
        When provided, *strategy* is used as the long strategy and
        *short_strategy* as the short strategy. They are run independently,
        validated for mutual exclusivity (no overlapping positions), and
        merged into a single result.
    trade_side
        Trade direction (single-strategy mode only). Overridden by the
        entry condition's ``trade_side`` if set. Ignored when
        *short_strategy* is provided.
    calendar
        Exchange calendar for market-hours filtering. When provided, the
        DataFrame is filtered to market hours before signal computation.
    flatten
        When and how open positions are force-closed. Accepts
        ``"eod"`` (every session's last bar), ``"eow"`` (the last session
        of each week), ``True``/``False``, or a
        :class:`~mktlib.backtest.FlattenSchedule` for full control.
        Requires *calendar*.
    flatten_eod
        Legacy spelling of ``flatten="eod"``. Force-close positions at each
        session's last bar, eliminating overnight exposure. Requires
        *calendar*. Fully supported; prefer *flatten* in new code. Setting
        both *flatten* and ``flatten_eod=True`` raises.
    cost
        Optional :class:`~mktlib.backtest.Cost` describing per-side
        transaction costs in **basis points of notional**. Charged at the
        fill — on the entry bar and on the exit bar — in both the returns
        series and ``trades.pnl``; holding bars are unaffected. ``None``
        (default) and ``Cost()`` are both exact no-ops.
    bracket
        Optional :class:`~mktlib.backtest.Bracket` describing protective
        take-profit / stop-loss levels resting against every position from
        its entry fill bar. The first leg to be tagged closes the position
        *inside* that bar, ahead of the strategy's own exit condition.
        Requires ``high`` and ``low`` columns. Not supported together with
        *short_strategy* or with a ``Limit(...)`` exit condition — both
        raise :class:`NotImplementedError`.
    instrument_col
        Column name identifying the symbol/ticker in a multi-symbol
        DataFrame. When provided, returns a
        :class:`~mktlib.backtest.MultiBacktestResult` that stores
        per-symbol results for O(1) access (``result["AAPL"]``).
        Combined DataFrames with the symbol column prepended are
        available via ``.returns``, ``.trades``, ``.signals`` properties.
    instrument_weights
        Optional portfolio weights for multi-instrument runs. Accepts
        either a ``Mapping[str, float]`` (``{"TQQQ": 0.5, "AAPL": 0.1}``)
        or a ``pl.DataFrame`` with columns ``(instrument, weight)``. When
        supplied, :attr:`MultiBacktestResult.returns` is the weighted
        portfolio time series (``(date, return)``) instead of the
        per-symbol concatenation. Requires *instrument_col*. See
        :mod:`mktlib.backtest._weights` for schema and renormalization
        semantics.

    Returns
    -------
    BacktestResult
        When *instrument_col* is ``None`` (default).
    MultiBacktestResult
        When *instrument_col* is set. Supports ``result[symbol]`` for O(1)
        per-symbol access, iteration, and lazy-cached combined views.

    Notes
    -----
    Signal at bar *t* → market order fills at bar *t+1*'s open.

    - **Entry bar** (*t+1*): return = ``(close - open) / open``
    - **Middle bars**: return = ``close / prev_close - 1``
    - **Exit bar** (first bar where position drops to 0): return =
      ``(open - prev_close) / prev_close`` (gap to fill price only)

    When *cost* is supplied, ``cost_bps / 1e4`` is **subtracted** from the
    entry-bar, exit-bar and limit-fill returns — never from holding bars,
    and never multiplied by the trade side.

    When *bracket* is supplied, a bracketed exit replaces the signal exit
    for that position and fills on the bar that tagged the level. The bars
    between a bracket exit and the signal exit that would have followed it
    return ``0.0``: the position is closed, and a still-true entry signal
    does **not** re-open it until the block ends. See
    :class:`~mktlib.backtest.Bracket` for the full fill table and for why
    same-bar both-touch resolution is an assumption rather than a result.

    When *instrument_col* is set, each symbol is backtested independently —
    indicators (e.g. rolling SMA) do not bleed across symbols. Calendar
    filtering is applied once on the full DataFrame before partitioning.

    If *instrument_weights* is omitted, aggregation stays per-symbol and
    the caller decides how to combine::

        result.returns.group_by("date").agg(pl.col("return").mean())
    """
    flatten_schedule = resolve_flatten(flatten, flatten_eod=flatten_eod)
    if flatten_schedule is not None and calendar is None:
        # Name the spelling the caller actually used — being told that
        # "flatten requires a calendar" when you wrote flatten_eod=True sends
        # you looking for a parameter you never passed.
        used = "flatten_eod=True" if flatten is None else "flatten=..."
        msg = (
            f"{used} requires a calendar: sessions, and therefore session "
            "closes, are defined by the calendar"
        )
        raise ValueError(msg)

    if instrument_weights is not None and instrument_col is None:
        # Canonical column name for multi-instrument runs. Callers in the
        # quant-finance convention use ``instrument`` for the ticker column;
        # default to that when they've signaled multi via weights but not
        # named the column explicitly.
        instrument_col = "instrument"

    weights_df: pl.DataFrame | None = None
    if instrument_weights is not None:
        # Validate early so input errors surface before expensive compute.
        weights_df = to_portfolio_weights_df(instrument_weights)

    if short_strategy is not None:
        if trade_side is not TradeSide.LONG:
            msg = "trade_side is ignored when short_strategy is provided"
            raise ValueError(msg)
        if bracket is not None:
            raise NotImplementedError(_DUAL_BRACKET_MSG)
        if instrument_col is not None:
            return _run_multi(
                df,
                strategy,
                short_strategy=short_strategy,
                instrument_col=instrument_col,
                trade_side=TradeSide.LONG,
                calendar=calendar,
                flatten=flatten_schedule,
                weights=weights_df,
                cost=cost,
                bracket=bracket,
            )
        if calendar is not None:
            df = calendar.filter_market_hours(df, "date")
        return _run_dual(
            df,
            strategy,
            short_strategy,
            calendar=calendar,
            flatten=flatten_schedule,
            cost=cost,
            bracket=bracket,
        )

    if instrument_col is not None:
        return _run_multi(
            df,
            strategy,
            instrument_col=instrument_col,
            trade_side=trade_side,
            calendar=calendar,
            flatten=flatten_schedule,
            weights=weights_df,
            cost=cost,
            bracket=bracket,
        )

    # Single-symbol path: calendar filter → _run_core
    if calendar is not None:
        df = calendar.filter_market_hours(df, "date")

    return _run_core(
        df,
        strategy,
        trade_side=trade_side,
        calendar=calendar,
        flatten=flatten_schedule,
        cost=cost,
        bracket=bracket,
    )


def _extract_trades(
    signals: pl.DataFrame,
    *,
    flatten_active: bool = False,
) -> pl.DataFrame:
    """Extract per-trade PnL from position transitions.

    Fill prices use the *next* bar's open (fill-at-next-open model).
    For session-forced exits (flatten_eod), the exit fill is the
    session-last bar's own open (can't trade during the close minute).

    Side is extracted from ``_side`` at each entry bar.

    When the engine materialized a per-bar cost column, each leg's cost is
    read with **exactly the same alignment as the price it pays**: the entry
    leg from the next bar (where ``entry_price`` comes from), the exit leg
    mirroring ``exit_price``'s bracket / limit / session-last / next-open
    priority.

    A bracket exit fills *inside* the bar that tagged the level, so its row
    is that fill bar and ``exit_date`` names it directly — unlike a signal
    exit, whose row is the signal bar and whose fill is the next bar's open.
    """
    has_cost = COST_COLUMN in signals.columns

    # Pre-compute next bar's open for fill price
    signals_with_next = signals.with_columns(
        pl.col("open").shift(-1).alias("_next_open"),
        # Row position, taken BEFORE the entry/exit filters. Taking it after
        # would number the surviving rows 0,1,2,... — a trade ordinal, not a
        # bar position — and the difference between two such ordinals is not a
        # holding period.
        pl.int_range(pl.len(), dtype=pl.Int64).alias("_bar_idx"),
    )
    if has_cost:
        signals_with_next = signals_with_next.with_columns(
            pl.col(COST_COLUMN).shift(-1).alias("_next_cost_bps"),
        )

    entry_selects = [
        pl.col("date").alias("entry_date"),
        pl.col("_next_open").alias("entry_price"),
        pl.col("_side").alias("_trade_side"),
        # An entry always fills at the next bar's open — the same bar
        # ``entry_price`` is read from.
        (pl.col("_bar_idx") + 1).alias("_entry_fill_idx"),
    ]
    if has_cost:
        # Entry fills at _next_open, so its cost is the *next* bar's cost.
        entry_selects.append(pl.col("_next_cost_bps").alias("_entry_cost_bps"))
    entries = signals_with_next.filter(pl.col("_entry_clean")).select(entry_selects)

    # Exit fill price priority:
    #   1. _bracket_level when a bracket leg closed the position on this bar
    #   2. _limit_price when a same-bar limit exit fired on this bar
    #   3. session-last bar's own open (flatten_eod path)
    #   4. next bar's open (default fill-at-next-open)
    has_limit = "_limit_price" in signals.columns
    has_bracket = BRACKET_LEVEL_COLUMN in signals.columns
    if flatten_active:
        base_expr = (
            pl.when(pl.col(FLATTEN_BAR_COLUMN))
            .then(pl.col("open"))
            .otherwise(pl.col("_next_open"))
        )
    else:
        base_expr = pl.col("_next_open")

    if has_limit:
        base_expr = (
            pl.when(pl.col("_limit_price").is_not_null())
            .then(pl.col("_limit_price"))
            .otherwise(base_expr)
        )
    if has_bracket:
        base_expr = (
            pl.when(pl.col(BRACKET_LEVEL_COLUMN).is_not_null())
            .then(pl.col(BRACKET_LEVEL_COLUMN))
            .otherwise(base_expr)
        )
    exit_price_expr = base_expr.alias("exit_price")

    # The bar the exit FILLS on, mirroring the priority order directly above.
    # Price, cost and index are three mirrors of one ordering: a bracket or a
    # limit fills inside its own bar, a session-forced exit at that bar's open,
    # and everything else at the next bar's open. Deliberately not factored
    # into a shared helper — the branch *values* differ (a price, a cost, an
    # index), only the branch *conditions* are common, and hiding three
    # sequences behind one abstraction would make a future divergence harder
    # to see, not easier.
    if flatten_active:
        base_idx = (
            pl.when(pl.col(FLATTEN_BAR_COLUMN))
            .then(pl.col("_bar_idx"))
            .otherwise(pl.col("_bar_idx") + 1)
        )
    else:
        base_idx = pl.col("_bar_idx") + 1

    if has_limit:
        base_idx = (
            pl.when(pl.col("_limit_price").is_not_null())
            .then(pl.col("_bar_idx"))
            .otherwise(base_idx)
        )
    if has_bracket:
        base_idx = (
            pl.when(pl.col(BRACKET_LEVEL_COLUMN).is_not_null())
            .then(pl.col("_bar_idx"))
            .otherwise(base_idx)
        )

    exit_selects = [
        pl.col("date").alias("exit_date"),
        exit_price_expr,
        base_idx.cast(pl.Int64).alias("_exit_fill_idx"),
    ]
    if has_cost:
        # Mirror the exit_price priority exactly — a same-bar fill pays this
        # bar's cost, a next-open fill pays the next bar's.
        if flatten_active:
            base_cost = (
                pl.when(pl.col(FLATTEN_BAR_COLUMN))
                .then(pl.col(COST_COLUMN))
                .otherwise(pl.col("_next_cost_bps"))
            )
        else:
            base_cost = pl.col("_next_cost_bps")
        if has_limit:
            base_cost = (
                pl.when(pl.col("_limit_price").is_not_null())
                .then(pl.col(COST_COLUMN))
                .otherwise(base_cost)
            )
        if has_bracket:
            base_cost = (
                pl.when(pl.col(BRACKET_LEVEL_COLUMN).is_not_null())
                .then(pl.col(COST_COLUMN))
                .otherwise(base_cost)
            )
        exit_selects.append(base_cost.alias("_exit_cost_bps"))

    exits = signals_with_next.filter(pl.col("_exit_clean")).select(exit_selects)

    # Pair entries with exits by ordinal position
    n_trades = min(entries.height, exits.height)
    if n_trades == 0:
        return pl.DataFrame(
            schema={
                "entry_date": signals["date"].dtype,
                "exit_date": signals["date"].dtype,
                "side": pl.Int8,
                "pnl": pl.Float64,
                "bars_held": pl.Int64,
            }
        )

    entries = entries.head(n_trades)
    exits = exits.head(n_trades)

    trade_cols = {
        "entry_date": entries["entry_date"],
        "exit_date": exits["exit_date"],
        "side": entries["_trade_side"],
        "entry_price": entries["entry_price"],
        "exit_price": exits["exit_price"],
        "_entry_fill_idx": entries["_entry_fill_idx"],
        "_exit_fill_idx": exits["_exit_fill_idx"],
    }
    if has_cost:
        trade_cols["_entry_cost_bps"] = entries["_entry_cost_bps"]
        trade_cols["_exit_cost_bps"] = exits["_exit_cost_bps"]
    trades = pl.DataFrame(trade_cols)

    pnl_expr = pl.col("side") * (pl.col("exit_price") / pl.col("entry_price") - 1)
    if has_cost:
        pnl_expr = pnl_expr - (
            pl.col("_entry_cost_bps") + pl.col("_exit_cost_bps")
        ) / 1e4

    trades = trades.with_columns(
        pnl_expr.alias("pnl"),
        # Bars, not calendar days: the fill-to-fill distance in rows. Zero when
        # a bracket or limit closes the position on the very bar the entry
        # filled on, which is a real same-bar round trip rather than a
        # degenerate one.
        (
            pl.col("_exit_fill_idx") - pl.col("_entry_fill_idx")
        ).alias("bars_held"),
    ).select("entry_date", "exit_date", "side", "pnl", "bars_held")

    return trades
