# Add `geometric: bool` parameter to FRW and OU generators

## Context

All three data generators sample from normal distributions, but construct prices differently:
- **GBM**: already geometric — `S₀ * exp(cumsum(log_returns))`, prices always positive
- **FRW**: additive — `base + cumsum(increments)`, prices can go negative
- **OU**: additive — mean-reverting AR(1), prices can go negative

For stock price simulation, lognormal (geometric) prices are more realistic: they can't go negative and exhibit multiplicative returns. The standard approach is **not** to sample from a lognormal distribution, but to exponentiate the additive process: `price = base * exp(cumsum(increments))`.

## Design decisions

1. **`geometric: bool = False`** on FRW and OU. Simple, binary. Default preserves current behavior.
2. **GBM is exempt** — already inherently geometric.
3. **Each generator applies `exp()` in its own pipeline** — no shared helper needed for a single `.exp()` call.
4. **OU `geometric=True`**: params (`mu`, `x0`) stay in log-space (standard convention, Schwartz 1997). Output column changes from `"value"` to `"price"`.
5. **Monte Carlo**: `geometric` flows through `**process_kwargs` to vectorized helpers — no changes to `monte_carlo()` signature.

## Files

### `mktlib/data/_random_walk.py`

Add `geometric: bool = False` to `fractional_random_walk()` signature.

**hurst == 0.5 path** (L80–86): Replace the additive price construction with a conditional:
```python
cumulative = increments.cum_sum()
if geometric:
    prices = (pl.Series("price", [0.0]).append(cumulative)[:n].exp() * base_price)
else:
    prices = pl.Series("price", [base_price]).append(cumulative + base_price)[:n]
```

**Davies-Harte path** (L102–106): Replace the price expression:
```python
cumulative = (pl.col("increment") * scale).cum_sum().shift(1).fill_null(0.0)
price_expr = (base_price * cumulative.exp() if geometric else base_price + cumulative).alias("price")
```

### `mktlib/data/_ornstein_uhlenbeck.py`

Add `geometric: bool = False` to `ornstein_uhlenbeck()` signature.

After L50 `.with_columns(_ou_value_expr(...))`, conditionally exponentiate:
```python
if geometric:
    return df.with_columns(pl.col("value").exp().alias("price")).select("step", "price")
return df.select("step", "value")
```

Update docstring: when `geometric=True`, `mu` and `x0` are in log-space, output is `exp(OU)`.

### `mktlib/data/_monte_carlo.py`

**`_vectorized_frw`** (L160): Add `geometric: bool = False`. Both code paths (hurst==0.5 at L183, Davies-Harte at L245) get conditional `exp()`:
```python
cumulative = <existing cumsum expression>.over("simulation")
price_expr = (base_price * cumulative.exp() if geometric else base_price + cumulative).alias("price")
```

**`_vectorized_ou`** (L128): Add `geometric: bool = False`. After L154:
```python
if geometric:
    return df.with_columns(pl.col("value").exp().alias("price")).select("simulation", "step", "price")
return df.select("simulation", "step", "value")
```

**`_vectorized_gbm`, `monte_carlo()`**: No changes.

### Tests

**`tests/data/test_random_walk.py`** — new class:
- `geometric=True` → prices always positive
- `geometric=True` → first price == `base_price`
- `geometric=False` (default) output unchanged from current
- Works for hurst > 0.5 (Davies-Harte path)
- Seeded reproducibility

**`tests/data/test_ornstein_uhlenbeck.py`** — new class:
- `geometric=True` → column is `"price"`, always positive
- `geometric=True` output == `exp(additive output)` (exact equivalence)
- Default unchanged

**`tests/data/test_monte_carlo.py`** — new tests:
- FRW geometric → prices positive
- OU geometric → column is `"price"`, prices positive

## Verification

```bash
pytest tests/data/ -v
```
