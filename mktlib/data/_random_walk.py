from __future__ import annotations

import random

import polars as pl
from polars_sdist import sample_normal


def _ensure_rfft() -> None:
    """Import polars_rfft to register the .rfft namespace on pl.Expr."""
    import polars_rfft  # type: ignore[import-untyped]  # noqa: F401


def _build_covariance_row(n: int, hurst: float) -> pl.Series:
    """Toeplitz coefficients + circulant embedding (length 2n)."""
    h2 = 2 * hurst
    k = pl.arange(0, n, eager=True).alias("k")
    c = 0.5 * (
        (k + 1).cast(pl.Float64).pow(h2)
        - 2 * k.cast(pl.Float64).pow(h2)
        + (k - 1).cast(pl.Float64).abs().pow(h2)
    )
    # Embed in circulant matrix: [c_0, ..., c_{n-1}, 0, c_{n-1}, ..., c_1]
    padding = pl.Series("pad", [0.0])
    return pl.concat([c, padding, c[1:].reverse()], rechunk=True).rename("cov_row")


def _sqrt_eigenvalue_expr(col: str = "cov_row") -> pl.Expr:
    """FFT of circulant row → real part → clamp ≥ 0 → sqrt."""
    _ensure_rfft()
    return (
        pl.col(col)
        .rfft.fft()  # type: ignore[attr-defined]
        .struct.field("re")
        .clip(lower_bound=0.0)
        .sqrt()
        .alias("sqrt_eig")
    )


def _frw_increments_expr() -> pl.Expr:
    """Build complex product and IFFT, assuming sqrt_eig/z_re/z_im columns.

    Returns a real-valued expression of the IFFT result.
    """
    return (
        pl.struct(
            re=pl.col("sqrt_eig") * pl.col("z_re"),
            im=pl.col("sqrt_eig") * pl.col("z_im"),
        )
        .rfft.ifft_real()  # type: ignore[attr-defined]
        .alias("increment")
    )


def _derive_seeds(seed: int | None) -> tuple[int, int]:
    """Derive two child seeds from a parent seed."""
    rng = random.Random(seed)
    return rng.randrange(2**63), rng.randrange(2**63)


def fractional_random_walk(
    n: int,
    hurst: float = 0.5,
    base_price: float = 100.0,
    step_size: float = 1.0,
    seed: int | None = None,
) -> pl.DataFrame:
    """Discrete-time fractional random walk using Davies-Harte circulant embedding.

    For ``hurst=0.5`` this is a standard random walk.  ``hurst > 0.5`` produces
    trending (persistent) paths; ``hurst < 0.5`` produces mean-reverting
    (anti-persistent) paths.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    if not 0 < hurst < 1:
        raise ValueError("hurst must be in (0, 1)")

    if hurst == 0.5:
        z = sample_normal(n, seed=seed)
        increments = z * step_size
        prices = pl.Series("price", [base_price]).append(
            (increments.cum_sum() + base_price)
        )[:n]
        return pl.DataFrame({"step": range(n), "price": prices})

    # Davies-Harte / circulant embedding — O(n log n)
    cov_row = _build_covariance_row(n, hurst)
    m = len(cov_row)
    seed_re, seed_im = _derive_seeds(seed)
    z_re = sample_normal(m, seed=seed_re)
    z_im = sample_normal(m, seed=seed_im)

    scale = m**0.5 * step_size
    return (
        pl.DataFrame({"cov_row": cov_row, "z_re": z_re, "z_im": z_im})
        .with_columns(_sqrt_eigenvalue_expr())
        .select(_frw_increments_expr())
        .head(n)
        .with_row_index("step")
        .with_columns(
            (
                base_price
                + (pl.col("increment") * scale).cum_sum().shift(1).fill_null(0.0)
            ).alias("price")
        )
        .cast({"step": pl.Int64})
        .select("step", "price")
    )
