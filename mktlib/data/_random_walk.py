from __future__ import annotations

import polars as pl


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

    try:
        import numpy as np
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "fractional_random_walk requires numpy. Install with: pip install numpy"
        ) from None

    rng = np.random.default_rng(seed)

    if hurst == 0.5:
        increments = rng.normal(0, step_size, n)
    else:
        # Davies-Harte / circulant embedding method — O(n log n)
        h2 = 2 * hurst
        k = np.arange(n)
        # First row of Toeplitz covariance of fBm increments
        c = 0.5 * (np.abs(k + 1) ** h2 - 2 * np.abs(k) ** h2 + np.abs(k - 1) ** h2)
        # Embed in circulant matrix (size 2n)
        row = np.concatenate([c, [0.0], c[-1:0:-1]])
        # Eigenvalues via FFT (real since row is symmetric)
        eigenvalues = np.fft.fft(row).real
        eigenvalues = np.maximum(eigenvalues, 0.0)  # clamp numerical noise
        sqrt_eig = np.sqrt(eigenvalues)
        # Generate complex normal in frequency domain
        m = len(row)
        z = rng.standard_normal(m) + 1j * rng.standard_normal(m)
        # Multiply and IFFT
        w = np.fft.ifft(sqrt_eig * z)
        increments = w[:n].real * step_size

    prices = base_price + np.cumsum(increments)
    prices = np.insert(prices, 0, base_price)[:n]

    return pl.DataFrame(
        {
            "step": range(n),
            "price": prices,
        }
    )
