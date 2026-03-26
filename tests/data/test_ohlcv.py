from __future__ import annotations

import polars as pl
import pytest

from mktlib.data import geometric_brownian_motion, ornstein_uhlenbeck, ticks_to_ohlcv


class TestTicksToOhlcv:
    def test_shape_and_columns(self):
        gbm = geometric_brownian_motion(100, seed=42)
        df = ticks_to_ohlcv(gbm, bar_size=10, seed=1)
        assert df.shape == (10, 6)

    def test_ohlc_consistency(self):
        gbm = geometric_brownian_motion(200, seed=7)
        df = ticks_to_ohlcv(gbm, bar_size=20, seed=1)
        for row in df.iter_rows(named=True):
            assert row["high"] >= max(row["open"], row["close"])
            assert row["low"] <= min(row["open"], row["close"])

    def test_adjacency(self):
        """close[i] and open[i+1] are adjacent ticks (1 tick apart)."""
        gbm = geometric_brownian_motion(50, seed=42)
        df = ticks_to_ohlcv(gbm, bar_size=10, seed=1)
        ticks_list = gbm["price"].to_list()
        closes = df["close"].to_list()
        opens = df["open"].to_list()
        for i in range(len(closes) - 1):
            # close[i] is the last tick of bar i, open[i+1] is the first tick of bar i+1
            close_idx = (i + 1) * 10 - 1
            open_idx = (i + 1) * 10
            assert closes[i] == ticks_list[close_idx]
            assert opens[i + 1] == ticks_list[open_idx]

    def test_no_volume_column(self):
        gbm = geometric_brownian_motion(100, seed=42)
        df = ticks_to_ohlcv(gbm, bar_size=10, volume=False)
        assert "volume" not in df.columns

    def test_reproducibility(self):
        gbm = geometric_brownian_motion(100, seed=42)
        df1 = ticks_to_ohlcv(gbm, bar_size=10, seed=99)
        df2 = ticks_to_ohlcv(gbm, bar_size=10, seed=99)
        assert df1.equals(df2)

    def test_incomplete_tail_dropped(self):
        """n=105, bar_size=10 → 10 bars (5 leftover ticks dropped)."""
        gbm = geometric_brownian_motion(105, seed=42)
        df = ticks_to_ohlcv(gbm, bar_size=10, seed=1)
        assert df.shape[0] == 10

    def test_bar_size_less_than_one_raises(self):
        gbm = geometric_brownian_motion(10, seed=42)
        with pytest.raises(ValueError, match="bar_size must be >= 1"):
            ticks_to_ohlcv(gbm, bar_size=0)

    def test_missing_price_column_raises(self):
        df = pl.DataFrame({"step": [1, 2, 3], "value": [10.0, 11.0, 12.0]})
        with pytest.raises(ValueError, match="'price'"):
            ticks_to_ohlcv(df, bar_size=1)

    def test_missing_custom_column_raises(self):
        df = pl.DataFrame({"step": [1, 2, 3], "price": [10.0, 11.0, 12.0]})
        with pytest.raises(ValueError, match="'spread'"):
            ticks_to_ohlcv(df, bar_size=1, column="spread")

    def test_bar_index_zero_based(self):
        gbm = geometric_brownian_motion(31, seed=42)
        df = ticks_to_ohlcv(gbm, bar_size=10, volume=False)
        assert df["bar"].to_list() == [0, 1, 2]

    def test_exact_multiple_no_drop(self):
        """n=60, bar_size=10 → 6 bars (no ticks dropped)."""
        gbm = geometric_brownian_motion(60, seed=42)
        df = ticks_to_ohlcv(gbm, bar_size=10, seed=1)
        assert df.shape[0] == 6

    def test_ou_with_value_column(self):
        """OU process outputs 'value' column — pass column='value'."""
        ou = ornstein_uhlenbeck(100, seed=42)
        df = ticks_to_ohlcv(ou, bar_size=10, column="value", seed=1)
        assert df.shape == (10, 6)
        for row in df.iter_rows(named=True):
            assert row["high"] >= max(row["open"], row["close"])
            assert row["low"] <= min(row["open"], row["close"])
