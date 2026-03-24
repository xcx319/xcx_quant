from __future__ import annotations
import numpy as np
from collections import deque
from datetime import datetime, timezone


def _gini(x):
    if len(x) == 0:
        return 0.0
    x = np.sort(np.asarray(x, dtype=np.float64))
    n = len(x)
    s = x.sum()
    if s == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float(((2 * idx - n - 1) * x).sum() / (n * s))


class BarAggregator:
    """Accumulates raw trades into 1-minute OHLCV bars with microstructure features."""

    def __init__(self, on_bar_complete):
        self.on_bar_complete = on_bar_complete
        self._current_minute: datetime | None = None
        self._reset_accum()
        self._avg_trade_size = deque(maxlen=1000)
        self._tick_buffer: deque[tuple[int, float, float]] = deque(maxlen=5000)

    def _reset_accum(self):
        self._open = self._high = self._low = self._close = 0.0
        self._volume = 0.0
        self._buy_vol = 0.0
        self._sell_vol = 0.0
        self._trade_sizes: list[float] = []
        self._count = 0

    def ingest_trade(self, price: float, size: float, side: str, ts_ms: int):
        """Ingest a single trade tick.
        Gate.io: size is signed (positive=buy, negative=sell), pass abs(size) here.
        side: 'buy' or 'sell'
        ts_ms: millisecond timestamp
        """
        dt = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)
        minute = dt.replace(second=0, microsecond=0)

        if self._current_minute is None:
            self._current_minute = minute

        if minute > self._current_minute:
            self._finalize()
            self._current_minute = minute

        if self._count == 0:
            self._open = price
            self._high = price
            self._low = price
        else:
            self._high = max(self._high, price)
            self._low = min(self._low, price)
        self._close = price
        self._volume += size
        if side == "buy":
            self._buy_vol += size
        else:
            self._sell_vol += size
        self._trade_sizes.append(size)
        self._avg_trade_size.append(size)
        self._count += 1
        self._tick_buffer.append((ts_ms * 1_000_000, price, size))

    def _finalize(self):
        if self._count == 0:
            return
        vol = max(self._volume, 1e-12)
        aggressor_ratio = self._buy_vol / vol
        net_taker = (self._buy_vol - self._sell_vol) / vol

        avg_sz = float(np.mean(list(self._avg_trade_size))) if self._avg_trade_size else 1.0
        large_mask = np.array(self._trade_sizes) > avg_sz * 5
        large_ratio = float(np.sum(np.array(self._trade_sizes)[large_mask]) / vol) if vol > 0 else 0.0

        bar = {
            "datetime": self._current_minute,
            "open": self._open,
            "high": self._high,
            "low": self._low,
            "close": self._close,
            "volume": self._volume,
            "aggressor_ratio": aggressor_ratio,
            "net_taker_vol_ratio": net_taker,
            "trade_gini": _gini(self._trade_sizes),
            "large_trade_vol_ratio": large_ratio,
            "trade_intensity": float(np.log1p(self._volume)),
        }
        self._reset_accum()
        self.on_bar_complete(bar)
