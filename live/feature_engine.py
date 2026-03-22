from __future__ import annotations
import sys, logging
from collections import deque
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from pipline_modified import add_features  # noqa: E402
from . import config

logger = logging.getLogger(__name__)


class FeatureEngine:
    """Rolling window feature computation. Reuses add_features() for exact parity."""

    def __init__(self, warmup_bars: int = config.WARMUP_BARS):
        self.warmup_bars = warmup_bars
        self._bars: deque[dict] = deque(maxlen=config.BAR_WINDOW)
        self._bar_count = 0
        self.is_warm = False

    def add_bar(self, bar: dict) -> Optional[pd.Series]:
        self._bars.append(bar)
        self._bar_count += 1
        if self._bar_count < self.warmup_bars:
            return None
        self.is_warm = True
        return self._compute()

    def _compute(self) -> pd.Series:
        df = pd.DataFrame(list(self._bars))
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.set_index("datetime").sort_index()
        feats = add_features(df)
        return feats.iloc[-1]

    @property
    def bar_count(self) -> int:
        return self._bar_count

    @property
    def last_bar(self) -> Optional[dict]:
        return self._bars[-1] if self._bars else None
