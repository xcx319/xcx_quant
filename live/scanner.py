from __future__ import annotations
import logging
from collections import deque
from typing import Optional

import numpy as np
import pandas as pd

from . import config

logger = logging.getLogger(__name__)


class FlowReversalScanner:
    """Evaluates flow_reversal scanner conditions on a feature row."""

    def __init__(self, params: dict | None = None):
        p = params or config.SCANNER_PARAMS
        self.range_hi = p.get("range_hi", 0.7)
        self.range_lo = p.get("range_lo", 0.3)
        self.flow_abs = p.get("flow_abs", 0.05)
        self.obi_abs = p.get("obi_abs", 0.0)
        self.spread_mult = p.get("spread_mult", 0.8)
        self._spread_history: deque[float] = deque(maxlen=30)

    def evaluate_detailed(self, feat: pd.Series) -> dict:
        """Return all 6 indicator values with thresholds and met status for both directions."""
        spread_bps = feat.get("ob_spread_bps", np.nan)
        if np.isfinite(spread_bps):
            self._spread_history.append(spread_bps)
        spread_ref = float(np.median(list(self._spread_history))) if self._spread_history else 0.0

        rp20 = feat.get("range_pos_20", np.nan)
        pv1 = feat.get("price_velocity_1", np.nan)
        ntvr = feat.get("net_taker_vol_ratio", np.nan)
        sfa = feat.get("signed_flow_accel", np.nan)
        obi = feat.get("obi", np.nan)

        vals_finite = all(np.isfinite(v) for v in [rp20, pv1, ntvr, sfa, obi, spread_bps])
        spread_thresh = spread_ref * self.spread_mult

        indicators = [
            {"name": "range_pos_20", "value": float(rp20) if np.isfinite(rp20) else None,
             "long_threshold": f"< {self.range_lo}", "short_threshold": f"> {self.range_hi}",
             "long_met": vals_finite and rp20 < self.range_lo,
             "short_met": vals_finite and rp20 > self.range_hi},
            {"name": "price_velocity_1", "value": float(pv1) if np.isfinite(pv1) else None,
             "long_threshold": "> 0", "short_threshold": "< 0",
             "long_met": vals_finite and pv1 > 0,
             "short_met": vals_finite and pv1 < 0},
            {"name": "net_taker_vol_ratio", "value": float(ntvr) if np.isfinite(ntvr) else None,
             "long_threshold": f"> {self.flow_abs}", "short_threshold": f"< -{self.flow_abs}",
             "long_met": vals_finite and ntvr > self.flow_abs,
             "short_met": vals_finite and ntvr < -self.flow_abs},
            {"name": "signed_flow_accel", "value": float(sfa) if np.isfinite(sfa) else None,
             "long_threshold": "> 0", "short_threshold": "< 0",
             "long_met": vals_finite and sfa > 0,
             "short_met": vals_finite and sfa < 0},
            {"name": "obi", "value": float(obi) if np.isfinite(obi) else None,
             "long_threshold": f"> {self.obi_abs}", "short_threshold": f"< -{self.obi_abs}",
             "long_met": vals_finite and obi > self.obi_abs,
             "short_met": vals_finite and obi < -self.obi_abs},
            {"name": "spread_bps", "value": float(spread_bps) if np.isfinite(spread_bps) else None,
             "long_threshold": f"> {spread_thresh:.2f}", "short_threshold": f"> {spread_thresh:.2f}",
             "long_met": vals_finite and spread_bps > spread_thresh,
             "short_met": vals_finite and spread_bps > spread_thresh},
        ]

        long_met = sum(1 for i in indicators if i["long_met"])
        short_met = sum(1 for i in indicators if i["short_met"])

        trigger = None
        if vals_finite and long_met == 6:
            score = float(ntvr + pv1)
            trigger = {"event_dir": 1, "scanner_score": score, "event_level": feat.get("close", 0.0)}
        elif vals_finite and short_met == 6:
            score = float(-ntvr - pv1)
            trigger = {"event_dir": -1, "scanner_score": score, "event_level": feat.get("close", 0.0)}

        return {
            "indicators": indicators,
            "long_met": long_met,
            "short_met": short_met,
            "trigger": trigger,
        }

    def evaluate(self, feat: pd.Series) -> Optional[dict]:
        result = self.evaluate_detailed(feat)
        return result["trigger"]
