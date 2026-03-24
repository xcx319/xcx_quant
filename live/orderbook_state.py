from __future__ import annotations
import numpy as np
from collections import deque


class OrderbookState:
    """Maintains live orderbook state from Gate.io futures.order_book and computes OB features."""

    def __init__(self):
        self._bids: list[tuple[float, float]] = []
        self._asks: list[tuple[float, float]] = []
        self._obi_history: deque[float] = deque(maxlen=30)
        self._spread_history: deque[float] = deque(maxlen=30)
        self._update_count = 0

    def update(self, data: dict):
        """Parse Gate.io futures.order_book update.
        Gate format: {"asks": [{"p": "3001.1", "s": 100}, ...], "bids": [...]}
        """
        bids_raw = data.get("bids", [])
        asks_raw = data.get("asks", [])
        self._bids = [(float(b["p"]), float(b["s"])) for b in bids_raw if b.get("s", 0) != 0]
        self._asks = [(float(a["p"]), float(a["s"])) for a in asks_raw if a.get("s", 0) != 0]
        # Sort: bids descending, asks ascending
        self._bids.sort(key=lambda x: -x[0])
        self._asks.sort(key=lambda x: x[0])
        self._update_count += 1

    def get_features(self) -> dict:
        if not self._bids or not self._asks:
            return {}

        bid_prices, bid_sizes = zip(*self._bids) if self._bids else ([], [])
        ask_prices, ask_sizes = zip(*self._asks) if self._asks else ([], [])
        bp, bs = np.array(bid_prices), np.array(bid_sizes)
        ap, a_s = np.array(ask_prices), np.array(ask_sizes)

        best_bid, best_ask = bp[0], ap[0]
        mid = (best_bid + best_ask) / 2.0
        spread = best_ask - best_bid
        spread_bps = (spread / mid) * 1e4 if mid > 0 else 0.0

        total_bid = float(bs.sum())
        total_ask = float(a_s.sum())
        total = total_bid + total_ask
        obi = (total_bid - total_ask) / total if total > 0 else 0.0

        self._obi_history.append(obi)
        self._spread_history.append(spread_bps)

        depth_bid_1 = float(bs[0]) if len(bs) > 0 else 0.0
        depth_ask_1 = float(a_s[0]) if len(a_s) > 0 else 0.0
        depth_bid_5 = float(bs[:5].sum())
        depth_ask_5 = float(a_s[:5].sum())
        depth_bid_20 = float(bs[:20].sum())
        depth_ask_20 = float(a_s[:20].sum())

        microprice = (best_bid * a_s[0] + best_ask * bs[0]) / (bs[0] + a_s[0]) if (bs[0] + a_s[0]) > 0 else mid

        max_bid_size = float(bs.max()) if len(bs) > 0 else 0.0
        max_ask_size = float(a_s.max()) if len(a_s) > 0 else 0.0
        wall_conc_bid = max_bid_size / total_bid if total_bid > 0 else 0.0
        wall_conc_ask = max_ask_size / total_ask if total_ask > 0 else 0.0

        return {
            "obi": obi,
            "obi_1": (bs[0] - a_s[0]) / (bs[0] + a_s[0]) if (bs[0] + a_s[0]) > 0 else 0.0,
            "obi_5": obi,
            "obi_20": obi,
            "ob_spread_bps": spread_bps,
            "spread": spread,
            "ob_depth_bid_1": depth_bid_1,
            "ob_depth_ask_1": depth_ask_1,
            "ob_depth_bid_5": depth_bid_5,
            "ob_depth_ask_5": depth_ask_5,
            "ob_depth_bid_20": depth_bid_20,
            "ob_depth_ask_20": depth_ask_20,
            "ob_microprice": microprice,
            "ob_mid_close": mid,
            "ob_ask_wall_size_20": max_ask_size,
            "ob_bid_wall_size_20": max_bid_size,
            "ob_ask_wall_conc_20": wall_conc_ask,
            "ob_bid_wall_conc_20": wall_conc_bid,
            "ob_ask_wall_levels_20": int((a_s > a_s.mean() * 2).sum()) if len(a_s) > 0 else 0,
            "ob_bid_wall_levels_20": int((bs > bs.mean() * 2).sum()) if len(bs) > 0 else 0,
            "ob_quote_count": self._update_count,
            "data_from_orderbook": 1,
        }

    def get_snapshot(self) -> dict:
        if not self._bids or not self._asks:
            return {"bids": [], "asks": [], "obi": 0.0, "spread": 0.0, "spread_bps": 0.0, "mid": 0.0}
        bids = [{"price": p, "size": s} for p, s in self._bids[:5]]
        asks = [{"price": p, "size": s} for p, s in self._asks[:5]]
        best_bid, best_ask = self._bids[0][0], self._asks[0][0]
        mid = (best_bid + best_ask) / 2.0
        spread = best_ask - best_bid
        spread_bps = (spread / mid) * 1e4 if mid > 0 else 0.0
        total_bid = sum(s for _, s in self._bids[:5])
        total_ask = sum(s for _, s in self._asks[:5])
        total = total_bid + total_ask
        obi = (total_bid - total_ask) / total if total > 0 else 0.0
        return {"bids": bids, "asks": asks, "obi": obi, "spread": spread, "spread_bps": spread_bps, "mid": mid}

    def reset_quote_count(self):
        self._update_count = 0
