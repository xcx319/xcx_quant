from __future__ import annotations
import base64, hashlib, hmac, json, logging, time
from datetime import datetime, timezone
from typing import Optional

import httpx

from . import config

logger = logging.getLogger(__name__)


class OrderExecutor:
    """OKX REST API client for order placement and position management."""

    def __init__(self):
        self._client = httpx.Client(base_url=config.REST_BASE, timeout=10)

    def _sign(self, timestamp: str, method: str, path: str, body: str = "") -> dict:
        prehash = timestamp + method.upper() + path + body
        mac = hmac.new(
            config.OKX_SECRET_KEY.encode(), prehash.encode(), hashlib.sha256
        )
        sig = base64.b64encode(mac.digest()).decode()
        return {
            "OK-ACCESS-KEY": config.OKX_API_KEY,
            "OK-ACCESS-SIGN": sig,
            "OK-ACCESS-TIMESTAMP": timestamp,
            "OK-ACCESS-PASSPHRASE": config.OKX_PASSPHRASE,
            "Content-Type": "application/json",
            "x-simulated-trading": config.FLAG,
        }

    def _ts(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

    def _get(self, path: str, params: dict | None = None) -> dict:
        qs = ""
        if params:
            qs = "?" + "&".join(f"{k}={v}" for k, v in params.items() if v)
        ts = self._ts()
        headers = self._sign(ts, "GET", path + qs)
        resp = self._client.get(path + qs, headers=headers)
        return resp.json()

    def _post(self, path: str, body: dict) -> dict:
        ts = self._ts()
        body_str = json.dumps(body)
        headers = self._sign(ts, "POST", path, body_str)
        resp = self._client.post(path, content=body_str, headers=headers)
        return resp.json()

    # --- Public endpoints ---

    def get_positions(self) -> list[dict]:
        r = self._get("/api/v5/account/positions", {"instId": config.INST_ID})
        return r.get("data", [])

    def get_all_positions(self) -> list[dict]:
        """Fetch all positions across all instruments."""
        r = self._get("/api/v5/account/positions", {})
        return r.get("data", [])

    def get_balance(self) -> dict:
        r = self._get("/api/v5/account/balance")
        return r.get("data", [{}])[0] if r.get("data") else {}

    def get_candles(self, bar: str = "1m", limit: int = 300) -> list[list]:
        r = self._get("/api/v5/market/candles", {"instId": config.INST_ID, "bar": bar, "limit": str(limit)})
        return r.get("data", [])

    def get_instruments(self) -> dict:
        r = self._get("/api/v5/public/instruments", {"instType": "SWAP", "instId": config.INST_ID})
        data = r.get("data", [])
        return data[0] if data else {}

    # --- Account setup ---

    def set_leverage(self, lever: int) -> dict:
        """Set leverage for INST_ID."""
        body = {
            "instId": config.INST_ID,
            "lever": str(lever),
            "mgnMode": config.TD_MODE,
        }
        logger.info(f"Setting leverage to {lever}x for {config.INST_ID}")
        result = self._post("/api/v5/account/set-leverage", body)
        logger.info(f"Set leverage result: {result}")
        return result

    # --- Trading ---

    def place_market_order(self, side: str, size: str, tp_price: float, sl_price: float) -> dict:
        body = {
            "instId": config.INST_ID,
            "tdMode": config.TD_MODE,
            "side": side,
            "ordType": "market",
            "sz": size,
            "attachAlgoOrds": [{
                "tpTriggerPx": f"{tp_price:.2f}",
                "tpOrdPx": "-1",
                "slTriggerPx": f"{sl_price:.2f}",
                "slOrdPx": "-1",
            }],
        }
        logger.info(f"Placing order: {side} {size} @ market, TP={tp_price:.2f}, SL={sl_price:.2f}")
        result = self._post("/api/v5/trade/order", body)
        logger.info(f"Order result: {result}")
        return result

    def get_order_history(self, inst_type: str = "SWAP", limit: int = 50) -> list[dict]:
        """Fetch recent filled order history (last 7 days)."""
        r = self._get("/api/v5/trade/orders-history-archive", {
            "instType": inst_type,
            "instId": config.INST_ID,
            "limit": str(limit),
        })
        return r.get("data", [])

    def get_order_detail(self, ord_id: str) -> dict:
        """Fetch order detail by ordId to get fillTime/fillPx."""
        r = self._get("/api/v5/trade/order", {"instId": config.INST_ID, "ordId": ord_id})
        data = r.get("data", [])
        return data[0] if data else {}

    def close_position(self) -> dict:
        """Market-close the current position on INST_ID."""
        body = {
            "instId": config.INST_ID,
            "mgnMode": config.TD_MODE,
        }
        logger.info(f"Closing position: {config.INST_ID}")
        result = self._post("/api/v5/trade/close-position", body)
        logger.info(f"Close result: {result}")
        return result

    def close(self):
        self._client.close()
