from __future__ import annotations
import hashlib, hmac, json, logging, time
from typing import Optional

import httpx

from . import config

logger = logging.getLogger(__name__)


class OrderExecutor:
    """Gate.io REST API client for futures order placement and position management."""

    def __init__(self):
        self._client = httpx.Client(base_url=config.REST_BASE, timeout=10)

    def _sign(self, method: str, path: str, query: str = "", body: str = "") -> dict:
        """Gate.io APIv4 HMAC-SHA512 signing.
        sign_str = METHOD\nPATH\nQUERY_STRING\nSHA512(body)\nTIMESTAMP
        """
        timestamp = str(int(time.time()))
        body_hash = hashlib.sha512(body.encode()).hexdigest()
        sign_str = f"{method}\n{path}\n{query}\n{body_hash}\n{timestamp}"
        sig = hmac.new(
            config.GATE_SECRET_KEY.encode(),
            sign_str.encode(),
            hashlib.sha512,
        ).hexdigest()
        return {
            "KEY": config.GATE_API_KEY,
            "SIGN": sig,
            "Timestamp": timestamp,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _get(self, path: str, params: dict | None = None) -> dict | list:
        query = "&".join(f"{k}={v}" for k, v in (params or {}).items() if v is not None)
        headers = self._sign("GET", path, query)
        url = path + ("?" + query if query else "")
        resp = self._client.get(url, headers=headers)
        return resp.json()

    def _post(self, path: str, body: dict) -> dict:
        body_str = json.dumps(body)
        headers = self._sign("POST", path, "", body_str)
        resp = self._client.post(path, content=body_str, headers=headers)
        return resp.json()

    def _delete(self, path: str, params: dict | None = None) -> dict:
        query = "&".join(f"{k}={v}" for k, v in (params or {}).items() if v is not None)
        headers = self._sign("DELETE", path, query)
        url = path + ("?" + query if query else "")
        resp = self._client.delete(url, headers=headers)
        return resp.json()

    # --- Market data ---

    def get_candles(self, interval: str = "1m", limit: int = 300) -> list[dict]:
        """Fetch historical 1-min candles. Returns list of dicts with t,o,h,l,c,v."""
        r = self._get(f"/api/v4/futures/{config.SETTLE}/candlesticks", {
            "contract": config.INST_ID,
            "interval": interval,
            "limit": str(limit),
        })
        return r if isinstance(r, list) else []

    def get_instruments(self) -> dict:
        """Fetch contract info (quanto_multiplier, order_size_min, etc.)."""
        r = self._get(f"/api/v4/futures/{config.SETTLE}/contracts/{config.INST_ID}")
        return r if isinstance(r, dict) else {}

    # --- Account ---

    def get_balance(self) -> dict:
        """Fetch futures account balance."""
        r = self._get(f"/api/v4/futures/{config.SETTLE}/accounts")
        return r if isinstance(r, dict) else {}

    def get_positions(self) -> list[dict]:
        """Fetch all open positions."""
        r = self._get(f"/api/v4/futures/{config.SETTLE}/positions")
        return r if isinstance(r, list) else []

    def get_position(self) -> dict:
        """Fetch position for INST_ID."""
        r = self._get(f"/api/v4/futures/{config.SETTLE}/positions/{config.INST_ID}")
        return r if isinstance(r, dict) else {}

    # --- Leverage ---

    def set_leverage(self, lever: int) -> dict:
        """Set leverage for INST_ID (isolated margin).
        Gate.io expects leverage as query parameter, not JSON body.
        """
        logger.info(f"Setting leverage to {lever}x for {config.INST_ID}")
        path = f"/api/v4/futures/{config.SETTLE}/positions/{config.INST_ID}/leverage"
        query = f"leverage={lever}&cross_leverage_limit=0"
        headers = self._sign("POST", path, query, "")
        url = path + "?" + query
        resp = self._client.post(url, content="", headers=headers)
        r = resp.json()
        logger.info(f"Set leverage result: {r}")
        return r

    # --- Trading ---

    def place_market_order(self, size: int, tp_price: float, sl_price: float) -> dict:
        """Place a market order.
        size: positive = long, negative = short (Gate.io convention)
        Market order: price=0, tif=ioc
        """
        body = {
            "contract": config.INST_ID,
            "size": size,
            "price": "0",
            "tif": "ioc",
            "text": "t-quant-live",
        }
        logger.info(f"Placing order: size={size} @ market, TP={tp_price:.2f}, SL={sl_price:.2f}")
        result = self._post(f"/api/v4/futures/{config.SETTLE}/orders", body)
        logger.info(f"Order result: {result}")
        return result

    def close_position(self) -> dict:
        """Market-close the current position using reduce_only reverse order."""
        pos = self.get_position()
        pos_size = int(pos.get("size", 0))
        if pos_size == 0:
            logger.info("No position to close")
            return {}
        # Reverse direction, reduce_only
        close_size = -pos_size
        body = {
            "contract": config.INST_ID,
            "size": close_size,
            "price": "0",
            "tif": "ioc",
            "reduce_only": True,
            "text": "t-quant-close",
        }
        logger.info(f"Closing position: size={close_size}")
        result = self._post(f"/api/v4/futures/{config.SETTLE}/orders", body)
        logger.info(f"Close result: {result}")
        return result

    def get_order_detail(self, order_id: str) -> dict:
        """Fetch order detail by order_id."""
        r = self._get(f"/api/v4/futures/{config.SETTLE}/orders/{order_id}")
        return r if isinstance(r, dict) else {}

    def get_order_history(self, limit: int = 50) -> list[dict]:
        """Fetch recent finished orders."""
        r = self._get(f"/api/v4/futures/{config.SETTLE}/orders", {
            "contract": config.INST_ID,
            "status": "finished",
            "limit": str(limit),
        })
        return r if isinstance(r, list) else []

    # --- TP/SL conditional orders ---

    def place_tpsl(self, size: int, tp_price: float, sl_price: float) -> tuple[dict, dict]:
        """Place TP and SL as separate price-triggered orders.
        Returns (tp_result, sl_result).
        size: closing size (opposite sign of position)
        """
        # TP order: triggers when price reaches tp_price
        # For long (size < 0 to close): trigger when price >= tp_price
        # For short (size > 0 to close): trigger when price <= tp_price
        is_closing_long = size < 0
        tp_rule = 1 if is_closing_long else 2   # 1=>=, 2=<=
        sl_rule = 2 if is_closing_long else 1

        tp_body = {
            "initial": {
                "contract": config.INST_ID,
                "size": size,
                "price": "0",
                "tif": "ioc",
                "reduce_only": True,
                "text": "t-quant-tp",
            },
            "trigger": {
                "strategy_type": 0,
                "price_type": 0,  # 0=last price
                "price": f"{tp_price:.2f}",
                "rule": tp_rule,
                "expiration": config.HORIZON * 60 * 2,  # 2x horizon as expiry
            },
        }
        sl_body = {
            "initial": {
                "contract": config.INST_ID,
                "size": size,
                "price": "0",
                "tif": "ioc",
                "reduce_only": True,
                "text": "t-quant-sl",
            },
            "trigger": {
                "strategy_type": 0,
                "price_type": 0,
                "price": f"{sl_price:.2f}",
                "rule": sl_rule,
                "expiration": config.HORIZON * 60 * 2,
            },
        }
        tp_result = self._post(f"/api/v4/futures/{config.SETTLE}/price_orders", tp_body)
        sl_result = self._post(f"/api/v4/futures/{config.SETTLE}/price_orders", sl_body)
        logger.info(f"TP order: {tp_result}, SL order: {sl_result}")
        return tp_result, sl_result

    def cancel_price_order(self, order_id: int) -> dict:
        """Cancel a price-triggered (TP/SL) order."""
        r = self._delete(f"/api/v4/futures/{config.SETTLE}/price_orders/{order_id}")
        logger.info(f"Cancel price order {order_id}: {r}")
        return r

    def get_pending_price_orders(self) -> list[dict]:
        """Fetch pending price-triggered orders for INST_ID."""
        r = self._get(f"/api/v4/futures/{config.SETTLE}/price_orders", {
            "contract": config.INST_ID,
            "status": "open",
        })
        return r if isinstance(r, list) else []

    def update_tpsl(self, new_tp: float | None, new_sl: float | None,
                    close_size: int) -> bool:
        """Cancel existing TP/SL orders and place new ones."""
        try:
            pending = self.get_pending_price_orders()
            for o in pending:
                oid = o.get("id")
                if oid:
                    self.cancel_price_order(oid)
        except Exception as e:
            logger.warning(f"Failed to cancel existing price orders: {e}")
            return False
        try:
            if new_tp is None or new_sl is None:
                return False
            self.place_tpsl(close_size, new_tp, new_sl)
            return True
        except Exception as e:
            logger.error(f"Failed to place new TP/SL: {e}")
            return False

    def close(self):
        self._client.close()
