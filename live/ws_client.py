from __future__ import annotations
import asyncio, base64, hashlib, hmac, json, logging, time
from typing import Callable

import websockets
import websockets.asyncio.client

from . import config

logger = logging.getLogger(__name__)


class OKXWebSocket:
    """Async WebSocket client for OKX public channels with auto-reconnect."""

    def __init__(self, url: str, channels: list[dict], on_message: Callable, name: str = "ws"):
        self.url = url
        self.channels = channels
        self.on_message = on_message
        self.name = name
        self._ws = None
        self._running = False
        self._backoff = 1

    async def start(self):
        self._running = True
        while self._running:
            try:
                async with websockets.asyncio.client.connect(
                    self.url, ping_interval=25, ping_timeout=10,
                    additional_headers={"User-Agent": "quant-live/1.0"},
                ) as ws:
                    self._ws = ws
                    self._backoff = 1
                    logger.info(f"[{self.name}] Connected to {self.url}")
                    await self._subscribe(ws)
                    async for msg in ws:
                        if not self._running:
                            break
                        try:
                            data = json.loads(msg)
                            if "data" in data:
                                self.on_message(data)
                        except json.JSONDecodeError:
                            pass
            except (websockets.ConnectionClosed, ConnectionError, OSError) as e:
                logger.warning(f"[{self.name}] Disconnected: {e}. Reconnecting in {self._backoff}s...")
                await asyncio.sleep(self._backoff)
                self._backoff = min(self._backoff * 2, 30)
            except Exception as e:
                logger.error(f"[{self.name}] Unexpected error: {e}")
                await asyncio.sleep(self._backoff)
                self._backoff = min(self._backoff * 2, 30)

    async def _subscribe(self, ws):
        payload = json.dumps({"op": "subscribe", "args": self.channels})
        await ws.send(payload)
        logger.info(f"[{self.name}] Subscribed: {self.channels}")

    async def stop(self):
        self._running = False
        if self._ws:
            await self._ws.close()


class OKXPrivateWebSocket(OKXWebSocket):
    """Async WebSocket client for OKX private channels (requires login)."""

    async def _subscribe(self, ws):
        # Login first
        ts = str(int(time.time()))
        sign_str = ts + "GET" + "/users/self/verify"
        mac = hmac.new(config.OKX_SECRET_KEY.encode(), sign_str.encode(), hashlib.sha256)
        sig = base64.b64encode(mac.digest()).decode()
        login_msg = json.dumps({
            "op": "login",
            "args": [{
                "apiKey": config.OKX_API_KEY,
                "passphrase": config.OKX_PASSPHRASE,
                "timestamp": ts,
                "sign": sig,
            }]
        })
        await ws.send(login_msg)
        # Wait for login response
        resp = await ws.recv()
        resp_data = json.loads(resp)
        if resp_data.get("event") == "login" and resp_data.get("code") == "0":
            logger.info(f"[{self.name}] Login successful")
        else:
            logger.error(f"[{self.name}] Login failed: {resp_data}")
            return
        # Subscribe to channels
        await super()._subscribe(ws)
