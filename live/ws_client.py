from __future__ import annotations
import asyncio, hashlib, hmac, json, logging, time
from typing import Callable

import websockets
import websockets.asyncio.client

from . import config

logger = logging.getLogger(__name__)


def _gate_ws_sign(channel: str, event: str, timestamp: int) -> str:
    """Gate.io WebSocket HMAC-SHA512 signature.
    sign_str = "channel=<channel>&event=<event>&time=<timestamp>"
    """
    sign_str = f"channel={channel}&event={event}&time={timestamp}"
    return hmac.new(
        config.GATE_SECRET_KEY.encode(),
        sign_str.encode(),
        hashlib.sha512,
    ).hexdigest()


class GateWebSocket:
    """Async WebSocket client for Gate.io futures public channels with auto-reconnect."""

    def __init__(self, url: str, channels: list[dict], on_message: Callable, name: str = "ws"):
        """channels: list of dicts with keys 'channel' and 'payload'."""
        self.url = url
        self.channels = channels
        self.on_message = on_message
        self.name = name
        self._ws = None
        self._running = False
        self._backoff = 1
        self._ping_task = None

    async def start(self):
        self._running = True
        while self._running:
            try:
                async with websockets.asyncio.client.connect(
                    self.url, ping_interval=None, ping_timeout=None,
                    additional_headers={"User-Agent": "quant-live/1.0"},
                ) as ws:
                    self._ws = ws
                    self._backoff = 1
                    logger.info(f"[{self.name}] Connected to {self.url}")
                    await self._subscribe(ws)
                    self._ping_task = asyncio.create_task(self._heartbeat(ws))
                    async for msg in ws:
                        if not self._running:
                            break
                        try:
                            data = json.loads(msg)
                            # Skip pong and subscribe ack messages
                            if data.get("event") in ("pong", "subscribe"):
                                continue
                            if data.get("event") in ("update", "all") and "result" in data:
                                self.on_message(data)
                        except json.JSONDecodeError:
                            pass
            except (websockets.ConnectionClosed, ConnectionError, OSError) as e:
                logger.warning(f"[{self.name}] Disconnected: {e}. Reconnecting in {self._backoff}s...")
                if self._ping_task:
                    self._ping_task.cancel()
                await asyncio.sleep(self._backoff)
                self._backoff = min(self._backoff * 2, 30)
            except Exception as e:
                logger.error(f"[{self.name}] Unexpected error: {e}")
                if self._ping_task:
                    self._ping_task.cancel()
                await asyncio.sleep(self._backoff)
                self._backoff = min(self._backoff * 2, 30)

    async def _heartbeat(self, ws):
        """Send application-level ping every 20s (Gate.io requires this)."""
        while True:
            await asyncio.sleep(20)
            try:
                ts = int(time.time())
                ping_msg = json.dumps({"time": ts, "channel": "futures.ping"})
                await ws.send(ping_msg)
            except Exception:
                break

    async def _subscribe(self, ws):
        ts = int(time.time())
        for ch in self.channels:
            channel = ch["channel"]
            payload = ch.get("payload", [])
            sub_msg = json.dumps({
                "time": ts,
                "channel": channel,
                "event": "subscribe",
                "payload": payload,
            })
            await ws.send(sub_msg)
            logger.info(f"[{self.name}] Subscribed: {channel} {payload}")

    async def stop(self):
        self._running = False
        if self._ping_task:
            self._ping_task.cancel()
        if self._ws:
            await self._ws.close()


class GatePrivateWebSocket(GateWebSocket):
    """Async WebSocket client for Gate.io futures private channels (requires auth per subscription)."""

    async def _subscribe(self, ws):
        ts = int(time.time())
        for ch in self.channels:
            channel = ch["channel"]
            payload = ch.get("payload", [])
            sig = _gate_ws_sign(channel, "subscribe", ts)
            sub_msg = json.dumps({
                "time": ts,
                "channel": channel,
                "event": "subscribe",
                "payload": payload,
                "auth": {
                    "method": "api_key",
                    "KEY": config.GATE_API_KEY,
                    "SIGN": sig,
                },
            })
            await ws.send(sub_msg)
            # Wait briefly for ack
            try:
                resp = await asyncio.wait_for(ws.recv(), timeout=5.0)
                resp_data = json.loads(resp)
                if resp_data.get("event") == "subscribe":
                    err = resp_data.get("error")
                    if err:
                        logger.error(f"[{self.name}] Subscribe error for {channel}: {err}")
                    else:
                        logger.info(f"[{self.name}] Subscribed (private): {channel}")
                else:
                    logger.info(f"[{self.name}] Subscribed (private): {channel} ack={resp_data.get('event')}")
            except asyncio.TimeoutError:
                logger.warning(f"[{self.name}] No ack for {channel} within 5s")
