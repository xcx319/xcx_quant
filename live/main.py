from __future__ import annotations
import asyncio, json, logging, time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from starlette.requests import Request

from . import config
from .state import AppState, append_log, load_log, rewrite_log
from .bar_aggregator import BarAggregator
from .orderbook_state import OrderbookState
from .feature_engine import FeatureEngine
from .scanner import FlowReversalScanner
from .model_inference import ModelInference
from .execution import OrderExecutor
from .ws_client import OKXWebSocket, OKXPrivateWebSocket
from .event_aligner import compute_event_features, POST_WINDOW_S

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s %(message)s")
logger = logging.getLogger("live")

# --- Globals ---
state = AppState()
ob_state = OrderbookState()
engine = FeatureEngine()
scanner = FlowReversalScanner()
model: ModelInference | None = None
executor: OrderExecutor | None = None
ws_clients: set[WebSocket] = set()
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
_last_ob_broadcast: float = 0.0
_live_bars_count: int = 0  # bars received from live data (not warmup)
_ct_val: float = 0.01  # contract value, fetched on startup
_lot_sz: float = 0.01  # lot size (min increment), fetched on startup
_min_sz: float = 0.01  # minimum order size, fetched on startup
_lever: int = 10  # leverage, fetched on startup


def _calc_position_size(price: float) -> str:
    """Compute number of contracts, rounded down to lotSz. Returns string for API.

    margin = INITIAL_CAPITAL * RISK_PER_TRADE  (e.g. 100 * 0.20 = 20 USDT)
    notional = margin * leverage              (e.g. 20 * 10 = 200 USDT)
    contracts = notional / (ctVal * price)    (e.g. 200 / (0.1 * 2123) = 0.94)
    """
    if price <= 0 or _ct_val <= 0:
        return str(_min_sz)
    margin = config.INITIAL_CAPITAL * config.RISK_PER_TRADE
    notional = margin * _lever
    contracts = notional / (_ct_val * price)
    # Round down to lot size
    contracts = int(contracts / _lot_sz) * _lot_sz
    contracts = max(_min_sz, contracts)
    return f"{contracts:.2f}".rstrip('0').rstrip('.')

# --- Broadcast to dashboard clients ---
async def broadcast(event_type: str, data: dict):
    global ws_clients
    msg = json.dumps({"type": event_type, "data": data}, default=str)
    dead = set()
    for ws in ws_clients:
        try:
            await ws.send_text(msg)
        except Exception:
            dead.add(ws)
    ws_clients -= dead


# --- Bar completion callback ---
def on_bar_complete(bar: dict):
    asyncio.get_event_loop().create_task(_process_bar(bar))


async def _process_bar(bar: dict):
    global state, _live_bars_count
    _live_bars_count += 1
    # Merge orderbook features into bar
    ob_feats = ob_state.get_features()
    bar.update(ob_feats)
    ob_state.reset_quote_count()

    state.bars_received = engine.bar_count + 1
    state.current_price = bar["close"]
    state.last_bar_time = str(bar["datetime"])
    state.recent_bars.append({
        "time": int(bar["datetime"].timestamp()) if isinstance(bar["datetime"], datetime) else int(time.time()),
        "open": bar["open"], "high": bar["high"], "low": bar["low"], "close": bar["close"],
        "volume": bar.get("volume", 0),
    })

    # Feature computation
    feat_row = engine.add_bar(bar)
    state.is_warm = engine.is_warm

    await broadcast("price", state.recent_bars[-1])
    await broadcast("health", state.snapshot())

    if feat_row is None:
        return  # still warming up

    state.current_atr = float(feat_row.get("atr", 0)) if np.isfinite(feat_row.get("atr", np.nan)) else 0.0

    # Scanner evaluation (detailed for dashboard)
    scanner_result = scanner.evaluate_detailed(feat_row)
    state.last_scanner_state = scanner_result
    await broadcast("scanner_state", scanner_result)

    trigger = scanner_result["trigger"]
    if trigger is None:
        return

    # Guard: require 30+ live bars for spread_ref/OBI deques to stabilize
    if _live_bars_count < 30:
        logger.info(f"Scanner trigger ignored: only {_live_bars_count}/30 live bars (features not stable)")
        return

    event_dir = trigger["event_dir"]
    scanner_score = trigger["scanner_score"]
    scanner_trigger_ts = time.time()

    # Delay model inference: wait POST_WINDOW_S seconds to collect tick data
    # for event-aligned features, then run model + trade in background
    bar_start_ns = int(bar["datetime"].timestamp() * 1e9) if isinstance(bar["datetime"], datetime) else int(time.time() * 1e9)
    logger.info(f"Scanner trigger: {'long' if event_dir == 1 else 'short'}, waiting {POST_WINDOW_S}s for event alignment...")
    asyncio.get_event_loop().create_task(
        _delayed_inference(feat_row, event_dir, scanner_score, bar, bar_start_ns, scanner_trigger_ts)
    )


async def _delayed_inference(feat_row, event_dir: int, scanner_score: float, bar: dict, bar_start_ns: int, scanner_trigger_ts: float):
    """Wait for post_window ticks, compute event-aligned features, then run model."""
    await asyncio.sleep(POST_WINDOW_S + 1)  # +1s buffer for tick arrival

    # Compute event-aligned features from tick buffer
    ev_feats = compute_event_features(
        tick_buffer=aggregator._tick_buffer,
        bar_start_ns=bar_start_ns,
        event_level=bar["close"],
        event_dir=event_dir,
        atr=state.current_atr,
    )
    logger.info(f"Event features: sec_in_bar={ev_feats['sec_in_bar']:.1f}, "
                f"event_return={ev_feats['event_return']:.6f}, "
                f"rejection={ev_feats['event_rejection_strength']:.4f}, "
                f"time_to_reject={ev_feats['time_to_reject_s']:.2f}s")

    # Model inference with real event-aligned features
    result = model.predict(feat_row, event_dir, scanner_score, event_features=ev_feats)
    model_inference_ts = time.time()
    signal_record = {
        "time": state.last_bar_time,
        "direction": result["direction"],
        "prob": round(result["prob"], 6),
        "threshold": result["threshold"],
        "signal": result["signal"],
        "price": bar["close"],
        "atr": state.current_atr,
    }
    state.signals.append(signal_record)
    append_log("signals", signal_record)
    await broadcast("signal", signal_record)
    logger.info(f"Model result: {result['direction']} prob={result['prob']:.4f} signal={result['signal']}")

    if not result["signal"]:
        return

    # Check trading toggle
    if not state.trading_enabled:
        logger.info("Trading disabled, skipping order")
        return

    # Check daily loss limit and max trades per day
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    today_trades = [t for t in state.trades if str(t.get("time", "")).startswith(today_str)]
    if len(today_trades) >= config.MAX_TRADES_PER_DAY:
        logger.info(f"Daily trade limit reached ({config.MAX_TRADES_PER_DAY}), skipping")
        return
    today_pnl_r = sum(
        t.get("pnl", 0) / max(t.get("atr", 1) * _ct_val * float(t.get("size", 1)), 1e-9)
        for t in today_trades if t.get("pnl") is not None
    )
    if today_pnl_r <= config.DAILY_LOSS_LIMIT_R:
        logger.info(f"Daily loss limit hit ({today_pnl_r:.1f}R <= {config.DAILY_LOSS_LIMIT_R}R), skipping")
        return

    # Check position limit
    if state.open_position is not None:
        logger.info("Signal skipped: already in position")
        return

    # Execute trade
    atr = state.current_atr
    if atr <= 0:
        logger.warning("ATR is zero, skipping trade")
        return

    entry = bar["close"]
    # Use current market price (after ~16s delay) for TP/SL, matching backtest entry_price_delayed
    entry_for_tpsl = state.current_price if state.current_price else entry
    if event_dir == 1:  # long
        tp_price = entry_for_tpsl + atr * config.TP_MULT
        sl_price = entry_for_tpsl - atr * config.SL_MULT
        side = "buy"
    else:  # short
        tp_price = entry_for_tpsl - atr * config.TP_MULT
        sl_price = entry_for_tpsl + atr * config.SL_MULT
        side = "sell"

    try:
        size = _calc_position_size(entry_for_tpsl)
        order_result = executor.place_market_order(side, size, tp_price, sl_price)
        if order_result.get("code") == "0":
            ord_id = order_result.get("data", [{}])[0].get("ordId", "")
            order_ts = time.time()
            # Fetch fill details from OKX
            fill_ts = None
            fill_px = None
            try:
                detail = executor.get_order_detail(ord_id)
                if detail:
                    ft = detail.get("fillTime", "")
                    if ft:
                        fill_ts = int(ft) / 1000  # ms -> s
                    fp = detail.get("fillPx", "")
                    if fp:
                        fill_px = float(fp)
            except Exception as e:
                logger.warning(f"Failed to fetch order detail: {e}")
            # Recompute TP/SL based on actual fill price if available
            actual_entry = fill_px or entry_for_tpsl
            if fill_px:
                if event_dir == 1:
                    tp_price = actual_entry + atr * config.TP_MULT
                    sl_price = actual_entry - atr * config.SL_MULT
                else:
                    tp_price = actual_entry - atr * config.TP_MULT
                    sl_price = actual_entry + atr * config.SL_MULT
            state.open_position = {
                "side": side, "entry": actual_entry, "tp": tp_price, "sl": sl_price,
                "size": size, "time": state.last_bar_time,
                "open_ts": time.time(),
                "order_id": ord_id,
                "atr": atr,
                "max_fav_r": 0.0,
            }
            trade_record = {
                **state.open_position,
                "atr": atr, "prob": result["prob"],
                "scanner_ts": scanner_trigger_ts,
                "model_ts": model_inference_ts,
                "order_ts": order_ts,
                "fill_ts": fill_ts,
                "fill_px": fill_px,
            }
            state.trades.append(trade_record)
            append_log("trades", trade_record)
            await broadcast("trade", trade_record)
            logger.info(f"Order placed: {side} @ {actual_entry:.2f}, TP={tp_price:.2f}, SL={sl_price:.2f}")
        else:
            logger.error(f"Order failed: {order_result}")
    except Exception as e:
        logger.error(f"Order execution error: {e}")


# --- OKX message handlers ---
_last_tick_broadcast: float = 0.0

def on_trades_message(msg: dict):
    global _last_tick_broadcast
    state.ws_trades_connected = True
    for trade in msg.get("data", []):
        px = float(trade["px"])
        aggregator.ingest_trade(
            price=px,
            size=float(trade["sz"]),
            side=trade["side"],
            ts_ms=int(trade["ts"]),
        )
        state.last_trade_ts = time.time()
        prev_price = state.current_price
        state.current_price = px
    # Throttled tick broadcast (max once per 500ms)
    now = time.time()
    if state.current_price and now - _last_tick_broadcast >= 0.5:
        _last_tick_broadcast = now
        try:
            loop = asyncio.get_event_loop()
            ob_snap = ob_state.get_snapshot()
            tick_data = {
                "price": state.current_price,
                "time": now,
                "bid": ob_snap["bids"][0]["price"] if ob_snap["bids"] else None,
                "ask": ob_snap["asks"][0]["price"] if ob_snap["asks"] else None,
                "bar_time": int(aggregator._current_minute.timestamp()) if aggregator._current_minute else None,
                "bar_open": aggregator._open,
                "bar_high": aggregator._high,
                "bar_low": aggregator._low,
                "bar_close": aggregator._close,
            }
            loop.create_task(broadcast("tick", tick_data))
        except RuntimeError:
            pass


def on_books_message(msg: dict):
    global _last_ob_broadcast
    state.ws_books_connected = True
    for snap in msg.get("data", []):
        ob_state.update(snap)
        state.last_book_ts = time.time()
    # Throttled orderbook broadcast (max once per 500ms)
    now = time.time()
    if now - _last_ob_broadcast >= 0.5:
        _last_ob_broadcast = now
        try:
            loop = asyncio.get_event_loop()
            loop.create_task(broadcast("orderbook", ob_state.get_snapshot()))
        except RuntimeError:
            pass


aggregator = BarAggregator(on_bar_complete=on_bar_complete)


# --- Warmup from REST candles ---
def warmup_from_candles():
    logger.info("Warming up from REST candles...")
    candles = executor.get_candles(bar="1m", limit=300)
    if not candles:
        logger.warning("No candles returned for warmup")
        return
    # OKX returns newest first, reverse to chronological
    candles = list(reversed(candles))
    for c in candles:
        ts_ms = int(c[0])
        bar = {
            "datetime": datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc).replace(second=0, microsecond=0),
            "open": float(c[1]), "high": float(c[2]), "low": float(c[3]), "close": float(c[4]),
            "volume": float(c[5]),
            "aggressor_ratio": 0.5, "net_taker_vol_ratio": 0.0,
            "trade_gini": 0.0, "large_trade_vol_ratio": 0.0,
            "trade_intensity": float(np.log1p(float(c[5]))),
            "data_from_orderbook": 0,
        }
        engine.add_bar(bar)
        state.recent_bars.append({
            "time": int(bar["datetime"].timestamp()),
            "open": bar["open"], "high": bar["high"], "low": bar["low"], "close": bar["close"],
        })
    state.bars_received = engine.bar_count
    state.is_warm = engine.is_warm
    state.current_price = float(candles[-1][4]) if candles else 0.0
    logger.info(f"Warmup complete: {engine.bar_count} bars, is_warm={engine.is_warm}")


# --- Private WebSocket handler (balance + positions, real-time push) ---
def on_private_message(msg: dict):
    """Handle balance_and_position and positions channel pushes."""
    channel = msg.get("arg", {}).get("channel", "")
    for d in msg.get("data", []):
        if channel == "balance_and_position":
            _handle_balance_and_position(d)
        elif channel == "positions":
            _handle_positions_push(d)
        elif channel == "account":
            _handle_account_push(d)


def _handle_balance_and_position(d: dict):
    """Process balance_and_position push."""
    # Balance data
    bal_list = d.get("balData", [])
    usdt_bal = next((b for b in bal_list if b.get("ccy") == "USDT"), None)

    # Position data
    pos_list = d.get("posData", [])
    active = [p for p in pos_list if p.get("instId") == config.INST_ID and float(p.get("pos", "0")) != 0]

    # Build account data
    if usdt_bal:
        account_data = {
            "totalEq": usdt_bal.get("eq", "0"),
            "availBal": usdt_bal.get("availBal", "0"),
            "upl": usdt_bal.get("upl", "0"),
            "ctVal": _ct_val,
            "lever": _lever,
        }
        state.last_account = account_data
        try:
            loop = asyncio.get_event_loop()
            loop.create_task(broadcast("account", account_data))
        except RuntimeError:
            pass

    # Build position data — only if posData was present in the push
    # (OKX omits posData or sends [] when only balance changed)
    if pos_list:
        _process_position_data(active)


def _handle_account_push(d: dict):
    """Process account channel push (totalEq level)."""
    details = d.get("details", [])
    usdt = next((x for x in details if x.get("ccy") == "USDT"), None)
    if not usdt:
        return
    account_data = {
        "totalEq": d.get("totalEq", "0"),
        "availBal": usdt.get("availBal", "0"),
        "upl": d.get("upl", "0"),
        "ctVal": _ct_val,
        "lever": _lever,
    }
    state.last_account = account_data
    try:
        loop = asyncio.get_event_loop()
        loop.create_task(broadcast("account", account_data))
    except RuntimeError:
        pass


def _handle_positions_push(d: dict):
    """Process positions channel push."""
    if d.get("instId") != config.INST_ID:
        return
    pos_size = float(d.get("pos", "0"))
    active = [d] if pos_size != 0 else []
    _process_position_data(active)


def _record_trade_close(exit_type: str = None):
    """Update the last trade record with close price, PnL, and exit type."""
    if not state.trades or not state.open_position:
        return
    pos = state.open_position
    entry = pos.get("entry", 0)
    exit_px = state.current_price
    side = pos.get("side", "buy")
    size = float(pos.get("size", 0))

    # PnL in USDT: (exit - entry) * contracts * ctVal, flip sign for short
    if side == "buy":
        pnl = (exit_px - entry) * size * _ct_val
    else:
        pnl = (entry - exit_px) * size * _ct_val

    # Determine exit type if not provided
    if exit_type is None:
        tp, sl = pos.get("tp", 0), pos.get("sl", 0)
        if side == "buy":
            exit_type = "tp" if exit_px >= tp else ("sl" if exit_px <= sl else "close")
        else:
            exit_type = "tp" if exit_px <= tp else ("sl" if exit_px >= sl else "close")

    # Update last trade record in memory
    trade = state.trades[-1]
    trade["close_price"] = exit_px
    trade["close_time"] = time.time()
    trade["pnl"] = round(pnl, 4)
    trade["exit_type"] = exit_type

    # Persist and broadcast
    rewrite_log("trades", state.trades)
    try:
        loop = asyncio.get_event_loop()
        loop.create_task(broadcast("trade_update", trade))
    except RuntimeError:
        pass
    logger.info(f"Trade closed: exit={exit_px:.2f} pnl={pnl:.4f} USDT ({exit_type})")


def _backfill_trade_pnl():
    """Backfill PnL and timestamps for historical trades using OKX order history."""
    needs_pnl = [t for t in state.trades if t.get("pnl") is None]
    needs_ts = [t for t in state.trades if t.get("fill_ts") is None]
    if not needs_pnl and not needs_ts:
        return
    try:
        orders = executor.get_order_history(limit=100)
    except Exception as e:
        logger.error(f"Backfill: failed to fetch order history: {e}")
        return
    filled = [o for o in orders if o.get("state") == "filled"]
    updated = False

    for trade in state.trades:
        ord_id = trade.get("order_id", "")
        entry = trade.get("entry", 0)
        side = trade.get("side", "buy")
        open_ts_ms = int(trade.get("open_ts", 0) * 1000)

        # --- Backfill opening timestamps via order_id ---
        if trade.get("fill_ts") is None and ord_id:
            # Find the opening order by ordId
            open_order = None
            for o in filled:
                if o.get("ordId") == ord_id:
                    open_order = o
                    break
            if open_order is None:
                # Also try fetching directly from API
                try:
                    open_order = executor.get_order_detail(ord_id)
                    if open_order and open_order.get("state") != "filled":
                        open_order = None
                except Exception:
                    pass
            if open_order:
                ft = open_order.get("fillTime", "")
                if ft:
                    trade["fill_ts"] = int(ft) / 1000
                fp = open_order.get("fillPx", "")
                if fp:
                    trade["fill_px"] = float(fp)
                # order_ts: use open_ts as best approximation (order was sent just before)
                if not trade.get("order_ts"):
                    trade["order_ts"] = trade.get("open_ts")
                # scanner_ts/model_ts: estimate from open_ts
                # scanner triggers ~17s before order (POST_WINDOW_S+1 delay)
                if not trade.get("scanner_ts"):
                    trade["scanner_ts"] = trade.get("open_ts", 0) - (POST_WINDOW_S + 1)
                if not trade.get("model_ts"):
                    trade["model_ts"] = trade.get("open_ts", 0) - 0.5  # model runs ~0.5s before order
                updated = True
                logger.info(f"Backfill timestamps: trade {ord_id} @ {entry:.2f}")

        # --- Backfill PnL for trades missing close info ---
        if trade.get("pnl") is None:
            close_order = None
            for o in filled:
                o_side = o.get("side", "")
                o_ts = int(o.get("fillTime", "0") or "0")
                if o_side != side and o_ts > open_ts_ms:
                    if close_order is None or o_ts < int(close_order.get("fillTime", "0") or "0"):
                        close_order = o
            if close_order:
                fill_px = float(close_order.get("fillPx", "0") or close_order.get("avgPx", "0") or "0")
                pnl_val = float(close_order.get("pnl", "0") or "0")
                fee = float(close_order.get("fee", "0") or "0")
                fill_ts = int(close_order.get("fillTime", "0") or "0") / 1000
                if pnl_val != 0:
                    total_pnl = pnl_val + fee
                else:
                    size = float(trade.get("size", 0))
                    if side == "buy":
                        total_pnl = (fill_px - entry) * size * _ct_val + fee
                    else:
                        total_pnl = (entry - fill_px) * size * _ct_val + fee
                tp, sl = trade.get("tp", 0), trade.get("sl", 0)
                if side == "buy":
                    exit_type = "tp" if fill_px >= tp else ("sl" if fill_px <= sl else "close")
                else:
                    exit_type = "tp" if fill_px <= tp else ("sl" if fill_px >= sl else "close")
                trade["close_price"] = fill_px
                trade["close_time"] = fill_ts
                trade["pnl"] = round(total_pnl, 4)
                trade["exit_type"] = exit_type
                updated = True
                logger.info(f"Backfill PnL: trade @ {entry:.2f} -> close {fill_px:.2f}, pnl={total_pnl:.4f} ({exit_type})")

    if updated:
        rewrite_log("trades", state.trades)
        logger.info(f"Backfill complete")


def _check_breakeven_stop():
    """Adjust SL to breakeven or lock-profit level based on favorable price excursion."""
    pos = state.open_position
    if pos is None or state.current_price is None:
        return
    entry = pos.get("entry", 0)
    atr = pos.get("atr", state.current_atr)
    if atr <= 0:
        return
    side = pos.get("side", "buy")
    price = state.current_price

    # Favorable excursion in R-multiples
    if side == "buy":
        fav_r = (price - entry) / atr
    else:
        fav_r = (entry - price) / atr

    # Track max favorable excursion
    prev_max = pos.get("max_fav_r", 0)
    pos["max_fav_r"] = max(prev_max, fav_r)

    BREAKEVEN_TRIGGER = 1.0   # move SL to entry after 1.0R favorable
    LOCK_TRIGGER = 1.5        # move SL to entry+0.5R after 1.5R favorable
    LOCK_LEVEL = 0.5          # locked profit level in R

    current_sl = pos.get("sl", 0)
    new_sl = current_sl

    if pos["max_fav_r"] >= LOCK_TRIGGER:
        # Lock profit: SL at entry + LOCK_LEVEL * ATR
        if side == "buy":
            new_sl = max(current_sl, entry + LOCK_LEVEL * atr)
        else:
            new_sl = min(current_sl, entry - LOCK_LEVEL * atr)
    elif pos["max_fav_r"] >= BREAKEVEN_TRIGGER:
        # Breakeven: SL at entry
        if side == "buy":
            new_sl = max(current_sl, entry)
        else:
            new_sl = min(current_sl, entry)

    if new_sl != current_sl:
        logger.info(f"Breakeven stop: SL moved {current_sl:.2f} -> {new_sl:.2f} (max_fav={pos['max_fav_r']:.2f}R)")
        pos["sl"] = new_sl
        # Check if current price already breaches new SL
        breached = (side == "buy" and price <= new_sl) or (side == "sell" and price >= new_sl)
        if breached:
            logger.info(f"Breakeven SL breached at {price:.2f}, closing position")
            try:
                close_result = executor.close_position()
                if close_result.get("code") == "0":
                    _record_trade_close(exit_type="breakeven")
                    state.open_position = None
            except Exception as e:
                logger.error(f"Breakeven close error: {e}")


def _process_position_data(active: list):
    """Shared logic for position updates from REST or WS."""
    if not active:
        if state.open_position is not None:
            # Position just closed — compute PnL and update trade record
            _record_trade_close()
            logger.info("Position closed (TP/SL hit)")
            state.open_position = None
        state.last_position_data = None
        try:
            loop = asyncio.get_event_loop()
            loop.create_task(broadcast("position", {"status": "none"}))
        except RuntimeError:
            pass
        return

    p = active[0]
    pos_size = float(p.get("pos", "0"))
    pos_side = "long" if pos_size > 0 else "short"
    upnl = float(p.get("upl", "0") or "0")
    avg_px = float(p.get("avgPx", "0") or "0")
    liq_px = p.get("liqPx", "")
    lever = p.get("lever", "")
    margin = float(p.get("margin", "0") or "0")

    pos_data = {
        "status": "active",
        "side": pos_side,
        "size": abs(pos_size),
        "avgPx": avg_px,
        "upl": upnl,
        "liqPx": liq_px,
        "lever": lever,
        "margin": margin,
    }

    if state.open_position is not None:
        pos_data["source"] = "system"
        pos_data["entry"] = state.open_position.get("entry", avg_px)
        pos_data["tp"] = state.open_position.get("tp", 0)
        pos_data["sl"] = state.open_position.get("sl", 0)
        pos_data["open_time"] = state.open_position.get("time", "")
        # Breakeven stop: dynamically adjust SL based on favorable excursion
        _check_breakeven_stop()
        # Horizon timeout check
        open_ts = state.open_position.get("open_ts", 0)
        elapsed_s = time.time() - open_ts if open_ts else 0
        horizon_s = config.HORIZON * 60
        pos_data["elapsed_s"] = round(elapsed_s)
        pos_data["horizon_s"] = horizon_s
        if open_ts and elapsed_s >= horizon_s:
            logger.info(f"Horizon timeout: position held {elapsed_s:.0f}s >= {horizon_s}s, closing at market")
            try:
                close_result = executor.close_position()
                if close_result.get("code") == "0":
                    logger.info("Horizon timeout close successful")
                    _record_trade_close(exit_type="timeout")
                    state.open_position = None
                else:
                    logger.error(f"Horizon timeout close failed: {close_result}")
            except Exception as e:
                logger.error(f"Horizon timeout close error: {e}")
    else:
        pos_data["source"] = "foreign"

    state.last_position_data = pos_data
    try:
        loop = asyncio.get_event_loop()
        loop.create_task(broadcast("position", pos_data))
    except RuntimeError:
        pass


def _fetch_foreign_positions():
    """Fetch all positions and broadcast non-SWAP ones as foreign."""
    try:
        all_pos = executor.get_all_positions()
        foreign = []
        for p in (all_pos or []):
            if float(p.get("pos", "0")) == 0:
                continue
            if p.get("instId") == config.INST_ID:
                continue  # SWAP position handled by _process_position_data
            pos_size = float(p.get("pos", "0"))
            foreign.append({
                "instId": p.get("instId", ""),
                "instType": p.get("instType", ""),
                "side": "long" if pos_size > 0 else "short",
                "size": abs(pos_size),
                "posCcy": p.get("posCcy", ""),
                "avgPx": float(p.get("avgPx", "0") or "0"),
                "upl": float(p.get("upl", "0") or "0"),
                "liqPx": p.get("liqPx", ""),
                "lever": p.get("lever", ""),
                "margin": float(p.get("margin", "0") or "0"),
                "mgnMode": p.get("mgnMode", ""),
            })
        state.foreign_positions = foreign
        try:
            loop = asyncio.get_event_loop()
            loop.create_task(broadcast("foreign_positions", foreign))
        except RuntimeError:
            pass
    except Exception as e:
        logger.error(f"Fetch foreign positions error: {e}")


# --- REST fallback monitor (backup for private WS) ---
async def rest_fallback_monitor():
    """Polls positions + balance via REST as fallback if private WS is slow/disconnected."""
    while True:
        await asyncio.sleep(30)
        try:
            # Positions
            positions = executor.get_positions()
            active = [p for p in (positions or []) if float(p.get("pos", "0")) != 0]
            _process_position_data(active)
            # Foreign positions (non-SWAP)
            _fetch_foreign_positions()
            # Balance
            bal = executor.get_balance()
            if bal:
                details = bal.get("details", [])
                usdt = next((d for d in details if d.get("ccy") == "USDT"), {})
                account_data = {
                    "totalEq": bal.get("totalEq", "0"),
                    "availBal": usdt.get("availBal", "0"),
                    "upl": bal.get("upl", "0"),
                    "ctVal": _ct_val,
                    "lever": _lever,
                }
                state.last_account = account_data
                await broadcast("account", account_data)
        except Exception as e:
            logger.error(f"REST fallback monitor error: {e}")


# --- FastAPI app ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, executor, _ct_val, _lot_sz, _min_sz, _lever
    model = ModelInference()
    executor = OrderExecutor()
    # Fetch contract info
    try:
        inst = executor.get_instruments()
        if inst:
            _ct_val = float(inst.get("ctVal", "0.01"))
            _lot_sz = float(inst.get("lotSz", "0.01"))
            _min_sz = float(inst.get("minSz", "0.01"))
    except Exception:
        pass
    # Set leverage via API to ensure it matches our config
    try:
        lev_result = executor.set_leverage(config.LEVERAGE)
        if lev_result.get("code") == "0":
            _lever = config.LEVERAGE
            logger.info(f"Leverage set successfully: _lever={_lever}")
        else:
            logger.warning(f"Failed to set leverage: {lev_result}, using default _lever={_lever}")
    except Exception as e:
        logger.warning(f"set_leverage error: {e}, using default _lever={_lever}")
    warmup_from_candles()
    # Fetch initial account + position via REST (before WS connects)
    try:
        bal = executor.get_balance()
        if bal:
            details = bal.get("details", [])
            usdt = next((d for d in details if d.get("ccy") == "USDT"), {})
            state.last_account = {
                "totalEq": bal.get("totalEq", "0"),
                "availBal": usdt.get("availBal", "0"),
                "upl": bal.get("upl", "0"),
                "ctVal": _ct_val, "lever": _lever,
            }
    except Exception:
        pass
    try:
        positions = executor.get_positions()
        active = [p for p in (positions or []) if float(p.get("pos", "0")) != 0]
        _process_position_data(active)
    except Exception:
        pass
    _fetch_foreign_positions()
    # Restore logs
    state.signals = load_log("signals")
    state.trades = load_log("trades")
    _backfill_trade_pnl()
    # Start WebSocket connections
    ws_trades = OKXWebSocket(
        config.WS_PUBLIC,
        [{"channel": "trades", "instId": config.INST_ID}],
        on_trades_message, name="trades",
    )
    ws_books = OKXWebSocket(
        config.WS_PUBLIC,
        [{"channel": "books5", "instId": config.INST_ID}],
        on_books_message, name="books5",
    )
    # Private WS for real-time account + position updates
    ws_private = OKXPrivateWebSocket(
        config.WS_PRIVATE,
        [
            {"channel": "account"},
            {"channel": "positions", "instType": "SWAP"},
            {"channel": "balance_and_position"},
        ],
        on_private_message, name="private",
    )
    t1 = asyncio.create_task(ws_trades.start())
    t2 = asyncio.create_task(ws_books.start())
    t3 = asyncio.create_task(ws_private.start())
    t4 = asyncio.create_task(rest_fallback_monitor())
    logger.info(f"System started. Dashboard at http://localhost:{config.DASHBOARD_PORT}")
    yield
    ws_trades._running = False
    ws_books._running = False
    ws_private._running = False
    t1.cancel(); t2.cancel(); t3.cancel(); t4.cancel()
    executor.close()


app = FastAPI(lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/state")
async def api_state():
    return {
        **state.snapshot(),
        "signals": state.signals[-50:],
        "trades": state.trades[-50:],
        "bars": list(state.recent_bars)[-200:],
    }


class TradingToggle(BaseModel):
    enabled: bool


@app.post("/api/trading-toggle")
async def trading_toggle(body: TradingToggle):
    state.trading_enabled = body.enabled
    logger.info(f"Trading {'enabled' if body.enabled else 'disabled'} via dashboard")
    await broadcast("health", state.snapshot())
    return {"trading_enabled": state.trading_enabled}


def _dedup_bars(bars: list) -> list:
    """Deduplicate bars by time, keeping last occurrence (most up-to-date)."""
    seen = {}
    for b in bars:
        seen[b["time"]] = b
    return sorted(seen.values(), key=lambda x: x["time"])


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    ws_clients.add(ws)
    try:
        # Send initial state
        bars = _dedup_bars(list(state.recent_bars)[-200:])
        await ws.send_text(json.dumps({
            "type": "init",
            "data": {
                "bars": bars,
                "signals": state.signals[-50:],
                "trades": state.trades[-50:],
                "health": state.snapshot(),
                "scanner_state": state.last_scanner_state,
                "account": state.last_account,
                "orderbook": ob_state.get_snapshot(),
                "position": state.last_position_data,
                "foreign_positions": state.foreign_positions,
                "trading_enabled": state.trading_enabled,
            }
        }, default=str))
        while True:
            await ws.receive_text()  # keep alive
    except WebSocketDisconnect:
        pass
    finally:
        ws_clients.discard(ws)


if __name__ == "__main__":
    uvicorn.run("live.main:app", host="0.0.0.0", port=config.DASHBOARD_PORT, reload=False)
