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
from .ws_client import GateWebSocket, GatePrivateWebSocket
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
_live_bars_count: int = 0
_ct_val: float = 0.0001   # Gate ETH_USDT quanto_multiplier (fetched on startup)
_lot_sz: float = 1.0      # Gate uses integer contract sizes
_min_sz: float = 1.0
_lever: int = 10


def _calc_position_size(price: float, atr: float, sl_mult: float) -> int:
    """Risk-based position sizing with auto-leverage.
    Gate.io uses integer contract sizes. Returns signed int (positive=long).
    """
    import math
    global _lever
    if price <= 0 or _ct_val <= 0 or atr <= 0 or sl_mult <= 0:
        return int(_min_sz)
    max_loss = state.trade_notional * state.risk_per_trade
    loss_per_contract = sl_mult * atr * _ct_val
    contracts = max_loss / loss_per_contract
    contracts = max(int(_min_sz), int(contracts))

    # Auto-adjust leverage
    notional = contracts * _ct_val * price
    required_lever = max(1, math.ceil(notional / state.max_capital))
    if required_lever != _lever:
        logger.info(f"Adjusting leverage: {_lever}x -> {required_lever}x "
                    f"(notional={notional:.1f}, max_capital={state.max_capital})")
        try:
            result = executor.set_leverage(required_lever)
            if isinstance(result, dict) and not result.get("label"):
                _lever = required_lever
                state.leverage = _lever
            else:
                logger.warning(f"Failed to set leverage to {required_lever}x: {result}")
        except Exception as e:
            logger.warning(f"set_leverage error: {e}")

    return contracts


# --- Broadcast to dashboard clients ---
async def broadcast(event_type: str, data: dict):
    global ws_clients
    msg = json.dumps({"type": event_type, "data": data}, default=str)
    dead = set()
    for ws in list(ws_clients):
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
    try:
        _live_bars_count += 1
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

        feat_row = engine.add_bar(bar)
        state.is_warm = engine.is_warm

        await broadcast("price", state.recent_bars[-1])
        await broadcast("health", state.snapshot())

        if feat_row is None:
            return

        state.current_atr = float(feat_row.get("atr", 0)) if np.isfinite(feat_row.get("atr", np.nan)) else 0.0

        scanner_result = scanner.evaluate_detailed(feat_row)
        state.last_scanner_state = scanner_result
        logger.info(f"Bar #{_live_bars_count}: atr={state.current_atr:.2f}, scanner L={scanner_result['long_met']}/6 S={scanner_result['short_met']}/6")
        await broadcast("scanner_state", scanner_result)

        trigger = scanner_result["trigger"]
        if trigger is None:
            return

        if _live_bars_count < 30:
            logger.info(f"Scanner trigger ignored: only {_live_bars_count}/30 live bars")
            return

        event_dir = trigger["event_dir"]
        scanner_score = trigger["scanner_score"]
        scanner_trigger_ts = time.time()

        if config.LONG_ONLY and event_dir != 1:
            logger.info("Short signal skipped (long_only=True)")
            return

        bar_start_ns = int(bar["datetime"].timestamp() * 1e9) if isinstance(bar["datetime"], datetime) else int(time.time() * 1e9)
        logger.info(f"Scanner trigger: {'long' if event_dir == 1 else 'short'}, waiting {POST_WINDOW_S}s for event alignment...")
        asyncio.get_event_loop().create_task(
            _delayed_inference(feat_row, event_dir, scanner_score, bar, bar_start_ns, scanner_trigger_ts)
        )
    except Exception as e:
        logger.error(f"_process_bar error: {e}", exc_info=True)


async def _delayed_inference(feat_row, event_dir: int, scanner_score: float, bar: dict, bar_start_ns: int, scanner_trigger_ts: float):
    """Wait for post_window ticks, compute event-aligned features, then run model."""
    await asyncio.sleep(POST_WINDOW_S + 1)

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

    if not state.trading_enabled:
        logger.info("Trading disabled, skipping order")
        return

    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    today_trades = [t for t in state.trades if str(t.get("time", "")).startswith(today_str)]
    if len(today_trades) >= state.daily_trade_limit:
        logger.info(f"Daily trade limit reached ({state.daily_trade_limit}), skipping")
        return
    today_pnl_r = sum(
        t.get("pnl", 0) / max(t.get("atr", 1) * _ct_val * float(t.get("size", 1)), 1e-9)
        for t in today_trades if t.get("pnl") is not None
    )
    if today_pnl_r <= state.daily_loss_limit_r:
        logger.info(f"Daily loss limit hit ({today_pnl_r:.1f}R <= {state.daily_loss_limit_r}R), skipping")
        return

    if state.open_position is not None:
        logger.info("Signal skipped: already in position")
        return

    atr = state.current_atr
    if atr <= 0:
        logger.warning("ATR is zero, skipping trade")
        return

    entry = bar["close"]
    entry_for_tpsl = state.current_price if state.current_price else entry
    if config.SPLIT_MODEL:
        tp_mult = config.LONG_TP if event_dir == 1 else config.SHORT_TP
        sl_mult = config.LONG_SL if event_dir == 1 else config.SHORT_SL
    else:
        tp_mult, sl_mult = config.TP_MULT, config.SL_MULT
    if event_dir == 1:  # long
        tp_price = entry_for_tpsl + atr * tp_mult
        sl_price = entry_for_tpsl - atr * sl_mult
        gate_size = _calc_position_size(entry_for_tpsl, atr, sl_mult)  # positive = long
    else:  # short
        tp_price = entry_for_tpsl - atr * tp_mult
        sl_price = entry_for_tpsl + atr * sl_mult
        gate_size = -_calc_position_size(entry_for_tpsl, atr, sl_mult)  # negative = short

    try:
        order_result = executor.place_market_order(gate_size, tp_price, sl_price)
        order_id = str(order_result.get("id", ""))
        order_status = order_result.get("status", "")
        if order_id and order_status not in ("error",):
            order_ts = time.time()
            fill_px = None
            fill_ts = None
            try:
                detail = executor.get_order_detail(order_id)
                if detail:
                    fp = detail.get("fill_price", "") or detail.get("price", "")
                    if fp and float(fp) > 0:
                        fill_px = float(fp)
                    ft = detail.get("finish_time", "") or detail.get("create_time", "")
                    if ft:
                        fill_ts = float(ft)
            except Exception as e:
                logger.warning(f"Failed to fetch order detail: {e}")

            actual_entry = fill_px or entry_for_tpsl
            if fill_px:
                if event_dir == 1:
                    tp_price = actual_entry + atr * tp_mult
                    sl_price = actual_entry - atr * sl_mult
                else:
                    tp_price = actual_entry - atr * tp_mult
                    sl_price = actual_entry + atr * sl_mult

            # Place TP/SL conditional orders
            close_size = -gate_size  # opposite direction to close
            try:
                executor.place_tpsl(close_size, tp_price, sl_price)
            except Exception as e:
                logger.warning(f"Failed to place TP/SL orders: {e}")

            side = "buy" if gate_size > 0 else "sell"
            state.open_position = {
                "side": side, "entry": actual_entry, "tp": tp_price, "sl": sl_price,
                "size": abs(gate_size), "gate_size": gate_size,
                "time": state.last_bar_time,
                "open_ts": time.time(),
                "order_id": order_id,
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
            logger.info(f"Order placed: {side} {abs(gate_size)} @ {actual_entry:.2f}, TP={tp_price:.2f}, SL={sl_price:.2f}")
        else:
            logger.error(f"Order failed: {order_result}")
    except Exception as e:
        logger.error(f"Order execution error: {e}")


# --- Gate.io message handlers ---
_last_tick_broadcast: float = 0.0

aggregator = BarAggregator(on_bar_complete=on_bar_complete)


def on_trades_message(msg: dict):
    """Handle Gate.io futures.trades update.
    Gate format: {"channel": "futures.trades", "event": "update",
                  "result": [{"contract": "ETH_USDT", "size": 10, "price": "3001.1",
                               "id": 123, "create_time": 1234567890,
                               "create_time_ms": 1234567890123}]}
    size > 0 = buy (long aggressor), size < 0 = sell (short aggressor)
    """
    global _last_tick_broadcast
    state.ws_trades_connected = True
    for trade in msg.get("result", []):
        raw_size = trade.get("size", 0)
        px = float(trade["price"])
        size = abs(float(raw_size))
        side = "buy" if raw_size > 0 else "sell"
        ts_ms = int(trade.get("create_time_ms", trade.get("create_time", 0) * 1000))
        aggregator.ingest_trade(price=px, size=size, side=side, ts_ms=ts_ms)
        state.last_trade_ts = time.time()
        state.current_price = px

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
    """Handle Gate.io futures.order_book update.
    Gate format: {"channel": "futures.order_book", "event": "update",
                  "result": {"t": 1234567890123, "contract": "ETH_USDT",
                              "asks": [{"p": "3001.1", "s": 100}, ...],
                              "bids": [{"p": "3000.9", "s": 200}, ...]}}
    """
    global _last_ob_broadcast
    state.ws_books_connected = True
    result = msg.get("result", {})
    if isinstance(result, dict):
        ob_state.update(result)
        state.last_book_ts = time.time()
    now = time.time()
    if now - _last_ob_broadcast >= 0.5:
        _last_ob_broadcast = now
        try:
            loop = asyncio.get_event_loop()
            loop.create_task(broadcast("orderbook", ob_state.get_snapshot()))
        except RuntimeError:
            pass


def on_private_message(msg: dict):
    """Handle Gate.io private channel pushes (orders, positions, balances)."""
    channel = msg.get("channel", "")
    result = msg.get("result", {})
    if channel == "futures.orders":
        _handle_orders_push(result if isinstance(result, list) else [result])
    elif channel == "futures.positions":
        _handle_positions_push(result if isinstance(result, list) else [result])
    elif channel == "futures.balances":
        _handle_balances_push(result if isinstance(result, list) else [result])


def _handle_orders_push(orders: list):
    """Process futures.orders push — update fill info for open position."""
    for o in orders:
        if o.get("contract") != config.INST_ID:
            continue
        status = o.get("status", "")
        if status == "finished" and state.open_position:
            oid = str(o.get("id", ""))
            if oid == state.open_position.get("order_id", ""):
                fp = o.get("fill_price", "") or o.get("price", "")
                if fp and float(fp) > 0 and not state.open_position.get("fill_px"):
                    state.open_position["fill_px"] = float(fp)
                    if state.trades:
                        state.trades[-1]["fill_px"] = float(fp)


def _handle_positions_push(positions: list):
    """Process futures.positions push."""
    active = [p for p in positions
              if p.get("contract") == config.INST_ID and int(p.get("size", 0)) != 0]
    _process_position_data(active)


def _handle_balances_push(balances: list):
    """Process futures.balances push."""
    for b in balances:
        account_data = {
            "totalEq": b.get("balance", "0"),
            "availBal": b.get("available", "0"),
            "upl": b.get("unrealised_pnl", "0"),
            "ctVal": _ct_val,
            "lever": _lever,
        }
        state.last_account = account_data
        try:
            loop = asyncio.get_event_loop()
            loop.create_task(broadcast("account", account_data))
        except RuntimeError:
            pass


def _record_trade_close(exit_type: str = None):
    if not state.trades or not state.open_position:
        return
    pos = state.open_position
    entry = pos.get("entry", 0)
    exit_px = state.current_price
    side = pos.get("side", "buy")
    size = float(pos.get("size", 0))

    if side == "buy":
        pnl = (exit_px - entry) * size * _ct_val
    else:
        pnl = (entry - exit_px) * size * _ct_val

    if exit_type is None:
        tp, sl = pos.get("tp", 0), pos.get("sl", 0)
        if side == "buy":
            exit_type = "tp" if exit_px >= tp else ("sl" if exit_px <= sl else "close")
        else:
            exit_type = "tp" if exit_px <= tp else ("sl" if exit_px >= sl else "close")

    trade = state.trades[-1]
    trade["close_price"] = exit_px
    trade["close_time"] = time.time()
    trade["pnl"] = round(pnl, 4)
    trade["exit_type"] = exit_type

    rewrite_log("trades", state.trades)
    try:
        loop = asyncio.get_event_loop()
        loop.create_task(broadcast("trade_update", trade))
    except RuntimeError:
        pass
    logger.info(f"Trade closed: exit={exit_px:.2f} pnl={pnl:.4f} USDT ({exit_type})")


def _backfill_trade_pnl():
    """Backfill PnL and timestamps for historical trades using Gate order history."""
    needs_pnl = [t for t in state.trades if t.get("pnl") is None]
    if not needs_pnl:
        return
    try:
        orders = executor.get_order_history(limit=100)
    except Exception as e:
        logger.error(f"Backfill: failed to fetch order history: {e}")
        return
    filled = [o for o in orders if o.get("status") == "finished" and float(o.get("fill_price", 0) or 0) > 0]
    updated = False

    for trade in state.trades:
        ord_id = str(trade.get("order_id", ""))
        entry = trade.get("entry", 0)
        side = trade.get("side", "buy")
        open_ts = trade.get("open_ts", 0)

        if trade.get("fill_ts") is None and ord_id:
            open_order = next((o for o in filled if str(o.get("id", "")) == ord_id), None)
            if open_order:
                fp = open_order.get("fill_price", "")
                if fp and float(fp) > 0:
                    trade["fill_px"] = float(fp)
                ft = open_order.get("finish_time", "") or open_order.get("create_time", "")
                if ft:
                    trade["fill_ts"] = float(ft)
                if not trade.get("scanner_ts"):
                    trade["scanner_ts"] = open_ts - (POST_WINDOW_S + 1)
                if not trade.get("model_ts"):
                    trade["model_ts"] = open_ts - 0.5
                updated = True

        if trade.get("pnl") is None:
            close_order = None
            for o in filled:
                o_size = int(o.get("size", 0))
                o_ts = float(o.get("finish_time", 0) or o.get("create_time", 0))
                o_is_close = (side == "buy" and o_size < 0) or (side == "sell" and o_size > 0)
                if o_is_close and o_ts > open_ts:
                    if close_order is None or o_ts < float(close_order.get("finish_time", 0) or 0):
                        close_order = o
            if close_order:
                fill_px = float(close_order.get("fill_price", 0) or 0)
                size = float(trade.get("size", 0))
                if side == "buy":
                    total_pnl = (fill_px - entry) * size * _ct_val
                else:
                    total_pnl = (entry - fill_px) * size * _ct_val
                tp, sl = trade.get("tp", 0), trade.get("sl", 0)
                if side == "buy":
                    exit_type = "tp" if fill_px >= tp else ("sl" if fill_px <= sl else "close")
                else:
                    exit_type = "tp" if fill_px <= tp else ("sl" if fill_px >= sl else "close")
                trade["close_price"] = fill_px
                trade["close_time"] = float(close_order.get("finish_time", 0) or 0)
                trade["pnl"] = round(total_pnl, 4)
                trade["exit_type"] = exit_type
                updated = True
                logger.info(f"Backfill PnL: trade @ {entry:.2f} -> close {fill_px:.2f}, pnl={total_pnl:.4f} ({exit_type})")

    if updated:
        rewrite_log("trades", state.trades)
        logger.info("Backfill complete")


def _check_breakeven_stop():
    pos = state.open_position
    if pos is None or state.current_price is None:
        return
    entry = pos.get("entry", 0)
    atr = pos.get("atr", state.current_atr)
    if atr <= 0:
        return
    side = pos.get("side", "buy")
    price = state.current_price

    if side == "buy":
        fav_r = (price - entry) / atr
    else:
        fav_r = (entry - price) / atr

    prev_max = pos.get("max_fav_r", 0)
    pos["max_fav_r"] = max(prev_max, fav_r)

    BREAKEVEN_TRIGGER = 1.0
    LOCK_TRIGGER = 1.5
    LOCK_LEVEL = 0.5

    current_sl = pos.get("sl", 0)
    new_sl = current_sl

    if pos["max_fav_r"] >= LOCK_TRIGGER:
        if side == "buy":
            new_sl = max(current_sl, entry + LOCK_LEVEL * atr)
        else:
            new_sl = min(current_sl, entry - LOCK_LEVEL * atr)
    elif pos["max_fav_r"] >= BREAKEVEN_TRIGGER:
        if side == "buy":
            new_sl = max(current_sl, entry)
        else:
            new_sl = min(current_sl, entry)

    if new_sl != current_sl:
        logger.info(f"Breakeven stop: SL moved {current_sl:.2f} -> {new_sl:.2f} (max_fav={pos['max_fav_r']:.2f}R)")
        pos["sl"] = new_sl
        tp = pos.get("tp", 0)
        close_size = -pos.get("gate_size", 0)
        try:
            ok = executor.update_tpsl(new_tp=tp, new_sl=new_sl, close_size=close_size)
            if ok:
                logger.info(f"Gate price orders updated: TP={tp:.2f}, SL={new_sl:.2f}")
            else:
                logger.warning("Failed to update Gate price orders, local SL still active")
        except Exception as e:
            logger.warning(f"update_tpsl error: {e}")
        breached = (side == "buy" and price <= new_sl) or (side == "sell" and price >= new_sl)
        if breached:
            logger.info(f"Breakeven SL breached at {price:.2f}, closing position")
            try:
                close_result = executor.close_position()
                if close_result.get("id"):
                    _record_trade_close(exit_type="breakeven")
                    state.open_position = None
            except Exception as e:
                logger.error(f"Breakeven close error: {e}")


def _process_position_data(active: list):
    """Shared logic for position updates from REST or WS.
    Gate position fields: size (signed int), entry_price, unrealised_pnl, leverage, liq_price
    """
    if not active:
        if state.open_position is not None:
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
    pos_size = int(p.get("size", 0))
    pos_side = "long" if pos_size > 0 else "short"
    upnl = float(p.get("unrealised_pnl", 0) or 0)
    avg_px = float(p.get("entry_price", 0) or 0)
    liq_px = p.get("liq_price", "")
    lever = p.get("leverage", "")
    margin = float(p.get("margin", 0) or 0)

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
        # Breakeven stop disabled — evaluated and found to hurt performance
        # _check_breakeven_stop()
        open_ts = state.open_position.get("open_ts", 0)
        elapsed_s = time.time() - open_ts if open_ts else 0
        pos_side = pos_data["side"]
        if config.SPLIT_MODEL:
            horizon = config.LONG_HORIZON if pos_side == "long" else config.SHORT_HORIZON
        else:
            horizon = config.HORIZON
        horizon_s = horizon * 60
        pos_data["elapsed_s"] = round(elapsed_s)
        pos_data["horizon_s"] = horizon_s
        if open_ts and elapsed_s >= horizon_s:
            logger.info(f"Horizon timeout: position held {elapsed_s:.0f}s >= {horizon_s}s, closing at market")
            try:
                close_result = executor.close_position()
                if close_result.get("id"):
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


# --- Warmup from REST candles ---
def warmup_from_candles():
    logger.info("Warming up from REST candles...")
    candles = executor.get_candles(interval="1m", limit=300)
    if not candles:
        logger.warning("No candles returned for warmup")
        return
    # Gate returns oldest first (ascending by time)
    for c in candles:
        ts = int(c.get("t", 0))
        bar = {
            "datetime": datetime.fromtimestamp(ts, tz=timezone.utc).replace(second=0, microsecond=0),
            "open": float(c.get("o", 0)),
            "high": float(c.get("h", 0)),
            "low": float(c.get("l", 0)),
            "close": float(c.get("c", 0)),
            "volume": float(c.get("v", 0)),
            "aggressor_ratio": 0.5, "net_taker_vol_ratio": 0.0,
            "trade_gini": 0.0, "large_trade_vol_ratio": 0.0,
            "trade_intensity": float(np.log1p(float(c.get("v", 0)))),
            "data_from_orderbook": 0,
        }
        engine.add_bar(bar)
        state.recent_bars.append({
            "time": int(bar["datetime"].timestamp()),
            "open": bar["open"], "high": bar["high"], "low": bar["low"], "close": bar["close"],
        })
    state.bars_received = engine.bar_count
    state.is_warm = engine.is_warm
    state.current_price = float(candles[-1].get("c", 0)) if candles else 0.0
    logger.info(f"Warmup complete: {engine.bar_count} bars, is_warm={engine.is_warm}")


# --- REST fallback monitor ---
async def rest_fallback_monitor():
    while True:
        await asyncio.sleep(30)
        try:
            positions = executor.get_positions()
            active = [p for p in (positions or [])
                      if p.get("contract") == config.INST_ID and int(p.get("size", 0)) != 0]
            _process_position_data(active)

            bal = executor.get_balance()
            if bal:
                account_data = {
                    "totalEq": bal.get("total", "0"),
                    "availBal": bal.get("available", "0"),
                    "upl": bal.get("unrealised_pnl", "0"),
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
            _ct_val = float(inst.get("quanto_multiplier", "0.0001"))
            _lot_sz = float(inst.get("order_size_min", "1"))
            _min_sz = float(inst.get("order_size_min", "1"))
            logger.info(f"Contract info: quanto_multiplier={_ct_val}, min_size={_min_sz}")
    except Exception as e:
        logger.warning(f"Failed to fetch contract info: {e}")

    # Set leverage
    try:
        lev_result = executor.set_leverage(config.LEVERAGE)
        if isinstance(lev_result, dict) and lev_result.get("leverage"):
            _lever = int(lev_result["leverage"])
            state.leverage = _lever
            logger.info(f"Leverage set: {_lever}x")
        else:
            logger.warning(f"set_leverage response: {lev_result}")
    except Exception as e:
        logger.warning(f"set_leverage error: {e}")

    warmup_from_candles()

    # Initial account + position via REST
    try:
        bal = executor.get_balance()
        if bal:
            state.last_account = {
                "totalEq": bal.get("total", "0"),
                "availBal": bal.get("available", "0"),
                "upl": bal.get("unrealised_pnl", "0"),
                "ctVal": _ct_val, "lever": _lever,
            }
    except Exception:
        pass
    try:
        positions = executor.get_positions()
        active = [p for p in (positions or [])
                  if p.get("contract") == config.INST_ID and int(p.get("size", 0)) != 0]
        _process_position_data(active)
    except Exception:
        pass

    # Restore logs
    state.signals = load_log("signals")
    state.trades = load_log("trades")
    _backfill_trade_pnl()

    # Start WebSocket connections
    ws_trades = GateWebSocket(
        config.WS_PUBLIC,
        [{"channel": "futures.trades", "payload": [config.INST_ID]}],
        on_trades_message, name="trades",
    )
    ws_books = GateWebSocket(
        config.WS_PUBLIC,
        [{"channel": "futures.order_book", "payload": [config.INST_ID, "20", "0"]}],
        on_books_message, name="books",
    )
    ws_private = GatePrivateWebSocket(
        config.WS_PRIVATE,
        [
            {"channel": "futures.orders", "payload": [config.INST_ID]},
            {"channel": "futures.positions", "payload": [config.INST_ID]},
            {"channel": "futures.balances", "payload": []},
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


class DailyTradeLimit(BaseModel):
    limit: int


@app.post("/api/daily-trade-limit")
async def set_daily_trade_limit(body: DailyTradeLimit):
    state.daily_trade_limit = max(0, body.limit)
    logger.info(f"Daily trade limit set to {state.daily_trade_limit} via dashboard")
    await broadcast("health", state.snapshot())
    return {"daily_trade_limit": state.daily_trade_limit}


class RiskParams(BaseModel):
    trade_notional: float | None = None
    risk_per_trade: float | None = None
    max_capital: float | None = None
    daily_loss_limit_r: float | None = None


@app.post("/api/risk-params")
async def set_risk_params(body: RiskParams):
    import math
    if body.trade_notional is not None:
        state.trade_notional = max(0, body.trade_notional)
    if body.risk_per_trade is not None:
        state.risk_per_trade = max(0, min(1, body.risk_per_trade))
    if body.max_capital is not None:
        state.max_capital = max(0, body.max_capital)
    if body.daily_loss_limit_r is not None:
        state.daily_loss_limit_r = min(0, body.daily_loss_limit_r)
    preview_lever = _lever
    if state.current_price > 0 and _ct_val > 0 and state.max_capital > 0 and state.current_atr > 0:
        sl_for_preview = config.LONG_SL if config.SPLIT_MODEL else config.SL_MULT
        if sl_for_preview > 0:
            max_loss = state.trade_notional * state.risk_per_trade
            loss_per_contract = sl_for_preview * state.current_atr * _ct_val
            contracts = max(int(_min_sz), int(max_loss / loss_per_contract))
            notional = contracts * _ct_val * state.current_price
            preview_lever = max(1, math.ceil(notional / state.max_capital))
    logger.info(f"Risk params updated: notional={state.trade_notional} "
                f"R={state.risk_per_trade} max_capital={state.max_capital} "
                f"daily_loss={state.daily_loss_limit_r} preview_lever={preview_lever}x")
    state.leverage = preview_lever
    await broadcast("health", state.snapshot())
    return {
        "trade_notional": state.trade_notional,
        "risk_per_trade": state.risk_per_trade,
        "max_capital": state.max_capital,
        "daily_loss_limit_r": state.daily_loss_limit_r,
        "leverage": preview_lever,
    }


def _dedup_bars(bars: list) -> list:
    seen = {}
    for b in bars:
        seen[b["time"]] = b
    return sorted(seen.values(), key=lambda x: x["time"])


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    ws_clients.add(ws)
    try:
        bars = _dedup_bars(list(state.recent_bars)[-200:])
        await ws.send_text(json.dumps({
            "type": "init",
            "data": {
                "bars": bars,
                "signals": state.signals[-50:],
                "trades": state.trades,
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
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        ws_clients.discard(ws)


if __name__ == "__main__":
    uvicorn.run("live.main:app", host="0.0.0.0", port=config.DASHBOARD_PORT, reload=False)

