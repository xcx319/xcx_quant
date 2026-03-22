from __future__ import annotations
import json, time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from . import config


@dataclass
class AppState:
    # Connection health
    ws_trades_connected: bool = False
    ws_books_connected: bool = False
    last_trade_ts: Optional[float] = None
    last_book_ts: Optional[float] = None

    # Bar state
    bars_received: int = 0
    is_warm: bool = False
    last_bar_time: Optional[str] = None
    current_price: float = 0.0
    current_atr: float = 0.0

    # Position
    open_position: Optional[dict] = None
    has_foreign_position: bool = False  # pre-existing position not opened by this system
    last_position_data: Optional[dict] = None  # latest exchange position snapshot
    foreign_positions: list = field(default_factory=list)  # non-SWAP positions from other instruments

    # Account & scanner state for dashboard
    last_account: Optional[dict] = None
    last_scanner_state: Optional[dict] = None

    # Trading toggle (default OFF — must be enabled via dashboard)
    trading_enabled: bool = False
    daily_trade_limit: int = config.MAX_TRADES_PER_DAY
    leverage: int = config.LEVERAGE
    trade_notional: float = config.TRADE_NOTIONAL
    risk_per_trade: float = config.RISK_PER_TRADE
    max_capital: float = config.MAX_CAPITAL
    daily_loss_limit_r: float = config.DAILY_LOSS_LIMIT_R

    # Logs
    signals: list = field(default_factory=list)
    trades: list = field(default_factory=list)
    equity_curve: list = field(default_factory=list)

    # Recent bars for dashboard chart
    recent_bars: deque = field(default_factory=lambda: deque(maxlen=500))

    def snapshot(self) -> dict:
        return {
            "ws_trades": self.ws_trades_connected,
            "ws_books": self.ws_books_connected,
            "bars_received": self.bars_received,
            "is_warm": self.is_warm,
            "last_bar_time": self.last_bar_time,
            "current_price": self.current_price,
            "current_atr": self.current_atr,
            "open_position": self.open_position,
            "signals_count": len(self.signals),
            "trades_count": len(self.trades),
            "trading_enabled": self.trading_enabled,
            "daily_trade_limit": self.daily_trade_limit,
            "leverage": self.leverage,
            "trade_notional": self.trade_notional,
            "risk_per_trade": self.risk_per_trade,
            "max_capital": self.max_capital,
            "daily_loss_limit_r": self.daily_loss_limit_r,
        }


# --- JSONL persistence ---

def _jsonl_path(name: str) -> Path:
    return config.DATA_DIR / f"{name}.jsonl"


def append_log(name: str, record: dict):
    record.setdefault("ts", time.time())
    with open(_jsonl_path(name), "a") as f:
        f.write(json.dumps(record, default=str) + "\n")


def load_log(name: str) -> list[dict]:
    p = _jsonl_path(name)
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def rewrite_log(name: str, records: list[dict]):
    """Rewrite entire JSONL file (for updating trade records with close info)."""
    with open(_jsonl_path(name), "w") as f:
        for r in records:
            f.write(json.dumps(r, default=str) + "\n")
