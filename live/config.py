from __future__ import annotations
import json, os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- Gate.io API credentials ---
GATE_API_KEY = os.environ.get("GATE_API_KEY", "")
GATE_SECRET_KEY = os.environ.get("GATE_SECRET_KEY", "")

# --- Trading ---
INST_ID = "ETH_USDT"          # Gate.io futures contract name
SETTLE = "usdt"               # Gate.io futures settle currency

# --- Gate.io URLs ---
REST_BASE = "https://api.gateio.ws"
WS_PUBLIC = "wss://fx-ws.gateio.ws/v4/ws/usdt"
WS_PRIVATE = "wss://fx-ws.gateio.ws/v4/ws/usdt"

# Testnet (set USE_TESTNET=1 env var to enable)
if os.environ.get("USE_TESTNET", "0") == "1":
    REST_BASE = "https://fx-api-testnet.gateio.ws"
    WS_PUBLIC = "wss://fx-ws-testnet.gateio.ws/v4/ws/usdt"
    WS_PRIVATE = "wss://fx-ws-testnet.gateio.ws/v4/ws/usdt"

# --- Load best_config.json ---
_cfg_path = PROJECT_ROOT / "best_config.json"
with open(_cfg_path) as f:
    BEST = json.load(f)

SCANNER_NAME: str = BEST.get("scanner_name", "flow_reversal")
SCANNER_VARIANT: str = BEST.get("scanner_variant", "")
_param_str = SCANNER_VARIANT.split("|", 1)[1] if "|" in SCANNER_VARIANT else ""
SCANNER_PARAMS: dict = {}
for kv in _param_str.split(","):
    if "=" in kv:
        k, v = kv.split("=", 1)
        SCANNER_PARAMS[k.strip()] = float(v.strip())

HORIZON: int = int(BEST["h"])
TP_MULT: float = float(BEST["tp"])
SL_MULT: float = float(BEST["sl"])
THRESHOLD: float = float(BEST["threshold"])
LABEL_MODE: str = BEST.get("label_mode", "first_touch")
LONG_ONLY: bool = BEST.get("long_only", False)

# --- Split long/short model support ---
SPLIT_MODEL: bool = BEST.get("split_model", False)
LONG_CFG: dict = BEST.get("long_config", {})
SHORT_CFG: dict = BEST.get("short_config", {})
LONG_TP: float = float(LONG_CFG.get("tp", TP_MULT))
LONG_SL: float = float(LONG_CFG.get("sl", SL_MULT))
LONG_THRESHOLD: float = float(LONG_CFG.get("threshold", THRESHOLD))
LONG_HORIZON: int = int(LONG_CFG.get("h", HORIZON))
SHORT_TP: float = float(SHORT_CFG.get("tp", TP_MULT))
SHORT_SL: float = float(SHORT_CFG.get("sl", SL_MULT))
SHORT_THRESHOLD: float = float(SHORT_CFG.get("threshold", THRESHOLD))
SHORT_HORIZON: int = int(SHORT_CFG.get("h", HORIZON))

MODEL_PATH = str(PROJECT_ROOT / "model_sniper_v3_first_touch.json")

# --- Multi-model support ---
MODEL_TYPE: str = BEST.get("model_type", "xgb")
MODEL_PATH_XGB = str(PROJECT_ROOT / "model_xgb.json")
MODEL_PATH_XGB_LONG = str(PROJECT_ROOT / "model_xgb_long.json")
MODEL_PATH_XGB_SHORT = str(PROJECT_ROOT / "model_xgb_short.json")
MODEL_PATH_LGB = str(PROJECT_ROOT / "model_lgb.txt")
MODEL_PATH_CB = str(PROJECT_ROOT / "model_catboost.cbm")

# --- Feature engine ---
WARMUP_BARS = 300
BAR_WINDOW = 500

# --- Risk ---
LEVERAGE = 1
MAX_POSITIONS = 1
TRADE_NOTIONAL = 100
RISK_PER_TRADE = 0.08
MAX_CAPITAL = 200
DAILY_LOSS_LIMIT_R = -5.0
MAX_TRADES_PER_DAY = 10

# --- Persistence ---
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# --- Dashboard ---
DASHBOARD_PORT = 8080
