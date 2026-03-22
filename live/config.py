from __future__ import annotations
import json, os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- OKX API credentials ---
OKX_API_KEY = os.environ.get("OKX_API_KEY", "20f35e70-e967-474a-a486-88dcbd4d754a")
OKX_SECRET_KEY = os.environ.get("OKX_SECRET_KEY", "ED96880ED8EEA294F490CE8B71802B3B")
OKX_PASSPHRASE = os.environ.get("OKX_PASSPHRASE", "Xcx1294500682@")  # TODO: fill your passphrase

# --- Trading ---
INST_ID = "ETH-USDT-SWAP"
TD_MODE = "cross"
FLAG = "0"  # 0=live, 1=demo

# --- OKX URLs ---
REST_BASE = "https://www.okx.com"
WS_PUBLIC = "wss://ws.okx.com:8443/ws/v5/public"
WS_PRIVATE = "wss://ws.okx.com:8443/ws/v5/private"

# --- Load best_config.json ---
_cfg_path = PROJECT_ROOT / "best_config.json"
with open(_cfg_path) as f:
    BEST = json.load(f)

SCANNER_NAME: str = BEST.get("scanner_name", "flow_reversal")
SCANNER_VARIANT: str = BEST.get("scanner_variant", "")
# Parse "flow_reversal|flow_abs=0.05,obi_abs=0.0,..." into dict
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

MODEL_PATH = str(PROJECT_ROOT / "model_sniper_v3_first_touch.json")

# --- Multi-model support ---
MODEL_TYPE: str = BEST.get("model_type", "xgb")
MODEL_PATH_XGB = str(PROJECT_ROOT / "model_xgb.json")
MODEL_PATH_LGB = str(PROJECT_ROOT / "model_lgb.txt")
MODEL_PATH_CB = str(PROJECT_ROOT / "model_catboost.cbm")

# --- Feature engine ---
WARMUP_BARS = 300
BAR_WINDOW = 500

# --- Risk ---
LEVERAGE = 1
MAX_POSITIONS = 1
INITIAL_CAPITAL = 100   # total initial capital in USD
RISK_PER_TRADE = 0.20   # fraction of capital per trade (20%)
DAILY_LOSS_LIMIT_R = -5.0  # stop trading after cumulative daily loss exceeds this (in R-multiples)
MAX_TRADES_PER_DAY = 10    # max trades per calendar day

# --- Persistence ---
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# --- Dashboard ---
DASHBOARD_PORT = 8080
