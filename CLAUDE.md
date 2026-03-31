# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

```bash
conda activate quant
```

Dependencies: pandas, numpy, xgboost, scikit-learn, pandas_ta, matplotlib. Optional: tqdm, lightgbm, catboost.

Live system additional deps: fastapi, uvicorn, websockets, httpx, jinja2.

## Running Scripts

```bash
# 1. Build enhanced dataset from raw data
python pipline_modified.py

# 2. Grid search for best horizon/TP/SL config → best_config.json
python tune_all_modified.py

# 3. Quick scanner variant evaluation
python scanner_grid_search.py

# 4. Robust out-of-sample validation (multi-fold)
python robust_oos_search.py --scanner flow_reversal --label-mode first_touch

# 5. Train final model using best_config.json → model + plots
python train_xgb.py          # XGBoost only
python train_multi_model.py  # XGBoost + LightGBM + CatBoost + ensemble

# 6. Hyperparameter tuning for XGBoost
python tune_xgb_params.py

# 7. Compare event-aligned post-window durations (5s/10s/15s/20s)
python compare_post_windows.py

# 8. Run live trading system (Gate.io ETH_USDT futures)
python -m live.main
```

Most scripts accept `--data-path`, `--label-mode`, `--same-bar-policy`, and parameter range flags (e.g. `--horizons`, `--tp-values`, `--sl-values`). Use `--help` for details.

Tests (require `dataset_enhanced.parquet`, `best_config.json`, and a trained model):

```bash
python test_event_aligned.py   # Compare predictions with/without event-aligned features
python test_horizon_close.py   # Compare horizon-close vs hold-until-TP/SL PnL
```

## Architecture

**Data flow:** raw data → `pipline_modified.py` → `dataset_enhanced.parquet` → tuning/search/training scripts → models + reports. For Gate.io data: `download_gate_data.py` → `build_gate_dataset.py` → `dataset_gate_enhanced.parquet`.

### Core Modules

- **`quant_modeling.py`** — Shared library imported by all scripts. Defines `BASE_FEATURES` (87 features), labeling logic (`build_labels`, `build_realized_pnl`), `LabelingCache` for vectorized label computation, and `add_directional_features`. This is the single source of truth for feature lists and label definitions.

- **`pipline_modified.py`** — Data pipeline. Loads raw OHLCV + orderbook data, computes technical indicators via pandas_ta, generates scanner signals (5 types: breakout_60, derivative_reversal, wick_reversal, flow_reversal, trend_exhaustion_confirmed), and builds forward-looking price columns (`future_high/low/close_{1..30}m`). The `add_features()` function is reused by the live system for feature parity.

- **`tune_all_modified.py`** — Exhaustive grid search over horizon/TP/SL combinations. Evaluates via Average Precision. Writes winner to `best_config.json`.

- **`robust_oos_search.py`** — Multi-fold walk-forward OOS testing with purge gaps (`PURGE_GAP_BARS=30`). Optimizes threshold on validation fold, tests on held-out fold. Outputs summary CSVs.

- **`train_xgb.py`** — Trains final XGBoost classifier using `best_config.json`. XGBoost defaults: `n_estimators=1500`, `max_depth=4`, `learning_rate=0.003`, early stopping at 120 rounds. Generates equity curves and performance plots to `./plots/`.

- **`train_multi_model.py`** — Trains XGBoost + LightGBM + CatBoost and an ensemble (average of 3). Reuses evaluation functions from `train_xgb.py`.

- **`tune_xgb_params.py`** — Hyperparameter grid search for XGBoost using the same multi-fold OOS framework. Outputs `xgb_param_search.csv`.

- **`train_long_short_split.py`** — Trains separate long/short models with per-direction TP/SL/threshold. Splits dataset by `event_dir`, trains independent XGBoost classifiers for each direction.

- **`compare_post_windows.py`** — Compares event-aligned feature windows (5s/10s/15s/20s) by running pipeline → grid search → robust OOS for each. Outputs `robust_oos_pw{N}.csv`.

### Live Trading System (`live/`)

Real-time trading system for Gate.io USDT perpetual futures. Runs as a FastAPI app with a WebSocket-powered dashboard at `http://localhost:8080`.

**Pipeline:** Gate.io WebSocket trades → `BarAggregator` (1-min OHLCV + microstructure) → `FeatureEngine` (calls `pipline_modified.add_features()` directly) → `FlowReversalScanner` → `ModelInference` (XGBoost/LGB/CatBoost/ensemble) → `OrderExecutor` (REST API market orders with TP/SL).

Key modules:
- **`live/config.py`** — Loads `best_config.json` for threshold/TP/SL/scanner params. Gate.io API credentials from env vars. Constants: `WARMUP_BARS=300`, `BAR_WINDOW=500`, `MAX_POSITIONS=1`. Supports `MODEL_TYPE` in `{xgb, lgb, catboost, ensemble}`.
- **`live/bar_aggregator.py`** — Builds 1-min bars from raw trades with microstructure features (aggressor_ratio, trade_gini, large_trade_vol_ratio, trade_intensity).
- **`live/orderbook_state.py`** — Maintains orderbook state, computes OBI, spread, depth, microprice, wall detection features.
- **`live/feature_engine.py`** — Rolling window (500 bars, 300 warmup). Calls `add_features()` from `pipline_modified.py` on each bar to match backtest feature computation exactly.
- **`live/event_aligner.py`** — Computes event-aligned features from live tick data after a `POST_WINDOW_S=15` second delay post bar-close. `POST_WINDOW_S` is defined here, not in config.
- **`live/scanner.py`** — `FlowReversalScanner` evaluates range_pos, flow, OBI conditions. Parameterized via `best_config.json` scanner_variant string.
- **`live/model_inference.py`** — Loads model, applies directional feature transforms (mirrors `add_directional_features`), predicts probability, compares to threshold. Supports multi-model with fallback logic.
- **`live/execution.py`** — Gate.io REST client with HMAC-SHA512 signing. Places market orders with separate TP/SL price orders.
- **`live/ws_client.py`** — Async Gate.io WebSocket client with auto-reconnect, exponential backoff, and per-subscription HMAC-SHA512 auth for private channels.
- **`live/state.py`** — `AppState` dataclass + JSONL persistence for signals/trades in `live/data/`.
- **`live/main.py`** — FastAPI app. Warms up from 300 REST candles on startup, then processes live bars. Dashboard broadcasts via WebSocket.

Warmup: On startup, fetches 300 historical 1-min candles via REST to fill the feature engine before processing live data. Orderbook features are unavailable during warmup (defaults to zero).

Signal delay: When the scanner triggers at bar close, the system waits ~16 seconds (`POST_WINDOW_S + 1`) to collect tick data for event-aligned features before running model inference. This delay is intentional and mirrors how backtest event features are computed.

### Key Concepts

- **Label modes:** `first_touch` (TP/SL race with same-bar ambiguity handling) and `window_tp` (any TP hit within horizon = win). Controlled by `--label-mode` and `--same-bar-policy` (drop/neutral/tp_first/sl_first).
- **R-multiples:** PnL is measured in ATR units (risk multiples), not raw price.
- **Directional features:** Raw features are multiplied by `event_dir` (+1 long, -1 short) to make them direction-agnostic for the model. The 8 directional features are: `dir_obi`, `dir_aggressor`, `dir_net_taker`, `dir_ema20_dist`, `dir_z_vwap`, `dir_ret_1/5/20`, `dir_event_return`.
- **Purge gap:** 30-bar gap between train/valid/test folds to prevent lookahead leakage.
- **Event-aligned features:** 5 features computed from tick-level data in a post-event window (sec_in_bar, event_return, event_effort_vs_result, event_rejection_strength, time_to_reject_s). In live, these require a ~16s delay after bar close.
- **Sample weighting:** Positive class weighted by `(1 - pos_ratio) / pos_ratio` to handle class imbalance.
- **Scanner variant format:** `scanner_name|param1=val1,param2=val2,...` (e.g. `flow_reversal|flow_abs=0.05,obi_abs=0.0,range_hi=0.7,range_lo=0.3,spread_mult=1.0`).

### Critical Invariants

- **Feature parity:** The live system imports `add_features()` directly from `pipline_modified.py`. Never reimplement feature logic in `live/` — always reuse the pipeline function.
- **Entry price:** Uses `entry_price_delayed` column if available, falls back to `close`. This matters for label computation.
- **Threshold selection:** Smoothed over a rolling window (`threshold_smooth_window`) on the validation fold, then tested on a held-out fold. The threshold in `best_config.json` is the final selected value.
- **Model format:** XGBoost saves as JSON (`model_sniper_v3_*.json` or `model_xgb.json`), LightGBM as text (`model_lgb.txt`), CatBoost as binary (`model_catboost.cbm`). Live inference has fallback logic that tries multi-model paths first, then legacy single-model path.

### `best_config.json` Structure

```json
{
  "h": 30,                    // horizon in minutes
  "tp": 2.0,                  // TP multiple (in ATR)
  "sl": 0.5,                  // SL multiple (in ATR)
  "threshold": 0.5062,        // model probability threshold
  "label_mode": "first_touch",
  "same_bar_policy": "drop",
  "scanner_name": "flow_reversal",
  "scanner_variant": "flow_reversal|flow_abs=0.05,...",
  "model_type": "xgb",        // xgb | lgb | catboost | ensemble
  "ap": 0.243,                // average precision (backtest metric)
  "avg_r_per_trade": 0.070    // average R per trade (backtest metric)
}
```

This file is the central config consumed by `train_xgb.py`, `train_multi_model.py`, and `live/config.py`. Updated by `tune_all_modified.py`.

## Gate.io Migration (Completed)

The live trading system has been migrated from OKX to Gate.io. Historical data pipeline scripts are also provided.

**Historical data pipeline:**
```bash
# Download raw data to /Volumes/TU280Pro/quant/raw_data
python download_gate_data.py --market ETH_USDT --start-month 202401 --end-month 202603

# Build parquet dataset (compatible with train_xgb.py)
python build_gate_dataset.py --market ETH_USDT --start-month 202401 --end-month 202603 \
    --input-dir /Volumes/TU280Pro/quant/raw_data --output dataset_gate_enhanced.parquet
```

**Live system env vars (Gate.io):**
```bash
export GATE_API_KEY="your_key"
export GATE_SECRET_KEY="your_secret"
export USE_TESTNET=1  # optional: use Gate testnet
```

**Key Gate.io differences vs OKX:**
- Auth: HMAC-SHA512, no passphrase — headers: `KEY`, `SIGN`, `Timestamp`
- WS auth: per-subscription `auth` field with HMAC-SHA512 of `channel=X&event=subscribe&time=T`
- Order direction: `size` positive=long, negative=short (no `side` field)
- Contract size: `quanto_multiplier` from `GET /api/v4/futures/usdt/contracts/ETH_USDT`
- TP/SL: `POST /api/v4/futures/usdt/price_orders` (separate TP and SL orders)
- Close position: reverse order with `reduce_only: true`
- WS channels: `futures.trades`, `futures.order_book`, `futures.orders`, `futures.positions`, `futures.balances`
- Candles response: list of `{"t": ts, "o": open, "h": high, "l": low, "c": close, "v": vol}`
- Testnet: `wss://fx-ws-testnet.gateio.ws/v4/ws/usdt`, `https://fx-api-testnet.gateio.ws/api/v4`

**Modules NOT modified (pure compute):** `feature_engine.py`, `scanner.py`, `model_inference.py`, `event_aligner.py`, `state.py`

## Conventions

- Dataset file: `dataset_enhanced.parquet` (gitignored, must be generated locally via `pipline_modified.py`)
- Config file: `best_config.json` (checked in, updated by `tune_all_modified.py`)
- Model files: `model_sniper_v3_*.json`, `model_xgb.json`, `model_lgb.txt`, `model_catboost.cbm`
- Plots output to `./plots/`
- Live logs: `live/data/signals.jsonl`, `live/data/trades.jsonl`
- Gate.io API credentials: use env vars (`GATE_API_KEY`, `GATE_SECRET_KEY`). Set `USE_TESTNET=1` for testnet.
