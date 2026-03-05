#run "conda activate quant" first
import pandas as pd
import pandas_ta as ta
import numpy as np
import json
import glob
import os
import argparse
import warnings
import time
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed

from quant_modeling import MAX_FORWARD_HORIZON, optimize_numeric_dtypes, resolve_data_root

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None

warnings.filterwarnings('ignore')

SCANNER_DESCRIPTIONS = {
    "breakout_60": "60分钟区间突破，保留原始 breakout scanner。",
    "derivative_reversal": "基于一阶/二阶导数、RSI回落与位置扩张的短期顶部/底部反转 scanner。",
    "wick_reversal": "基于长上影/下影和回落确认的 exhaustion reversal scanner。",
    "flow_reversal": "基于 orderbook / taker flow 反转和价格拐点的 scanner。",
    "trend_exhaustion_confirmed": "趋势末端衰竭 + 反向确认，强化短期冲顶回落/探底回升信号。",
}

SCANNER_DEFAULT_PARAMS = {
    "breakout_60": {
        "lookback": 60,
        "min_break_atr": 0.05,
    },
    "derivative_reversal": {
        "range_hi": 0.65,
        "range_lo": 0.35,
        "rsi_hi": 55.0,
        "rsi_lo": 45.0,
        "vel3_abs": 0.02,
        "obi_hi": 0.10,
        "obi_lo": -0.10,
    },
    "wick_reversal": {
        "wick_frac": 0.45,
        "range_hi": 0.70,
        "range_lo": 0.30,
    },
    "flow_reversal": {
        "range_hi": 0.70,
        "range_lo": 0.30,
        "flow_abs": 0.05,
        "obi_abs": 0.0,
        "spread_mult": 1.0,
    },
    "trend_exhaustion_confirmed": {
        "range_hi": 0.72,
        "range_lo": 0.28,
        "trend_strength": 0.0005,
        "trend_slow": 0.03,
        "accel_abs": 0.02,
        "confirm_ret": 0.0002,
        "confirm_rsi_slope": 1.0,
        "wick_frac": 0.20,
    },
}

SCANNER_PARAM_GRID = {
    "breakout_60": {
        "lookback": [30, 60, 90],
        "min_break_atr": [0.0, 0.05, 0.10],
    },
    "derivative_reversal": {
        "range_hi": [0.60, 0.65, 0.70],
        "range_lo": [0.30, 0.35, 0.40],
        "rsi_hi": [52.0, 55.0, 58.0],
        "rsi_lo": [42.0, 45.0, 48.0],
        "vel3_abs": [0.01, 0.02, 0.04],
        "obi_hi": [0.00, 0.10, 0.20],
        "obi_lo": [-0.20, -0.10, 0.00],
    },
    "wick_reversal": {
        "wick_frac": [0.35, 0.45, 0.55],
        "range_hi": [0.65, 0.70, 0.75],
        "range_lo": [0.25, 0.30, 0.35],
    },
    "flow_reversal": {
        "range_hi": [0.65, 0.70, 0.75],
        "range_lo": [0.25, 0.30, 0.35],
        "flow_abs": [0.03, 0.05, 0.08],
        "obi_abs": [0.0, 0.05, 0.10],
        "spread_mult": [0.8, 1.0, 1.2],
    },
    "trend_exhaustion_confirmed": {
        "range_hi": [0.72, 0.78, 0.84],
        "range_lo": [0.16, 0.22, 0.28],
        "trend_strength": [0.0003, 0.0005, 0.0010],
        "trend_slow": [0.03, 0.05, 0.08],
        "accel_abs": [0.02, 0.04, 0.06],
        "confirm_ret": [0.0002, 0.0005, 0.0010],
        "confirm_rsi_slope": [1.0, 2.0, 3.0],
        "wick_frac": [0.20, 0.28, 0.36],
    },
}

# ==========================================
# Console logger
# ==========================================
def log(msg: str):
    ts = datetime.now().strftime('%H:%M:%S')
    print(f"[{ts}] {msg}", flush=True)

# ==========================================
# 0. 辅助工具 (已修复)
# ==========================================
def _safe_ratio(numerator: pd.Series, denominator: pd.Series, default: float = np.nan) -> pd.Series:
    """
    [FIXED] 默认值改为 np.nan，避免将数据缺失(NaN)误认为是 0.0 (完美均值回归)
    """
    denom = denominator.replace(0, np.nan)
    result = numerator / denom
    return result.replace([np.inf, -np.inf], np.nan).fillna(default)

def gini_coefficient(x):
    """计算基尼系数 (0=均匀, 1=极度集中)"""
    if len(x) == 0:
        return 0.0
    x = np.sort(np.array(x))
    n = len(x)
    s = x.sum()
    if s == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return ((2 * index - n - 1) * x).sum() / (n * s)

def _minute_of_day_index(dt_index: pd.DatetimeIndex) -> pd.Series:
    return (dt_index.hour * 60 + dt_index.minute).astype(int)

# ==========================================
# 1. 核心处理逻辑 (数据清洗 + 微观结构)
# ==========================================
def process_trades_to_ohlcv(df_trades: pd.DataFrame, freq: str = '1min') -> pd.DataFrame:
    """
    Trade数据 -> OHLCV + Advanced Microstructure Features (1m聚合)
    """
    if df_trades.empty:
        return pd.DataFrame()

    # 确保索引是 DatetimeIndex
    if not isinstance(df_trades.index, pd.DatetimeIndex):
        if 'datetime' in df_trades.columns:
            df_trades = df_trades.set_index('datetime')
        else:
            df_trades.index = pd.to_datetime(df_trades.index)
        
    df_trades = df_trades.sort_index()

    # 基础 OHLCV
    ohlc = df_trades['price'].resample(freq).ohlc()
    vol = df_trades['size'].resample(freq).sum()
    ohlc['volume'] = vol

    # --- Aggressor Flow ---
    buy_trades = df_trades[df_trades['side'] == 'buy']
    sell_trades = df_trades[df_trades['side'] == 'sell']

    taker_buy = buy_trades['size'].resample(freq).sum().fillna(0)
    taker_sell = sell_trades['size'].resample(freq).sum().fillna(0)

    ohlc['taker_buy_vol'] = taker_buy
    ohlc['taker_sell_vol'] = taker_sell
    # [FIXED] 使用修复后的 _safe_ratio
    ohlc['aggressor_ratio'] = _safe_ratio(taker_buy, ohlc['volume']) 
    ohlc['net_taker_vol_ratio'] = _safe_ratio(taker_buy - taker_sell, ohlc['volume'])

    # --- Trade Asymmetry / Gini ---
    ohlc['trade_gini'] = df_trades['size'].resample(freq).apply(gini_coefficient)

    # --- Large Trade Dominance ---
    day_avg_size = df_trades.groupby(df_trades.index.date)['size'].transform(
        lambda s: s.expanding().mean().shift(1)
    )
    day_avg_size = day_avg_size.fillna(day_avg_size.median())
    threshold = 5 * day_avg_size.replace(0, 1e9)

    large_trades = df_trades[df_trades['size'] > threshold]
    large_vol = large_trades['size'].resample(freq).sum().fillna(0)

    ohlc['large_trade_count'] = large_trades['size'].resample(freq).count().fillna(0)
    ohlc['large_trade_vol_ratio'] = _safe_ratio(large_vol, ohlc['volume'])
    ohlc['trade_gini'] = ohlc['trade_gini'].fillna(0.0)
    ohlc['large_trade_vol_ratio'] = ohlc['large_trade_vol_ratio'].fillna(0.0)

    return ohlc


def process_orderbook(
    file_path: str,
    freq: str = "1min",
    top_n: int = 20,
    sample_ms: int = 1000,
    return_meta: bool = False,
) -> object:
    """
    解析 Orderbook JSONL 并生成特征。
    """
    import heapq
    records = []
    meta = {
        "file": file_path,
        "top_n": top_n,
        "sample_ms": sample_ms,
        "lines_total": 0,
        "lines_json_ok": 0,
        "lines_used": 0,
        "samples": 0,
        "error": None
    }

    asks = {}
    bids = {}
    last_sample_ts = None

    def _apply(side: dict, updates):
        for lvl in updates:
            if len(lvl) < 2: continue
            try:
                p = float(lvl[0])
                s = float(lvl[1])
            except Exception: continue
            if s <= 0.0: side.pop(p, None)
            else: side[p] = s

    def _compute_features(ts_ms: int):
        if not asks or not bids: return None

        # 使用 try-except 防止空字典导致的 min/max 错误
        try:
            best_ask = min(asks.keys())
            best_bid = max(bids.keys())
        except ValueError:
            return None

        # 简单的交叉验证，防止坏数据
        if best_ask <= best_bid:
             # 出现交叉盘，可能是数据乱序或快照错误，跳过该帧
            return None

        top_asks_p = heapq.nsmallest(top_n, asks.keys())
        top_bids_p = heapq.nlargest(top_n, bids.keys())

        def _sum_sizes(side_dict, prices, n):
            s = 0.0
            for p in prices[:n]: s += side_dict.get(p, 0.0)
            return s

        bid_qty_1 = _sum_sizes(bids, top_bids_p, 1)
        ask_qty_1 = _sum_sizes(asks, top_asks_p, 1)
        denom_1 = bid_qty_1 + ask_qty_1
        obi_1 = (bid_qty_1 - ask_qty_1) / denom_1 if denom_1 > 0 else 0.0

        n5 = 5 if top_n >= 5 else top_n
        bid_qty_5 = _sum_sizes(bids, top_bids_p, n5)
        ask_qty_5 = _sum_sizes(asks, top_asks_p, n5)
        denom_5 = bid_qty_5 + ask_qty_5
        obi_5 = (bid_qty_5 - ask_qty_5) / denom_5 if denom_5 > 0 else 0.0

        bid_qty_n = _sum_sizes(bids, top_bids_p, top_n)
        ask_qty_n = _sum_sizes(asks, top_asks_p, top_n)
        denom_n = bid_qty_n + ask_qty_n
        obi_n = (bid_qty_n - ask_qty_n) / denom_n if denom_n > 0 else 0.0

        # Wall features
        ask_sizes = [asks.get(p, 0.0) for p in top_asks_p]
        bid_sizes = [bids.get(p, 0.0) for p in top_bids_p]
        
        ask_sum = sum(ask_sizes)
        bid_sum = sum(bid_sizes)

        if ask_sizes:
            ask_wall_idx = int(np.argmax(ask_sizes))
            ask_wall_size = float(ask_sizes[ask_wall_idx])
            ask_wall_conc = ask_wall_size / ask_sum if ask_sum > 0 else 0.0
            ask_wall_levels = float(ask_wall_idx)
        else:
            ask_wall_size, ask_wall_conc, ask_wall_levels = 0.0, 0.0, 0.0

        if bid_sizes:
            bid_wall_idx = int(np.argmax(bid_sizes))
            bid_wall_size = float(bid_sizes[bid_wall_idx])
            bid_wall_conc = bid_wall_size / bid_sum if bid_sum > 0 else 0.0
            bid_wall_levels = float(bid_wall_idx)
        else:
            bid_wall_size, bid_wall_conc, bid_wall_levels = 0.0, 0.0, 0.0

        spread = best_ask - best_bid
        mid = (best_ask + best_bid) / 2.0
        spread_bps = (spread / mid) * 1e4 if mid > 0 else 0.0
        microprice = (
            (best_ask * bid_qty_1 + best_bid * ask_qty_1) / denom_1
            if denom_1 > 0 else mid
        )

        return {
            "ts": ts_ms,
            "obi": float(obi_n),
            "obi_1": float(obi_1),
            "obi_5": float(obi_5),
            f"obi_{top_n}": float(obi_n),
            
            "ob_spread": float(spread),
            "spread": float(spread), # duplicate for safety
            "ob_spread_bps": float(spread_bps),
            "ob_mid_price": float(mid),
            "ob_microprice": float(microprice),
            
            "ob_depth_bid_1": float(bid_qty_1),
            "ob_depth_ask_1": float(ask_qty_1),
            "ob_depth_bid_5": float(bid_qty_5),
            "ob_depth_ask_5": float(ask_qty_5),
            f"ob_depth_bid_{top_n}": float(bid_qty_n),
            f"ob_depth_ask_{top_n}": float(ask_qty_n),
            
            f"ob_ask_wall_size_{top_n}": float(ask_wall_size),
            f"ob_ask_wall_conc_{top_n}": float(ask_wall_conc),
            f"ob_ask_wall_levels_{top_n}": float(ask_wall_levels),
            f"ob_bid_wall_size_{top_n}": float(bid_wall_size),
            f"ob_bid_wall_conc_{top_n}": float(bid_wall_conc),
            f"ob_bid_wall_levels_{top_n}": float(bid_wall_levels),
        }

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                meta["lines_total"] += 1
                if not line.strip(): continue
                try:
                    data = json.loads(line)
                    meta["lines_json_ok"] += 1
                except Exception: continue

                ts_raw = data.get("ts")
                if ts_raw is None: continue
                try: ts_ms = int(ts_raw)
                except Exception: continue

                action = data.get("action", "")
                meta["lines_used"] += 1
                a = data.get("asks") or []
                b = data.get("bids") or []

                if action == "snapshot":
                    asks.clear(); bids.clear()
                    _apply(asks, a); _apply(bids, b)
                else:
                    if a: _apply(asks, a)
                    if b: _apply(bids, b)

                if last_sample_ts is None or (ts_ms - last_sample_ts) >= sample_ms:
                    feat = _compute_features(ts_ms)
                    if feat is not None:
                        records.append(feat)
                        meta["samples"] += 1
                        last_sample_ts = ts_ms

        if not records:
            if return_meta: return pd.DataFrame(), meta
            return pd.DataFrame()

        df = pd.DataFrame.from_records(records)
        df["datetime"] = pd.to_datetime(df["ts"], unit="ms")
        df = df.set_index("datetime").sort_index()

        # Resample to align with OHLC
        mean_cols = [col for col in df.columns if col not in {"ts", "ob_mid_price"}]
        out = df[mean_cols].resample(freq).mean(numeric_only=True)
        mid_ohlc = df["ob_mid_price"].resample(freq).ohlc()
        mid_ohlc = mid_ohlc.rename(
            columns={
                "open": "ob_mid_open",
                "high": "ob_mid_high",
                "low": "ob_mid_low",
                "close": "ob_mid_close",
            }
        )
        out = out.join(mid_ohlc, how="left")
        out["ob_quote_count"] = df["ob_mid_price"].resample(freq).size().astype("float64")
        # 订单簿可以 ffill，但不能无限 ffill，设置 limit 防止过度填充
        out = out.ffill(limit=5) 

        if return_meta:
            meta["resampled_rows"] = int(len(out))
            meta["columns"] = list(out.columns)
            for k in ["obi", f"obi_{top_n}"]:
                if k in out.columns: meta[f"non_null_{k}"] = float(out[k].notna().mean())
            return out, meta
        return out

    except Exception as e:
        if return_meta:
            meta["error"] = str(e)
            return pd.DataFrame(), meta
        return pd.DataFrame()


# ==========================================
# 2. 特征工程 (Scanner & Dynamics)
# ==========================================
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    o = df['open']; c = df['close']; h = df['high']; l = df['low']; v = df['volume']

    df['high_60m'] = h.rolling(60, min_periods=30).max().shift(1)
    df['low_60m'] = l.rolling(60, min_periods=30).min().shift(1)
    df['high_240m'] = h.rolling(240, min_periods=60).max().shift(1)
    df['low_240m'] = l.rolling(240, min_periods=60).min().shift(1)

    range_60 = df['high_60m'] - df['low_60m']
    range_240 = df['high_240m'] - df['low_240m']
    df['range_compression'] = _safe_ratio(range_60, range_240)

    if 'obi' in df.columns:
        df['obi_shock'] = df['obi'].diff()
        df['obi_slope_5'] = df['obi'].diff(5)
        df['obi_ma5'] = df['obi'].rolling(5).mean().shift(1)
        df['obi_std5'] = df['obi'].rolling(5).std().shift(1)
        df['obi_zscore'] = _safe_ratio(df['obi'] - df['obi_ma5'], df['obi_std5'])
    else:
        df['obi_shock'] = np.nan
        df['obi_slope_5'] = np.nan
        df['obi_zscore'] = np.nan

    if 'ob_depth_bid_5' in df.columns and 'ob_depth_ask_5' in df.columns:
        depth_total = df['ob_depth_bid_5'] + df['ob_depth_ask_5']
        df['depth_ratio_5'] = _safe_ratio(df['ob_depth_bid_5'], depth_total)
        df['depth_delta_5'] = depth_total.pct_change(5)
    if 'ob_spread_bps' in df.columns:
        df['spread_bps_ma10'] = df['ob_spread_bps'].rolling(10).mean().shift(1)
        df['spread_bps_ratio'] = _safe_ratio(df['ob_spread_bps'], df['spread_bps_ma10'])
    if 'ob_microprice' in df.columns and 'ob_mid_close' in df.columns:
        df['ob_microprice_dev_bps'] = _safe_ratio(
            df['ob_microprice'] - df['ob_mid_close'],
            df['ob_mid_close'],
            default=0.0,
        ) * 1e4
    else:
        df['ob_microprice_dev_bps'] = np.nan
    if 'ob_mid_close' in df.columns:
        df['ob_mid_close_dist'] = _safe_ratio(c - df['ob_mid_close'], df['ob_mid_close'])
    else:
        df['ob_mid_close_dist'] = np.nan
    if 'ob_quote_count' not in df.columns:
        df['ob_quote_count'] = 0.0

    df['atr'] = ta.atr(h, l, c, length=20)
    df['natr_20'] = _safe_ratio(df['atr'], c)

    rsi_7 = ta.rsi(c, length=7)
    df['rsi_7'] = rsi_7
    df['rsi_slope_3'] = rsi_7.diff(3)
    adx_df = ta.adx(h, l, c, length=14)
    if adx_df is not None:
        df['adx_14'] = adx_df.iloc[:, 0]

    ema20 = ta.ema(c, length=20)
    ema50 = ta.ema(c, length=50)
    ema200 = ta.ema(c, length=200)
    df['ema_spread'] = _safe_ratio(ema20 - ema50, c)
    df['ema20_dist'] = _safe_ratio(c - ema20, ema20)
    df['ema50_dist'] = _safe_ratio(c - ema50, ema50)
    df['ema200_dist'] = _safe_ratio(c - ema200, ema200)
    df['ema_trend'] = (ema20 > ema50).astype(int) * 2 - 1
    df['ema_spread_slope'] = df['ema_spread'].diff(3)

    pv = c * v
    if isinstance(df.index, pd.DatetimeIndex):
        grouper = df.index.date
    else:
        grouper = pd.to_datetime(df.index).date

    vwap_daily = pv.groupby(grouper).cumsum() / v.groupby(grouper).cumsum()
    vol_std = c.rolling(60).std().clip(lower=c * 1e-5)
    df['z_vwap'] = _safe_ratio(c - vwap_daily, vol_std)
    df['dist_vwap_daily'] = _safe_ratio(c - vwap_daily, vwap_daily)

    bb = ta.bbands(c, length=20, std=2.0)
    if bb is not None:
        width = bb.iloc[:, 2] - bb.iloc[:, 0]
        df['bb_width'] = width
        df['bb_percent_b'] = _safe_ratio(c - bb.iloc[:, 0], width)

    df['rvol'] = _safe_ratio(v, v.rolling(20).median())
    df['vol_change_rate'] = v.pct_change(5)
    df['trade_intensity'] = np.log1p(v.clip(lower=0.0))

    df['ret_1'] = c.pct_change(1)
    df['ret_5'] = c.pct_change(5)
    df['ret_20'] = c.pct_change(20)
    df['price_velocity_1'] = _safe_ratio(c.diff(1), df['atr'])
    df['price_velocity_3'] = _safe_ratio(c.diff(3), df['atr'])
    df['price_accel_1'] = df['price_velocity_1'].diff()
    df['price_accel_3'] = df['price_velocity_3'].diff(3)
    df['realized_vol_5'] = df['ret_1'].rolling(5).std()
    df['realized_vol_20'] = df['ret_1'].rolling(20).std()
    df['ret_vol_ratio'] = _safe_ratio(df['realized_vol_5'], df['realized_vol_20'])

    vol_20 = c.rolling(20).std()
    vol_60 = c.rolling(60).std()
    df['vol_regime'] = _safe_ratio(vol_20, vol_60)
    range_20 = h.rolling(20).max() - l.rolling(20).min()
    df['range_pos_20'] = _safe_ratio(c - l.rolling(20).min(), range_20)
    path_len_20 = c.diff().abs().rolling(20).sum()
    df['price_efficiency_20'] = _safe_ratio((c - c.shift(20)).abs(), path_len_20)
    bar_range = (h - l).replace(0, np.nan)
    df['upper_wick_frac'] = _safe_ratio(h - np.maximum(o, c), bar_range, default=0.0)
    df['lower_wick_frac'] = _safe_ratio(np.minimum(o, c) - l, bar_range, default=0.0)

    if 'net_taker_vol_ratio' in df.columns:
        df['signed_flow_accel'] = df['net_taker_vol_ratio'].diff()
        signed_flow = df['net_taker_vol_ratio'].fillna(0.0) * v.fillna(0.0)
        df['volume_pressure_5'] = _safe_ratio(signed_flow.rolling(5).sum(), v.rolling(5).sum(), default=0.0)
    else:
        df['signed_flow_accel'] = np.nan
        df['volume_pressure_5'] = np.nan

    if isinstance(df.index, pd.DatetimeIndex):
        minute_idx = _minute_of_day_index(df.index)
        df['time_sin'] = np.sin(2 * np.pi * minute_idx / 1440.0)
        df['time_cos'] = np.cos(2 * np.pi * minute_idx / 1440.0)

    return df


def fill_price_from_orderbook(ohlc: pd.DataFrame) -> pd.DataFrame:
    out = ohlc.copy()
    trade_missing = out['close'].isna()

    fill_map = {
        'open': 'ob_mid_open',
        'high': 'ob_mid_high',
        'low': 'ob_mid_low',
        'close': 'ob_mid_close',
    }
    for price_col, ob_col in fill_map.items():
        if ob_col in out.columns:
            out[price_col] = out[price_col].fillna(out[ob_col])

    out['volume'] = out['volume'].fillna(0.0)
    out['data_from_orderbook'] = trade_missing.astype(np.int8)
    return out


def add_forward_price_path(df: pd.DataFrame, max_horizon: int = MAX_FORWARD_HORIZON) -> pd.DataFrame:
    out = df.copy()
    for step in range(1, max_horizon + 1):
        out[f'future_high_{step}m'] = out['high'].shift(-step)
        out[f'future_low_{step}m'] = out['low'].shift(-step)
        out[f'future_close_{step}m'] = out['close'].shift(-step)
    return out


def parse_scanner_names(scanner_arg: str) -> list[str]:
    if not scanner_arg or scanner_arg == "all":
        return list(SCANNER_DESCRIPTIONS)

    names = [name.strip() for name in scanner_arg.split(",") if name.strip()]
    invalid = [name for name in names if name not in SCANNER_DESCRIPTIONS]
    if invalid:
        raise ValueError(f"Unknown scanner(s): {invalid}. Valid scanners: {list(SCANNER_DESCRIPTIONS)}")
    return names


def _format_scanner_variant(scanner_name: str, params: dict) -> str:
    if not params:
        return scanner_name
    ordered = ",".join(f"{k}={params[k]}" for k in sorted(params))
    return f"{scanner_name}|{ordered}"


def expand_scanner_specs(
    scanner_names: list[str],
    use_grid: bool = False,
    max_variants_per_scanner: int = 0,
) -> list[dict]:
    specs = []
    for scanner_name in scanner_names:
        defaults = dict(SCANNER_DEFAULT_PARAMS.get(scanner_name, {}))
        if not use_grid:
            specs.append(
                {
                    "scanner_name": scanner_name,
                    "params": defaults,
                    "scanner_variant": _format_scanner_variant(scanner_name, defaults),
                }
            )
            continue

        grid = SCANNER_PARAM_GRID.get(scanner_name, {})
        variants = [
            {
                "scanner_name": scanner_name,
                "params": defaults,
                "scanner_variant": _format_scanner_variant(scanner_name, defaults),
            }
        ]
        for key, candidates in grid.items():
            for value in candidates:
                if key in defaults and defaults[key] == value:
                    continue
                params = dict(defaults)
                params[key] = value
                variants.append(
                    {
                        "scanner_name": scanner_name,
                        "params": params,
                        "scanner_variant": _format_scanner_variant(scanner_name, params),
                    }
                )
        if max_variants_per_scanner > 0:
            variants = variants[:max_variants_per_scanner]
        specs.extend(variants)
    return specs


def _scanner_frame(
    feats: pd.DataFrame,
    mask: pd.Series,
    scanner_name: str,
    scanner_variant: str,
    scanner_params: dict,
    event_dir: int,
    score: pd.Series,
    event_level: pd.Series,
    trigger_source: str,
) -> pd.DataFrame:
    if mask is None or not bool(mask.fillna(False).any()):
        return pd.DataFrame()

    out = feats.loc[mask.fillna(False)].copy()
    out["scanner_name"] = scanner_name
    out["scanner_variant"] = scanner_variant
    out["scanner_params"] = json.dumps(scanner_params, sort_keys=True)
    out["scanner_triggered"] = 1
    out["event_dir"] = np.int8(event_dir)
    out["event_level"] = event_level.loc[out.index].astype("float32")
    out["trigger_source"] = trigger_source
    out["scanner_score"] = score.loc[out.index].astype("float32").fillna(0.0)
    return out


def _build_single_scanner_events(feats: pd.DataFrame, scanner_name: str, params: dict, scanner_variant: str) -> list[pd.DataFrame]:
    frames = []
    atr = feats["atr"].replace(0, np.nan)
    close = feats["close"]
    obi = feats["obi"] if "obi" in feats.columns else pd.Series(0.0, index=feats.index)
    spread_ref = feats["ob_spread_bps"].rolling(30, min_periods=10).median()

    if scanner_name == "breakout_60":
        lookback = int(params["lookback"])
        min_break_atr = float(params["min_break_atr"])
        high_level = feats["high"].rolling(lookback, min_periods=max(lookback // 2, 10)).max().shift(1)
        low_level = feats["low"].rolling(lookback, min_periods=max(lookback // 2, 10)).min().shift(1)
        score_up = _safe_ratio(feats["high"] - high_level, atr, default=0.0)
        score_down = _safe_ratio(low_level - feats["low"], atr, default=0.0)
        defs = [
            ((feats["high"] > high_level) & (score_up >= min_break_atr), 1, score_up, high_level, "level_cross"),
            ((feats["low"] < low_level) & (score_down >= min_break_atr), -1, score_down, low_level, "level_cross"),
        ]
    elif scanner_name == "derivative_reversal":
        defs = [
            (
                (feats["range_pos_20"] < params["range_lo"])
                & (feats["rsi_7"] < params["rsi_lo"])
                & (feats["price_velocity_3"] > params["vel3_abs"])
                & (feats["price_accel_1"] >= 0)
                & (feats["ema_spread_slope"] >= 0)
                & (obi > params["obi_lo"]),
                1,
                (0.5 - feats["range_pos_20"]) + feats["price_velocity_3"] + feats["price_accel_1"],
                close,
                "bar_close",
            ),
            (
                (feats["range_pos_20"] > params["range_hi"])
                & (feats["rsi_7"] > params["rsi_hi"])
                & (feats["price_velocity_3"] < -params["vel3_abs"])
                & (feats["price_accel_1"] <= 0)
                & (feats["ema_spread_slope"] <= 0)
                & (obi < params["obi_hi"]),
                -1,
                (feats["range_pos_20"] - 0.5) - feats["price_velocity_3"] - feats["price_accel_1"],
                close,
                "bar_close",
            ),
        ]
    elif scanner_name == "wick_reversal":
        defs = [
            (
                (feats["lower_wick_frac"] > params["wick_frac"])
                & (feats["range_pos_20"] < params["range_lo"])
                & (feats["ret_1"] > 0)
                & (feats["rsi_slope_3"] > 0),
                1,
                feats["lower_wick_frac"] + feats["ret_1"].fillna(0.0),
                close,
                "bar_close",
            ),
            (
                (feats["upper_wick_frac"] > params["wick_frac"])
                & (feats["range_pos_20"] > params["range_hi"])
                & (feats["ret_1"] < 0)
                & (feats["rsi_slope_3"] < 0),
                -1,
                feats["upper_wick_frac"] - feats["ret_1"].fillna(0.0),
                close,
                "bar_close",
            ),
        ]
    elif scanner_name == "flow_reversal":
        defs = [
            (
                (feats["range_pos_20"] < params["range_lo"])
                & (feats["price_velocity_1"] > 0)
                & (feats["net_taker_vol_ratio"] > params["flow_abs"])
                & (feats["signed_flow_accel"] > 0)
                & (obi > params["obi_abs"])
                & (feats["ob_spread_bps"] > spread_ref * params["spread_mult"]),
                1,
                feats["net_taker_vol_ratio"].fillna(0.0) + feats["price_velocity_1"].fillna(0.0),
                close,
                "bar_close",
            ),
            (
                (feats["range_pos_20"] > params["range_hi"])
                & (feats["price_velocity_1"] < 0)
                & (feats["net_taker_vol_ratio"] < -params["flow_abs"])
                & (feats["signed_flow_accel"] < 0)
                & (obi < -params["obi_abs"])
                & (feats["ob_spread_bps"] > spread_ref * params["spread_mult"]),
                -1,
                -feats["net_taker_vol_ratio"].fillna(0.0) - feats["price_velocity_1"].fillna(0.0),
                close,
                "bar_close",
            ),
        ]
    elif scanner_name == "trend_exhaustion_confirmed":
        long_trend = (
            (feats["ema_spread"] > params["trend_strength"])
            & (feats["price_velocity_3"] > params["trend_slow"])
            & (feats["range_pos_20"] > params["range_hi"])
        )
        short_trend = (
            (feats["ema_spread"] < -params["trend_strength"])
            & (feats["price_velocity_3"] < -params["trend_slow"])
            & (feats["range_pos_20"] < params["range_lo"])
        )
        top_exhaustion = (
            long_trend
            & (feats["price_accel_1"] < -params["accel_abs"])
            & (feats["ret_1"] < -params["confirm_ret"])
            & (feats["rsi_slope_3"] < -params["confirm_rsi_slope"])
            & (
                (feats["upper_wick_frac"] > params["wick_frac"])
                | (feats["ema_spread_slope"] < 0)
            )
        )
        bottom_exhaustion = (
            short_trend
            & (feats["price_accel_1"] > params["accel_abs"])
            & (feats["ret_1"] > params["confirm_ret"])
            & (feats["rsi_slope_3"] > params["confirm_rsi_slope"])
            & (
                (feats["lower_wick_frac"] > params["wick_frac"])
                | (feats["ema_spread_slope"] > 0)
            )
        )
        defs = [
            (
                bottom_exhaustion,
                1,
                -feats["ema_spread"].fillna(0.0) + feats["price_accel_1"].fillna(0.0) + feats["ret_1"].fillna(0.0) * 100.0,
                close,
                "bar_close",
            ),
            (
                top_exhaustion,
                -1,
                feats["ema_spread"].fillna(0.0) - feats["price_accel_1"].fillna(0.0) - feats["ret_1"].fillna(0.0) * 100.0,
                close,
                "bar_close",
            ),
        ]
    else:
        raise ValueError(f"Unknown scanner_name: {scanner_name}")

    for mask, event_dir, score, event_level, trigger_source in defs:
        frames.append(
            _scanner_frame(
                feats=feats,
                mask=mask,
                scanner_name=scanner_name,
                scanner_variant=scanner_variant,
                scanner_params=params,
                event_dir=event_dir,
                score=score,
                event_level=event_level,
                trigger_source=trigger_source,
            )
        )
    return frames


def build_scanner_events(feats: pd.DataFrame, scanner_specs: list[dict]) -> pd.DataFrame:
    frames = []
    for spec in scanner_specs:
        frames.extend(
            _build_single_scanner_events(
                feats=feats,
                scanner_name=spec["scanner_name"],
                params=spec["params"],
                scanner_variant=spec["scanner_variant"],
            )
        )

    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, axis=0).sort_index()
    if out.empty:
        return out

    out = (
        out.reset_index()
        .drop_duplicates(subset=["datetime", "scanner_variant", "event_dir"], keep="last")
        .set_index("datetime")
    )
    return out.sort_index()

# ==========================================
# 2.1 事件对齐特征 (逻辑增强版)
# ==========================================
def add_event_aligned_features(
    df_trades: pd.DataFrame,
    events: pd.DataFrame,
    post_window_s: int = 15,
) -> pd.DataFrame:
    """
    [FIXED] 修复了 0秒拒绝 bug 和 effort 值爆炸 bug
    """
    if events.empty: return events
    if df_trades.empty:
        out = pd.DataFrame(index=events.index)
        for col in ['sec_in_bar', 'sec_frac', 'event_return', 'event_effort_vs_result', 'event_rejection_strength']:
            out[col] = 0.0
        out['time_to_reject_s'] = float(post_window_s)
        return events.join(out, how='left')

    # 1. 统一时间戳为 int64 nanoseconds
    if 'datetime' in df_trades.columns:
        t = df_trades['datetime'].values.astype('datetime64[ns]')
    else:
        t = df_trades.index.values.astype('datetime64[ns]')

    # 排序检查 (SearchSorted 要求有序)
    if not df_trades.index.is_monotonic_increasing and 'datetime' not in df_trades.columns:
        order = np.argsort(t)
        t = t[order]
        price = df_trades['price'].to_numpy(dtype='float64')[order]
        size = df_trades['size'].to_numpy(dtype='float64')[order]
    else:
        price = df_trades['price'].to_numpy(dtype='float64')
        size = df_trades['size'].to_numpy(dtype='float64')

    t_ns = t.view('int64')

    ev_index = events.index
    ev_ns = ev_index.values.astype('datetime64[ns]').view('int64')

    level_arr = events['event_level'].to_numpy(dtype='float64') if 'event_level' in events.columns else events['close'].to_numpy(dtype='float64')
    dir_arr = events['event_dir'].to_numpy(dtype='int8') if 'event_dir' in events.columns else np.ones(len(events), dtype='int8')
    trigger_arr = events['trigger_source'].astype(str).to_numpy() if 'trigger_source' in events.columns else np.array(['bar_close'] * len(events), dtype=object)
    atr_arr = events['atr'].to_numpy(dtype='float64') if 'atr' in events.columns else np.zeros(len(events), dtype='float64')

    out = np.zeros((len(events), 6), dtype='float64')
    # 默认拒绝时间为窗口最大值
    out[:, 5] = float(post_window_s)

# === [诊断准备] 新增一个数组记录原因 ===
    # 0=Normal, 1=Fallback(Mismatch), 2=Collision(Same Time)
    diag_arr = np.zeros(len(events), dtype=int) 

    one_min_ns = int(60 * 1e9)
    post_ns = int(max(post_window_s, 0) * 1e9)

    for i in range(len(events)):
        level = level_arr[i]
        direction = 'up' if dir_arr[i] > 0 else 'down'
        trigger_source = trigger_arr[i]
        if not np.isfinite(level): continue

        bar_start_ns = ev_ns[i]
        bar_end_ns = bar_start_ns + one_min_ns

        j0 = np.searchsorted(t_ns, bar_start_ns, side='left')
        j1 = np.searchsorted(t_ns, bar_end_ns, side='right')
        if j0 >= j1: continue

        if trigger_source == 'level_cross':
            p_bar = price[j0:j1]
            if direction == 'up':
                mask = p_bar > level
            else:
                mask = p_bar < level

            if not mask.any():
                out[i, :] = np.nan
                continue

            rel = int(np.argmax(mask))
            t0_pos = j0 + rel
            t0_ns = t_ns[t0_pos]
            break_price = price[t0_pos]
        else:
            t0_pos = j1 - 1
            t0_ns = t_ns[t0_pos]
            break_price = price[t0_pos]

        sec_in_bar = (t0_ns - bar_start_ns) / 1e9
        out[i, 0] = sec_in_bar
        out[i, 1] = sec_in_bar / 60.0

        if post_ns <= 0: continue

        # 定义观测窗口 (从突破时刻 t0 开始，向后推 post_window_s)
        end_ns = t0_ns + post_ns
        k0 = np.searchsorted(t_ns, t0_ns, side='left')
        k1 = np.searchsorted(t_ns, end_ns, side='right')
        
               
        if k0 >= k1 - 1: 
            continue       
        # 跳过 k0 (突破那一刻的交易)，从 k0 开始看后续反应
        p_w = price[k0 + 1: k1]
        s_w = size[k0 + 1: k1]
        
        if len(p_w) == 0: continue

        price_end = float(p_w[-1])
        vol_w = float(s_w.sum())
        max_p = float(p_w.max())
        min_p = float(p_w.min())

        # 特征 3: 窗口内回报率
        out[i, 2] = (price_end - break_price) / (break_price + 1e-12)

        # 特征 4: Effort vs Result [FIXED]
        # 使用 log1p(Volume) / (AbsPriceMove + Epsilon)
        # 增加 1e-9 防止除零，取对数压缩量级
        price_move = abs(price_end - break_price)
        out[i, 3] = np.log1p(vol_w) / (price_move + 1e-9)

        atr = float(atr_arr[i]) if np.isfinite(atr_arr[i]) and atr_arr[i] > 0 else 1.0

        # 特征 5: Rejection Strength (反向移动幅度 / ATR)
        # 并在此时计算 time_to_reject
        if direction == 'up':
            rejection_mag = max_p - price_end
            back_mask = p_w < level # 价格跌回 level 下方
        else:
            rejection_mag = price_end - min_p
            back_mask = p_w > level # 价格涨回 level 上方

        out[i, 4] = rejection_mag / atr

        # 特征 6: Time to Reject [FIXED]
        if back_mask.any():
            back_rel = int(np.argmax(back_mask))
            # p_w 是从 k0+1 开始的，所以对应的真实索引偏移要加上
            # t_ns 的索引是 k0 + 1 + back_rel
            back_t_ns = t_ns[k0 + 1 + back_rel]
            diff_ns = back_t_ns - t0_ns
            if diff_ns <= 0:
                out[i, 5] = 0.001  # 1ms，或者你可以用 1e-6 (1微秒)
            else:
                out[i, 5] = diff_ns / 1e9
            time_diff = out[i, 5]
            # === [检测点 B] 可能性一：时间戳完全重叠 ===
            # 如果前面已经是 1 (Mismatch)，就不覆盖了，因为 Mismatch 必然导致 0
            if time_diff == 0.0 and diag_arr[i] != 1:
                diag_arr[i] = 2  # 标记为：Timestamp Collision
        else:
            # 保持默认值 (post_window_s)
            pass

    out_df = pd.DataFrame(
        out, index=events.index,
        columns=['sec_in_bar', 'sec_frac', 'event_return', 'event_effort_vs_result', 'event_rejection_strength', 'time_to_reject_s']
    )
    out_df['diagnosis_code'] = diag_arr
    return events.join(out_df, how='left')

# ==========================================
# 3. 预计算未来结果
# ==========================================
def add_multi_horizon_outcomes(df: pd.DataFrame, horizons=[5, 10, 15, 20, 30]) -> pd.DataFrame:
    df = df.sort_index()
    for h in horizons:
        # Fix lookahead bias: Shift -1 then roll exactly h periods window forward
        indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=h)
        df[f'next_high_{h}m'] = df['high'].shift(-1).rolling(window=indexer).max()
        df[f'next_low_{h}m'] = df['low'].shift(-1).rolling(window=indexer).min()
    return df


def find_orderbook_file(ob_dir: str, date_str: str) -> str:
    cand = os.path.join(ob_dir, f"ETH-USDT-orderbook-{date_str}.jsonl")
    if os.path.exists(cand): return cand

    clean_date = date_str.replace("-", "") 
    patterns = [
        os.path.join(ob_dir, f"*{date_str}*.jsonl"),
        os.path.join(ob_dir, f"*{date_str}*.data"),
        os.path.join(ob_dir, f"*{clean_date}*.jsonl"),
        os.path.join(ob_dir, f"*{clean_date}*.data"),
    ]
    hits = []
    for p in patterns:
        hits.extend(glob.glob(p))
    hits = sorted(set(hits))

    if hits:
        hits.sort(key=lambda x: os.path.getsize(x), reverse=True)
        return hits[0]
    return ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scanner", default="all", help="Scanner name or comma-separated scanner names. Use 'all' for all scanners.")
    parser.add_argument("--list-scanners", action="store_true", help="List available scanners and exit.")
    parser.add_argument("--scanner-grid", action="store_true", help="Expand selected scanners into parameter grid variants.")
    parser.add_argument("--max-variants-per-scanner", type=int, default=0, help="Cap variants per scanner when using --scanner-grid.")
    parser.add_argument("--data-dir", default=os.environ.get("QUANT_DATA_DIR"))
    parser.add_argument("--event-post-window-s", type=int, default=15)
    parser.add_argument("--max-files", type=int, default=int(os.environ.get("PIPELINE_MAX_FILES", "0")))
    parser.add_argument("--serial", action="store_true", help="Run serially for debugging.")
    return parser.parse_args()


# ==========================================
# Main & Process Day
# ==========================================

def _process_one_day(t_file: str, ob_dir: str, event_post_window_s: int, scanner_specs: list[dict]):
    """
    单日处理：读取 Trades + Orderbook
    """
    date_str = t_file.split("trades-")[-1].replace(".csv", "")
    
    meta_day = {
        "date": date_str,
        "trades_file": t_file,
        "events": 0,
        "orderbook_file_curr": "MISSING",
        "orderbook_file_prev": "MISSING",
        "non_null_obi_20": 0.0,
        "error": None
    }

    try:
        # 1. Process Trades
        df_trades_raw = pd.read_csv(
            t_file,
            usecols=lambda c: c in {"instrument_name", "trade_id", "side", "price", "size", "created_time", "datetime"},
            dtype={"side": "category", "price": "float64", "size": "float64"},
        )
        
        if 'created_time' in df_trades_raw.columns:
            df_trades_raw['datetime'] = pd.to_datetime(df_trades_raw['created_time'], unit='ms')
        else:
            df_trades_raw['datetime'] = pd.to_datetime(df_trades_raw['datetime'])

        df_trades_raw = df_trades_raw.drop_duplicates(subset=['trade_id'])
        df_trades_raw = df_trades_raw.sort_values('datetime')
        
        # 副本用于 OHLC
        df_trades_indexed = df_trades_raw.set_index('datetime')
        ohlc = process_trades_to_ohlcv(df_trades_indexed, freq='1min')
        
        # 2. Process Orderbook
        try:
            dt_curr = datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            dt_curr = datetime.now() 
        
        dt_prev = dt_curr - timedelta(days=1)
        prev_date_str = dt_prev.strftime("%Y-%m-%d")

        f_curr = find_orderbook_file(ob_dir, date_str)
        f_prev = find_orderbook_file(ob_dir, prev_date_str)

        meta_day["orderbook_file_curr"] = f_curr if f_curr else ""
        meta_day["orderbook_file_prev"] = f_prev if f_prev else ""

        ob_dfs = []
        
        if f_prev:
            ob_p, _ = process_orderbook(f_prev, freq='1min', top_n=20, return_meta=True)
            if not ob_p.empty: ob_dfs.append(ob_p)
            
        if f_curr:
            ob_c, meta_ob = process_orderbook(f_curr, freq='1min', top_n=20, return_meta=True)
            if not ob_c.empty: ob_dfs.append(ob_c)
            meta_day["ob_meta_curr"] = meta_ob

        ob_combined = pd.DataFrame()
        if ob_dfs:
            ob_combined = pd.concat(ob_dfs).sort_index()
            ob_combined = ob_combined[~ob_combined.index.duplicated(keep='last')]

        # 3. Merge OB to OHLC [FIXED]
        if not ob_combined.empty:
            if ohlc.index.tz is not None: ohlc.index = ohlc.index.tz_localize(None)
            if ob_combined.index.tz is not None: ob_combined.index = ob_combined.index.tz_localize(None)

            # Left Join
            ohlc = ohlc.join(ob_combined, how='left')
            # Forward fill limit 3 to cover temporary websocket snapshot failures
            ohlc = ohlc.ffill(limit=3)
        else:
            # 如果完全没有 OB 文件，这一列必须是 NaN
            ohlc['obi'] = np.nan
            ohlc['ob_mid_close'] = np.nan

        ohlc = fill_price_from_orderbook(ohlc)

        # 4. Feature Engineering
        feats = add_features(ohlc)
        feats = add_forward_price_path(feats, max_horizon=MAX_FORWARD_HORIZON)
        feats = add_multi_horizon_outcomes(feats, horizons=[2, 3, 4, 5, 10, 15, 20, 30])
        
        # [FIXED] 丢弃预热期(EMA200导致)和数据缺失行
        # 我们只保留那些所有特征都齐全的行，确保训练数据质量
        # 重点检查 'ema200_dist' 和 'obi' 是否为空
        valid_mask = feats['ema200_dist'].notna()
        if 'obi' in feats.columns:
            valid_mask = valid_mask & feats['obi'].notna()
        valid_mask = (
            valid_mask
            & feats[f'future_high_{MAX_FORWARD_HORIZON}m'].notna()
            & feats[f'future_low_{MAX_FORWARD_HORIZON}m'].notna()
            & feats[f'future_close_{MAX_FORWARD_HORIZON}m'].notna()
        )
            
        scanner_input = feats[valid_mask].copy()
        events = build_scanner_events(scanner_input, scanner_specs)

        if events.empty:
            meta_day["events"] = 0
            return pd.DataFrame(), meta_day

        # 5. Event Alignment
        out = add_event_aligned_features(df_trades_raw, events, post_window_s=event_post_window_s).sort_index()

        meta_day["events"] = int(len(out))
        if 'obi_20' in out.columns:
            meta_day["non_null_obi_20"] = float(out['obi_20'].notna().mean())

        return optimize_numeric_dtypes(out), meta_day

    except Exception as e:
        print(f"Error processing {date_str}: {e}")
        meta_day['error'] = str(e)
        return pd.DataFrame(), meta_day


def main():
    args = parse_args()

    if args.list_scanners:
        print("Available scanners:")
        for name, desc in SCANNER_DESCRIPTIONS.items():
            print(f"  {name}: {desc}")
            print(f"    defaults: {SCANNER_DEFAULT_PARAMS.get(name, {})}")
            print(f"    grid: {SCANNER_PARAM_GRID.get(name, {})}")
        return

    scanner_names = parse_scanner_names(args.scanner)
    scanner_specs = expand_scanner_specs(
        scanner_names,
        use_grid=args.scanner_grid,
        max_variants_per_scanner=args.max_variants_per_scanner,
    )
    base_dir = resolve_data_root(args.data_dir)
    trades_dir = os.path.join(base_dir, "trades")
    ob_dir = os.path.join(base_dir, "orderbook")

    trade_files = sorted(glob.glob(os.path.join(trades_dir, "ETH-USDT-trades-*.csv")))
    if args.max_files > 0:
        trade_files = trade_files[:args.max_files]
    log(f"Found {len(trade_files)} trade files in {trades_dir}")
    log(f"Selected scanners: {', '.join(scanner_names)}")
    log(f"Scanner variants: {len(scanner_specs)}")

    if not trade_files:
        log("No trade files found.")
        return

    all_events = []
    run_report = []

    EVENT_POST_WINDOW_S = args.event_post_window_s
    debug_serial = args.serial or os.environ.get("PIPELINE_DEBUG_SERIAL", "0") == "1"

    if debug_serial:
        log("Running in SERIAL mode")
        for t_file in tqdm(trade_files) if tqdm else trade_files:
            df, meta = _process_one_day(t_file, ob_dir, EVENT_POST_WINDOW_S, scanner_specs)
            if meta: run_report.append(meta)
            if not df.empty: all_events.append(df)
    else:
        max_workers = max(1, os.cpu_count() or 1)
        log(f"Running in PARALLEL mode (workers={max_workers})")
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            fut_map = {
                ex.submit(_process_one_day, f, ob_dir, EVENT_POST_WINDOW_S, scanner_specs): f
                for f in trade_files
            }
            
            iterator = as_completed(fut_map)
            if tqdm: iterator = tqdm(iterator, total=len(fut_map), desc="Processing")
            
            for fut in iterator:
                try:
                    df, meta = fut.result()
                    if meta: run_report.append(meta)
                    if not df.empty: all_events.append(df)
                except Exception as e:
                    log(f"Worker Error: {e}")

    # Save outputs
    if run_report:
        pd.DataFrame(run_report).to_csv("pipeline_run_report.csv", index=False)
        log("Saved report.")

    final_df = pd.DataFrame()
    if all_events:
        final_df = optimize_numeric_dtypes(pd.concat(all_events).sort_index())
        final_df.to_parquet("dataset_enhanced.parquet")
        log(f"Saved dataset. Rows: {len(final_df)}")
        if "scanner_name" in final_df.columns:
            scanner_counts = final_df["scanner_name"].value_counts().to_dict()
            log(f"Scanner counts: {scanner_counts}")
        
        # Validation Log
        log("--- Coverage Check (Should be 0% zeros for indicators) ---")
        check_cols = ['obi_20', 'ob_spread_bps', 'ema200_dist', 'time_to_reject_s']
        for k in check_cols:
            if k in final_df.columns:
                nulls = final_df[k].isna().mean()
                # 只有真正的 0.0 才会在这里显示，NaN 不算
                zeros = (final_df[k] == 0).mean()
                log(f"{k}: NaNs={(nulls):.1%} | Zeros={zeros:.1%}")
    else:
        log("No events generated.")

    # 在 main() 函数最后，或者单独的检查脚本中
    if not final_df.empty and 'diagnosis_code' in final_df.columns:
        print("\n=== 0值原因诊断报告 ===")
        
        # 筛选出 time_to_reject_s 为 0 的行
        zeros_df = final_df[final_df['time_to_reject_s'] == 0]
        
        print(f"总计 0 值行数: {len(zeros_df)}")
        
        # 统计原因分布
        counts = zeros_df['diagnosis_code'].value_counts()
        
        print("\n分布情况:")
        if 1 in counts:
            print(f"可能性二 (OHLC与逐笔数据不匹配/回退逻辑): {counts[1]} 行 (占比 {counts[1]/len(zeros_df):.1%})")
            print("   -> 说明: OHLC显示突破了Level，但逐笔数据里没有任何一笔交易超过Level。")
        
        if 2 in counts:
            print(f"可能性一 (同一毫秒时间戳重叠): {counts[2]} 行 (占比 {counts[2]/len(zeros_df):.1%})")
            print("   -> 说明: 确实找到了突破交易，但在同一毫秒内的后续交易(或同一笔)立刻满足了拒绝条件。")
            
        if 0 in counts:
            print(f"其他原因 (Logic Error): {counts[0]} 行")

if __name__ == "__main__":
    main()
