from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

MAX_FORWARD_HORIZON = 30

BASE_FEATURES = [
    "scanner_score",
    "dir_obi",
    "obi_1",
    "obi_5",
    "obi_20",
    "obi_shock",
    "obi_zscore",
    "obi_slope_5",
    "ob_spread_bps",
    "spread",
    "spread_bps_ratio",
    "ob_depth_bid_1",
    "ob_depth_ask_1",
    "ob_depth_bid_5",
    "ob_depth_ask_5",
    "ob_depth_bid_20",
    "ob_depth_ask_20",
    "depth_ratio_5",
    "depth_delta_5",
    "ob_ask_wall_size_20",
    "ob_bid_wall_size_20",
    "ob_ask_wall_conc_20",
    "ob_bid_wall_conc_20",
    "ob_ask_wall_levels_20",
    "ob_bid_wall_levels_20",
    "ob_microprice_dev_bps",
    "ob_mid_close_dist",
    "ob_quote_count",
    "trade_gini",
    "dir_aggressor",
    "dir_net_taker",
    "large_trade_vol_ratio",
    "volume_pressure_5",
    "signed_flow_accel",
    "trade_intensity",
    "dir_event_return",
    "event_effort_vs_result",
    "event_rejection_strength",
    "time_to_reject_s",
    "sec_in_bar",
    "rvol",
    "vol_change_rate",
    "natr_20",
    "dir_ema20_dist",
    "dir_ema50_dist",
    "ema_trend",
    "dir_z_vwap",
    "bb_percent_b",
    "bb_width",
    "rsi_7",
    "adx_14",
    "dir_ret_1",
    "dir_ret_5",
    "dir_ret_20",
    "realized_vol_5",
    "realized_vol_20",
    "ret_vol_ratio",
    "range_pos_20",
    "price_efficiency_20",
    "price_velocity_1",
    "price_velocity_3",
    "price_accel_1",
    "price_accel_3",
    "ema_spread",
    "ema_spread_slope",
    "rsi_slope_3",
    "upper_wick_frac",
    "lower_wick_frac",
    "time_sin",
    "time_cos",
    "event_dir",
    "breakout_strength",
    "range_compression",
    "vol_regime",
    "data_from_orderbook",
]

DEFAULT_DATA_ROOTS = [
    "/Users/xuchuanxi/lianghua/raw_data/data",
    "./raw_data/data",
]


def safe_ratio(
    numerator: pd.Series | np.ndarray,
    denominator: pd.Series | np.ndarray,
    default: float = np.nan,
) -> pd.Series:
    numerator_s = pd.Series(numerator)
    denominator_s = pd.Series(denominator).replace(0, np.nan)
    out = numerator_s / denominator_s
    return out.replace([np.inf, -np.inf], np.nan).fillna(default)


def resolve_data_root(explicit: str | None = None) -> str:
    candidates = []
    if explicit:
        candidates.append(explicit)
    env_root = Path.cwd()
    candidates.append(str(env_root / "raw_data" / "data"))
    candidates.extend(DEFAULT_DATA_ROOTS)

    seen = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        path = Path(candidate).expanduser()
        if path.exists():
            return str(path)
    return explicit or DEFAULT_DATA_ROOTS[0]


def optimize_numeric_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.select_dtypes(include=["float64"]).columns:
        out[col] = pd.to_numeric(out[col], downcast="float")
    for col in out.select_dtypes(include=["int64", "int32"]).columns:
        out[col] = pd.to_numeric(out[col], downcast="integer")
    for col in out.select_dtypes(include=["bool"]).columns:
        out[col] = out[col].astype(np.int8)
    return out


def add_directional_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "event_dir" not in df.columns:
        df["event_dir"] = -1

    d = df["event_dir"]
    if "obi" in df.columns:
        df["dir_obi"] = df["obi"] * d
    if "aggressor_ratio" in df.columns:
        df["dir_aggressor"] = (df["aggressor_ratio"] - 0.5) * d
    if "net_taker_vol_ratio" in df.columns:
        df["dir_net_taker"] = df["net_taker_vol_ratio"] * d
    if "event_return" in df.columns:
        df["dir_event_return"] = df["event_return"] * d
    if "ema20_dist" in df.columns:
        df["dir_ema20_dist"] = df["ema20_dist"] * d
    if "ema50_dist" in df.columns:
        df["dir_ema50_dist"] = df["ema50_dist"] * d
    if "z_vwap" in df.columns:
        df["dir_z_vwap"] = df["z_vwap"] * d
    for key in ["ret_1", "ret_5", "ret_20"]:
        if key in df.columns:
            df[f"dir_{key}"] = df[key] * d

    is_long = df["event_dir"] == 1
    is_short = df["event_dir"] == -1
    if "high_60m" in df.columns and "low_60m" in df.columns and "atr" in df.columns:
        atr = df["atr"].replace(0, np.nan)
        breakout = pd.Series(0.0, index=df.index, dtype="float64")
        breakout.loc[is_long] = (
            (df.loc[is_long, "high"] - df.loc[is_long, "high_60m"]) / atr.loc[is_long]
        )
        breakout.loc[is_short] = (
            (df.loc[is_short, "low_60m"] - df.loc[is_short, "low"]) / atr.loc[is_short]
        )
        df["breakout_strength"] = breakout

    return df


@dataclass
class LabelingCache:
    index: pd.Index
    close: np.ndarray
    atr: np.ndarray
    is_long: np.ndarray
    is_short: np.ndarray
    future_high: np.ndarray
    future_low: np.ndarray
    future_close: np.ndarray


def build_labeling_cache(df: pd.DataFrame, max_horizon: int = MAX_FORWARD_HORIZON) -> LabelingCache:
    high_cols = [f"future_high_{step}m" for step in range(1, max_horizon + 1)]
    low_cols = [f"future_low_{step}m" for step in range(1, max_horizon + 1)]
    close_cols = [f"future_close_{step}m" for step in range(1, max_horizon + 1)]
    missing_high = [col for col in high_cols if col not in df.columns]
    missing_low = [col for col in low_cols if col not in df.columns]
    missing_close = [col for col in close_cols if col not in df.columns]
    if missing_high or missing_low or missing_close:
        raise ValueError(
            "Missing forward path columns. Re-run pipeline_modified.py to rebuild dataset_enhanced.parquet."
        )

    return LabelingCache(
        index=df.index,
        close=df["close"].to_numpy(dtype=np.float64, copy=False),
        atr=df["atr"].to_numpy(dtype=np.float64, copy=False),
        is_long=(df["event_dir"] == 1).to_numpy(copy=False),
        is_short=(df["event_dir"] == -1).to_numpy(copy=False),
        future_high=df[high_cols].to_numpy(dtype=np.float64, copy=False),
        future_low=df[low_cols].to_numpy(dtype=np.float64, copy=False),
        future_close=df[close_cols].to_numpy(dtype=np.float64, copy=False),
    )


def build_labels(
    cache: LabelingCache,
    horizon: int,
    tp_mult: float,
    sl_mult: float,
    label_mode: str = "first_touch",
    same_bar_policy: str = "drop",
) -> dict[str, pd.Series]:
    if horizon < 1 or horizon > cache.future_high.shape[1]:
        raise ValueError(f"horizon must be between 1 and {cache.future_high.shape[1]}")

    entry = cache.close
    atr = cache.atr
    is_long = cache.is_long
    is_short = cache.is_short

    tp_level = np.full(len(entry), np.nan, dtype=np.float64)
    sl_level = np.full(len(entry), np.nan, dtype=np.float64)
    tp_level[is_long] = entry[is_long] + atr[is_long] * tp_mult
    sl_level[is_long] = entry[is_long] - atr[is_long] * sl_mult
    tp_level[is_short] = entry[is_short] - atr[is_short] * tp_mult
    sl_level[is_short] = entry[is_short] + atr[is_short] * sl_mult

    future_high = cache.future_high[:, :horizon]
    future_low = cache.future_low[:, :horizon]

    tp_hits = np.where(
        is_long[:, None],
        future_high >= tp_level[:, None],
        future_low <= tp_level[:, None],
    )
    sl_hits = np.where(
        is_long[:, None],
        future_low <= sl_level[:, None],
        future_high >= sl_level[:, None],
    )

    hit_tp = tp_hits.any(axis=1)
    hit_sl = sl_hits.any(axis=1)
    ambiguous = np.zeros(len(entry), dtype=bool)

    if label_mode == "window_tp":
        label = hit_tp.astype(np.float32)
    elif label_mode == "first_touch":
        label = np.zeros(len(entry), dtype=np.float32)
        unresolved = np.ones(len(entry), dtype=bool)
        for step in range(horizon):
            tp_here = tp_hits[:, step]
            sl_here = sl_hits[:, step]
            both_here = unresolved & tp_here & sl_here
            tp_first = unresolved & tp_here & ~sl_here
            sl_first = unresolved & sl_here & ~tp_here

            label[tp_first] = 1.0
            label[sl_first] = 0.0
            ambiguous |= both_here

            if same_bar_policy == "tp_first":
                label[both_here] = 1.0
            elif same_bar_policy in {"sl_first", "neutral"}:
                label[both_here] = 0.0
            elif same_bar_policy == "drop":
                label[both_here] = np.nan
            else:
                raise ValueError("same_bar_policy must be one of: drop, neutral, tp_first, sl_first")

            resolved = tp_first | sl_first | both_here
            unresolved &= ~resolved
    else:
        raise ValueError("label_mode must be one of: window_tp, first_touch")

    valid = np.isfinite(label)
    return {
        "label": pd.Series(label, index=cache.index, dtype="float32"),
        "hit_tp": pd.Series(hit_tp, index=cache.index, dtype=bool),
        "hit_sl": pd.Series(hit_sl, index=cache.index, dtype=bool),
        "tp_level": pd.Series(tp_level, index=cache.index, dtype="float64"),
        "sl_level": pd.Series(sl_level, index=cache.index, dtype="float64"),
        "ambiguous": pd.Series(ambiguous, index=cache.index, dtype=bool),
        "valid": pd.Series(valid, index=cache.index, dtype=bool),
    }


def build_realized_pnl(
    cache: LabelingCache,
    horizon: int,
    tp_mult: float,
    sl_mult: float,
    label_mode: str = "first_touch",
    same_bar_policy: str = "drop",
) -> pd.Series:
    if horizon < 1 or horizon > cache.future_high.shape[1]:
        raise ValueError(f"horizon must be between 1 and {cache.future_high.shape[1]}")

    entry = cache.close
    atr = np.where(np.isfinite(cache.atr) & (cache.atr > 0), cache.atr, np.nan)
    direction = np.where(cache.is_long, 1.0, -1.0)

    tp_level = np.full(len(entry), np.nan, dtype=np.float64)
    sl_level = np.full(len(entry), np.nan, dtype=np.float64)
    tp_level[cache.is_long] = entry[cache.is_long] + cache.atr[cache.is_long] * tp_mult
    sl_level[cache.is_long] = entry[cache.is_long] - cache.atr[cache.is_long] * sl_mult
    tp_level[cache.is_short] = entry[cache.is_short] - cache.atr[cache.is_short] * tp_mult
    sl_level[cache.is_short] = entry[cache.is_short] + cache.atr[cache.is_short] * sl_mult

    future_high = cache.future_high[:, :horizon]
    future_low = cache.future_low[:, :horizon]
    future_close = cache.future_close[:, :horizon]

    tp_hits = np.where(
        cache.is_long[:, None],
        future_high >= tp_level[:, None],
        future_low <= tp_level[:, None],
    )
    sl_hits = np.where(
        cache.is_long[:, None],
        future_low <= sl_level[:, None],
        future_high >= sl_level[:, None],
    )

    horizon_close = future_close[:, horizon - 1]
    horizon_pnl_r = direction * (horizon_close - entry) / atr
    pnl_r = horizon_pnl_r.copy()

    if label_mode == "window_tp":
        hit_tp = tp_hits.any(axis=1)
        pnl_r[hit_tp] = tp_mult
    elif label_mode == "first_touch":
        pnl_r = np.full(len(entry), -sl_mult, dtype=np.float64)
        unresolved = np.ones(len(entry), dtype=bool)
        for step in range(horizon):
            tp_here = tp_hits[:, step]
            sl_here = sl_hits[:, step]
            both_here = unresolved & tp_here & sl_here
            tp_first = unresolved & tp_here & ~sl_here
            sl_first = unresolved & sl_here & ~tp_here

            pnl_r[tp_first] = tp_mult
            pnl_r[sl_first] = -sl_mult

            if same_bar_policy == "tp_first":
                pnl_r[both_here] = tp_mult
            elif same_bar_policy in {"sl_first", "neutral"}:
                pnl_r[both_here] = -sl_mult
            elif same_bar_policy == "drop":
                pnl_r[both_here] = np.nan
            else:
                raise ValueError("same_bar_policy must be one of: drop, neutral, tp_first, sl_first")

            resolved = tp_first | sl_first | both_here
            unresolved &= ~resolved
    else:
        raise ValueError("label_mode must be one of: window_tp, first_touch")

    return pd.Series(pnl_r, index=cache.index, dtype="float64")
