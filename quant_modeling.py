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
    # --- Cross-timeframe features ---
    "rsi_7_5m",
    "range_pos_20_5m",
    "ema_trend_15m",
    "tf_alignment",
    "momentum_divergence_5m",
    # --- Order flow dynamics ---
    "obi_momentum_3",
    "depth_absorption_rate",
    "depth_divergence",
    "flow_persistence_5",
    # --- Liquidity stress ---
    "spread_vol_10",
    "quote_intensity_change",
    "price_impact",
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
    entry: np.ndarray  # delayed entry price (or close if unavailable)
    atr: np.ndarray
    is_long: np.ndarray
    is_short: np.ndarray
    future_high: np.ndarray
    future_low: np.ndarray
    future_close: np.ndarray
    vol_regime: np.ndarray | None = None  # vol_20 / vol_60 ratio for dynamic TP/SL


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

    # Use entry_price_delayed if available, otherwise fall back to close
    close_arr = df["close"].to_numpy(dtype=np.float64, copy=False)
    if "entry_price_delayed" in df.columns:
        entry_raw = df["entry_price_delayed"].to_numpy(dtype=np.float64, copy=True)
        nan_mask = ~np.isfinite(entry_raw)
        entry_raw[nan_mask] = close_arr[nan_mask]
        entry_arr = entry_raw
    else:
        entry_arr = close_arr

    vol_regime_arr = None
    if "vol_regime" in df.columns:
        vol_regime_arr = df["vol_regime"].to_numpy(dtype=np.float64, copy=True)

    future_high_arr = df[high_cols].to_numpy(dtype=np.float64, copy=True)
    future_low_arr = df[low_cols].to_numpy(dtype=np.float64, copy=True)
    future_close_arr = df[close_cols].to_numpy(dtype=np.float64, copy=True)

    # Partial bar correction: override step 0 (first future bar) with
    # tick-level data from [entry_time, next_bar_end) so that the ~15s
    # pre-entry portion of the bar does not pollute TP/SL evaluation.
    if "partial_bar_high" in df.columns:
        pb_h = df["partial_bar_high"].to_numpy(dtype=np.float64)
        valid = np.isfinite(pb_h)
        future_high_arr[valid, 0] = pb_h[valid]

    if "partial_bar_low" in df.columns:
        pb_l = df["partial_bar_low"].to_numpy(dtype=np.float64)
        valid = np.isfinite(pb_l)
        future_low_arr[valid, 0] = pb_l[valid]

    if "partial_bar_close" in df.columns:
        pb_c = df["partial_bar_close"].to_numpy(dtype=np.float64)
        valid = np.isfinite(pb_c)
        future_close_arr[valid, 0] = pb_c[valid]

    return LabelingCache(
        index=df.index,
        close=close_arr,
        entry=entry_arr,
        atr=df["atr"].to_numpy(dtype=np.float64, copy=False),
        is_long=(df["event_dir"] == 1).to_numpy(copy=False),
        is_short=(df["event_dir"] == -1).to_numpy(copy=False),
        future_high=future_high_arr,
        future_low=future_low_arr,
        future_close=future_close_arr,
        vol_regime=vol_regime_arr,
    )


def _compute_dynamic_tpsl(
    vol_regime: np.ndarray | None,
    tp_mult: float,
    sl_mult: float,
    n: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Scale TP/SL multipliers per row based on vol_regime.

    Low vol (< 0.8): tighter TP (0.7x), same SL
    High vol (> 1.2): wider TP (1.3x), wider SL (1.5x)
    Normal: no change
    """
    tp_arr = np.full(n, tp_mult, dtype=np.float64)
    sl_arr = np.full(n, sl_mult, dtype=np.float64)
    if vol_regime is None:
        return tp_arr, sl_arr
    low = vol_regime < 0.8
    high = vol_regime > 1.2
    tp_arr[low] = tp_mult * 0.7
    tp_arr[high] = tp_mult * 1.3
    sl_arr[high] = sl_mult * 1.5
    return tp_arr, sl_arr


def build_labels(
    cache: LabelingCache,
    horizon: int,
    tp_mult: float,
    sl_mult: float,
    label_mode: str = "first_touch",
    same_bar_policy: str = "drop",
    dynamic_tpsl: bool = False,
) -> dict[str, pd.Series]:
    if horizon < 1 or horizon > cache.future_high.shape[1]:
        raise ValueError(f"horizon must be between 1 and {cache.future_high.shape[1]}")

    entry = cache.entry
    atr = cache.atr
    is_long = cache.is_long
    is_short = cache.is_short

    if dynamic_tpsl:
        tp_mult_arr, sl_mult_arr = _compute_dynamic_tpsl(cache.vol_regime, tp_mult, sl_mult, len(entry))
    else:
        tp_mult_arr = np.full(len(entry), tp_mult, dtype=np.float64)
        sl_mult_arr = np.full(len(entry), sl_mult, dtype=np.float64)

    tp_level = np.full(len(entry), np.nan, dtype=np.float64)
    sl_level = np.full(len(entry), np.nan, dtype=np.float64)
    tp_level[is_long] = entry[is_long] + atr[is_long] * tp_mult_arr[is_long]
    sl_level[is_long] = entry[is_long] - atr[is_long] * sl_mult_arr[is_long]
    tp_level[is_short] = entry[is_short] - atr[is_short] * tp_mult_arr[is_short]
    sl_level[is_short] = entry[is_short] + atr[is_short] * sl_mult_arr[is_short]

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

        # Unresolved samples: neither TP nor SL hit within horizon.
        # Label by horizon-close PnL direction instead of defaulting to 0 (loss).
        if unresolved.any():
            future_close = cache.future_close[:, :horizon]
            horizon_close = future_close[:, horizon - 1]
            direction = np.where(is_long, 1.0, -1.0)
            horizon_pnl = direction * (horizon_close - entry)
            label[unresolved] = np.where(horizon_pnl[unresolved] > 0, 1.0, 0.0).astype(np.float32)
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
    breakeven_trigger_r: float = 0.0,
    lock_profit_trigger_r: float = 0.0,
    lock_profit_level_r: float = 0.0,
    dynamic_tpsl: bool = False,
) -> pd.Series:
    """Compute realized PnL in R-multiples.

    Breakeven stop parameters (set to 0 to disable):
        breakeven_trigger_r: move SL to entry after this favorable excursion (e.g. 1.0)
        lock_profit_trigger_r: move SL to lock_profit_level_r after this excursion (e.g. 1.5)
        lock_profit_level_r: locked SL level in R above entry (e.g. 0.5)
    """
    if horizon < 1 or horizon > cache.future_high.shape[1]:
        raise ValueError(f"horizon must be between 1 and {cache.future_high.shape[1]}")

    entry = cache.entry
    atr = np.where(np.isfinite(cache.atr) & (cache.atr > 0), cache.atr, np.nan)
    direction = np.where(cache.is_long, 1.0, -1.0)

    if dynamic_tpsl:
        tp_mult_arr, sl_mult_arr = _compute_dynamic_tpsl(cache.vol_regime, tp_mult, sl_mult, len(entry))
    else:
        tp_mult_arr = np.full(len(entry), tp_mult, dtype=np.float64)
        sl_mult_arr = np.full(len(entry), sl_mult, dtype=np.float64)

    tp_level = np.full(len(entry), np.nan, dtype=np.float64)
    sl_level = np.full(len(entry), np.nan, dtype=np.float64)
    tp_level[cache.is_long] = entry[cache.is_long] + cache.atr[cache.is_long] * tp_mult_arr[cache.is_long]
    sl_level[cache.is_long] = entry[cache.is_long] - cache.atr[cache.is_long] * sl_mult_arr[cache.is_long]
    tp_level[cache.is_short] = entry[cache.is_short] - cache.atr[cache.is_short] * tp_mult_arr[cache.is_short]
    sl_level[cache.is_short] = entry[cache.is_short] + cache.atr[cache.is_short] * sl_mult_arr[cache.is_short]

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
        pnl_r[hit_tp] = tp_mult_arr[hit_tp]
    elif label_mode == "first_touch":
        use_breakeven = breakeven_trigger_r > 0 or lock_profit_trigger_r > 0
        pnl_r = -sl_mult_arr.copy()  # default: lose SL amount (per-row)
        unresolved = np.ones(len(entry), dtype=bool)
        # Dynamic SL level (may be adjusted by breakeven logic)
        dyn_sl = sl_level.copy()
        for step in range(horizon):
            tp_here = tp_hits[:, step]
            # Recompute SL hits dynamically when breakeven is active
            if use_breakeven:
                step_high = future_high[:, step]
                step_low = future_low[:, step]
                sl_here = np.where(
                    cache.is_long,
                    step_low <= dyn_sl,
                    step_high >= dyn_sl,
                )
            else:
                sl_here = sl_hits[:, step]
            both_here = unresolved & tp_here & sl_here
            tp_first = unresolved & tp_here & ~sl_here
            sl_first = unresolved & sl_here & ~tp_here

            pnl_r[tp_first] = tp_mult_arr[tp_first]
            # SL PnL: when breakeven is active, compute actual PnL at dynamic SL level
            if use_breakeven:
                sl_pnl = direction[sl_first] * (dyn_sl[sl_first] - entry[sl_first]) / atr[sl_first]
                pnl_r[sl_first] = sl_pnl
            else:
                pnl_r[sl_first] = -sl_mult_arr[sl_first]

            if same_bar_policy == "tp_first":
                pnl_r[both_here] = tp_mult_arr[both_here]
            elif same_bar_policy in {"sl_first", "neutral"}:
                if use_breakeven:
                    both_pnl = direction[both_here] * (dyn_sl[both_here] - entry[both_here]) / atr[both_here]
                    pnl_r[both_here] = both_pnl
                else:
                    pnl_r[both_here] = -sl_mult_arr[both_here]
            elif same_bar_policy == "drop":
                pnl_r[both_here] = np.nan
            else:
                raise ValueError("same_bar_policy must be one of: drop, neutral, tp_first, sl_first")

            resolved = tp_first | sl_first | both_here
            unresolved &= ~resolved

            # Breakeven stop: adjust SL for unresolved rows based on favorable excursion
            if use_breakeven and unresolved.any():
                step_high = future_high[:, step]
                step_low = future_low[:, step]
                # Favorable excursion in R-multiples
                fav_r = np.where(
                    cache.is_long,
                    (step_high - entry) / atr,
                    (entry - step_low) / atr,
                )
                # Level 1: move SL to entry (breakeven)
                if breakeven_trigger_r > 0:
                    be_mask = unresolved & (fav_r >= breakeven_trigger_r)
                    dyn_sl[be_mask] = entry[be_mask]
                # Level 2: lock profit (move SL to entry + lock_profit_level_r * atr)
                if lock_profit_trigger_r > 0:
                    lp_mask = unresolved & (fav_r >= lock_profit_trigger_r)
                    lp_sl = np.where(
                        cache.is_long,
                        entry + lock_profit_level_r * atr,
                        entry - lock_profit_level_r * atr,
                    )
                    # Only move SL if new level is more favorable than current
                    better = np.where(cache.is_long, lp_sl > dyn_sl, lp_sl < dyn_sl)
                    lp_mask = lp_mask & better
                    dyn_sl[lp_mask] = lp_sl[lp_mask]

        # Unresolved samples: neither TP nor SL hit within horizon.
        # Use actual horizon-close PnL instead of defaulting to -sl_mult.
        if unresolved.any():
            pnl_r[unresolved] = horizon_pnl_r[unresolved]
    else:
        raise ValueError("label_mode must be one of: window_tp, first_touch")

    return pd.Series(pnl_r, index=cache.index, dtype="float64")
