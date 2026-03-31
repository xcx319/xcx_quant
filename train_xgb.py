# run "conda activate quant" first
from __future__ import annotations

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve
from sklearn.inspection import permutation_importance

from quant_modeling import (
    BASE_FEATURES,
    add_directional_features,
    build_labeling_cache,
    build_labels,
    build_realized_pnl,
)

try:
    with open("best_config.json", "r") as f:
        BEST = json.load(f)
    print(f"Loaded Config: {BEST}")
except Exception:
    print("Config not found, using defaults.")
    BEST = {"h": 30, "tp": 1.5, "sl": 1.5}

PURGE_GAP_BARS = 30

CONFIG = {
    "data_path": "dataset_enhanced.parquet",
    "scanner": str(BEST.get("scanner_name", "all")),
    "scanner_variant": str(BEST.get("scanner_variant", "all")),
    "horizon": int(BEST["h"]),
    "tp_mult": float(BEST["tp"]),
    "sl_mult": float(BEST["sl"]),
    "label_mode": str(BEST.get("label_mode", "first_touch")),
    "same_bar_policy": str(BEST.get("same_bar_policy", "drop")),
    "features": list(BASE_FEATURES),
    "model_out": "model_sniper_v3.json",
    "plot_dir": "./plots",
    "min_valid_trades": int(BEST.get("min_valid_trades", 80)),
    "threshold_smooth_window": int(BEST.get("threshold_smooth_window", 3)),
    "long_only": bool(BEST.get("long_only", False)),
}

os.makedirs(CONFIG["plot_dir"], exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default=CONFIG["data_path"])
    parser.add_argument("--scanner", default=CONFIG["scanner"], help="Filter scanner_name by one or more comma-separated names.")
    parser.add_argument("--scanner-variant", default=CONFIG["scanner_variant"], help="Filter exact scanner_variant by one or more comma-separated values.")
    parser.add_argument("--label-mode", choices=["window_tp", "first_touch"], default=CONFIG["label_mode"])
    parser.add_argument("--same-bar-policy", choices=["drop", "neutral", "tp_first", "sl_first"], default=CONFIG["same_bar_policy"])
    parser.add_argument("--horizon", type=int, default=CONFIG["horizon"])
    parser.add_argument("--tp-mult", type=float, default=CONFIG["tp_mult"])
    parser.add_argument("--sl-mult", type=float, default=CONFIG["sl_mult"])
    parser.add_argument("--min-valid-trades", type=int, default=CONFIG["min_valid_trades"])
    parser.add_argument("--threshold-smooth-window", type=int, default=CONFIG["threshold_smooth_window"])
    parser.add_argument("--live", action="store_true", help="Live mode: use all data for train+valid (no test holdout), select threshold from validation, update best_config.json.")
    parser.add_argument("--dynamic-tpsl", action="store_true", help="Enable dynamic TP/SL scaling based on vol_regime.")
    parser.add_argument("--breakeven", action="store_true", help="Enable breakeven stop (move SL to entry after 1.0R, lock 0.5R after 1.5R).")
    parser.add_argument("--long-only", action="store_true", default=CONFIG["long_only"], help="Only train on long (event_dir==1) signals.")
    return parser.parse_args()


def filter_by_scanner(df: pd.DataFrame, scanner_arg: str, scanner_variant_arg: str) -> pd.DataFrame:
    out = df
    if scanner_arg != "all" and "scanner_name" in out.columns:
        names = [name.strip() for name in scanner_arg.split(",") if name.strip()]
        out = out[out["scanner_name"].isin(names)]
    if scanner_variant_arg != "all" and "scanner_variant" in out.columns:
        variant_series = out["scanner_variant"].astype(str)
        exact_variant = scanner_variant_arg.strip()
        if (variant_series == exact_variant).any():
            out = out[variant_series == exact_variant]
        else:
            # Fallback for manually supplied multiple variants separated by ';'
            variants = [name.strip() for name in scanner_variant_arg.split(";") if name.strip()]
            out = out[variant_series.isin(variants)]
    return out.copy()


def load_and_label(path: str, scanner_arg: str, scanner_variant_arg: str, label_mode: str, same_bar_policy: str, horizon: int, tp_mult: float, sl_mult: float, dynamic_tpsl: bool = False, breakeven: bool = False, long_only: bool = False) -> pd.DataFrame:
    print(f"Loading data... label_mode={label_mode}, horizon={horizon}m")
    print(f"Scanner filter... scanner={scanner_arg}, variant={scanner_variant_arg}")
    try:
        df = pd.read_parquet(path)
    except FileNotFoundError:
        print(f"Error: File {path} not found.")
        return pd.DataFrame()

    if "event_dir" not in df.columns:
        df["event_dir"] = -1

    df = filter_by_scanner(df, scanner_arg, scanner_variant_arg)
    if df.empty:
        print(f"No rows left after scanner filter: scanner={scanner_arg}, variant={scanner_variant_arg}")
        return df

    if long_only:
        df = df[df["event_dir"] == 1].copy()
        print(f"Long-only filter: {len(df)} rows remaining")
        if df.empty:
            print("No long signals found.")
            return df

    df = add_directional_features(df)
    cache = build_labeling_cache(df)
    label_info = build_labels(
        cache=cache,
        horizon=horizon,
        tp_mult=tp_mult,
        sl_mult=sl_mult,
        label_mode=label_mode,
        same_bar_policy=same_bar_policy,
        dynamic_tpsl=dynamic_tpsl,
    )

    df["label"] = label_info["label"]
    df["label_valid"] = label_info["valid"]
    df["hit_tp"] = label_info["hit_tp"]
    df["hit_sl"] = label_info["hit_sl"]
    df["ambiguous_same_bar"] = label_info["ambiguous"]

    be_trigger = 1.0 if breakeven else 0.0
    lp_trigger = 1.5 if breakeven else 0.0
    lp_level = 0.5 if breakeven else 0.0
    df["realized_pnl_r"] = build_realized_pnl(
        cache=cache,
        horizon=horizon,
        tp_mult=tp_mult,
        sl_mult=sl_mult,
        label_mode=label_mode,
        same_bar_policy=same_bar_policy,
        breakeven_trigger_r=be_trigger,
        lock_profit_trigger_r=lp_trigger,
        lock_profit_level_r=lp_level,
        dynamic_tpsl=dynamic_tpsl,
    )

    dropped = int((~df["label_valid"]).sum())
    if dropped:
        print(f"Dropping ambiguous rows: {dropped}")
    df = df[df["label_valid"]].copy()
    df["label"] = df["label"].astype(np.int8)

    df = df[df["realized_pnl_r"].notna()].copy()

    tp_only = df["hit_tp"] & ~df["hit_sl"]
    sl_only = ~df["hit_tp"] & df["hit_sl"]
    both = df["hit_tp"] & df["hit_sl"]
    neither = ~df["hit_tp"] & ~df["hit_sl"]

    pos_ratio = df["label"].mean()
    weights = np.ones(len(df), dtype=np.float32)
    if 0 < pos_ratio < 1:
        pos_weight = (1 - pos_ratio) / pos_ratio
        weights = np.where(df["label"] == 1, pos_weight, 1.0).astype(np.float32)
    df["sample_weight"] = weights

    print(f"Data Loaded. Rows: {len(df)}. Win Rate (Base): {pos_ratio:.2%}")
    print(
        "  TP-only: "
        f"{tp_only.sum()}, SL-only: {sl_only.sum()}, Both: {both.sum()}, Neither: {neither.sum()}, "
        f"Ambiguous(same bar): {int(df['ambiguous_same_bar'].sum())}"
    )
    if label_mode == "window_tp":
        print(
            "  Window-TP realized R stats: "
            f"mean={df['realized_pnl_r'].mean():.3f}, min={df['realized_pnl_r'].min():.3f}, "
            f"p05={df['realized_pnl_r'].quantile(0.05):.3f}"
        )
    return df


def train_final_model(df: pd.DataFrame):
    available_feats = [feature for feature in CONFIG["features"] if feature in df.columns]
    missing = sorted(set(CONFIG["features"]) - set(available_feats))
    if missing:
        print(f"Warning: Missing features (will skip): {missing}")

    X = df[available_feats].astype(np.float32)
    y = df["label"]
    w = df["sample_weight"]

    train_end = int(len(df) * 0.70)
    valid_start = train_end + PURGE_GAP_BARS
    valid_end = int(len(df) * 0.85)
    test_start = valid_end + PURGE_GAP_BARS

    X_train = X.iloc[:train_end]
    y_train = y.iloc[:train_end]
    w_train = w.iloc[:train_end]

    X_valid = X.iloc[valid_start:valid_end]
    y_valid = y.iloc[valid_start:valid_end]

    X_test = X.iloc[test_start:]
    y_test = y.iloc[test_start:]

    print(f"\nTrain: {len(X_train)}, Valid: {len(X_valid)}, Test: {len(X_test)} (purge gap={PURGE_GAP_BARS})")
    print(f"Train label dist: {y_train.value_counts().to_dict()}")
    print(f"Valid label dist: {y_valid.value_counts().to_dict()}")
    print(f"Test  label dist: {y_test.value_counts().to_dict()}")

    fit_kwargs = {
        "X": X_train,
        "y": y_train,
        "sample_weight": w_train,
        "verbose": 50,
    }
    early_stopping_rounds = 120 if len(X_valid) > 0 and y_valid.nunique() > 1 else None
    if early_stopping_rounds is not None:
        fit_kwargs["eval_set"] = [(X_train, y_train), (X_valid, y_valid)]

    model = xgb.XGBClassifier(
        n_estimators=1500,
        max_depth=4,
        learning_rate=0.003,
        subsample=0.5,
        colsample_bytree=0.7,
        colsample_bylevel=0.7,
        gamma=0.5,
        min_child_weight=50,
        reg_alpha=0.5,
        reg_lambda=2.0,
        max_delta_step=1,
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
        early_stopping_rounds=early_stopping_rounds,
    )
    model.fit(**fit_kwargs)

    model.get_booster().save_model(CONFIG["model_out"])
    model.get_booster().save_model("model_xgb.json")

    print("\nFeature Importance (top 20):")
    fi = pd.Series(model.feature_importances_, index=available_feats).sort_values(ascending=False)
    print(fi.head(20))

    zero_imp = fi[fi == 0].index.tolist()
    if zero_imp:
        print(f"\nZero-importance features (consider removing): {zero_imp}")

    # Permutation importance on validation set
    if len(X_valid) > 0 and y_valid.nunique() > 1:
        print("\nPermutation Importance (validation set, 5 repeats)...")
        perm_result = permutation_importance(
            model, X_valid, y_valid, n_repeats=5,
            scoring="average_precision", random_state=42, n_jobs=-1,
        )
        perm_fi = pd.Series(perm_result.importances_mean, index=available_feats).sort_values(ascending=False)
        print("Top 20 by permutation importance:")
        print(perm_fi.head(20))
        negative_perm = perm_fi[perm_fi < 0].index.tolist()
        if negative_perm:
            print(f"\nNegative permutation importance (hurting OOS, consider removing): {negative_perm}")

    split_data = {
        "X_valid": X_valid,
        "y_valid": y_valid,
        "df_valid": df.iloc[valid_start:valid_end].copy(),
        "X_test": X_test,
        "y_test": y_test,
        "df_test": df.iloc[test_start:].copy(),
    }
    return model, split_data


def train_live_model(df: pd.DataFrame):
    """Train model for live trading: 85% train + 15% valid (early stopping only), no test holdout."""
    available_feats = [feature for feature in CONFIG["features"] if feature in df.columns]
    missing = sorted(set(CONFIG["features"]) - set(available_feats))
    if missing:
        print(f"Warning: Missing features (will skip): {missing}")

    X = df[available_feats].astype(np.float32)
    y = df["label"]
    w = df["sample_weight"]

    train_end = int(len(df) * 0.85)
    valid_start = train_end + PURGE_GAP_BARS

    X_train = X.iloc[:train_end]
    y_train = y.iloc[:train_end]
    w_train = w.iloc[:train_end]

    X_valid = X.iloc[valid_start:]
    y_valid = y.iloc[valid_start:]

    print(f"\n[LIVE MODE] Train: {len(X_train)}, Valid: {len(X_valid)} (purge gap={PURGE_GAP_BARS}), Test: 0")
    print(f"Train label dist: {y_train.value_counts().to_dict()}")
    print(f"Valid label dist: {y_valid.value_counts().to_dict()}")

    fit_kwargs = {
        "X": X_train,
        "y": y_train,
        "sample_weight": w_train,
        "verbose": 50,
    }
    early_stopping_rounds = 120 if len(X_valid) > 0 and y_valid.nunique() > 1 else None
    if early_stopping_rounds is not None:
        fit_kwargs["eval_set"] = [(X_train, y_train), (X_valid, y_valid)]

    model = xgb.XGBClassifier(
        n_estimators=1500,
        max_depth=4,
        learning_rate=0.003,
        subsample=0.5,
        colsample_bytree=0.7,
        colsample_bylevel=0.7,
        gamma=0.5,
        min_child_weight=50,
        reg_alpha=0.5,
        reg_lambda=2.0,
        max_delta_step=1,
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
        early_stopping_rounds=early_stopping_rounds,
    )
    model.fit(**fit_kwargs)

    model.get_booster().save_model(CONFIG["model_out"])
    model.get_booster().save_model("model_xgb.json")
    print(f"\nModel saved to {CONFIG['model_out']} + model_xgb.json")

    print("\nFeature Importance (top 20):")
    fi = pd.Series(model.feature_importances_, index=available_feats).sort_values(ascending=False)
    print(fi.head(20))

    # Select threshold from validation set
    valid_probs = model.predict_proba(X_valid)[:, 1]
    summarize_scores("Valid", valid_probs)

    tp_r = CONFIG["tp_mult"]
    sl_r = CONFIG["sl_mult"]
    realized_pnl_valid = None
    df_valid = df.iloc[valid_start:].copy()
    if "realized_pnl_r" in df_valid.columns:
        realized_pnl_valid = df_valid["realized_pnl_r"].to_numpy(dtype=np.float64, copy=False)

    thresholds = build_threshold_grid(valid_probs)
    valid_df = evaluate_thresholds(valid_probs, y_valid, realized_pnl_valid, tp_r, sl_r, thresholds)
    print_threshold_context("Validation", valid_df)

    picked = select_threshold(
        valid_df,
        min_valid_trades=CONFIG["min_valid_trades"],
        smooth_window=CONFIG["threshold_smooth_window"],
    )
    if picked is not None:
        best_threshold = float(picked["threshold"])
        print(f"\nSelected threshold: {best_threshold:.4f}")
        print(f"  Trades: {int(picked['trades'])}, Win Rate: {picked['win_rate']:.2%}, "
              f"Net Profit: {picked['net_profit_r']:.1f} R, Avg R: {picked['avg_r']:.2f}")
    else:
        best_threshold = float(valid_df.loc[valid_df["net_profit_r"].idxmax(), "threshold"])
        print(f"\nFallback threshold (max profit): {best_threshold:.4f}")

    # Update best_config.json with new threshold
    try:
        with open("best_config.json", "r") as f:
            cfg = json.load(f)
        cfg["threshold"] = best_threshold
        with open("best_config.json", "w") as f:
            json.dump(cfg, f, indent=2)
        print(f"Updated best_config.json threshold -> {best_threshold:.4f}")
    except Exception as e:
        print(f"Warning: could not update best_config.json: {e}")

    return model


def build_threshold_grid(probs: np.ndarray) -> np.ndarray:
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs[np.isfinite(probs)]
    if probs.size == 0:
        return np.array([], dtype=np.float64)

    quantiles = np.linspace(0.10, 0.995, 80)
    thresholds = np.unique(np.round(np.quantile(probs, quantiles), 4))
    thresholds = thresholds[(thresholds > 0.0) & (thresholds < 1.0)]
    return thresholds


def summarize_scores(name: str, probs: np.ndarray) -> None:
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs[np.isfinite(probs)]
    if probs.size == 0:
        print(f"{name} score dist: empty")
        return

    q = np.quantile(probs, [0.00, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.00])
    print(
        f"{name} score dist: "
        f"min={q[0]:.4f}, p01={q[1]:.4f}, p05={q[2]:.4f}, p10={q[3]:.4f}, "
        f"p25={q[4]:.4f}, p50={q[5]:.4f}, p75={q[6]:.4f}, p90={q[7]:.4f}, "
        f"p95={q[8]:.4f}, p99={q[9]:.4f}, max={q[10]:.4f}, std={probs.std():.6f}"
    )


def print_threshold_context(name: str, df: pd.DataFrame, selected_threshold: float | None = None) -> None:
    if df.empty:
        print(f"{name}: no threshold rows")
        return

    top = df.sort_values("net_profit_r", ascending=False).head(5).copy()
    print(f"\n{name} top thresholds by net profit:")
    print(
        top.loc[:, ["threshold", "trades", "win_rate", "avg_r", "net_profit_r", "max_drawdown_r", "recall"]]
        .to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )

    if selected_threshold is None:
        return

    nearest_idx = (df["threshold"] - selected_threshold).abs().idxmin()
    center = df.index.get_loc(nearest_idx)
    lo = max(center - 2, 0)
    hi = min(center + 3, len(df))
    window = df.iloc[lo:hi].copy()
    print(f"\n{name} around selected threshold {selected_threshold:.4f}:")
    print(
        window.loc[:, ["threshold", "trades", "win_rate", "avg_r", "net_profit_r", "max_drawdown_r", "recall"]]
        .to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )


def print_bin_diagnostics(probs: np.ndarray, y_true: pd.Series, realized_pnl: np.ndarray | None, label_mode: str) -> None:
    probs = np.asarray(probs, dtype=np.float64)
    y_arr = np.asarray(y_true, dtype=np.float64)
    if probs.size == 0:
        print("\nTest score bins: empty")
        return

    quantiles = np.linspace(0.0, 1.0, 11)
    edges = np.unique(np.quantile(probs, quantiles))
    if edges.size < 3:
        print("\nTest score bins: insufficient score spread for deciles")
        return

    bins = pd.cut(probs, bins=edges, include_lowest=True, duplicates="drop")
    metric_label = "win_rate" if label_mode == "first_touch" else "positive_pnl_rate"
    frame = pd.DataFrame({"prob": probs, "label": y_arr, "bin": bins})
    if realized_pnl is not None:
        frame["realized_pnl_r"] = realized_pnl

    grouped = frame.groupby("bin", observed=True)
    rows: list[dict[str, float | int | str]] = []
    for bucket, grp in grouped:
        row: dict[str, float | int | str] = {
            "bin": str(bucket),
            "count": int(len(grp)),
            "mean_prob": float(grp["prob"].mean()),
        }
        if realized_pnl is not None:
            row[metric_label] = float((grp["realized_pnl_r"] > 0).mean())
            row["avg_r"] = float(grp["realized_pnl_r"].mean())
        else:
            row[metric_label] = float(grp["label"].mean())
            row["avg_r"] = np.nan
        rows.append(row)

    out = pd.DataFrame(rows)
    print("\nTest score bins:")
    print(out.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def evaluate_thresholds(probs: np.ndarray, y_true: pd.Series, realized_pnl: np.ndarray | None, tp_r: float, sl_r: float, thresholds: np.ndarray) -> pd.DataFrame:
    results = []
    for thresh in thresholds:
        preds = (probs > thresh).astype(int)
        if preds.sum() == 0:
            continue

        cm = confusion_matrix(y_true, preds, labels=[0, 1])
        _, fp, fn, tp = cm.ravel()
        trades = tp + fp
        if trades == 0:
            continue

        if realized_pnl is not None:
            trade_pnl = realized_pnl[preds == 1]
            net_profit_r = float(np.nansum(trade_pnl))
            avg_r_per_trade = float(np.nanmean(trade_pnl))
            win_rate = float(np.mean(trade_pnl > 0))
            worst_trade = float(np.nanmin(trade_pnl))
            cum_r = np.cumsum(trade_pnl)
        else:
            net_profit_r = (tp * tp_r) - (fp * sl_r)
            avg_r_per_trade = net_profit_r / trades
            win_rate = tp / trades
            worst_trade = -sl_r
            trade_pnl = np.where(preds == 1, np.where(y_true.to_numpy(copy=False) == 1, tp_r, -sl_r), 0.0)
            cum_r = np.cumsum(trade_pnl[trade_pnl != 0.0])
        if cum_r.size > 0:
            peak = np.maximum.accumulate(cum_r)
            max_drawdown_r = float((cum_r - peak).min())
        else:
            max_drawdown_r = 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        results.append(
            {
                "threshold": float(thresh),
                "trades": int(trades),
                "win_rate": float(win_rate),
                "recall": float(recall),
                "net_profit_r": float(net_profit_r),
                "avg_r": float(avg_r_per_trade),
                "tp": int(tp),
                "fp": int(fp),
                "worst_trade_r": float(worst_trade),
                "max_drawdown_r": float(max_drawdown_r),
            }
        )
    return pd.DataFrame(results)


def select_threshold(valid_df: pd.DataFrame, min_valid_trades: int, smooth_window: int) -> pd.Series | None:
    if valid_df.empty:
        return None

    cand = valid_df[valid_df["trades"] >= min_valid_trades].sort_values("threshold").copy()
    if cand.empty:
        return None

    if smooth_window > 1:
        cand["smoothed_profit_r"] = cand["net_profit_r"].rolling(window=smooth_window, min_periods=1, center=True).mean()
        cand["smoothed_avg_r"] = cand["avg_r"].rolling(window=smooth_window, min_periods=1, center=True).mean()
    else:
        cand["smoothed_profit_r"] = cand["net_profit_r"]
        cand["smoothed_avg_r"] = cand["avg_r"]

    picked = cand.sort_values(
        by=["smoothed_profit_r", "smoothed_avg_r", "trades"],
        ascending=[False, False, False],
    ).iloc[0]
    return picked


def select_high_precision_threshold(
    valid_df: pd.DataFrame,
    min_win_rate: float,
    min_valid_trades: int,
    smooth_window: int,
) -> pd.Series | None:
    if valid_df.empty:
        return None

    cand = valid_df[
        (valid_df["trades"] >= min_valid_trades) & (valid_df["win_rate"] >= min_win_rate)
    ].sort_values("threshold").copy()
    if cand.empty:
        return None

    if smooth_window > 1:
        cand["smoothed_profit_r"] = cand["net_profit_r"].rolling(window=smooth_window, min_periods=1, center=True).mean()
        cand["smoothed_avg_r"] = cand["avg_r"].rolling(window=smooth_window, min_periods=1, center=True).mean()
    else:
        cand["smoothed_profit_r"] = cand["net_profit_r"]
        cand["smoothed_avg_r"] = cand["avg_r"]

    picked = cand.sort_values(
        by=["smoothed_profit_r", "smoothed_avg_r", "trades"],
        ascending=[False, False, False],
    ).iloc[0]
    return picked


def select_quality_threshold(
    valid_df: pd.DataFrame,
    min_valid_trades: int,
    smooth_window: int,
) -> pd.Series | None:
    """Select threshold that maximises a quality score balancing avg_r, win_rate
    and drawdown.  score = avg_r * win_rate / (-max_drawdown_r + 1).

    Compared to select_threshold (which maximises total net_profit and tends to
    pick lower thresholds with more trades), this favours higher thresholds that
    produce fewer but higher-quality trades with smaller drawdowns.
    """
    if valid_df.empty:
        return None

    cand = valid_df[valid_df["trades"] >= min_valid_trades].sort_values("threshold").copy()
    if cand.empty:
        return None

    if smooth_window > 1:
        cand["_avg_r"] = cand["avg_r"].rolling(window=smooth_window, min_periods=1, center=True).mean()
        cand["_wr"] = cand["win_rate"].rolling(window=smooth_window, min_periods=1, center=True).mean()
        cand["_dd"] = cand["max_drawdown_r"].rolling(window=smooth_window, min_periods=1, center=True).mean()
    else:
        cand["_avg_r"] = cand["avg_r"]
        cand["_wr"] = cand["win_rate"]
        cand["_dd"] = cand["max_drawdown_r"]

    # quality = avg_r * win_rate / (-max_drawdown + 1)
    # higher avg_r, higher win_rate, smaller |drawdown| → higher score
    cand["_quality"] = cand["_avg_r"] * cand["_wr"] / (-cand["_dd"] + 1.0)
    # only keep candidates with positive avg_r
    cand = cand[cand["_avg_r"] > 0]
    if cand.empty:
        return None

    picked = cand.sort_values("_quality", ascending=False).iloc[0]
    return picked


def evaluate_strategy(model, split_data):
    print("\n" + "=" * 40)
    print("STRATEGY PERFORMANCE REPORT")
    print("=" * 40)

    X_valid = split_data["X_valid"]
    y_valid = split_data["y_valid"]
    df_valid = split_data["df_valid"]
    X_test = split_data["X_test"]
    y_test = split_data["y_test"]
    df_test = split_data["df_test"]

    if len(X_test) == 0 or len(y_test) == 0:
        print("Test split is empty after purge gap. Skip evaluation.")
        return

    valid_probs = model.predict_proba(X_valid)[:, 1] if len(X_valid) > 0 else np.array([], dtype=np.float64)
    probs = model.predict_proba(X_test)[:, 1]
    ap = average_precision_score(y_test, probs)
    print(f"Average Precision (AP): {ap:.4f}")
    if len(valid_probs) > 0:
        summarize_scores("Valid", valid_probs)
    summarize_scores("Test ", probs)

    tp_r = CONFIG["tp_mult"]
    sl_r = CONFIG["sl_mult"]
    realized_pnl_valid = None
    realized_pnl_test = None
    if "realized_pnl_r" in df_valid.columns:
        realized_pnl_valid = df_valid["realized_pnl_r"].to_numpy(dtype=np.float64, copy=False)
    if "realized_pnl_r" in df_test.columns:
        realized_pnl_test = df_test["realized_pnl_r"].to_numpy(dtype=np.float64, copy=False)

    thresholds = build_threshold_grid(valid_probs if len(valid_probs) > 0 else probs)
    if len(thresholds) == 0:
        print("No valid thresholds generated.")
        return
    print(f"Threshold range: {thresholds[0]:.4f} -> {thresholds[-1]:.4f} ({len(thresholds)} points)")
    if len(thresholds) > 1:
        print(
            "Threshold step summary: "
            f"min={np.diff(thresholds).min():.6f}, median={np.median(np.diff(thresholds)):.6f}, "
            f"max={np.diff(thresholds).max():.6f}"
        )

    valid_df = evaluate_thresholds(valid_probs, y_valid, realized_pnl_valid, tp_r, sl_r, thresholds) if len(valid_probs) > 0 else pd.DataFrame()
    res_df = evaluate_thresholds(probs, y_test, realized_pnl_test, tp_r, sl_r, thresholds)
    if res_df.empty:
        print("No trades taken at any evaluated threshold.")
        return

    if not valid_df.empty:
        picked = select_threshold(
            valid_df,
            min_valid_trades=CONFIG["min_valid_trades"],
            smooth_window=CONFIG["threshold_smooth_window"],
        )
        if picked is not None:
            best_threshold = float(picked["threshold"])
            matched = res_df[res_df["threshold"] == best_threshold]
            if matched.empty:
                nearest_idx = (res_df["threshold"] - best_threshold).abs().idxmin()
                best_profit = res_df.loc[nearest_idx]
                best_threshold = float(best_profit["threshold"])
            else:
                best_profit = matched.iloc[0]
            print(
                "Threshold selection: "
                f"validation smoothed profit, min_valid_trades={CONFIG['min_valid_trades']}, "
                f"smooth_window={CONFIG['threshold_smooth_window']}"
            )
        else:
            print(
                "Threshold selection fallback: "
                f"no validation threshold met min_valid_trades={CONFIG['min_valid_trades']}, "
                "using validation max net profit without trade floor."
            )
            best_threshold = float(valid_df.loc[valid_df["net_profit_r"].idxmax(), "threshold"])
            matched = res_df[res_df["threshold"] == best_threshold]
            if matched.empty:
                nearest_idx = (res_df["threshold"] - best_threshold).abs().idxmin()
                best_profit = res_df.loc[nearest_idx]
                best_threshold = float(best_profit["threshold"])
            else:
                best_profit = matched.iloc[0]
    else:
        best_profit = res_df.loc[res_df["net_profit_r"].idxmax()]
        best_threshold = float(best_profit["threshold"])

    if CONFIG["label_mode"] == "first_touch":
        be_win_rate = sl_r / (tp_r + sl_r)
        high_prec_min_wr = max(be_win_rate + 0.05, 0.40)
        best_prec_valid = (
            select_high_precision_threshold(
                valid_df,
                min_win_rate=high_prec_min_wr,
                min_valid_trades=CONFIG["min_valid_trades"],
                smooth_window=CONFIG["threshold_smooth_window"],
            )
            if not valid_df.empty
            else None
        )
        if best_prec_valid is not None:
            best_prec_threshold = float(best_prec_valid["threshold"])
            matched = res_df[res_df["threshold"] == best_prec_threshold]
            if matched.empty:
                nearest_idx = (res_df["threshold"] - best_prec_threshold).abs().idxmin()
                best_prec = res_df.loc[nearest_idx]
                best_prec_threshold = float(best_prec["threshold"])
            else:
                best_prec = matched.iloc[0]
        else:
            best_prec_threshold = None
            best_prec = None
    else:
        be_win_rate = None
        best_prec = None
        best_prec_valid = None
        best_prec_threshold = None

    print(f"\nScenario A: VALID-SELECTED THRESHOLD")
    print(f"Threshold    : {best_threshold:.4f}")
    print(f"Trades Taken : {int(best_profit['trades'])}")
    print(f"Win Rate     : {best_profit['win_rate']:.2%}")
    print(f"Net Profit   : {best_profit['net_profit_r']:.1f} R")
    print(f"Avg R/Trade  : {best_profit['avg_r']:.2f} R")
    print(f"Worst Trade  : {best_profit['worst_trade_r']:.2f} R")
    print(f"Max Drawdown : {best_profit['max_drawdown_r']:.1f} R")

    if not valid_df.empty:
        print_threshold_context("Validation", valid_df, selected_threshold=best_threshold)
    print_threshold_context("Test", res_df, selected_threshold=best_threshold)
    print_bin_diagnostics(probs, y_test, realized_pnl_test, CONFIG["label_mode"])

    if best_prec is not None and be_win_rate is not None and best_prec_valid is not None and best_prec_threshold is not None:
        print(f"\nScenario B: VALID-SELECTED HIGH PRECISION (>{high_prec_min_wr:.0%})")
        print(
            "Threshold selection: "
            f"validation smoothed profit under win-rate floor, min_valid_trades={CONFIG['min_valid_trades']}, "
            f"smooth_window={CONFIG['threshold_smooth_window']}"
        )
        print(f"Threshold    : {best_prec_threshold:.4f}")
        print(f"Trades Taken : {int(best_prec['trades'])}")
        print(f"Win Rate     : {best_prec['win_rate']:.2%}")
        print(f"Net Profit   : {best_prec['net_profit_r']:.1f} R")
        print(f"Worst Trade  : {best_prec['worst_trade_r']:.2f} R")
        print(f"Max Drawdown : {best_prec['max_drawdown_r']:.1f} R")
    elif CONFIG["label_mode"] == "first_touch":
        valid_wr_max = valid_df["win_rate"].max() if not valid_df.empty else float("nan")
        print(f"\nScenario B: Not reachable on validation. Max validation WR: {valid_wr_max:.2%}")

    plot_curves(res_df, y_test, probs, valid_probs, best_threshold)

    if df_test is not None and len(df_test) == len(y_test):
        print("\n--- Equity Curve (Sequential Walk) ---")
        best_thresh = best_threshold
        signals = (probs > best_thresh).astype(int)
        if "realized_pnl_r" in df_test.columns:
            outcomes = df_test["realized_pnl_r"].to_numpy(dtype=np.float64, copy=False)
        else:
            outcomes = np.where(y_test == 1, tp_r, -sl_r)
        trade_pnl = signals * outcomes
        cum_r = trade_pnl.cumsum()
        peak = np.maximum.accumulate(cum_r)
        drawdown = cum_r - peak
        print(f"Total R: {cum_r.iloc[-1] if hasattr(cum_r, 'iloc') else cum_r[-1]:.1f}")
        print(f"Max Drawdown: {drawdown.min():.1f} R")
        print(f"Total trades: {signals.sum()}")


def plot_curves(df, y_test, probs, valid_probs, selected_threshold: float):
    _, axes = plt.subplots(1, 5, figsize=(30, 5))

    axes[0].plot(df["threshold"], df["net_profit_r"], color="green", linewidth=2, label="Net Profit (R)")
    axes[0].axhline(0, color="black", linestyle="--")
    axes[0].axvline(selected_threshold, color="orange", linestyle=":", linewidth=1.5, label=f"Selected {selected_threshold:.4f}")
    axes[0].set_title("Test Profit vs. Threshold")
    axes[0].set_xlabel("Threshold")
    axes[0].set_ylabel("Net Profit (R-Units)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    ax2 = axes[1]
    metric_label = "Win Rate" if CONFIG["label_mode"] == "first_touch" else "Positive PnL Rate"
    ax2.plot(df["threshold"], df["win_rate"], color="blue", label=metric_label)
    ax2.axvline(selected_threshold, color="orange", linestyle=":", linewidth=1.5)
    ax2.set_ylabel(metric_label)
    ax2.set_ylim(0, 1.0)

    ax2_twin = ax2.twinx()
    ax2_twin.fill_between(df["threshold"], df["trades"], color="gray", alpha=0.2, label="Trade Count")
    ax2_twin.set_ylabel("Number of Trades")

    axes[1].set_title("Win Rate & Opportunity Frequency")
    axes[1].set_xlabel("Threshold")

    if CONFIG["label_mode"] == "first_touch":
        be_win_rate = CONFIG["sl_mult"] / (CONFIG["tp_mult"] + CONFIG["sl_mult"])
        axes[1].axhline(be_win_rate, color="red", linestyle="--", label=f"Breakeven ({be_win_rate:.0%})")

    precision, recall, _ = precision_recall_curve(y_test, probs)
    axes[2].plot(recall, precision, color="purple", linewidth=2)
    axes[2].set_title("Precision-Recall Curve")
    axes[2].set_xlabel("Recall")
    axes[2].set_ylabel("Precision")
    axes[2].grid(True)

    dd_ax = axes[3]
    dd_ax.plot(df["threshold"], df["max_drawdown_r"], color="firebrick", linewidth=2, label="Max Drawdown (R)")
    dd_ax.axvline(selected_threshold, color="orange", linestyle=":", linewidth=1.5)
    dd_ax.axhline(0, color="black", linestyle="--", linewidth=1.0)
    dd_ax.set_title("Max Drawdown vs. Threshold")
    dd_ax.set_xlabel("Threshold")
    dd_ax.set_ylabel("Max Drawdown (R)")
    dd_ax.grid(True, alpha=0.3)
    dd_ax.legend()

    score_ax = axes[4]
    score_ax.hist(probs, bins=30, alpha=0.55, color="steelblue", label="Test", density=False)
    if len(valid_probs) > 0:
        score_ax.hist(valid_probs, bins=30, alpha=0.35, color="darkorange", label="Valid", density=False)
    score_ax.axvline(selected_threshold, color="black", linestyle="--", linewidth=1.5, label=f"Selected {selected_threshold:.4f}")
    score_ax.set_title("Score Distribution")
    score_ax.set_xlabel("Predicted Probability")
    score_ax.set_ylabel("Count")
    score_ax.grid(True, alpha=0.3)
    score_ax.legend()

    plt.tight_layout()
    save_path = os.path.join(CONFIG["plot_dir"], "strategy_performance.png")
    plt.savefig(save_path)
    print(f"\nPlots saved to: {save_path}")
    if plt.get_backend().lower() != "agg":
        plt.show()
    plt.close()


if __name__ == "__main__":
    args = parse_args()
    CONFIG["data_path"] = args.data_path
    CONFIG["label_mode"] = args.label_mode
    CONFIG["same_bar_policy"] = args.same_bar_policy
    CONFIG["horizon"] = args.horizon
    CONFIG["tp_mult"] = args.tp_mult
    CONFIG["sl_mult"] = args.sl_mult
    CONFIG["min_valid_trades"] = args.min_valid_trades
    CONFIG["threshold_smooth_window"] = args.threshold_smooth_window
    CONFIG["model_out"] = f"model_sniper_v3_{args.label_mode}.json"

    df = load_and_label(
        CONFIG["data_path"],
        scanner_arg=args.scanner,
        scanner_variant_arg=args.scanner_variant,
        label_mode=CONFIG["label_mode"],
        same_bar_policy=CONFIG["same_bar_policy"],
        horizon=CONFIG["horizon"],
        tp_mult=CONFIG["tp_mult"],
        sl_mult=CONFIG["sl_mult"],
        dynamic_tpsl=args.dynamic_tpsl,
        breakeven=args.breakeven,
        long_only=args.long_only,
    )

    if not df.empty:
        available = [feature for feature in CONFIG["features"] if feature in df.columns]
        df = df.dropna(subset=available).copy()
        if args.live:
            train_live_model(df)
        else:
            model, split_data = train_final_model(df)
            evaluate_strategy(model, split_data)
