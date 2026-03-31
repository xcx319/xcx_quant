# run "conda activate quant" first
# pip install lightgbm catboost
"""Train XGBoost + LightGBM + CatBoost with split long/short support.

Uses the same threshold selection and evaluation flow as train_xgb.py.
Trains each model type for both long and short directions independently,
then compares all combinations to pick the best overall configuration.
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score

from quant_modeling import (
    BASE_FEATURES,
    add_directional_features,
    build_labeling_cache,
    build_labels,
    build_realized_pnl,
)

# Reuse helpers from train_xgb
from train_xgb import (
    build_threshold_grid,
    evaluate_thresholds,
    filter_by_scanner,
    load_and_label,
    select_threshold,
    select_quality_threshold,
    summarize_scores,
)

try:
    with open("best_config.json", "r") as f:
        BEST = json.load(f)
    print(f"Loaded Config: {BEST}")
except Exception:
    print("Config not found, using defaults.")
    BEST = {"h": 30, "tp": 1.5, "sl": 1.5}

PURGE_GAP_BARS = 30
PLOT_DIR = "./plots"
os.makedirs(PLOT_DIR, exist_ok=True)

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
    "min_valid_trades": int(BEST.get("min_valid_trades", 80)),
    "threshold_smooth_window": int(BEST.get("threshold_smooth_window", 3)),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train XGB+LGB+CB with long/short split")
    parser.add_argument("--data-path", default=CONFIG["data_path"])
    parser.add_argument("--label-mode", choices=["window_tp", "first_touch"], default=CONFIG["label_mode"])
    parser.add_argument("--same-bar-policy", choices=["drop", "neutral", "tp_first", "sl_first"], default=CONFIG["same_bar_policy"])
    parser.add_argument("--horizon", type=int, default=CONFIG["horizon"])
    parser.add_argument("--tp-mult", type=float, default=CONFIG["tp_mult"])
    parser.add_argument("--sl-mult", type=float, default=CONFIG["sl_mult"])
    parser.add_argument("--min-valid-trades", type=int, default=CONFIG["min_valid_trades"])
    parser.add_argument("--threshold-smooth-window", type=int, default=CONFIG["threshold_smooth_window"])
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Split helper
# ---------------------------------------------------------------------------

def make_splits(df: pd.DataFrame, features: list[str]):
    available_feats = [f for f in features if f in df.columns]
    X = df[available_feats].astype(np.float32)
    y = df["label"]
    w = df["sample_weight"]
    rpnl = df["realized_pnl_r"].to_numpy(dtype=np.float64)

    train_end = int(len(df) * 0.70)
    valid_start = train_end + PURGE_GAP_BARS
    valid_end = int(len(df) * 0.85)
    test_start = valid_end + PURGE_GAP_BARS

    return {
        "X_train": X.iloc[:train_end], "y_train": y.iloc[:train_end], "w_train": w.iloc[:train_end],
        "X_valid": X.iloc[valid_start:valid_end], "y_valid": y.iloc[valid_start:valid_end],
        "X_test": X.iloc[test_start:], "y_test": y.iloc[test_start:],
        "rpnl_valid": rpnl[valid_start:valid_end],
        "rpnl_test": rpnl[test_start:],
        "df_valid": df.iloc[valid_start:valid_end].copy(),
        "df_test": df.iloc[test_start:].copy(),
        "available_feats": available_feats,
    }


# ---------------------------------------------------------------------------
# Model trainers
# ---------------------------------------------------------------------------

def train_xgb_model(splits, save_path: str = "model_xgb.json"):
    model = xgb.XGBClassifier(
        n_estimators=1500, max_depth=4, learning_rate=0.003, subsample=0.5,
        colsample_bytree=0.7, colsample_bylevel=0.7, gamma=0.5, min_child_weight=50,
        reg_alpha=0.5, reg_lambda=2.0, max_delta_step=1, objective="binary:logistic",
        eval_metric="aucpr", tree_method="hist", n_jobs=-1, random_state=42,
        early_stopping_rounds=120,
    )
    model.fit(
        splits["X_train"], splits["y_train"], sample_weight=splits["w_train"], verbose=50,
        eval_set=[(splits["X_train"], splits["y_train"]), (splits["X_valid"], splits["y_valid"])],
    )
    model.get_booster().save_model(save_path)
    return model


def train_lgb_model(splits, save_path: str = "model_lgb.txt"):
    callbacks = [lgb.log_evaluation(50), lgb.early_stopping(120)]
    model = lgb.LGBMClassifier(
        n_estimators=1500, max_depth=4, learning_rate=0.003, subsample=0.5,
        colsample_bytree=0.7, min_child_weight=50, reg_lambda=2.0,
        num_leaves=15, n_jobs=-1, random_state=42, verbose=-1,
    )
    model.fit(
        splits["X_train"], splits["y_train"], sample_weight=splits["w_train"],
        eval_set=[(splits["X_valid"], splits["y_valid"])],
        eval_metric="average_precision", callbacks=callbacks,
    )
    model.booster_.save_model(save_path)
    return model


def train_cb_model(splits, save_path: str = "model_catboost.cbm"):
    model = CatBoostClassifier(
        iterations=1500, depth=4, learning_rate=0.003, subsample=0.5,
        rsm=0.7, min_data_in_leaf=50, l2_leaf_reg=2.0,
        random_seed=42, eval_metric="AUC", verbose=50,
        early_stopping_rounds=120,
    )
    model.fit(
        splits["X_train"], splits["y_train"], sample_weight=splits["w_train"],
        eval_set=(splits["X_valid"], splits["y_valid"]),
    )
    model.save_model(save_path)
    return model


def get_probs(model, X, model_name: str) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


# ---------------------------------------------------------------------------
# Threshold selection (same as train_xgb.py)
# ---------------------------------------------------------------------------

def _lookup_test_row(test_df, thr):
    """Find the test_df row closest to thr. Returns (adjusted_thr, row_dict)."""
    matched = test_df[test_df["threshold"] == thr]
    if matched.empty:
        nearest_idx = (test_df["threshold"] - thr).abs().idxmin()
        row = test_df.loc[nearest_idx].to_dict()
        return float(row["threshold"]), row
    return thr, matched.iloc[0].to_dict()


def find_best_threshold(probs_valid, probs_test, splits, tp_r, sl_r,
                        min_valid_trades, smooth_window):
    """Select threshold on validation, evaluate on test.

    Returns dict with keys 'profit' and 'quality', each containing
    (threshold, test_row_dict).
    """
    thresholds = build_threshold_grid(probs_valid if len(probs_valid) > 0 else probs_test)
    if len(thresholds) == 0:
        empty = 0.5, {}
        return {"profit": empty, "quality": empty}

    rpnl_valid = splits["rpnl_valid"]
    rpnl_test = splits["rpnl_test"]

    valid_df = evaluate_thresholds(probs_valid, splits["y_valid"], rpnl_valid, tp_r, sl_r, thresholds)
    test_df = evaluate_thresholds(probs_test, splits["y_test"], rpnl_test, tp_r, sl_r, thresholds)

    if valid_df.empty or test_df.empty:
        empty = 0.5, {}
        return {"profit": empty, "quality": empty}

    # --- Profit threshold (max net_profit_r) ---
    picked = select_threshold(valid_df, min_valid_trades=min_valid_trades, smooth_window=smooth_window)
    if picked is not None:
        profit_thr = float(picked["threshold"])
    else:
        profit_thr = float(valid_df.loc[valid_df["net_profit_r"].idxmax(), "threshold"])
    profit_thr, profit_row = _lookup_test_row(test_df, profit_thr)

    # --- Quality threshold (avg_r * win_rate / drawdown) ---
    picked_q = select_quality_threshold(valid_df, min_valid_trades=min_valid_trades, smooth_window=smooth_window)
    if picked_q is not None:
        quality_thr = float(picked_q["threshold"])
    else:
        quality_thr = profit_thr  # fallback
    quality_thr, quality_row = _lookup_test_row(test_df, quality_thr)

    return {
        "profit": (profit_thr, profit_row),
        "quality": (quality_thr, quality_row),
    }


# ---------------------------------------------------------------------------
# Evaluate a single direction model
# ---------------------------------------------------------------------------

def evaluate_direction(name: str, model, splits, tp_r, sl_r, min_vt, sw) -> dict:
    """Train threshold on valid, evaluate on test for one model+direction.

    Returns dict with both 'profit' and 'quality' threshold results.
    """
    pv = get_probs(model, splits["X_valid"], name)
    pt = get_probs(model, splits["X_test"], name)
    ap = average_precision_score(splits["y_test"], pt) if splits["y_test"].nunique() > 1 else 0.0
    thresholds = find_best_threshold(pv, pt, splits, tp_r, sl_r, min_vt, sw)

    result = {"name": name, "ap": ap}
    for mode in ["profit", "quality"]:
        thr, row = thresholds[mode]
        result[f"{mode}_threshold"] = thr
        result[f"{mode}_trades"] = int(row.get("trades", 0))
        result[f"{mode}_win_rate"] = float(row.get("win_rate", 0))
        result[f"{mode}_net_profit_r"] = float(row.get("net_profit_r", 0))
        result[f"{mode}_avg_r"] = float(row.get("avg_r", 0))
        result[f"{mode}_max_drawdown_r"] = float(row.get("max_drawdown_r", 0))
    return result


# ---------------------------------------------------------------------------
# Merged evaluation (route long→long_model, short→short_model)
# ---------------------------------------------------------------------------

def evaluate_merged(name: str, model_long, model_short, feats,
                    df_test_long, df_test_short, thr_long, thr_short):
    """Evaluate merged long+short on their respective test sets."""
    parts = []
    for df_part, model, thr in [(df_test_long, model_long, thr_long),
                                 (df_test_short, model_short, thr_short)]:
        if len(df_part) == 0:
            continue
        X = df_part[feats].astype(np.float32)
        probs = model.predict_proba(X)[:, 1]
        mask = probs >= thr
        df_sel = df_part[mask].copy()
        if len(df_sel) > 0:
            parts.append(df_sel)

    if not parts:
        return {"name": name, "trades": 0, "win_rate": 0, "net_r": 0, "avg_r": 0,
                "max_dd": 0, "long_trades": 0, "short_trades": 0}

    merged = pd.concat(parts).sort_index()
    rpnl = merged["realized_pnl_r"].to_numpy(dtype=np.float64)
    cum = np.cumsum(rpnl)
    dd = float((cum - np.maximum.accumulate(cum)).min()) if len(cum) > 0 else 0
    dirs = merged["event_dir"].to_numpy()
    return {
        "name": name,
        "trades": len(merged),
        "win_rate": float(np.mean(rpnl > 0)),
        "net_r": float(np.nansum(rpnl)),
        "avg_r": float(np.nanmean(rpnl)),
        "max_dd": dd,
        "long_trades": int((dirs == 1).sum()),
        "short_trades": int((dirs == -1).sum()),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    CONFIG["data_path"] = args.data_path
    CONFIG["label_mode"] = args.label_mode
    CONFIG["same_bar_policy"] = args.same_bar_policy
    CONFIG["horizon"] = args.horizon
    CONFIG["tp_mult"] = args.tp_mult
    CONFIG["sl_mult"] = args.sl_mult
    CONFIG["min_valid_trades"] = args.min_valid_trades
    CONFIG["threshold_smooth_window"] = args.threshold_smooth_window

    h = CONFIG["horizon"]
    tp = CONFIG["tp_mult"]
    sl = CONFIG["sl_mult"]
    min_vt = CONFIG["min_valid_trades"]
    sw = CONFIG["threshold_smooth_window"]

    # Load full dataset
    df = load_and_label(
        CONFIG["data_path"],
        scanner_arg=CONFIG["scanner"],
        scanner_variant_arg=CONFIG["scanner_variant"],
        label_mode=CONFIG["label_mode"],
        same_bar_policy=CONFIG["same_bar_policy"],
        horizon=h, tp_mult=tp, sl_mult=sl,
    )
    if df.empty:
        print("No data. Exiting.")
        return

    available = [f for f in CONFIG["features"] if f in df.columns]
    missing = sorted(set(CONFIG["features"]) - set(available))
    if missing:
        print(f"Missing features (skipped): {missing}")
    df = df.dropna(subset=available).copy()

    df_long = df[df["event_dir"] == 1].copy()
    df_short = df[df["event_dir"] == -1].copy()

    # Recalculate sample weights per direction
    for sub in [df_long, df_short]:
        pr = sub["label"].mean()
        if 0 < pr < 1:
            sub["sample_weight"] = np.where(sub["label"] == 1, (1 - pr) / pr, 1.0).astype(np.float32)

    print(f"\nCombined: {len(df)} | Long: {len(df_long)} | Short: {len(df_short)}")
    print(f"Win rate — Combined: {df['label'].mean():.2%} | Long: {df_long['label'].mean():.2%} | Short: {df_short['label'].mean():.2%}")

    splits_all = make_splits(df, CONFIG["features"])
    splits_long = make_splits(df_long, CONFIG["features"])
    splits_short = make_splits(df_short, CONFIG["features"])
    feats = splits_all["available_feats"]

    # ── Train all models ──
    trainers = {
        "xgb": (train_xgb_model, "model_xgb{suffix}.json"),
        "lgb": (train_lgb_model, "model_lgb{suffix}.txt"),
        "catboost": (train_cb_model, "model_catboost{suffix}.cbm"),
    }

    models = {}  # {("xgb", "long"): model, ...}
    dir_evals = {}  # {("xgb", "long"): eval_dict, ...}

    test_start_l = int(len(df_long) * 0.85) + PURGE_GAP_BARS
    test_start_s = int(len(df_short) * 0.85) + PURGE_GAP_BARS
    df_test_l = df_long.iloc[test_start_l:].copy()
    df_test_s = df_short.iloc[test_start_s:].copy()

    for mtype, (trainer, path_tpl) in trainers.items():
        for direction, splits_dir, suffix in [
            ("combined", splits_all, ""),
            ("long", splits_long, "_long"),
            ("short", splits_short, "_short"),
        ]:
            print(f"\n{'='*60}")
            print(f"TRAINING: {mtype.upper()} {direction.upper()} (h={h}, tp={tp}, sl={sl})")
            print(f"{'='*60}")
            m = trainer(splits_dir, save_path=path_tpl.format(suffix=suffix))
            models[(mtype, direction)] = m
            ev = evaluate_direction(f"{mtype}_{direction}", m, splits_dir, tp, sl, min_vt, sw)
            dir_evals[(mtype, direction)] = ev

    # ── Ensemble: wrapper class ──
    class EnsembleModel:
        def __init__(self, model_list):
            self._models = model_list
        def predict_proba(self, X):
            probs = np.mean([m.predict_proba(X)[:, 1] for m in self._models], axis=0)
            return np.column_stack([1 - probs, probs])

    for direction, splits_dir in [("combined", splits_all), ("long", splits_long), ("short", splits_short)]:
        ens = EnsembleModel([models[(m, direction)] for m in ["xgb", "lgb", "catboost"]])
        models[("ensemble", direction)] = ens
        ev = evaluate_direction(f"ensemble_{direction}", ens, splits_dir, tp, sl, min_vt, sw)
        dir_evals[("ensemble", direction)] = ev

    # ── Build all candidate configurations ──
    # Each candidate = (name, model_type, threshold_mode, mode, ...)
    all_results = []
    model_types = ["xgb", "lgb", "catboost", "ensemble"]

    for mtype in model_types:
        for thr_mode in ["profit", "quality"]:
            # Combined
            ev = dir_evals[(mtype, "combined")]
            all_results.append({
                "name": f"{mtype}", "model_type": mtype, "mode": "combined", "thr_mode": thr_mode,
                "ap": ev["ap"],
                "threshold": ev[f"{thr_mode}_threshold"],
                "trades": ev[f"{thr_mode}_trades"],
                "win_rate": ev[f"{thr_mode}_win_rate"],
                "net_profit_r": ev[f"{thr_mode}_net_profit_r"],
                "avg_r": ev[f"{thr_mode}_avg_r"],
                "max_drawdown_r": ev[f"{thr_mode}_max_drawdown_r"],
            })

            # Split (merged long+short)
            ev_l = dir_evals[(mtype, "long")]
            ev_s = dir_evals[(mtype, "short")]
            thr_l = ev_l[f"{thr_mode}_threshold"]
            thr_s = ev_s[f"{thr_mode}_threshold"]

            m_long = models[(mtype, "long")]
            m_short = models[(mtype, "short")]
            r_merged = evaluate_merged(
                f"{mtype}_split", m_long, m_short, feats,
                df_test_l, df_test_s, thr_l, thr_s,
            )
            r_merged["model_type"] = mtype
            r_merged["mode"] = "split"
            r_merged["thr_mode"] = thr_mode
            r_merged["long_threshold"] = thr_l
            r_merged["short_threshold"] = thr_s
            r_merged["ap"] = (ev_l["ap"] + ev_s["ap"]) / 2
            # Normalize keys
            r_merged["net_profit_r"] = r_merged.pop("net_r", 0)
            r_merged["max_drawdown_r"] = r_merged.pop("max_dd", 0)
            all_results.append(r_merged)

    # ── Print comparison ──
    print(f"\n{'='*110}")
    print("MODEL COMPARISON — ALL CONFIGURATIONS")
    print(f"{'='*110}")
    header = f"{'Name':<16} {'Mode':<10} {'ThrMode':<8} {'AP':>6} {'Threshold':>16} {'Trades':>7} {'WinRate':>8} {'NetProfit':>10} {'AvgR':>7} {'MaxDD':>8}"
    print(header)
    print("-" * 110)
    for r in sorted(all_results, key=lambda x: -x.get("net_profit_r", 0)):
        mode = r.get("mode", "")
        thr_mode = r.get("thr_mode", "")
        if mode == "split" and "long_threshold" in r:
            thr_str = f"L{r['long_threshold']:.4f}/S{r['short_threshold']:.4f}"
        else:
            thr_str = f"{r.get('threshold', 0):.4f}"
        print(f"{r['name']:<16} {mode:<10} {thr_mode:<8} {r.get('ap', 0):>6.4f} {thr_str:>16} {r['trades']:>7} "
              f"{r.get('win_rate', 0):>7.2%} {r.get('net_profit_r', 0):>10.1f} {r.get('avg_r', 0):>7.3f} {r.get('max_drawdown_r', 0):>8.1f}")

    # ── Pick best by net_profit_r ──
    best_profit = max(all_results, key=lambda x: x.get("net_profit_r", 0))
    # ── Pick best quality (highest avg_r among those with win_rate > 50% and trades >= 500) ──
    quality_candidates = [r for r in all_results if r.get("win_rate", 0) > 0.50 and r.get("trades", 0) >= 500]
    best_quality = max(quality_candidates, key=lambda x: x.get("avg_r", 0)) if quality_candidates else best_profit

    print(f"\nBest by net_profit: {best_profit['name']} ({best_profit['mode']}/{best_profit['thr_mode']}) — "
          f"net={best_profit['net_profit_r']:.1f}R, avg_r={best_profit['avg_r']:.3f}, wr={best_profit['win_rate']:.1%}, dd={best_profit['max_drawdown_r']:.1f}")
    print(f"Best by quality:    {best_quality['name']} ({best_quality['mode']}/{best_quality['thr_mode']}) — "
          f"net={best_quality['net_profit_r']:.1f}R, avg_r={best_quality['avg_r']:.3f}, wr={best_quality['win_rate']:.1%}, dd={best_quality['max_drawdown_r']:.1f}")

    # Use quality as the final pick
    best = best_quality

    # ── Update best_config.json ──
    model_type_map = {"xgb": "xgb", "lgb": "lgb", "catboost": "catboost", "ensemble": "ensemble"}
    cfg = BEST.copy()
    mtype = best.get("model_type", "xgb")
    cfg["model_type"] = model_type_map.get(mtype, mtype)
    cfg["threshold_mode"] = best.get("thr_mode", "profit")

    if best.get("mode") == "split":
        cfg["split_model"] = True
        cfg["threshold"] = best.get("long_threshold", best.get("threshold", 0.5))
        cfg["long_config"] = {"h": h, "tp": tp, "sl": sl, "threshold": round(best["long_threshold"], 4)}
        cfg["short_config"] = {"h": h, "tp": tp, "sl": sl, "threshold": round(best["short_threshold"], 4)}
    else:
        cfg["split_model"] = False
        cfg["threshold"] = best.get("threshold", 0.5)
        cfg.pop("long_config", None)
        cfg.pop("short_config", None)

    cfg["total_test_trades"] = best["trades"]
    cfg["total_test_net_profit_r"] = best.get("net_profit_r", 0)
    cfg["avg_r_per_trade"] = best.get("avg_r", 0)
    cfg["ap"] = best.get("ap", 0)

    with open("best_config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"\nUpdated best_config.json: model_type={cfg['model_type']}, split={cfg.get('split_model', False)}, thr_mode={cfg.get('threshold_mode')}")

    # ── Equity curves plot ──
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    plot_idx = 0
    for r in sorted(all_results, key=lambda x: -x.get("net_profit_r", 0))[:6]:
        ax = axes[plot_idx // 3, plot_idx % 3]
        mtype_r = r.get("model_type", "xgb")
        mode = r.get("mode", "combined")
        if mode == "combined" and mtype_r != "ensemble":
            m = models.get((mtype_r, "combined"))
            if m:
                pt = get_probs(m, splits_all["X_test"], mtype_r)
                mask = pt >= r.get("threshold", 0.5)
                pnl = splits_all["rpnl_test"][mask]
                if len(pnl) > 0:
                    ax.plot(np.cumsum(pnl), linewidth=1.2)
        net = r.get("net_profit_r", 0)
        thr_mode = r.get("thr_mode", "")
        ax.set_title(f"{r['name']} ({mode}/{thr_mode})\n{r['trades']}t, {net:.0f}R, avg={r.get('avg_r',0):.3f}, wr={r.get('win_rate',0):.0%}", fontsize=8)
        ax.set_ylabel("Cumulative R")
        ax.grid(True, alpha=0.3)
        plot_idx += 1

    plt.suptitle(f"Multi-Model Comparison h={h}/tp={tp}/sl={sl}", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "multi_model_comparison.png"), dpi=150)
    print(f"Plot saved to: {PLOT_DIR}/multi_model_comparison.png")
    plt.close()


if __name__ == "__main__":
    main()
