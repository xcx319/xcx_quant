# run "conda activate quant" first
# pip install lightgbm catboost
from __future__ import annotations

import argparse
import json
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score
from sklearn.linear_model import LogisticRegression

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
    plot_curves,
    select_threshold,
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
    "plot_dir": "./plots",
    "min_valid_trades": int(BEST.get("min_valid_trades", 80)),
    "threshold_smooth_window": int(BEST.get("threshold_smooth_window", 3)),
}

os.makedirs(CONFIG["plot_dir"], exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train XGBoost + LightGBM + CatBoost + Ensemble")
    parser.add_argument("--data-path", default=CONFIG["data_path"])
    parser.add_argument("--scanner", default=CONFIG["scanner"])
    parser.add_argument("--scanner-variant", default=CONFIG["scanner_variant"])
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
    missing = sorted(set(features) - set(available_feats))
    if missing:
        print(f"Warning: Missing features (will skip): {missing}")

    X = df[available_feats].astype(np.float32)
    y = df["label"]
    w = df["sample_weight"]

    train_end = int(len(df) * 0.70)
    valid_start = train_end + PURGE_GAP_BARS
    valid_end = int(len(df) * 0.85)
    test_start = valid_end + PURGE_GAP_BARS

    splits = {
        "X_train": X.iloc[:train_end], "y_train": y.iloc[:train_end], "w_train": w.iloc[:train_end],
        "X_valid": X.iloc[valid_start:valid_end], "y_valid": y.iloc[valid_start:valid_end],
        "X_test": X.iloc[test_start:], "y_test": y.iloc[test_start:],
        "df_valid": df.iloc[valid_start:valid_end].copy(),
        "df_test": df.iloc[test_start:].copy(),
        "available_feats": available_feats,
    }
    print(f"\nTrain: {len(splits['X_train'])}, Valid: {len(splits['X_valid'])}, Test: {len(splits['X_test'])} (purge gap={PURGE_GAP_BARS})")
    return splits


# ---------------------------------------------------------------------------
# Individual model trainers
# ---------------------------------------------------------------------------

def train_xgb_model(splits):
    print("\n" + "=" * 50)
    print("Training XGBoost")
    print("=" * 50)
    early_stopping = 120 if len(splits["X_valid"]) > 0 and splits["y_valid"].nunique() > 1 else None
    fit_kwargs = {"X": splits["X_train"], "y": splits["y_train"], "sample_weight": splits["w_train"], "verbose": 50}
    if early_stopping:
        fit_kwargs["eval_set"] = [(splits["X_train"], splits["y_train"]), (splits["X_valid"], splits["y_valid"])]

    model = xgb.XGBClassifier(
        n_estimators=1500, max_depth=4, learning_rate=0.003, subsample=0.5,
        colsample_bytree=0.7, colsample_bylevel=0.7, gamma=0.5, min_child_weight=50,
        reg_alpha=0.5, reg_lambda=2.0, max_delta_step=1, objective="binary:logistic",
        eval_metric="aucpr", tree_method="hist", n_jobs=-1, random_state=42,
        early_stopping_rounds=early_stopping,
    )
    model.fit(**fit_kwargs)
    model.get_booster().save_model("model_xgb.json")
    print("Saved model_xgb.json")
    return model


def train_lgb_model(splits):
    print("\n" + "=" * 50)
    print("Training LightGBM")
    print("=" * 50)
    callbacks = [lgb.log_evaluation(50)]
    early_stopping = 120 if len(splits["X_valid"]) > 0 and splits["y_valid"].nunique() > 1 else None
    if early_stopping:
        callbacks.append(lgb.early_stopping(early_stopping))

    fit_kwargs = {
        "X": splits["X_train"], "y": splits["y_train"], "sample_weight": splits["w_train"],
        "callbacks": callbacks,
    }
    if early_stopping:
        fit_kwargs["eval_set"] = [(splits["X_valid"], splits["y_valid"])]
        fit_kwargs["eval_metric"] = "average_precision"

    model = lgb.LGBMClassifier(
        n_estimators=1500, max_depth=4, learning_rate=0.003, subsample=0.5,
        colsample_bytree=0.7, min_child_weight=50, reg_lambda=2.0,
        num_leaves=15, n_jobs=-1, random_state=42, verbose=-1,
    )
    model.fit(**fit_kwargs)
    model.booster_.save_model("model_lgb.txt")
    print("Saved model_lgb.txt")
    return model


def train_catboost_model(splits):
    print("\n" + "=" * 50)
    print("Training CatBoost")
    print("=" * 50)
    early_stopping = 120 if len(splits["X_valid"]) > 0 and splits["y_valid"].nunique() > 1 else None

    model = CatBoostClassifier(
        iterations=1500, depth=4, learning_rate=0.003, subsample=0.5,
        rsm=0.7, min_data_in_leaf=50, l2_leaf_reg=2.0,
        random_seed=42, eval_metric="AUC", verbose=50,
        early_stopping_rounds=early_stopping,
    )
    eval_set = None
    if early_stopping:
        eval_set = (splits["X_valid"], splits["y_valid"])
    model.fit(splits["X_train"], splits["y_train"], sample_weight=splits["w_train"], eval_set=eval_set)
    model.save_model("model_catboost.cbm")
    print("Saved model_catboost.cbm")
    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def get_probs(model, X, model_name: str) -> np.ndarray:
    if model_name == "catboost":
        return model.predict_proba(X)[:, 1]
    elif model_name == "lgb":
        return model.predict_proba(X)[:, 1]
    else:
        return model.predict_proba(X)[:, 1]


def evaluate_single(name: str, probs_valid: np.ndarray, probs_test: np.ndarray, splits) -> dict:
    """Evaluate a single model/ensemble and return summary dict."""
    y_valid = splits["y_valid"]
    y_test = splits["y_test"]
    df_test = splits["df_test"]

    ap = average_precision_score(y_test, probs_test)
    summarize_scores(f"{name} Test", probs_test)

    tp_r = CONFIG["tp_mult"]
    sl_r = CONFIG["sl_mult"]
    realized_pnl_valid = splits["df_valid"]["realized_pnl_r"].to_numpy(dtype=np.float64) if "realized_pnl_r" in splits["df_valid"].columns else None
    realized_pnl_test = df_test["realized_pnl_r"].to_numpy(dtype=np.float64) if "realized_pnl_r" in df_test.columns else None

    thresholds = build_threshold_grid(probs_valid if len(probs_valid) > 0 else probs_test)
    valid_df = evaluate_thresholds(probs_valid, y_valid, realized_pnl_valid, tp_r, sl_r, thresholds) if len(probs_valid) > 0 else pd.DataFrame()
    test_df = evaluate_thresholds(probs_test, y_test, realized_pnl_test, tp_r, sl_r, thresholds)

    if not valid_df.empty:
        picked = select_threshold(valid_df, min_valid_trades=CONFIG["min_valid_trades"], smooth_window=CONFIG["threshold_smooth_window"])
        if picked is not None:
            best_threshold = float(picked["threshold"])
        else:
            best_threshold = float(valid_df.loc[valid_df["net_profit_r"].idxmax(), "threshold"])
    else:
        best_threshold = float(test_df.loc[test_df["net_profit_r"].idxmax(), "threshold"]) if not test_df.empty else 0.5

    matched = test_df[test_df["threshold"] == best_threshold]
    if matched.empty and not test_df.empty:
        nearest_idx = (test_df["threshold"] - best_threshold).abs().idxmin()
        test_row = test_df.loc[nearest_idx]
        best_threshold = float(test_row["threshold"])
    elif not matched.empty:
        test_row = matched.iloc[0]
    else:
        test_row = None

    result = {
        "name": name, "ap": ap, "threshold": best_threshold,
        "trades": int(test_row["trades"]) if test_row is not None else 0,
        "win_rate": float(test_row["win_rate"]) if test_row is not None else 0.0,
        "net_profit_r": float(test_row["net_profit_r"]) if test_row is not None else 0.0,
        "avg_r": float(test_row["avg_r"]) if test_row is not None else 0.0,
        "max_drawdown_r": float(test_row["max_drawdown_r"]) if test_row is not None else 0.0,
    }
    return result


def print_comparison(results: list[dict]):
    print("\n" + "=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)
    header = f"{'Model':<12} {'AP':>8} {'Threshold':>10} {'Trades':>8} {'WinRate':>8} {'NetProfit':>10} {'AvgR':>8} {'MaxDD':>8}"
    print(header)
    print("-" * 80)
    for r in results:
        print(f"{r['name']:<12} {r['ap']:>8.4f} {r['threshold']:>10.4f} {r['trades']:>8d} "
              f"{r['win_rate']:>7.2%} {r['net_profit_r']:>10.1f} {r['avg_r']:>8.3f} {r['max_drawdown_r']:>8.1f}")
    print("=" * 80)


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

    df = load_and_label(
        CONFIG["data_path"],
        scanner_arg=CONFIG["scanner"],
        scanner_variant_arg=CONFIG["scanner_variant"],
        label_mode=CONFIG["label_mode"],
        same_bar_policy=CONFIG["same_bar_policy"],
        horizon=CONFIG["horizon"],
        tp_mult=CONFIG["tp_mult"],
        sl_mult=CONFIG["sl_mult"],
    )
    if df.empty:
        print("No data. Exiting.")
        return

    available = [f for f in CONFIG["features"] if f in df.columns]
    df = df.dropna(subset=available).copy()
    splits = make_splits(df, CONFIG["features"])

    # Train all three models
    xgb_model = train_xgb_model(splits)
    lgb_model = train_lgb_model(splits)
    cb_model = train_catboost_model(splits)

    # Get probabilities
    xgb_valid = get_probs(xgb_model, splits["X_valid"], "xgb")
    xgb_test = get_probs(xgb_model, splits["X_test"], "xgb")
    lgb_valid = get_probs(lgb_model, splits["X_valid"], "lgb")
    lgb_test = get_probs(lgb_model, splits["X_test"], "lgb")
    cb_valid = get_probs(cb_model, splits["X_valid"], "catboost")
    cb_test = get_probs(cb_model, splits["X_test"], "catboost")

    # Ensemble = average
    ens_valid = (xgb_valid + lgb_valid + cb_valid) / 3.0
    ens_test = (xgb_test + lgb_test + cb_test) / 3.0

    # Stacking ensemble: LogisticRegression meta-learner on validation OOF predictions
    print("\n" + "=" * 50)
    print("Training Stacking Meta-Learner")
    print("=" * 50)
    meta_X_valid = np.column_stack([xgb_valid, lgb_valid, cb_valid])
    meta_X_test = np.column_stack([xgb_test, lgb_test, cb_test])
    meta_model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    meta_model.fit(meta_X_valid, splits["y_valid"])
    stacked_valid = meta_model.predict_proba(meta_X_valid)[:, 1]
    stacked_test = meta_model.predict_proba(meta_X_test)[:, 1]
    print(f"Meta-learner weights: XGB={meta_model.coef_[0][0]:.3f}, LGB={meta_model.coef_[0][1]:.3f}, CB={meta_model.coef_[0][2]:.3f}")
    # Save meta-learner
    with open("model_meta_learner.pkl", "wb") as f:
        pickle.dump(meta_model, f)
    print("Saved model_meta_learner.pkl")

    # Evaluate all five
    results = [
        evaluate_single("XGBoost", xgb_valid, xgb_test, splits),
        evaluate_single("LightGBM", lgb_valid, lgb_test, splits),
        evaluate_single("CatBoost", cb_valid, cb_test, splits),
        evaluate_single("Ensemble", ens_valid, ens_test, splits),
        evaluate_single("Stacked", stacked_valid, stacked_test, splits),
    ]
    print_comparison(results)

    # Pick best by net_profit_r
    best = max(results, key=lambda r: r["net_profit_r"])
    model_type_map = {"XGBoost": "xgb", "LightGBM": "lgb", "CatBoost": "catboost", "Ensemble": "ensemble", "Stacked": "stacked"}
    best_model_type = model_type_map[best["name"]]

    print(f"\nBest model: {best['name']} (net_profit_r={best['net_profit_r']:.1f})")

    # Update best_config.json
    try:
        with open("best_config.json", "r") as f:
            cfg = json.load(f)
        cfg["model_type"] = best_model_type
        cfg["threshold"] = best["threshold"]
        with open("best_config.json", "w") as f:
            json.dump(cfg, f, indent=2)
        print(f"Updated best_config.json: model_type={best_model_type}, threshold={best['threshold']:.4f}")
    except Exception as e:
        print(f"Warning: could not update best_config.json: {e}")

    # Plot for the best model
    best_probs_map = {"XGBoost": (xgb_valid, xgb_test), "LightGBM": (lgb_valid, lgb_test),
                      "CatBoost": (cb_valid, cb_test), "Ensemble": (ens_valid, ens_test),
                      "Stacked": (stacked_valid, stacked_test)}
    bv, bt = best_probs_map[best["name"]]
    tp_r, sl_r = CONFIG["tp_mult"], CONFIG["sl_mult"]
    realized_pnl_test = splits["df_test"]["realized_pnl_r"].to_numpy(dtype=np.float64) if "realized_pnl_r" in splits["df_test"].columns else None
    thresholds = build_threshold_grid(bv)
    test_df = evaluate_thresholds(bt, splits["y_test"], realized_pnl_test, tp_r, sl_r, thresholds)
    if not test_df.empty:
        plot_curves(test_df, splits["y_test"], bt, bv, best["threshold"])


if __name__ == "__main__":
    main()



