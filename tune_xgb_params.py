"""XGBoost hyperparameter grid search using walk-forward OOS evaluation.

Tests combinations of model hyperparameters while keeping TP/SL/horizon fixed.
Uses the same multi-fold OOS framework as robust_oos_search.py.

Usage: python tune_xgb_params.py
"""
from __future__ import annotations
import argparse, itertools, json, time
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score

from quant_modeling import (
    BASE_FEATURES,
    add_directional_features,
    build_labeling_cache,
    build_labels,
    build_realized_pnl,
)

PURGE_GAP_BARS = 30

# --- Hyperparameter grid ---
PARAM_GRID = {
    "max_depth": [2, 3, 4, 5],
    "learning_rate": [0.003, 0.005, 0.01, 0.02],
    "subsample": [0.5, 0.65, 0.8],
    "colsample_bytree": [0.3, 0.5, 0.7],
    "min_child_weight": [20, 50, 100],
    "gamma": [0.5, 1.0, 2.0],
    "reg_lambda": [1.0, 2.0, 5.0],
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-path", default="dataset_enhanced.parquet")
    p.add_argument("--scanner", default="flow_reversal")
    p.add_argument("--scanner-variant", default="flow_reversal|flow_abs=0.05,obi_abs=0.0,range_hi=0.7,range_lo=0.3,spread_mult=1.0")
    p.add_argument("--label-mode", default="first_touch")
    p.add_argument("--same-bar-policy", default="drop")
    p.add_argument("--horizon", type=int, default=30)
    p.add_argument("--tp", type=float, default=2.0)
    p.add_argument("--sl", type=float, default=0.5)
    p.add_argument("--folds", type=int, default=3)
    p.add_argument("--valid-frac", type=float, default=0.10)
    p.add_argument("--test-frac", type=float, default=0.10)
    p.add_argument("--min-train-size", type=int, default=500)
    p.add_argument("--min-valid-trades", type=int, default=80)
    p.add_argument("--smooth-window", type=int, default=3)
    p.add_argument("--max-combos", type=int, default=200,
                   help="Max random combos to test (0=all)")
    p.add_argument("--output", default="xgb_param_search.csv")
    return p.parse_args()


def filter_by_scanner(df, scanner_arg, variant_arg):
    out = df
    if scanner_arg != "all" and "scanner_name" in out.columns:
        names = [n.strip() for n in scanner_arg.split(",") if n.strip()]
        out = out[out["scanner_name"].isin(names)]
    if variant_arg != "all" and "scanner_variant" in out.columns:
        vs = out["scanner_variant"].astype(str)
        exact = variant_arg.strip()
        if (vs == exact).any():
            out = out[vs == exact]
    return out.copy()


def build_fold_splits(n, folds, purge_gap, valid_frac, test_frac, min_train_size):
    valid_size = max(int(n * valid_frac), 100)
    test_size = max(int(n * test_frac), 100)
    splits = []
    for offset in range(folds - 1, -1, -1):
        test_end = n - offset * test_size
        test_start = test_end - test_size
        valid_end = test_start - purge_gap
        valid_start = valid_end - valid_size
        train_end = valid_start - purge_gap
        if min(train_end, valid_start, valid_end, test_start) < 0:
            continue
        if train_end < min_train_size:
            continue
        splits.append((train_end, valid_start, valid_end, test_start, test_end))
    return splits


def build_threshold_grid(probs):
    probs = probs[np.isfinite(probs)]
    if probs.size == 0:
        return np.array([], dtype=np.float64)
    quantiles = np.linspace(0.10, 0.995, 80)
    thresholds = np.unique(np.round(np.quantile(probs, quantiles), 4))
    return thresholds[(thresholds > 0.0) & (thresholds < 1.0)]


def evaluate_one_config(params, X, y, realized_pnl, splits, min_valid_trades, smooth_window):
    """Train on each fold with given params, return aggregated OOS metrics."""
    fold_results = []
    for train_end, vs, ve, ts, te in splits:
        X_tr, y_tr = X.iloc[:train_end], y.iloc[:train_end]
        X_va, y_va = X.iloc[vs:ve], y.iloc[vs:ve]
        X_te, y_te = X.iloc[ts:te], y.iloc[ts:te]

        if min(len(X_va), len(X_te)) == 0:
            return None
        if y_tr.nunique() < 2 or y_va.nunique() < 2 or y_te.nunique() < 2:
            return None

        pos_ratio = float(y_tr.mean())
        sw = np.ones(len(y_tr), dtype=np.float32)
        if 0 < pos_ratio < 1:
            sw = np.where(y_tr.to_numpy() == 1, (1 - pos_ratio) / pos_ratio, 1.0).astype(np.float32)

        model = xgb.XGBClassifier(
            n_estimators=1500,
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            subsample=params["subsample"],
            colsample_bytree=params["colsample_bytree"],
            colsample_bylevel=0.7,
            gamma=params["gamma"],
            min_child_weight=params["min_child_weight"],
            reg_alpha=0.5,
            reg_lambda=params["reg_lambda"],
            max_delta_step=1,
            objective="binary:logistic",
            eval_metric="aucpr",
            tree_method="hist",
            n_jobs=-1,
            random_state=42,
            early_stopping_rounds=120,
        )
        model.fit(X_tr, y_tr, sample_weight=sw,
                  eval_set=[(X_va, y_va)], verbose=False)

        va_probs = model.predict_proba(X_va)[:, 1]
        te_probs = model.predict_proba(X_te)[:, 1]

        # Select threshold on validation
        thresholds = build_threshold_grid(va_probs)
        if len(thresholds) == 0:
            return None

        rpnl_va = realized_pnl.iloc[vs:ve].to_numpy(dtype=np.float64)
        rpnl_te = realized_pnl.iloc[ts:te].to_numpy(dtype=np.float64)

        best_thresh, best_profit = None, -np.inf
        for th in thresholds:
            mask = va_probs > th
            trades = int(mask.sum())
            if trades < min_valid_trades:
                continue
            profit = float(np.nansum(rpnl_va[mask]))
            if profit > best_profit:
                best_profit = profit
                best_thresh = th

        if best_thresh is None:
            return None

        # Evaluate on test
        te_mask = te_probs > best_thresh
        te_trades = int(te_mask.sum())
        if te_trades == 0:
            return None
        te_pnl = rpnl_te[te_mask]
        te_profit = float(np.nansum(te_pnl))
        te_avg_r = float(np.nanmean(te_pnl))
        te_wr = float(np.mean(te_pnl > 0))
        try:
            te_ap = float(average_precision_score(y_te, te_probs))
        except Exception:
            te_ap = 0.0

        # Score spread (indicator of overfitting)
        score_std = float(te_probs.std())
        n_trees = model.best_iteration if hasattr(model, 'best_iteration') and model.best_iteration else 1500

        fold_results.append({
            "threshold": best_thresh,
            "test_trades": te_trades,
            "test_profit_r": te_profit,
            "test_avg_r": te_avg_r,
            "test_wr": te_wr,
            "test_ap": te_ap,
            "score_std": score_std,
            "n_trees": n_trees,
        })

    if len(fold_results) < len(splits):
        return None

    # Aggregate
    total_trades = sum(f["test_trades"] for f in fold_results)
    total_profit = sum(f["test_profit_r"] for f in fold_results)
    profitable_folds = sum(1 for f in fold_results if f["test_profit_r"] > 0)
    return {
        "profitable_fold_ratio": profitable_folds / len(fold_results),
        "total_test_trades": total_trades,
        "total_test_profit_r": total_profit,
        "avg_r_per_trade": total_profit / total_trades if total_trades > 0 else 0,
        "mean_test_ap": np.mean([f["test_ap"] for f in fold_results]),
        "mean_test_wr": np.mean([f["test_wr"] for f in fold_results]),
        "mean_score_std": np.mean([f["score_std"] for f in fold_results]),
        "mean_n_trees": np.mean([f["n_trees"] for f in fold_results]),
        "median_threshold": np.median([f["threshold"] for f in fold_results]),
    }


def main():
    args = parse_args()

    df = pd.read_parquet(args.data_path)
    if "scanner_triggered" in df.columns:
        df = df[df["scanner_triggered"] == 1].copy()
    df = filter_by_scanner(df, args.scanner, args.scanner_variant)
    if "event_dir" not in df.columns:
        df["event_dir"] = -1
    df = add_directional_features(df)

    features = [f for f in BASE_FEATURES if f in df.columns]
    cache = build_labeling_cache(df)
    label_info = build_labels(cache, args.horizon, args.tp, args.sl,
                              args.label_mode, args.same_bar_policy)
    realized = build_realized_pnl(cache, args.horizon, args.tp, args.sl,
                                  args.label_mode, args.same_bar_policy)
    valid = label_info["valid"] & realized.notna()
    valid_pos = np.flatnonzero(valid.to_numpy(dtype=bool))
    y = label_info["label"].iloc[valid_pos].reset_index(drop=True).astype(np.int8)
    X = df[features].astype(np.float32).iloc[valid_pos].reset_index(drop=True)
    pnl = realized.iloc[valid_pos].reset_index(drop=True)

    X = X.dropna()
    common = X.index
    y = y.loc[common]
    pnl = pnl.loc[common]

    print(f"Data: {len(X)} events, {y.mean():.2%} positive rate")

    splits = build_fold_splits(len(X), args.folds, PURGE_GAP_BARS,
                               args.valid_frac, args.test_frac, args.min_train_size)
    if not splits:
        print("No valid splits")
        return

    # Generate param combos
    keys = sorted(PARAM_GRID.keys())
    all_combos = list(itertools.product(*(PARAM_GRID[k] for k in keys)))
    np.random.seed(42)
    if args.max_combos > 0 and len(all_combos) > args.max_combos:
        idx = np.random.choice(len(all_combos), args.max_combos, replace=False)
        combos = [all_combos[i] for i in sorted(idx)]
    else:
        combos = all_combos
    print(f"Testing {len(combos)} / {len(all_combos)} param combos, {len(splits)} folds each\n")

    results = []
    for i, vals in enumerate(combos):
        params = dict(zip(keys, vals))
        t0 = time.time()
        agg = evaluate_one_config(params, X, y, pnl, splits,
                                  args.min_valid_trades, args.smooth_window)
        elapsed = time.time() - t0
        if agg is None:
            continue
        row = {**params, **agg, "elapsed_s": round(elapsed, 1)}
        results.append(row)
        tag = "OK" if agg["total_test_profit_r"] > 0 else "--"
        print(f"[{i+1}/{len(combos)}] {tag} depth={params['max_depth']} lr={params['learning_rate']} "
              f"sub={params['subsample']} col={params['colsample_bytree']} "
              f"mcw={params['min_child_weight']} gamma={params['gamma']} lam={params['reg_lambda']} "
              f"| profit={agg['total_test_profit_r']:+.1f}R "
              f"trades={agg['total_test_trades']} "
              f"avg_r={agg['avg_r_per_trade']:.4f} "
              f"wr={agg['mean_test_wr']:.2%} "
              f"score_std={agg['mean_score_std']:.4f} "
              f"trees={agg['mean_n_trees']:.0f} "
              f"({elapsed:.1f}s)")

    if not results:
        print("No valid results")
        return

    res_df = pd.DataFrame(results).sort_values(
        ["profitable_fold_ratio", "total_test_profit_r", "avg_r_per_trade"],
        ascending=[False, False, False],
    )
    res_df.to_csv(args.output, index=False)
    print(f"\nSaved {len(res_df)} results to {args.output}")

    print("\n" + "=" * 80)
    print("TOP 10 CONFIGURATIONS")
    print("=" * 80)
    top = res_df.head(10)
    show_cols = ["max_depth", "learning_rate", "subsample", "colsample_bytree",
                 "min_child_weight", "gamma", "reg_lambda",
                 "profitable_fold_ratio", "total_test_trades",
                 "total_test_profit_r", "avg_r_per_trade", "mean_test_wr",
                 "mean_score_std", "mean_n_trees"]
    print(top[show_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Best config
    best = res_df.iloc[0]
    print(f"\nBEST: depth={int(best['max_depth'])} lr={best['learning_rate']} "
          f"sub={best['subsample']} col={best['colsample_bytree']} "
          f"mcw={int(best['min_child_weight'])} gamma={best['gamma']} lam={best['reg_lambda']}")
    print(f"  profit={best['total_test_profit_r']:+.1f}R, "
          f"trades={int(best['total_test_trades'])}, "
          f"avg_r={best['avg_r_per_trade']:.4f}, "
          f"score_std={best['mean_score_std']:.4f}")


if __name__ == "__main__":
    main()
