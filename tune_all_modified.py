from __future__ import annotations

import argparse
import json
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score

from quant_modeling import BASE_FEATURES, add_directional_features, build_labeling_cache, build_labels

warnings.filterwarnings("ignore")

PURGE_GAP_BARS = 30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="dataset_enhanced.parquet")
    parser.add_argument("--scanner", default="all", help="Filter scanner_name by one or more comma-separated names.")
    parser.add_argument("--scanner-variant", default="all", help="Filter exact scanner_variant by one or more comma-separated values.")
    parser.add_argument("--label-mode", choices=["window_tp", "first_touch"], default="first_touch")
    parser.add_argument("--same-bar-policy", choices=["drop", "neutral", "tp_first", "sl_first"], default="drop")
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
            variants = [name.strip() for name in scanner_variant_arg.split(";") if name.strip()]
            out = out[variant_series.isin(variants)]
    return out.copy()


def quick_evaluate(X: pd.DataFrame, y: pd.Series) -> float:
    if y.sum() < 10 or (len(y) - y.sum()) < 10:
        return 0.0

    n = len(X)
    train_end = int(n * 0.70)
    valid_start = train_end + PURGE_GAP_BARS
    valid_end = int(n * 0.85)
    test_start = valid_end + PURGE_GAP_BARS

    if test_start >= n or valid_start >= valid_end:
        return 0.0

    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    X_valid, y_valid = X.iloc[valid_start:valid_end], y.iloc[valid_start:valid_end]
    X_test, y_test = X.iloc[test_start:], y.iloc[test_start:]

    if y_train.nunique() < 2 or y_test.nunique() < 2:
        return 0.0

    pos = float(y_train.sum())
    neg = float(len(y_train) - pos)
    scale_pos_weight = (neg / pos) if pos > 0 else 1.0

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.01,
        subsample=0.65,
        colsample_bytree=0.5,
        gamma=0.5,
        min_child_weight=20,
        reg_alpha=0.5,
        reg_lambda=2.0,
        max_delta_step=1,
        n_jobs=-1,
        eval_metric="aucpr",
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        early_stopping_rounds=40 if len(X_valid) > 0 and y_valid.nunique() > 1 else None,
    )
    fit_kwargs = {"X": X_train, "y": y_train, "verbose": False}
    if len(X_valid) > 0 and y_valid.nunique() > 1:
        fit_kwargs["eval_set"] = [(X_valid, y_valid)]
    model.fit(**fit_kwargs)

    probs = model.predict_proba(X_test)[:, 1]
    try:
        return float(average_precision_score(y_test, probs))
    except Exception:
        return 0.0


def main():
    args = parse_args()

    try:
        df = pd.read_parquet(args.data_path)
    except FileNotFoundError:
        print("Run pipeline first! (dataset_enhanced.parquet not found)")
        return

    if "scanner_triggered" in df.columns:
        df = df[df["scanner_triggered"] == 1].copy()
    df = filter_by_scanner(df, args.scanner, args.scanner_variant)
    if df.empty:
        print(f"No rows left after scanner filter: scanner={args.scanner}, variant={args.scanner_variant}")
        return

    if "event_dir" not in df.columns:
        df["event_dir"] = -1

    df = add_directional_features(df)
    cache = build_labeling_cache(df)

    features = [feature for feature in BASE_FEATURES if feature in df.columns]
    missing = [feature for feature in BASE_FEATURES if feature not in df.columns]

    if len(features) < 6:
        print("Too few usable features. Available columns:")
        print(list(df.columns)[:80])
        print(f"\nMissing: {missing[:30]}")
        return

    if missing:
        print(f"[Info] Missing {len(missing)} features (OK): {missing[:20]} ...")

    df = df.dropna(subset=features + ["close", "atr", "high", "low"]).copy()
    X = df[features].astype(np.float32)
    cache = build_labeling_cache(df)

    horizons = [2, 3, 4, 5, 10, 15, 30]
    tp_opts = [1.5, 2.0, 2.5, 3.0]
    sl_opts = [0.5, 1.0, 1.5]

    results = []

    print(f"Scanning {len(df)} events with {len(features)} features...")
    print(f"{'Horizon':<8} | {'TP':<4} | {'SL':<4} | {'Win%':<8} | {'AP':<8}")
    print("-" * 55)

    for horizon in horizons:
        for tp in tp_opts:
            for sl in sl_opts:
                if tp < sl:
                    continue

                label_info = build_labels(
                    cache=cache,
                    horizon=horizon,
                    tp_mult=tp,
                    sl_mult=sl,
                    label_mode=args.label_mode,
                    same_bar_policy=args.same_bar_policy,
                )
                valid = label_info["valid"]
                y = label_info["label"][valid].astype(np.int8)
                if len(y) < len(X) * 0.6:
                    continue

                X_valid = X.loc[y.index]
                win_rate = float(y.mean())
                if win_rate < 0.05 or win_rate > 0.60:
                    continue

                ap = quick_evaluate(X_valid, y)
                if ap > 0.20:
                    print(f"{horizon:<8} | {tp:<4} | {sl:<4} | {win_rate:.2%}   | {ap:.4f}")

                results.append(
                    {
                        "h": horizon,
                        "tp": tp,
                        "sl": sl,
                        "win": win_rate,
                        "ap": ap,
                        "label_mode": args.label_mode,
                        "same_bar_policy": args.same_bar_policy,
                    }
                )

    if not results:
        print("No valid configs.")
        return

    best = max(results, key=lambda item: item["ap"])

    print("\n" + "=" * 55)
    print("BEST CONFIGURATION")
    print(f"Horizon: {best['h']}m, TP: {best['tp']}R, SL: {best['sl']}R")
    print(f"Win%: {best['win']:.2%}, AP: {best['ap']:.4f}")
    print("=" * 55)

    with open("best_config.json", "w") as f:
        json.dump(best, f, indent=2)


if __name__ == "__main__":
    main()
