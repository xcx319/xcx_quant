from __future__ import annotations

import argparse

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="dataset_enhanced.parquet")
    parser.add_argument("--scanner", default="all", help="Filter scanner_name by one or more comma-separated names.")
    parser.add_argument("--label-mode", choices=["window_tp", "first_touch"], default="first_touch")
    parser.add_argument("--same-bar-policy", choices=["drop", "neutral", "tp_first", "sl_first"], default="drop")
    parser.add_argument("--min-events", type=int, default=80)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--output", default="scanner_grid_search_results.csv")
    return parser.parse_args()


def filter_by_scanner(df: pd.DataFrame, scanner_arg: str) -> pd.DataFrame:
    if scanner_arg == "all" or "scanner_name" not in df.columns:
        return df
    names = [name.strip() for name in scanner_arg.split(",") if name.strip()]
    return df[df["scanner_name"].isin(names)].copy()


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
    return float(average_precision_score(y_test, probs))


def main():
    args = parse_args()

    try:
        df = pd.read_parquet(args.data_path)
    except FileNotFoundError:
        print("Run pipeline first! (dataset_enhanced.parquet not found)")
        return

    df = filter_by_scanner(df, args.scanner)
    if df.empty:
        print(f"No rows left after scanner filter: {args.scanner}")
        return

    if "scanner_variant" not in df.columns:
        print("dataset_enhanced.parquet does not contain scanner_variant. Re-run pipeline with the new scanner framework.")
        return

    df = add_directional_features(df)
    features = [feature for feature in BASE_FEATURES if feature in df.columns]
    if len(features) < 6:
        print("Too few usable features.")
        return

    horizons = [2, 3, 5, 10, 15]
    tp_opts = [1.5, 2.0, 2.5, 3.0]
    sl_opts = [0.5, 1.0, 1.5]
    results: list[dict] = []

    for scanner_variant, df_variant in df.groupby("scanner_variant", sort=False):
        if len(df_variant) < args.min_events:
            continue

        df_variant = df_variant.dropna(subset=features + ["close", "atr", "high", "low"]).copy()
        if len(df_variant) < args.min_events:
            continue

        X = df_variant[features].astype(np.float32)
        cache = build_labeling_cache(df_variant)
        scanner_name = str(df_variant["scanner_name"].iloc[0]) if "scanner_name" in df_variant.columns else "unknown"
        scanner_params = str(df_variant["scanner_params"].iloc[0]) if "scanner_params" in df_variant.columns else ""

        best_row = None
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
                    realized = build_realized_pnl(
                        cache=cache,
                        horizon=horizon,
                        tp_mult=tp,
                        sl_mult=sl,
                        label_mode=args.label_mode,
                        same_bar_policy=args.same_bar_policy,
                    )

                    valid = label_info["valid"] & realized.notna()
                    if valid.sum() < args.min_events:
                        continue

                    y = label_info["label"][valid].astype(np.int8)
                    X_valid = X.loc[y.index]
                    pnl_valid = realized.loc[y.index]

                    if y.mean() < 0.05 or y.mean() > 0.60:
                        continue

                    ap = quick_evaluate(X_valid, y)
                    row = {
                        "scanner_name": scanner_name,
                        "scanner_variant": scanner_variant,
                        "scanner_params": scanner_params,
                        "events": int(len(y)),
                        "h": horizon,
                        "tp": tp,
                        "sl": sl,
                        "label_mode": args.label_mode,
                        "same_bar_policy": args.same_bar_policy,
                        "win_rate": float(y.mean()),
                        "ap": float(ap),
                        "mean_realized_r": float(pnl_valid.mean()),
                        "p05_realized_r": float(pnl_valid.quantile(0.05)),
                        "worst_realized_r": float(pnl_valid.min()),
                    }
                    if best_row is None or (row["ap"], row["mean_realized_r"]) > (best_row["ap"], best_row["mean_realized_r"]):
                        best_row = row

        if best_row is not None:
            results.append(best_row)

    if not results:
        print("No valid scanner variants found.")
        return

    result_df = pd.DataFrame(results).sort_values(
        by=["ap", "mean_realized_r", "events"],
        ascending=[False, False, False],
    )
    result_df.to_csv(args.output, index=False)

    print(f"Saved {len(result_df)} rows to {args.output}")
    print(result_df.head(args.top_k).to_string(index=False))


if __name__ == "__main__":
    main()
