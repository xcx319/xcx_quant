from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

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


@dataclass
class FoldSplit:
    train_end: int
    valid_start: int
    valid_end: int
    test_start: int
    test_end: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="dataset_enhanced.parquet")
    parser.add_argument("--scanner", default="all", help="Filter scanner_name by one or more comma-separated names.")
    parser.add_argument("--scanner-variant", default="all", help="Filter exact scanner_variant by one or more ';'-separated values.")
    parser.add_argument("--label-mode", choices=["window_tp", "first_touch"], default="first_touch")
    parser.add_argument("--same-bar-policy", choices=["drop", "neutral", "tp_first", "sl_first"], default="drop")
    parser.add_argument("--horizons", default="2,3,5,10,15")
    parser.add_argument("--tp-values", default="1.5,2.0,2.5,3.0")
    parser.add_argument("--sl-values", default="0.5,1.0,1.5")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--valid-frac", type=float, default=0.10)
    parser.add_argument("--test-frac", type=float, default=0.10)
    parser.add_argument("--min-train-size", type=int, default=500)
    parser.add_argument("--min-valid-trades", type=int, default=80)
    parser.add_argument("--min-test-trades", type=int, default=80)
    parser.add_argument("--min-total-test-trades", type=int, default=240)
    parser.add_argument("--min-events", type=int, default=500)
    parser.add_argument("--smooth-window", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--output", default="robust_oos_search_results.csv")
    return parser.parse_args()


def parse_num_list(raw: str, cast):
    return [cast(item.strip()) for item in raw.split(",") if item.strip()]


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


def build_threshold_grid(probs: np.ndarray) -> np.ndarray:
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs[np.isfinite(probs)]
    if probs.size == 0:
        return np.array([], dtype=np.float64)

    quantiles = np.linspace(0.10, 0.995, 80)
    thresholds = np.unique(np.round(np.quantile(probs, quantiles), 4))
    thresholds = thresholds[(thresholds > 0.0) & (thresholds < 1.0)]
    return thresholds


def build_fold_splits(
    n: int,
    folds: int,
    purge_gap: int,
    valid_frac: float,
    test_frac: float,
    min_train_size: int,
) -> list[FoldSplit]:
    valid_size = max(int(n * valid_frac), 100)
    test_size = max(int(n * test_frac), 100)
    splits: list[FoldSplit] = []
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
        splits.append(
            FoldSplit(
                train_end=train_end,
                valid_start=valid_start,
                valid_end=valid_end,
                test_start=test_start,
                test_end=test_end,
            )
        )
    return splits


def fit_fold_model(X_train: pd.DataFrame, y_train: pd.Series, X_valid: pd.DataFrame, y_valid: pd.Series) -> xgb.XGBClassifier | None:
    if len(X_train) == 0 or len(X_valid) == 0 or y_train.nunique() < 2:
        return None

    pos_ratio = float(y_train.mean())
    sample_weight = np.ones(len(y_train), dtype=np.float32)
    if 0 < pos_ratio < 1:
        pos_weight = (1.0 - pos_ratio) / pos_ratio
        sample_weight = np.where(y_train.to_numpy() == 1, pos_weight, 1.0).astype(np.float32)

    model = xgb.XGBClassifier(
        n_estimators=1500,
        max_depth=4,
        learning_rate=0.005,
        subsample=0.65,
        colsample_bytree=0.5,
        colsample_bylevel=0.7,
        gamma=0.5,
        min_child_weight=20,
        reg_alpha=0.5,
        reg_lambda=2.0,
        max_delta_step=1,
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
        early_stopping_rounds=120 if len(X_valid) > 0 and y_valid.nunique() > 1 else None,
    )
    fit_kwargs = {
        "X": X_train,
        "y": y_train,
        "sample_weight": sample_weight,
        "verbose": False,
    }
    if len(X_valid) > 0 and y_valid.nunique() > 1:
        fit_kwargs["eval_set"] = [(X_valid, y_valid)]
    model.fit(**fit_kwargs)
    return model


def evaluate_thresholds(
    probs: np.ndarray,
    y_true: pd.Series,
    realized_pnl: np.ndarray,
    thresholds: np.ndarray,
) -> pd.DataFrame:
    y_arr = y_true.to_numpy(dtype=np.int8, copy=False)
    out: list[dict[str, float | int]] = []
    for thresh in thresholds:
        mask = probs > thresh
        trades = int(mask.sum())
        if trades == 0:
            continue

        trade_pnl = realized_pnl[mask]
        positives = int((trade_pnl > 0).sum())
        net_profit_r = float(np.nansum(trade_pnl))
        avg_r = float(np.nanmean(trade_pnl))
        win_rate = float(np.mean(trade_pnl > 0))
        tp = int(((mask) & (y_arr == 1)).sum())
        fn = int(((~mask) & (y_arr == 1)).sum())
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

        out.append(
            {
                "threshold": float(thresh),
                "trades": trades,
                "wins": positives,
                "win_rate": win_rate,
                "avg_r": avg_r,
                "net_profit_r": net_profit_r,
                "recall": recall,
            }
        )
    return pd.DataFrame(out)


def select_threshold(valid_df: pd.DataFrame, min_valid_trades: int, smooth_window: int) -> pd.Series | None:
    if valid_df.empty:
        return None

    cand = valid_df[valid_df["trades"] >= min_valid_trades].sort_values("threshold").copy()
    if cand.empty:
        return None

    if smooth_window > 1:
        cand["smoothed_profit_r"] = (
            cand["net_profit_r"].rolling(window=smooth_window, min_periods=1, center=True).mean()
        )
        cand["smoothed_avg_r"] = cand["avg_r"].rolling(window=smooth_window, min_periods=1, center=True).mean()
    else:
        cand["smoothed_profit_r"] = cand["net_profit_r"]
        cand["smoothed_avg_r"] = cand["avg_r"]

    picked = cand.sort_values(
        by=["smoothed_profit_r", "smoothed_avg_r", "trades"],
        ascending=[False, False, False],
    ).iloc[0]
    return picked


def evaluate_fold(
    X: pd.DataFrame,
    y: pd.Series,
    realized_pnl: pd.Series,
    split: FoldSplit,
    min_valid_trades: int,
    min_test_trades: int,
    smooth_window: int,
) -> dict | None:
    X_train = X.iloc[: split.train_end]
    y_train = y.iloc[: split.train_end]
    X_valid = X.iloc[split.valid_start : split.valid_end]
    y_valid = y.iloc[split.valid_start : split.valid_end]
    X_test = X.iloc[split.test_start : split.test_end]
    y_test = y.iloc[split.test_start : split.test_end]

    if min(len(X_valid), len(X_test)) == 0:
        return None
    if min(y_train.sum(), len(y_train) - y_train.sum()) < 10:
        return None
    if min(y_valid.sum(), len(y_valid) - y_valid.sum()) < 5:
        return None
    if min(y_test.sum(), len(y_test) - y_test.sum()) < 5:
        return None

    model = fit_fold_model(X_train, y_train, X_valid, y_valid)
    if model is None:
        return None

    valid_probs = model.predict_proba(X_valid)[:, 1]
    test_probs = model.predict_proba(X_test)[:, 1]
    thresholds = build_threshold_grid(valid_probs)
    if len(thresholds) == 0:
        return None

    realized_valid = realized_pnl.iloc[split.valid_start : split.valid_end].to_numpy(dtype=np.float64, copy=False)
    realized_test = realized_pnl.iloc[split.test_start : split.test_end].to_numpy(dtype=np.float64, copy=False)
    valid_df = evaluate_thresholds(valid_probs, y_valid, realized_valid, thresholds)
    picked = select_threshold(valid_df, min_valid_trades=min_valid_trades, smooth_window=smooth_window)
    if picked is None:
        return None

    test_df = evaluate_thresholds(test_probs, y_test, realized_test, np.array([picked["threshold"]], dtype=np.float64))
    if test_df.empty:
        return None

    test_row = test_df.iloc[0]
    if int(test_row["trades"]) < min_test_trades:
        return None

    try:
        ap = float(average_precision_score(y_test, test_probs))
    except Exception:
        ap = 0.0

    return {
        "threshold": float(picked["threshold"]),
        "valid_trades": int(picked["trades"]),
        "valid_net_profit_r": float(picked["net_profit_r"]),
        "valid_smoothed_profit_r": float(picked["smoothed_profit_r"]),
        "test_trades": int(test_row["trades"]),
        "test_win_rate": float(test_row["win_rate"]),
        "test_avg_r": float(test_row["avg_r"]),
        "test_net_profit_r": float(test_row["net_profit_r"]),
        "test_recall": float(test_row["recall"]),
        "test_ap": ap,
    }


def aggregate_fold_rows(fold_rows: list[dict]) -> dict[str, float | int]:
    thresholds = np.array([row["threshold"] for row in fold_rows], dtype=np.float64)
    total_trades = int(sum(int(row["test_trades"]) for row in fold_rows))
    total_profit = float(sum(float(row["test_net_profit_r"]) for row in fold_rows))
    profitable_folds = int(sum(float(row["test_net_profit_r"]) > 0 for row in fold_rows))
    ap_values = [float(row["test_ap"]) for row in fold_rows]
    avg_r = total_profit / total_trades if total_trades > 0 else np.nan
    return {
        "folds_used": len(fold_rows),
        "profitable_folds": profitable_folds,
        "profitable_fold_ratio": profitable_folds / len(fold_rows) if fold_rows else 0.0,
        "total_test_trades": total_trades,
        "min_fold_test_trades": int(min(int(row["test_trades"]) for row in fold_rows)),
        "median_threshold": float(np.median(thresholds)),
        "threshold_std": float(np.std(thresholds)),
        "test_ap_mean": float(np.mean(ap_values)) if ap_values else 0.0,
        "test_ap_std": float(np.std(ap_values)) if ap_values else 0.0,
        "total_test_net_profit_r": total_profit,
        "avg_r_per_trade": float(avg_r),
        "mean_test_win_rate": float(np.mean([row["test_win_rate"] for row in fold_rows])),
        "mean_test_recall": float(np.mean([row["test_recall"] for row in fold_rows])),
        "mean_valid_profit_r": float(np.mean([row["valid_net_profit_r"] for row in fold_rows])),
    }


def main() -> None:
    args = parse_args()
    horizons = parse_num_list(args.horizons, int)
    tp_values = parse_num_list(args.tp_values, float)
    sl_values = parse_num_list(args.sl_values, float)

    try:
        df = pd.read_parquet(args.data_path)
    except FileNotFoundError:
        print("Run pipeline first! dataset_enhanced.parquet not found.")
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
    features = [feature for feature in BASE_FEATURES if feature in df.columns]
    if len(features) < 6:
        print("Too few usable features.")
        return

    results: list[dict] = []
    candidate_count = 0
    for scanner_variant, df_variant in df.groupby("scanner_variant", sort=False):
        df_variant = df_variant.dropna(subset=features + ["close", "atr", "high", "low"]).reset_index(drop=True)
        if len(df_variant) < args.min_events:
            continue

        scanner_name = str(df_variant["scanner_name"].iloc[0]) if "scanner_name" in df_variant.columns else "unknown"
        scanner_params = str(df_variant["scanner_params"].iloc[0]) if "scanner_params" in df_variant.columns else ""
        X = df_variant[features].astype(np.float32)
        cache = build_labeling_cache(df_variant)

        for horizon in horizons:
            for tp in tp_values:
                for sl in sl_values:
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
                    if int(valid.sum()) < args.min_events:
                        continue

                    valid_pos = np.flatnonzero(valid.to_numpy(dtype=bool, copy=False))
                    y = label_info["label"].iloc[valid_pos].reset_index(drop=True).astype(np.int8)
                    if y.mean() < 0.05 or y.mean() > 0.60:
                        continue
                    X_valid = X.iloc[valid_pos].reset_index(drop=True)
                    pnl_valid = realized.iloc[valid_pos].reset_index(drop=True)

                    splits = build_fold_splits(
                        n=len(X_valid),
                        folds=args.folds,
                        purge_gap=PURGE_GAP_BARS,
                        valid_frac=args.valid_frac,
                        test_frac=args.test_frac,
                        min_train_size=args.min_train_size,
                    )
                    if not splits:
                        continue

                    fold_rows: list[dict] = []
                    for split in splits:
                        fold_row = evaluate_fold(
                            X=X_valid,
                            y=y,
                            realized_pnl=pnl_valid,
                            split=split,
                            min_valid_trades=args.min_valid_trades,
                            min_test_trades=args.min_test_trades,
                            smooth_window=args.smooth_window,
                        )
                        if fold_row is not None:
                            fold_rows.append(fold_row)

                    if len(fold_rows) < args.folds:
                        continue

                    agg = aggregate_fold_rows(fold_rows)
                    if int(agg["total_test_trades"]) < args.min_total_test_trades:
                        continue

                    candidate_count += 1
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
                        **agg,
                        "fold_thresholds": json.dumps([round(float(item["threshold"]), 4) for item in fold_rows]),
                        "fold_test_trades": json.dumps([int(item["test_trades"]) for item in fold_rows]),
                        "fold_test_profit_r": json.dumps([round(float(item["test_net_profit_r"]), 4) for item in fold_rows]),
                    }
                    results.append(row)

    if not results:
        print("No robust OOS candidates found under current constraints.")
        return

    result_df = pd.DataFrame(results).sort_values(
        by=[
            "profitable_fold_ratio",
            "total_test_net_profit_r",
            "avg_r_per_trade",
            "test_ap_mean",
            "total_test_trades",
        ],
        ascending=[False, False, False, False, False],
    )
    result_df.to_csv(args.output, index=False)

    print(f"Evaluated {candidate_count} robust candidates. Saved to {args.output}")
    print(
        result_df.head(args.top_k).loc[
            :,
            [
                "scanner_name",
                "events",
                "h",
                "tp",
                "sl",
                "folds_used",
                "profitable_fold_ratio",
                "total_test_trades",
                "total_test_net_profit_r",
                "avg_r_per_trade",
                "test_ap_mean",
                "median_threshold",
                "threshold_std",
            ],
        ].to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )


if __name__ == "__main__":
    main()
