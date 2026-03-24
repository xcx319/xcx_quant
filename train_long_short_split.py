"""Train separate XGBoost models for long and short signals, compare with combined model.

Supports independent h/tp/sl per direction via --long-h/--long-tp/--long-sl and --short-h/--short-tp/--short-sl.
"""
from __future__ import annotations

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score

from quant_modeling import BASE_FEATURES, add_directional_features, build_labeling_cache, build_labels, build_realized_pnl

with open("best_config.json", "r") as f:
    BEST = json.load(f)

PURGE_GAP_BARS = 30
PLOT_DIR = "./plots"
os.makedirs(PLOT_DIR, exist_ok=True)

XGB_PARAMS = dict(
    n_estimators=1500, max_depth=4, learning_rate=0.003,
    subsample=0.5, colsample_bytree=0.7, colsample_bylevel=0.7,
    gamma=0.5, min_child_weight=50, reg_alpha=0.5, reg_lambda=2.0,
    max_delta_step=1, objective="binary:logistic", eval_metric="aucpr",
    tree_method="hist", n_jobs=-1, random_state=42, early_stopping_rounds=120,
)


def load_data(data_path: str, h: int, tp: float, sl: float) -> pd.DataFrame:
    df = pd.read_parquet(data_path)
    if "event_dir" not in df.columns:
        df["event_dir"] = -1
    variant = str(BEST.get("scanner_variant", "all"))
    if variant != "all" and "scanner_variant" in df.columns:
        df = df[df["scanner_variant"].astype(str) == variant].copy()
    df = add_directional_features(df)
    cache = build_labeling_cache(df)
    lm = str(BEST.get("label_mode", "first_touch"))
    sbp = str(BEST.get("same_bar_policy", "drop"))
    label_info = build_labels(cache=cache, horizon=h, tp_mult=tp, sl_mult=sl, label_mode=lm, same_bar_policy=sbp)
    df["label"] = label_info["label"]
    df["label_valid"] = label_info["valid"]
    df["realized_pnl_r"] = build_realized_pnl(cache=cache, horizon=h, tp_mult=tp, sl_mult=sl, label_mode=lm, same_bar_policy=sbp)
    df = df[df["label_valid"]].copy()
    df["label"] = df["label"].astype(np.int8)
    df = df[df["realized_pnl_r"].notna()].copy()
    pos_ratio = df["label"].mean()
    if 0 < pos_ratio < 1:
        df["sample_weight"] = np.where(df["label"] == 1, (1 - pos_ratio) / pos_ratio, 1.0).astype(np.float32)
    else:
        df["sample_weight"] = 1.0
    return df


def get_features(df: pd.DataFrame) -> list[str]:
    return [f for f in BASE_FEATURES if f in df.columns]


def split_data(df: pd.DataFrame, feats: list[str]):
    X = df[feats].astype(np.float32)
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
        "rpnl_test": rpnl[test_start:], "rpnl_valid": rpnl[valid_start:valid_end],
        "dirs_test": df["event_dir"].iloc[test_start:].to_numpy(),
    }


def train_model(s: dict) -> xgb.XGBClassifier:
    model = xgb.XGBClassifier(**XGB_PARAMS)
    model.fit(
        s["X_train"], s["y_train"], sample_weight=s["w_train"], verbose=50,
        eval_set=[(s["X_train"], s["y_train"]), (s["X_valid"], s["y_valid"])],
    )
    return model


def select_threshold(probs, y, rpnl, min_trades=80, smooth=3):
    thresholds = np.unique(np.round(np.quantile(probs[np.isfinite(probs)], np.linspace(0.10, 0.995, 80)), 4))
    thresholds = thresholds[(thresholds > 0) & (thresholds < 1)]
    best_thr, best_profit = 0.5, -np.inf
    profits = []
    for thr in thresholds:
        mask = probs >= thr
        if mask.sum() < min_trades:
            profits.append(np.nan)
            continue
        profit = float(np.nansum(rpnl[mask]))
        profits.append(profit)
    profits = pd.Series(profits, index=thresholds)
    smoothed = profits.rolling(smooth, min_periods=1, center=True).mean()
    valid = smoothed.dropna()
    if len(valid) > 0:
        best_thr = float(valid.idxmax())
        best_profit = float(valid.max())
    return best_thr


def evaluate(name: str, probs, y, rpnl, threshold, dirs=None):
    mask = probs >= threshold
    trades = mask.sum()
    if trades == 0:
        return {"name": name, "trades": 0, "win_rate": 0, "net_r": 0, "avg_r": 0, "ap": 0, "max_dd": 0}
    trade_pnl = rpnl[mask]
    net_r = float(np.nansum(trade_pnl))
    avg_r = float(np.nanmean(trade_pnl))
    wr = float(np.mean(trade_pnl > 0))
    ap = average_precision_score(y, probs)
    cum = np.cumsum(trade_pnl)
    dd = float((cum - np.maximum.accumulate(cum)).min()) if len(cum) > 0 else 0
    result = {"name": name, "trades": int(trades), "win_rate": wr, "net_r": net_r, "avg_r": avg_r, "ap": ap, "max_dd": dd}
    if dirs is not None:
        long_mask = mask & (dirs == 1)
        short_mask = mask & (dirs == -1)
        result["long_trades"] = int(long_mask.sum())
        result["short_trades"] = int(short_mask.sum())
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="dataset_gate_enhanced.parquet")
    parser.add_argument("--combined-h", type=int, default=int(BEST["h"]))
    parser.add_argument("--combined-tp", type=float, default=float(BEST["tp"]))
    parser.add_argument("--combined-sl", type=float, default=float(BEST["sl"]))
    parser.add_argument("--long-h", type=int, default=None)
    parser.add_argument("--long-tp", type=float, default=None)
    parser.add_argument("--long-sl", type=float, default=None)
    parser.add_argument("--short-h", type=int, default=None)
    parser.add_argument("--short-tp", type=float, default=None)
    parser.add_argument("--short-sl", type=float, default=None)
    args = parser.parse_args()

    # Defaults: if per-direction not specified, use combined
    lh = args.long_h or args.combined_h
    ltp = args.long_tp or args.combined_tp
    lsl = args.long_sl or args.combined_sl
    sh = args.short_h or args.combined_h
    stp = args.short_tp or args.combined_tp
    ssl_ = args.short_sl or args.combined_sl
    ch, ctp, csl = args.combined_h, args.combined_tp, args.combined_sl

    print("=" * 60)
    print("LONG/SHORT SPLIT MODEL EXPERIMENT (independent params)")
    print(f"Combined: h={ch}, tp={ctp}, sl={csl}")
    print(f"Long:     h={lh}, tp={ltp}, sl={lsl}")
    print(f"Short:    h={sh}, tp={stp}, sl={ssl_}")
    print("=" * 60)

    # Load with combined params for the combined model
    df_combined = load_data(args.data_path, ch, ctp, csl)
    # Load with per-direction params
    df_all_for_split = load_data(args.data_path, lh, ltp, lsl)  # will re-label for long
    df_long = df_all_for_split[df_all_for_split["event_dir"] == 1].copy()

    df_all_for_short = load_data(args.data_path, sh, stp, ssl_)
    df_short = df_all_for_short[df_all_for_short["event_dir"] == -1].copy()

    feats = get_features(df_combined)
    missing = sorted(set(BASE_FEATURES) - set(feats))
    if missing:
        print(f"Missing features (skipped): {missing}")

    print(f"\nCombined: {len(df_combined)} | Long: {len(df_long)} | Short: {len(df_short)}")
    print(f"Win rate — Combined: {df_combined['label'].mean():.2%} | Long: {df_long['label'].mean():.2%} | Short: {df_short['label'].mean():.2%}")

    # Recalculate sample weights per-subset
    for sub in [df_long, df_short]:
        pr = sub["label"].mean()
        if 0 < pr < 1:
            sub["sample_weight"] = np.where(sub["label"] == 1, (1 - pr) / pr, 1.0).astype(np.float32)

    # --- Train combined model ---
    print("\n" + "=" * 60)
    print(f"TRAINING: COMBINED MODEL (h={ch}, tp={ctp}, sl={csl})")
    print("=" * 60)
    s_all = split_data(df_combined, feats)
    print(f"Train: {len(s_all['X_train'])}, Valid: {len(s_all['X_valid'])}, Test: {len(s_all['X_test'])}")
    m_all = train_model(s_all)
    p_all_valid = m_all.predict_proba(s_all["X_valid"])[:, 1]
    p_all_test = m_all.predict_proba(s_all["X_test"])[:, 1]
    thr_all = select_threshold(p_all_valid, s_all["y_valid"], s_all["rpnl_valid"])
    r_all = evaluate("Combined", p_all_test, s_all["y_test"], s_all["rpnl_test"], thr_all, s_all["dirs_test"])

    # --- Train long-only model ---
    print("\n" + "=" * 60)
    print(f"TRAINING: LONG-ONLY MODEL (h={lh}, tp={ltp}, sl={lsl})")
    print("=" * 60)
    s_long = split_data(df_long, feats)
    print(f"Train: {len(s_long['X_train'])}, Valid: {len(s_long['X_valid'])}, Test: {len(s_long['X_test'])}")
    m_long = train_model(s_long)
    p_long_valid = m_long.predict_proba(s_long["X_valid"])[:, 1]
    p_long_test = m_long.predict_proba(s_long["X_test"])[:, 1]
    thr_long = select_threshold(p_long_valid, s_long["y_valid"], s_long["rpnl_valid"])
    r_long = evaluate("Long-only", p_long_test, s_long["y_test"], s_long["rpnl_test"], thr_long)

    # --- Train short-only model ---
    print("\n" + "=" * 60)
    print(f"TRAINING: SHORT-ONLY MODEL (h={sh}, tp={stp}, sl={ssl_})")
    print("=" * 60)
    s_short = split_data(df_short, feats)
    print(f"Train: {len(s_short['X_train'])}, Valid: {len(s_short['X_valid'])}, Test: {len(s_short['X_test'])}")
    m_short = train_model(s_short)
    p_short_valid = m_short.predict_proba(s_short["X_valid"])[:, 1]
    p_short_test = m_short.predict_proba(s_short["X_test"])[:, 1]
    thr_short = select_threshold(p_short_valid, s_short["y_valid"], s_short["rpnl_valid"])
    r_short = evaluate("Short-only", p_short_test, s_short["y_test"], s_short["rpnl_test"], thr_short)

    # --- Save split model files ---
    m_long.get_booster().save_model("model_xgb_long.json")
    m_short.get_booster().save_model("model_xgb_short.json")
    print(f"\nModels saved: model_xgb_long.json, model_xgb_short.json")

    # --- Update best_config.json with split model config ---
    BEST["split_model"] = True
    BEST["long_config"] = {"h": lh, "tp": ltp, "sl": lsl, "threshold": round(thr_long, 4)}
    BEST["short_config"] = {"h": sh, "tp": stp, "sl": ssl_, "threshold": round(thr_short, 4)}
    with open("best_config.json", "w") as f:
        json.dump(BEST, f, indent=2)
    print(f"best_config.json updated: split_model=True, long_thr={thr_long:.4f}, short_thr={thr_short:.4f}")

    # --- Merged: route long events to long model, short events to short model ---
    # We need a common test set. Use the combined test set timeframe, but apply
    # direction-specific labels. For fair comparison, we take the test portion of
    # each direction-specific dataset and merge them.
    test_start_long = int(len(df_long) * 0.85) + PURGE_GAP_BARS
    test_start_short = int(len(df_short) * 0.85) + PURGE_GAP_BARS
    df_test_long = df_long.iloc[test_start_long:].copy()
    df_test_short = df_short.iloc[test_start_short:].copy()

    # Predict with respective models
    if len(df_test_long) > 0:
        pl = m_long.predict_proba(df_test_long[feats].astype(np.float32))[:, 1]
        df_test_long["_prob"] = pl
        df_test_long["_pass"] = pl >= thr_long
    else:
        df_test_long["_prob"] = []
        df_test_long["_pass"] = []

    if len(df_test_short) > 0:
        ps = m_short.predict_proba(df_test_short[feats].astype(np.float32))[:, 1]
        df_test_short["_prob"] = ps
        df_test_short["_pass"] = ps >= thr_short
    else:
        df_test_short["_prob"] = []
        df_test_short["_pass"] = []

    df_merged_test = pd.concat([df_test_long, df_test_short]).sort_index()
    mask_merged = df_merged_test["_pass"].to_numpy().astype(bool)

    if mask_merged.sum() > 0:
        rpnl_merged = df_merged_test["realized_pnl_r"].to_numpy(dtype=np.float64)
        trade_pnl_m = rpnl_merged[mask_merged]
        cum_m = np.cumsum(trade_pnl_m)
        dd_m = float((cum_m - np.maximum.accumulate(cum_m)).min()) if len(cum_m) > 0 else 0
        dirs_m = df_merged_test["event_dir"].to_numpy()
        r_merged = {
            "name": "Merged (L+S models)",
            "trades": int(mask_merged.sum()),
            "win_rate": float(np.mean(trade_pnl_m > 0)),
            "net_r": float(np.nansum(trade_pnl_m)),
            "avg_r": float(np.nanmean(trade_pnl_m)),
            "max_dd": dd_m,
            "long_trades": int((mask_merged & (dirs_m == 1)).sum()),
            "short_trades": int((mask_merged & (dirs_m == -1)).sum()),
        }
    else:
        r_merged = {"name": "Merged (L+S models)", "trades": 0, "win_rate": 0, "net_r": 0, "avg_r": 0, "max_dd": 0, "long_trades": 0, "short_trades": 0}

    # --- Summary ---
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY (Test Set)")
    print("=" * 60)
    print(f"{'Model':<22} {'Params':>14} {'Thr':>6} {'Trades':>7} {'WinRate':>8} {'NetR':>10} {'AvgR':>7} {'MaxDD':>8} {'AP':>7}")
    print("-" * 95)
    rows = [
        (r_all, f"h{ch}/tp{ctp}/sl{csl}", thr_all),
        (r_long, f"h{lh}/tp{ltp}/sl{lsl}", thr_long),
        (r_short, f"h{sh}/tp{stp}/sl{ssl_}", thr_short),
        (r_merged, "L+S indep", None),
    ]
    for r, params, thr in rows:
        thr_str = f"{thr:.4f}" if thr is not None else "L+S"
        ap_str = f"{r.get('ap', 0):.4f}" if r.get("ap") else "  -"
        print(f"{r['name']:<22} {params:>14} {thr_str:>6} {r['trades']:>7} {r['win_rate']:>7.2%} {r['net_r']:>10.1f} {r['avg_r']:>7.3f} {r['max_dd']:>8.1f} {ap_str:>7}")

    if r_merged.get("long_trades") or r_merged.get("short_trades"):
        print(f"\nMerged breakdown: Long={r_merged['long_trades']}, Short={r_merged['short_trades']}")

    # --- Equity curves plot ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    mask_all = p_all_test >= thr_all
    pnl_all = s_all["rpnl_test"][mask_all] if mask_all.any() else np.array([])
    if len(pnl_all) > 0:
        axes[0, 0].plot(np.cumsum(pnl_all), color="blue", linewidth=1.5)
    axes[0, 0].set_title(f"Combined h={ch} (thr={thr_all:.4f}, {r_all['trades']}t, {r_all['net_r']:.0f}R)")
    axes[0, 0].set_ylabel("Cumulative R"); axes[0, 0].grid(True, alpha=0.3)

    mask_l = p_long_test >= thr_long
    pnl_l = s_long["rpnl_test"][mask_l] if mask_l.any() else np.array([])
    if len(pnl_l) > 0:
        axes[0, 1].plot(np.cumsum(pnl_l), color="green", linewidth=1.5)
    axes[0, 1].set_title(f"Long h={lh} (thr={thr_long:.4f}, {r_long['trades']}t, {r_long['net_r']:.0f}R)")
    axes[0, 1].set_ylabel("Cumulative R"); axes[0, 1].grid(True, alpha=0.3)

    mask_s = p_short_test >= thr_short
    pnl_s = s_short["rpnl_test"][mask_s] if mask_s.any() else np.array([])
    if len(pnl_s) > 0:
        axes[1, 0].plot(np.cumsum(pnl_s), color="red", linewidth=1.5)
    axes[1, 0].set_title(f"Short h={sh} (thr={thr_short:.4f}, {r_short['trades']}t, {r_short['net_r']:.0f}R)")
    axes[1, 0].set_ylabel("Cumulative R"); axes[1, 0].set_xlabel("Trade #"); axes[1, 0].grid(True, alpha=0.3)

    if mask_merged.sum() > 0:
        axes[1, 1].plot(np.cumsum(trade_pnl_m), color="purple", linewidth=1.5)
    axes[1, 1].set_title(f"Merged L+S ({r_merged['trades']}t, {r_merged['net_r']:.0f}R)")
    axes[1, 1].set_ylabel("Cumulative R"); axes[1, 1].set_xlabel("Trade #"); axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle(f"Long/Short Split — Combined h={ch}/tp{ctp}/sl{csl}, Long h={lh}/tp{ltp}/sl{lsl}, Short h={sh}/tp{stp}/sl{ssl_}", fontsize=12)
    plt.tight_layout()
    save_path = os.path.join(PLOT_DIR, "long_short_split_comparison.png")
    plt.savefig(save_path, dpi=150)
    print(f"\nPlot saved to: {save_path}")
    plt.close()


if __name__ == "__main__":
    main()
