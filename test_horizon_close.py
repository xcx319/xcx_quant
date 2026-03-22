"""
Test: Compare PnL when closing at horizon vs holding indefinitely.

Simulates two scenarios for trades where TP/SL is NOT hit within horizon:
  A) Close at horizon (backtest assumption for unresolved trades)
  B) Hold until TP/SL eventually hits (live behavior, using extended horizon)
"""
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from quant_modeling import (BASE_FEATURES, add_directional_features,
                            build_labeling_cache, build_labels, build_realized_pnl,
                            MAX_FORWARD_HORIZON)

# --- Load ---
df = pd.read_parquet("dataset_enhanced.parquet")
cfg = json.load(open("best_config.json"))

booster = xgb.Booster()
booster.load_model("model_sniper_v3_first_touch.json")
feat_names = booster.feature_names
threshold = float(cfg["threshold"])
horizon = int(cfg["h"])
tp = float(cfg["tp"])
sl = float(cfg["sl"])

print(f"Config: horizon={horizon}m, TP={tp}R, SL={sl}R, threshold={threshold:.4f}")

# --- Filter to flow_reversal ---
events = df[df["scanner_name"] == "flow_reversal"].copy()
print(f"flow_reversal events: {len(events)}")

# --- Get model predictions to filter to actual signals ---
events_dir = add_directional_features(events.copy())

def predict_batch(data, feature_names):
    vals = []
    for fname in feature_names:
        if fname in data.columns:
            vals.append(data[fname].values.astype(float))
        else:
            vals.append(np.full(len(data), np.nan))
    X = np.column_stack(vals)
    dmat = xgb.DMatrix(X, feature_names=feature_names, missing=np.nan)
    return booster.predict(dmat)

probs = predict_batch(events_dir, feat_names)
signals = probs >= threshold
events_sig = events[signals].copy()
events_sig["prob"] = probs[signals]
print(f"Signals (prob >= {threshold:.4f}): {len(events_sig)}")

# --- Build cache ---
cache = build_labeling_cache(events_sig)

# --- Scenario A: PnL at horizon (backtest) ---
pnl_at_horizon = build_realized_pnl(cache, horizon=horizon, tp_mult=tp, sl_mult=sl,
                                     label_mode="first_touch", same_bar_policy="drop")

# --- Identify unresolved trades ---
label_result = build_labels(cache, horizon=horizon, tp_mult=tp, sl_mult=sl,
                            label_mode="first_touch", same_bar_policy="drop")
hit_tp = label_result["hit_tp"]
hit_sl = label_result["hit_sl"]
resolved = hit_tp | hit_sl
unresolved = ~resolved

print(f"\n{'='*70}")
print(f"Trade resolution within {horizon} minutes:")
print(f"  TP hit:      {hit_tp.sum():6d} ({hit_tp.mean()*100:.1f}%)")
print(f"  SL hit:      {hit_sl.sum():6d} ({hit_sl.mean()*100:.1f}%)")
print(f"  Unresolved:  {unresolved.sum():6d} ({unresolved.mean()*100:.1f}%)")

# --- For unresolved trades, compute what happens if we hold longer ---
# Use max available horizon (30m) to see eventual outcome
max_h = min(MAX_FORWARD_HORIZON, cache.future_high.shape[1])
pnl_extended = build_realized_pnl(cache, horizon=max_h, tp_mult=tp, sl_mult=sl,
                                   label_mode="first_touch", same_bar_policy="drop")
label_extended = build_labels(cache, horizon=max_h, tp_mult=tp, sl_mult=sl,
                              label_mode="first_touch", same_bar_policy="drop")
hit_tp_ext = label_extended["hit_tp"]
hit_sl_ext = label_extended["hit_sl"]
resolved_ext = hit_tp_ext | hit_sl_ext
still_unresolved = ~resolved_ext

# --- Scenario B: Hold until TP/SL (use extended horizon PnL for resolved, horizon close for still unresolved) ---
pnl_hold = pnl_extended.copy()

# For trades still unresolved even at max horizon, use close at max horizon
entry = cache.close
atr = np.where(np.isfinite(cache.atr) & (cache.atr > 0), cache.atr, np.nan)
direction = np.where(cache.is_long, 1.0, -1.0)
close_at_max = cache.future_close[:, max_h - 1]
pnl_at_max_close = direction * (close_at_max - entry) / atr

# Also compute close-at-horizon PnL for unresolved
close_at_h = cache.future_close[:, horizon - 1]
pnl_close_at_h = direction * (close_at_h - entry) / atr

print(f"\n{'='*70}")
print(f"Extended horizon analysis ({max_h} minutes):")
print(f"  Resolved by {max_h}m:  {resolved_ext.sum()} ({resolved_ext.mean()*100:.1f}%)")
print(f"  Still unresolved:    {still_unresolved.sum()} ({still_unresolved.mean()*100:.1f}%)")
if unresolved.sum() > 0:
    ext_tp = (unresolved & hit_tp_ext).sum()
    ext_sl = (unresolved & hit_sl_ext).sum()
    ext_neither = (unresolved & ~resolved_ext).sum()
    print(f"\n  Of the {unresolved.sum()} unresolved at {horizon}m:")
    print(f"    Eventually hit TP ({horizon}-{max_h}m): {ext_tp}")
    print(f"    Eventually hit SL ({horizon}-{max_h}m): {ext_sl}")
    print(f"    Still unresolved at {max_h}m:          {ext_neither}")

# --- Compare PnL scenarios ---
print(f"\n{'='*70}")
print(f"PnL COMPARISON (all {len(events_sig)} signal trades)")
print(f"{'='*70}")

# Scenario A: close at horizon for unresolved
pnl_A = pnl_at_horizon.copy()
pnl_A_clean = pnl_A.dropna()

# Scenario B: hold until TP/SL (extended), close at max_h for still unresolved
pnl_B = pnl_extended.copy()
pnl_B_clean = pnl_B.dropna()

# Scenario C: close at horizon price for unresolved (market close)
pnl_C = pnl_at_horizon.copy()
pnl_C[unresolved] = pnl_close_at_h[unresolved]
pnl_C_clean = pnl_C.dropna()

print(f"\n{'':35s} {'A: Backtest':>14s} {'B: Hold→TP/SL':>14s} {'C: Close@{h}m':>14s}")
print(f"{'':35s} {'(default -SL)':>14s} {'(ext horizon)':>14s} {'(mkt close)':>14s}")
print(f"  {'Trades':35s} {len(pnl_A_clean):14d} {len(pnl_B_clean):14d} {len(pnl_C_clean):14d}")
print(f"  {'Total PnL (R)':35s} {pnl_A_clean.sum():14.2f} {pnl_B_clean.sum():14.2f} {pnl_C_clean.sum():14.2f}")
print(f"  {'Avg PnL per trade (R)':35s} {pnl_A_clean.mean():14.4f} {pnl_B_clean.mean():14.4f} {pnl_C_clean.mean():14.4f}")
print(f"  {'Win rate':35s} {(pnl_A_clean>0).mean():14.4f} {(pnl_B_clean>0).mean():14.4f} {(pnl_C_clean>0).mean():14.4f}")
print(f"  {'Avg win (R)':35s} {pnl_A_clean[pnl_A_clean>0].mean():14.4f} {pnl_B_clean[pnl_B_clean>0].mean():14.4f} {pnl_C_clean[pnl_C_clean>0].mean():14.4f}")
print(f"  {'Avg loss (R)':35s} {pnl_A_clean[pnl_A_clean<=0].mean():14.4f} {pnl_B_clean[pnl_B_clean<=0].mean():14.4f} {pnl_C_clean[pnl_C_clean<=0].mean():14.4f}")

# --- Focus on unresolved trades only ---
if unresolved.sum() > 0:
    print(f"\n{'='*70}")
    print(f"UNRESOLVED TRADES ONLY ({unresolved.sum()} trades, no TP/SL within {horizon}m)")
    print(f"{'='*70}")
    u_pnl_A = pnl_A[unresolved].dropna()
    u_pnl_B = pnl_B[unresolved].dropna()
    u_pnl_C = pnl_C[unresolved].dropna()
    print(f"\n{'':35s} {'A: Assume -SL':>14s} {'B: Hold→TP/SL':>14s} {'C: Close@{h}m':>14s}")
    if len(u_pnl_A) > 0:
        print(f"  {'Count':35s} {len(u_pnl_A):14d} {len(u_pnl_B):14d} {len(u_pnl_C):14d}")
        print(f"  {'Total PnL (R)':35s} {u_pnl_A.sum():14.2f} {u_pnl_B.sum():14.2f} {u_pnl_C.sum():14.2f}")
        print(f"  {'Avg PnL (R)':35s} {u_pnl_A.mean():14.4f} {u_pnl_B.mean():14.4f} {u_pnl_C.mean():14.4f}")
        print(f"  {'Win rate':35s} {(u_pnl_A>0).mean():14.4f} {(u_pnl_B>0).mean():14.4f} {(u_pnl_C>0).mean():14.4f}")

    # Distribution of close-at-horizon PnL for unresolved
    print(f"\n  Close-at-{horizon}m PnL distribution for unresolved trades:")
    pcts = [5, 25, 50, 75, 95]
    for p in pcts:
        print(f"    P{p:02d}: {np.nanpercentile(u_pnl_C.values, p):+.4f} R")
