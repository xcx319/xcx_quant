"""
Test: Compare model predictions WITH vs WITHOUT event-aligned features.

Shows how much the post_window event-aligned features affect model output.
"""
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from quant_modeling import BASE_FEATURES, add_directional_features, build_labeling_cache, build_labels

# --- Load ---
df = pd.read_parquet("dataset_enhanced.parquet")
cfg = json.load(open("best_config.json"))

booster = xgb.Booster()
booster.load_model("model_sniper_v3_first_touch.json")
feat_names = booster.feature_names
threshold = cfg["threshold"]

print(f"Dataset: {len(df)} rows")
print(f"Model features: {len(feat_names)}")
print(f"Threshold: {threshold}")

# --- Filter to flow_reversal scanner events ---
scanner_mask = df["scanner_name"] == "flow_reversal"
events = df[scanner_mask].copy()
print(f"\nflow_reversal events: {len(events)}")

# --- Build labels ---
cache = build_labeling_cache(events)
label_result = build_labels(cache, horizon=int(cfg["h"]), tp_mult=float(cfg["tp"]),
                            sl_mult=float(cfg["sl"]),
                            label_mode=cfg.get("label_mode", "first_touch"),
                            same_bar_policy=cfg.get("same_bar_policy", "drop"))
events["label"] = label_result["label"]
events = events.dropna(subset=["label"])
print(f"Events with labels: {len(events)}")

# --- Add directional features ---
events_dir = add_directional_features(events.copy())

# --- Event-aligned feature columns ---
EA_COLS = ["sec_in_bar", "event_return", "event_effort_vs_result",
           "event_rejection_strength", "time_to_reject_s"]

print(f"\n{'='*70}")
print("Event-aligned feature statistics (real values):")
print(events_dir[EA_COLS].describe().round(4).to_string())

# --- Predict WITH real event features ---
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

probs_real = predict_batch(events_dir, feat_names)

# --- Predict WITHOUT event features (hardcoded defaults like old live system) ---
events_no_ea = events_dir.copy()
events_no_ea["sec_in_bar"] = 59.0
events_no_ea["event_return"] = 0.0
events_no_ea["dir_event_return"] = 0.0
events_no_ea["event_effort_vs_result"] = 0.0
events_no_ea["event_rejection_strength"] = 0.0
events_no_ea["time_to_reject_s"] = 15.0

probs_default = predict_batch(events_no_ea, feat_names)

# --- Compare ---
print(f"\n{'='*70}")
print("PREDICTION COMPARISON: Real EA features vs Hardcoded defaults")
print(f"{'='*70}")

diff = probs_real - probs_default
abs_diff = np.abs(diff)

print(f"\nProbability statistics:")
print(f"  {'':30s} {'Real EA':>12s} {'Defaults':>12s} {'Diff':>12s}")
print(f"  {'Mean prob':30s} {probs_real.mean():12.6f} {probs_default.mean():12.6f} {diff.mean():+12.6f}")
print(f"  {'Median prob':30s} {np.median(probs_real):12.6f} {np.median(probs_default):12.6f} {np.median(diff):+12.6f}")
print(f"  {'Std prob':30s} {probs_real.std():12.6f} {probs_default.std():12.6f} {abs_diff.std():12.6f}")
print(f"  {'Mean |diff|':30s} {'':12s} {'':12s} {abs_diff.mean():12.6f}")
print(f"  {'Max |diff|':30s} {'':12s} {'':12s} {abs_diff.max():12.6f}")
print(f"  {'P90 |diff|':30s} {'':12s} {'':12s} {np.percentile(abs_diff, 90):12.6f}")

# --- Signal agreement ---
sig_real = probs_real >= threshold
sig_default = probs_default >= threshold
agree = (sig_real == sig_default).sum()
disagree = (sig_real != sig_default).sum()
real_only = (sig_real & ~sig_default).sum()
default_only = (~sig_real & sig_default).sum()

print(f"\nSignal agreement (threshold={threshold:.4f}):")
print(f"  Total events:          {len(probs_real)}")
print(f"  Signals (real EA):     {sig_real.sum()} ({sig_real.mean()*100:.1f}%)")
print(f"  Signals (defaults):    {sig_default.sum()} ({sig_default.mean()*100:.1f}%)")
print(f"  Agreement:             {agree} ({agree/len(probs_real)*100:.1f}%)")
print(f"  Disagreement:          {disagree} ({disagree/len(probs_real)*100:.1f}%)")
print(f"    Real EA only:        {real_only}")
print(f"    Defaults only:       {default_only}")

# --- Performance comparison ---
labels_arr = events["label"].values

def calc_metrics(probs, labels, threshold):
    signals = probs >= threshold
    if signals.sum() == 0:
        return {"trades": 0, "win_rate": 0, "avg_prob": 0}
    wins = labels[signals].sum()
    total = signals.sum()
    return {
        "trades": int(total),
        "win_rate": float(wins / total),
        "avg_prob": float(probs[signals].mean()),
    }

m_real = calc_metrics(probs_real, labels_arr, threshold)
m_default = calc_metrics(probs_default, labels_arr, threshold)

print(f"\nTrading performance:")
print(f"  {'':25s} {'Real EA':>12s} {'Defaults':>12s}")
print(f"  {'Trades':25s} {m_real['trades']:12d} {m_default['trades']:12d}")
print(f"  {'Win rate':25s} {m_real['win_rate']:12.4f} {m_default['win_rate']:12.4f}")
print(f"  {'Avg prob (signals)':25s} {m_real['avg_prob']:12.4f} {m_default['avg_prob']:12.4f}")

# --- Show biggest disagreements ---
print(f"\n{'='*70}")
print("Top 10 biggest probability differences:")
print(f"{'='*70}")
idx_sorted = np.argsort(-abs_diff)[:10]
for rank, i in enumerate(idx_sorted):
    label = "WIN" if labels_arr[i] == 1 else "LOSS"
    sig_r = "TRADE" if sig_real[i] else "skip"
    sig_d = "TRADE" if sig_default[i] else "skip"
    print(f"  #{rank+1}: real={probs_real[i]:.4f}({sig_r}) vs default={probs_default[i]:.4f}({sig_d}) "
          f"diff={diff[i]:+.4f} actual={label} "
          f"| sec_in_bar={events_dir.iloc[i].get('sec_in_bar', 'N/A'):.1f} "
          f"rej={events_dir.iloc[i].get('event_rejection_strength', 'N/A'):.4f} "
          f"t_rej={events_dir.iloc[i].get('time_to_reject_s', 'N/A'):.2f}")

# --- Feature importance of EA features ---
print(f"\n{'='*70}")
print("Feature importance of event-aligned features:")
print(f"{'='*70}")
importance = booster.get_score(importance_type="gain")
total_gain = sum(importance.values())
ea_features = ["sec_in_bar", "dir_event_return", "event_effort_vs_result",
                "event_rejection_strength", "time_to_reject_s"]
for f in ea_features:
    gain = importance.get(f, 0)
    pct = gain / total_gain * 100 if total_gain > 0 else 0
    print(f"  {f:35s} gain={gain:10.2f} ({pct:5.2f}%)")
ea_total = sum(importance.get(f, 0) for f in ea_features)
print(f"  {'TOTAL EA features':35s} gain={ea_total:10.2f} ({ea_total/total_gain*100:5.2f}%)")
