"""Compare post_window_s = 5, 10, 15, 20 on full dataset.

For each window value:
1. Run pipline_modified.py to rebuild dataset
2. Run grid search (tune_all_modified.py) for flow_reversal
3. Run robust OOS search for top configs
4. Print summary

Usage: python compare_post_windows.py
"""
from __future__ import annotations
import subprocess, json, sys, shutil, os
import pandas as pd

DATA_DIR = "/Volumes/TU280Pro/quant/raw_data/data"
WINDOWS = [5, 10, 15, 20]
SCANNER = "flow_reversal"
HORIZONS = "10,15,30"
TP_VALUES = "1.5,2.0"
SL_VALUES = "0.5,1.0,1.5"

results = []

for pw in WINDOWS:
    print(f"\n{'='*60}")
    print(f"  POST_WINDOW_S = {pw}")
    print(f"{'='*60}\n")

    # 1. Pipeline
    print(f"[{pw}s] Running pipeline...")
    r = subprocess.run(
        [sys.executable, "pipline_modified.py",
         "--data-dir", DATA_DIR,
         "--event-post-window-s", str(pw)],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        print(f"[{pw}s] Pipeline FAILED:\n{r.stderr[-500:]}")
        continue
    # Extract row count
    for line in r.stdout.split("\n"):
        if "Saved dataset" in line:
            print(f"[{pw}s] {line.strip()}")
        if "flow_reversal" in line and "Scanner counts" in line:
            print(f"[{pw}s] {line.strip()}")

    # Verify entry_price_delayed
    r2 = subprocess.run(
        [sys.executable, "-c",
         "import pandas as pd; df=pd.read_parquet('dataset_enhanced.parquet'); "
         "d=df['entry_price_delayed']-df['close']; "
         f"print(f'[{pw}s] entry_delay stats: mean={{d.mean():.4f}} abs_mean={{d.abs().mean():.4f}} std={{d.std():.4f}}')"],
        capture_output=True, text=True,
    )
    print(r2.stdout.strip())

    # 2. Robust OOS search (includes training internally)
    print(f"[{pw}s] Running robust OOS search...")
    oos_output = f"robust_oos_pw{pw}.csv"
    r3 = subprocess.run(
        [sys.executable, "robust_oos_search.py",
         "--scanner", SCANNER,
         "--label-mode", "first_touch",
         "--horizons", HORIZONS,
         "--tp-values", TP_VALUES,
         "--sl-values", SL_VALUES,
         "--output", oos_output],
        capture_output=True, text=True,
    )
    if r3.returncode != 0:
        print(f"[{pw}s] OOS search FAILED:\n{r3.stderr[-500:]}")
        continue

    # Parse OOS results
    try:
        oos_df = pd.read_csv(oos_output)
    except Exception as e:
        print(f"[{pw}s] Failed to read OOS results: {e}")
        continue

    if oos_df.empty:
        print(f"[{pw}s] No OOS candidates found")
        continue

    # Show top 5
    top = oos_df.head(5)
    print(f"\n[{pw}s] Top 5 OOS configs:")
    cols = ["h", "tp", "sl", "profitable_fold_ratio", "total_test_trades",
            "total_test_net_profit_r", "avg_r_per_trade", "test_ap_mean", "median_threshold"]
    print(top[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Record best
    best = oos_df.iloc[0]
    results.append({
        "post_window_s": pw,
        "h": int(best["h"]),
        "tp": float(best["tp"]),
        "sl": float(best["sl"]),
        "profitable_fold_ratio": float(best["profitable_fold_ratio"]),
        "total_test_trades": int(best["total_test_trades"]),
        "total_test_net_profit_r": float(best["total_test_net_profit_r"]),
        "avg_r_per_trade": float(best["avg_r_per_trade"]),
        "test_ap_mean": float(best["test_ap_mean"]),
        "median_threshold": float(best["median_threshold"]),
    })

# Final comparison
print(f"\n\n{'='*80}")
print("  FINAL COMPARISON: POST_WINDOW_S = 5 vs 10 vs 15 vs 20")
print(f"{'='*80}\n")

if results:
    comp_df = pd.DataFrame(results)
    print(comp_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    comp_df.to_csv("post_window_comparison.csv", index=False)
    print("\nSaved to post_window_comparison.csv")
else:
    print("No results collected!")
