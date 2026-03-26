#!/usr/bin/env python3
"""Convert Gate.io orderbook CSV files to parquet (zstd) for efficient storage and reading.

56MB CSV → 4MB parquet (14x compression), read speed 5x faster.

Usage:
    python convert_ob_to_parquet.py --input-dir /Volumes/TU280Pro/quant/raw_data/orderbooks
    python convert_ob_to_parquet.py --input-dir /Volumes/TU280Pro/quant/raw_data/orderbooks --delete-csv
"""
from __future__ import annotations
import argparse, logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

COLS = ["timestamp", "action", "price", "size", "begin_id", "merged"]
DTYPES = {"action": str, "price": float, "size": float}


def convert_file(csv_path: Path, delete_csv: bool) -> bool:
    pq_path = csv_path.with_suffix(".parquet")
    if pq_path.exists():
        if delete_csv and csv_path.exists():
            csv_path.unlink()
        return True
    try:
        df = pd.read_csv(csv_path, header=None, names=COLS,
                         usecols=["timestamp", "action", "price", "size"],
                         dtype=DTYPES)
        df.to_parquet(pq_path, compression="zstd", index=False)
        if delete_csv:
            csv_path.unlink()
        return True
    except Exception as e:
        logger.error(f"Failed {csv_path.name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="/Volumes/TU280Pro/quant/raw_data/orderbooks")
    parser.add_argument("--delete-csv", action="store_true",
                        help="Delete original CSV after successful conversion")
    args = parser.parse_args()

    ob_dir = Path(args.input_dir)
    csv_files = sorted(ob_dir.glob("*.csv"))
    logger.info(f"Found {len(csv_files)} CSV files in {ob_dir}")

    ok = skip = fail = 0
    for i, f in enumerate(csv_files, 1):
        if f.with_suffix(".parquet").exists():
            skip += 1
            if args.delete_csv and f.exists():
                f.unlink()
            continue
        if convert_file(f, args.delete_csv):
            ok += 1
        else:
            fail += 1
        if i % 100 == 0:
            logger.info(f"  Progress: {i}/{len(csv_files)} (ok={ok} skip={skip} fail={fail})")

    logger.info(f"Done. ok={ok} skip={skip} fail={fail}")

    # Report size savings
    pq_files = list(ob_dir.glob("*.parquet"))
    total_pq = sum(f.stat().st_size for f in pq_files)
    remaining_csv = list(ob_dir.glob("*.csv"))
    total_csv = sum(f.stat().st_size for f in remaining_csv)
    logger.info(f"Parquet total: {total_pq//1024//1024//1024}GB ({len(pq_files)} files)")
    if remaining_csv:
        logger.info(f"Remaining CSV: {total_csv//1024//1024//1024}GB ({len(remaining_csv)} files)")


if __name__ == "__main__":
    main()
