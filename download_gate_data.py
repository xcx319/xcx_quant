#!/usr/bin/env python3
"""Download Gate.io historical futures data (trades, orderbooks, candlesticks_1m).

Orderbooks are per-hour files: {MARKET}-{YYYYMMDDHH}.csv.gz → converted to .parquet
Trades and candlesticks are per-month files: {MARKET}-{YYYYMM}.csv.gz → .csv

Usage:
    python download_gate_data.py --market ETH_USDT --start-month 202510 --end-month 202602
    python download_gate_data.py --market ETH_USDT --start-month 202510 --end-month 202603 \
        --types orderbooks --output-dir /Volumes/TU280Pro/quant/raw_data
"""
from __future__ import annotations
import argparse, calendar, gzip, logging, shutil, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_progress_lock = threading.Lock()
_progress = {"done": 0, "total": 0, "ok": 0, "skip": 0, "fail": 0}

BASE_URL = "https://download.gatedata.org"
BIZ = "futures_usdt"


def iter_months(start: str, end: str):
    """Yield YYYYMM strings from start to end inclusive."""
    y, m = int(start[:4]), int(start[4:])
    ey, em = int(end[:4]), int(end[4:])
    while (y, m) <= (ey, em):
        yield f"{y}{m:02d}"
        m += 1
        if m > 12:
            m = 1
            y += 1


def download_file(url: str, dest: Path, client: httpx.Client, retries: int = 3) -> bool:
    """Download url to dest with retries. Returns True on success, False on 404/skip."""
    if dest.exists():
        logger.debug(f"  Skip (exists): {dest.name}")
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries):
        try:
            with client.stream("GET", url, follow_redirects=True) as r:
                if r.status_code == 404:
                    return False
                r.raise_for_status()
                tmp = dest.with_suffix(f".tmp.{threading.current_thread().ident}")
                with open(tmp, "wb") as f:
                    for chunk in r.iter_bytes(chunk_size=65536):
                        f.write(chunk)
                tmp.rename(dest)
            return True
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                logger.error(f"  Error downloading {dest.name}: {e}")
                return False
    return False


def decompress_gz(gz_path: Path) -> Path:
    """Decompress .csv.gz to .csv in same directory. Returns csv path."""
    csv_path = gz_path.with_suffix("")  # removes .gz → .csv
    if csv_path.exists():
        return csv_path
    logger.info(f"  Decompressing: {gz_path.name}")
    with gzip.open(gz_path, "rb") as f_in, open(csv_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    logger.info(f"  Decompressed: {csv_path.name} ({csv_path.stat().st_size // 1024} KB)")
    return csv_path


def _csv_to_parquet(csv_path: Path, pq_path: Path) -> None:
    """Convert orderbook CSV to parquet (zstd). Keeps only needed columns."""
    import pandas as pd
    df = pd.read_csv(csv_path, header=None,
                     names=["timestamp", "action", "price", "size", "begin_id", "merged"],
                     usecols=["timestamp", "action", "price", "size"],
                     dtype={"action": str, "price": float, "size": float})
    df.to_parquet(pq_path, compression="zstd", index=False)
    logger.info(f"  Parquet: {pq_path.name} ({pq_path.stat().st_size // 1024} KB)")


def download_monthly(output_dir: Path, market: str, dtype: str, months: list[str],
                     client: httpx.Client, keep_gz: bool, stats: dict):
    """Download monthly files (trades, candlesticks_1m)."""
    for month in months:
        filename = f"{market}-{month}.csv.gz"
        url = f"{BASE_URL}/{BIZ}/{dtype}/{month}/{filename}"
        dest = output_dir / dtype / filename

        ok = download_file(url, dest, client)
        if not ok:
            stats["fail"] += 1
            continue

        if dest.exists():
            decompress_gz(dest)
            if not keep_gz and dest.exists():
                dest.unlink()
            stats["ok"] += 1
        else:
            stats["skip"] += 1


def _download_one_ob(task: tuple, keep_gz: bool) -> str:
    """Download + convert one orderbook file. Returns 'ok', 'skip', or 'fail'."""
    url, dest, pq_dest = task
    if pq_dest.exists():
        return "skip"
    client = httpx.Client(timeout=300, follow_redirects=True)
    try:
        ok = download_file(url, dest, client)
        if not ok:
            return "fail"
        if dest.exists():
            csv_path = decompress_gz(dest)
            _csv_to_parquet(csv_path, pq_dest)
            csv_path.unlink(missing_ok=True)
            if not keep_gz and dest.exists():
                dest.unlink()
            return "ok"
        return "skip"
    finally:
        client.close()


def download_orderbooks_hourly(output_dir: Path, market: str, months: list[str],
                                client: httpx.Client, keep_gz: bool, stats: dict,
                                workers: int = 1):
    """Download hourly orderbook files for given months.
    URL pattern: {BASE_URL}/futures_usdt/orderbooks/{YYYYMM}/{MARKET}-{YYYYMMDDHH}.csv.gz
    Skips hours that haven't completed yet (current UTC hour and beyond).
    """
    now_utc = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)

    # Collect all tasks first
    tasks = []
    for month in months:
        year = int(month[:4])
        mon = int(month[4:])
        days_in_month = calendar.monthrange(year, mon)[1]
        for day in range(1, days_in_month + 1):
            for hour in range(24):
                dt = datetime(year, mon, day, hour, tzinfo=timezone.utc)
                if dt >= now_utc:
                    break
                tag = f"{year}{mon:02d}{day:02d}{hour:02d}"
                filename = f"{market}-{tag}.csv.gz"
                url = f"{BASE_URL}/{BIZ}/orderbooks/{month}/{filename}"
                dest = output_dir / "orderbooks" / filename
                pq_dest = dest.parent / f"{market}-{tag}.parquet"
                if pq_dest.exists():
                    stats["skip"] += 1
                    continue
                tasks.append((url, dest, pq_dest))

    if not tasks:
        logger.info("All orderbook files already downloaded")
        return

    logger.info(f"Downloading {len(tasks)} orderbook files with {workers} workers...")
    dest_dir = output_dir / "orderbooks"
    dest_dir.mkdir(parents=True, exist_ok=True)

    if workers <= 1:
        for i, task in enumerate(tasks):
            result = _download_one_ob(task, keep_gz)
            stats[result] += 1
            if (i + 1) % 50 == 0:
                logger.info(f"  Progress: {i+1}/{len(tasks)}")
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_download_one_ob, t, keep_gz): t for t in tasks}
            done_count = 0
            for fut in as_completed(futures):
                result = fut.result()
                stats[result] += 1
                done_count += 1
                if done_count % 50 == 0:
                    logger.info(f"  Progress: {done_count}/{len(tasks)} (ok={stats['ok']} fail={stats['fail']})")


def main():
    parser = argparse.ArgumentParser(description="Download Gate.io historical futures data")
    parser.add_argument("--market", default="ETH_USDT")
    parser.add_argument("--start-month", required=True, help="e.g. 202510")
    parser.add_argument("--end-month", required=True, help="e.g. 202602")
    parser.add_argument("--types", default="trades,orderbooks,candlesticks_1m",
                        help="Comma-separated data types")
    parser.add_argument("--output-dir", default="/Volumes/TU280Pro/quant/raw_data")
    parser.add_argument("--keep-gz", action="store_true", help="Keep .gz files after decompression")
    parser.add_argument("--workers", type=int, default=8, help="Parallel download workers for orderbooks")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    data_types = [t.strip() for t in args.types.split(",")]
    months = list(iter_months(args.start_month, args.end_month))

    logger.info(f"Market: {args.market}, months: {months[0]}..{months[-1]}, types: {data_types}")
    logger.info(f"Output dir: {output_dir}")

    stats = {"ok": 0, "skip": 0, "fail": 0}

    with httpx.Client(timeout=300, follow_redirects=True) as client:
        for dtype in data_types:
            if dtype == "orderbooks":
                download_orderbooks_hourly(output_dir, args.market, months, client,
                                           args.keep_gz, stats, workers=args.workers)
            else:
                download_monthly(output_dir, args.market, dtype, months, client,
                                 args.keep_gz, stats)

    logger.info(f"Done. ok={stats['ok']} skip={stats['skip']} fail={stats['fail']}")


if __name__ == "__main__":
    main()
