#!/usr/bin/env python3
"""Build dataset_gate_enhanced.parquet from downloaded Gate.io raw data.

Reads trades, orderbooks, candlesticks_1m CSVs from --input-dir,
merges them into a DataFrame compatible with dataset_enhanced.parquet,
then calls pipline_modified.add_features() to compute all features.

Usage:
    python build_gate_dataset.py \
        --input-dir /Volumes/TU280Pro/quant/raw_data \
        --output dataset_gate_enhanced.parquet \
        --market ETH_USDT \
        --start-month 202401 --end-month 202603
"""
from __future__ import annotations
import argparse, logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def iter_months(start: str, end: str):
    y, m = int(start[:4]), int(start[4:])
    ey, em = int(end[:4]), int(end[4:])
    while (y, m) <= (ey, em):
        yield f"{y}{m:02d}"
        m += 1
        if m > 12:
            m = 1
            y += 1


# ---------------------------------------------------------------------------
# 1. Load candlesticks
# ---------------------------------------------------------------------------

def load_candles(input_dir: Path, market: str, months: list[str]) -> pd.DataFrame:
    """Load Gate candlesticks_1m CSVs.
    Columns: timestamp, volume, close, high, low, open, amount
    """
    frames = []
    for month in months:
        p = input_dir / "candlesticks_1m" / f"{market}-{month}.csv"
        if not p.exists():
            logger.warning(f"Missing candles: {p}")
            continue
        df = pd.read_csv(p, header=None,
                         names=["timestamp", "volume", "close", "high", "low", "open", "amount"])
        frames.append(df)
    if not frames:
        raise FileNotFoundError("No candlestick files found")
    df = pd.concat(frames, ignore_index=True)
    df["datetime"] = pd.to_datetime(df["timestamp"].astype(float), unit="s", utc=True)
    df = df.sort_values("datetime").drop_duplicates("datetime")
    df = df.set_index("datetime")[["open", "high", "low", "close", "volume"]].astype(float)
    logger.info(f"Candles: {len(df)} rows, {df.index[0]} .. {df.index[-1]}")
    return df


# ---------------------------------------------------------------------------
# 2. Load trades → microstructure per minute
# ---------------------------------------------------------------------------

def _gini(x: np.ndarray) -> float:
    if len(x) == 0:
        return 0.0
    x = np.sort(np.abs(x))
    n = len(x)
    s = x.sum()
    if s == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float(((2 * idx - n - 1) * x).sum() / (n * s))


def _agg_minute_vectorized(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorized per-minute microstructure aggregation (no groupby.apply)."""
    df = df.copy()
    df["minute"] = pd.to_datetime(df["timestamp"].astype(float), unit="s", utc=True).dt.floor("1min")
    df["size"] = df["size"].astype(float)
    df["abs_size"] = df["size"].abs()
    df["buy_vol"] = df["size"].clip(lower=0)
    df["sell_vol"] = (-df["size"]).clip(lower=0)

    g = df.groupby("minute")
    vol = g["abs_size"].sum()
    buy = g["buy_vol"].sum()
    sell = g["sell_vol"].sum()

    # trade_intensity
    intensity = np.log1p(vol)

    # aggressor_ratio, net_taker_vol_ratio
    safe_vol = vol.clip(lower=1e-12)
    agg_ratio = buy / safe_vol
    agg_ratio[vol == 0] = 0.5
    net_taker = (buy - sell) / safe_vol
    net_taker[vol == 0] = 0.0

    # gini and large_trade_vol_ratio need per-group computation
    gini_vals = g["abs_size"].apply(lambda x: _gini(x.values))
    mean_sz = g["abs_size"].transform("mean")
    df["_large"] = df["abs_size"] * (df["abs_size"] > mean_sz * 5)
    large_sum = df.groupby("minute")["_large"].sum()
    large_ratio = large_sum / safe_vol
    large_ratio[vol == 0] = 0.0

    result = pd.DataFrame({
        "aggressor_ratio": agg_ratio,
        "net_taker_vol_ratio": net_taker,
        "trade_gini": gini_vals,
        "large_trade_vol_ratio": large_ratio,
        "trade_intensity": intensity,
    })
    result.index.name = "datetime"
    return result


def load_trades_microstructure(input_dir: Path, market: str, months: list[str]) -> pd.DataFrame:
    """Load Gate trades CSVs month-by-month and aggregate to 1-min microstructure.
    Processes each month independently to avoid OOM on large datasets.
    """
    micro_frames = []
    for month in months:
        p = input_dir / "trades" / f"{market}-{month}.csv"
        if not p.exists():
            logger.warning(f"Missing trades: {p}")
            continue
        logger.info(f"Processing trades: {p.name}")
        df = pd.read_csv(p, header=None, names=["timestamp", "dealid", "price", "size"],
                         usecols=["timestamp", "size"])
        micro = _agg_minute_vectorized(df)
        micro_frames.append(micro)
        logger.info(f"  {p.name}: {len(df)} trades → {len(micro)} minutes")
        del df

    if not micro_frames:
        logger.warning("No trade files found — microstructure features will be zero")
        return pd.DataFrame()

    result = pd.concat(micro_frames).sort_index()
    result = result[~result.index.duplicated(keep="last")]
    logger.info(f"Microstructure total: {len(result)} minutes")
    return result


# ---------------------------------------------------------------------------
# 3. Load orderbooks → OBI features per minute
# ---------------------------------------------------------------------------

def load_orderbook_features(input_dir: Path, market: str, months: list[str]) -> pd.DataFrame:
    """Load hourly Gate orderbook files and compute per-minute OB features.

    Supports both parquet (preferred, ~4MB/file) and CSV (~56MB/file).
    Parquet files take priority; falls back to CSV if parquet not found.

    Hourly file naming: {market}-{YYYYMMDDHH}.parquet / .csv
    Columns: timestamp, action, price, size
      action='make' → size>0 = new/updated bid, size<0 = new/updated ask, size=0 = cancel
      action='take' → order filled (remove from book)

    Strategy: vectorized per-file processing.
    For each minute, take the last known size at each price level from make rows.
    Positive size = bid, negative = ask. Use top-20 levels for OBI/depth features.
    """
    ob_dir = input_dir / "orderbooks"
    month_set = set(months)

    # Prefer parquet over CSV (parquet is 14x smaller and 5x faster to read)
    relevant = []
    for tag_len in [10]:  # YYYYMMDDHH = 10 chars
        for ext in [".parquet", ".csv"]:
            for f in sorted(ob_dir.glob(f"{market}-??????????{ext}")):
                stem = f.stem
                parts = stem.split("-", 1)
                if len(parts) < 2 or len(parts[1]) != tag_len:
                    continue
                if parts[1][:6] not in month_set:
                    continue
                # Skip CSV if parquet already in list for same tag
                tag = parts[1]
                if ext == ".csv" and (ob_dir / f"{stem}.parquet").exists():
                    continue
                relevant.append(f)
    relevant = sorted(set(relevant), key=lambda f: f.stem)

    if not relevant:
        logger.warning("No orderbook files found — OB features will be zero")
        return pd.DataFrame()

    n_pq = sum(1 for f in relevant if f.suffix == ".parquet")
    n_csv = sum(1 for f in relevant if f.suffix == ".csv")
    logger.info(f"Found {len(relevant)} hourly orderbook files ({n_pq} parquet, {n_csv} csv), computing OB features...")

    all_records = []

    for f in relevant:
        try:
            if f.suffix == ".parquet":
                df = pd.read_parquet(f, columns=["timestamp", "action", "price", "size"])
            else:
                df = pd.read_csv(f, header=None,
                                 names=["timestamp", "action", "price", "size", "begin_id", "merged"],
                                 usecols=["timestamp", "action", "price", "size"],
                                 dtype={"action": str, "price": float, "size": float})
        except Exception as e:
            logger.warning(f"Failed to read {f.name}: {e}")
            continue

        # Only make rows carry book state (take = removal, set = unreliable)
        make = df[df["action"] == "make"].copy()
        if make.empty:
            continue

        make["minute"] = pd.to_datetime(
            make["timestamp"].astype(float), unit="s", utc=True
        ).dt.floor("1min")

        # For each (minute, price), keep the last size update
        last_state = make.groupby(["minute", "price"])["size"].last().reset_index()

        # Remove cancelled levels (size == 0)
        last_state = last_state[last_state["size"] != 0]

        bids = last_state[last_state["size"] > 0]
        asks = last_state[last_state["size"] < 0].copy()
        asks["size"] = asks["size"].abs()

        # For each minute, compute OB features from top-20 bid/ask levels
        # Vectorized: rank within each (minute, side) group
        bids = bids.copy()
        asks = asks.copy()
        bids["_rank"] = bids.groupby("minute")["price"].rank(method="first", ascending=False)
        asks["_rank"] = asks.groupby("minute")["price"].rank(method="first", ascending=True)

        bids20 = bids[bids["_rank"] <= 20].copy()
        asks20 = asks[asks["_rank"] <= 20].copy()

        # Pure pandas aggregation — no Python-level groupby.apply
        # best price = rank-1 row
        bid_best = bids20[bids20["_rank"] == 1].set_index("minute")[["price", "size"]].rename(
            columns={"price": "best", "size": "s1"})
        ask_best = asks20[asks20["_rank"] == 1].set_index("minute")[["price", "size"]].rename(
            columns={"price": "best", "size": "s1"})

        # top-5 sum
        bid5 = bids20[bids20["_rank"] <= 5].groupby("minute")["size"].sum().rename("s5")
        ask5 = asks20[asks20["_rank"] <= 5].groupby("minute")["size"].sum().rename("s5")

        # total, max, mean (for walls)
        bid_stats = bids20.groupby("minute")["size"].agg(["sum", "max", "mean"]).rename(
            columns={"sum": "total", "max": "max_s", "mean": "avg_s"})
        ask_stats = asks20.groupby("minute")["size"].agg(["sum", "max", "mean"]).rename(
            columns={"sum": "total", "max": "max_s", "mean": "avg_s"})

        # walls: count levels where size > avg*2
        bids20["_wall"] = bids20["size"] > bids20.groupby("minute")["size"].transform("mean") * 2
        asks20["_wall"] = asks20["size"] > asks20.groupby("minute")["size"].transform("mean") * 2
        bid_walls = bids20.groupby("minute")["_wall"].sum().rename("walls")
        ask_walls = asks20.groupby("minute")["_wall"].sum().rename("walls")

        # Combine bid side
        ba = bid_best.join(bid5).join(bid_stats).join(bid_walls)
        aa = ask_best.join(ask5).join(ask_stats).join(ask_walls)

        common = ba.index.intersection(aa.index)
        if common.empty:
            continue

        ba, aa = ba.loc[common], aa.loc[common]
        mid = (ba["best"] + aa["best"]) / 2.0
        valid = mid > 0
        # For crossed book (bid >= ask), use mid as best estimate
        crossed = ba["best"] >= aa["best"]
        if crossed.any():
            ba.loc[crossed, "best"] = mid[crossed] - 0.005
            aa.loc[crossed, "best"] = mid[crossed] + 0.005
        ba, aa, mid = ba[valid], aa[valid], mid[valid]
        if ba.empty:
            continue

        spread_bps = (aa["best"] - ba["best"]) / mid * 1e4
        total = ba["total"] + aa["total"]
        obi = (ba["total"] - aa["total"]) / total.clip(lower=1e-12)
        obi_5 = (ba["s5"] - aa["s5"]) / (ba["s5"] + aa["s5"]).clip(lower=1e-12)
        obi_1 = (ba["s1"] - aa["s1"]) / (ba["s1"] + aa["s1"]).clip(lower=1e-12)
        mp = (ba["best"] * aa["s1"] + aa["best"] * ba["s1"]) / (ba["s1"] + aa["s1"]).clip(lower=1e-12)

        file_df = pd.DataFrame({
            "obi": obi, "obi_1": obi_1, "obi_5": obi_5, "obi_20": obi,
            "ob_spread_bps": spread_bps,
            "ob_depth_bid_1": ba["s1"], "ob_depth_ask_1": aa["s1"],
            "ob_depth_bid_5": ba["s5"], "ob_depth_ask_5": aa["s5"],
            "ob_depth_bid_20": ba["total"], "ob_depth_ask_20": aa["total"],
            "ob_microprice": mp, "ob_mid_close": mid,
            "ob_ask_wall_size_20": aa["max_s"], "ob_bid_wall_size_20": ba["max_s"],
            "ob_ask_wall_conc_20": aa["max_s"] / aa["total"].clip(lower=1e-12),
            "ob_bid_wall_conc_20": ba["max_s"] / ba["total"].clip(lower=1e-12),
            "ob_ask_wall_levels_20": aa["walls"].astype(int),
            "ob_bid_wall_levels_20": ba["walls"].astype(int),
            "ob_quote_count": (ba["total"] + aa["total"]).astype(int),
            "data_from_orderbook": 1,
        })
        file_df.index.name = "datetime"
        all_records.append(file_df)

    if not all_records:
        logger.warning("No OB features computed — OB features will be zero")
        return pd.DataFrame()

    result = pd.concat(all_records).sort_index()
    result = result[~result.index.duplicated(keep="last")]
    logger.info(f"OB features: {len(result)} minutes from {result.index[0]} to {result.index[-1]}")
    return result


# ---------------------------------------------------------------------------
# 4. Merge + add_features


# ---------------------------------------------------------------------------
# 4. Merge + add_features
# ---------------------------------------------------------------------------

def build_dataset(input_dir: Path, market: str, months: list[str], output: Path):
    candles = load_candles(input_dir, market, months)
    micro = load_trades_microstructure(input_dir, market, months)
    ob = load_orderbook_features(input_dir, market, months)

    df = candles.copy()

    if not micro.empty:
        df = df.join(micro, how="left")
    else:
        for col in ["aggressor_ratio", "net_taker_vol_ratio", "trade_gini",
                    "large_trade_vol_ratio", "trade_intensity"]:
            df[col] = 0.0
        df["aggressor_ratio"] = 0.5

    _ob_defaults = {
        "obi": 0.0, "obi_1": 0.0, "obi_5": 0.0, "obi_20": 0.0,
        "ob_spread_bps": 1.0,
        "ob_depth_bid_1": 1.0, "ob_depth_ask_1": 1.0,
        "ob_depth_bid_5": 5.0, "ob_depth_ask_5": 5.0,
        "ob_depth_bid_20": 20.0, "ob_depth_ask_20": 20.0,
        "ob_ask_wall_size_20": 0.0, "ob_bid_wall_size_20": 0.0,
        "ob_ask_wall_conc_20": 0.0, "ob_bid_wall_conc_20": 0.0,
        "ob_ask_wall_levels_20": 0.0, "ob_bid_wall_levels_20": 0.0,
        "ob_quote_count": 0.0, "data_from_orderbook": 0,
    }

    if not ob.empty:
        df = df.join(ob, how="left")
        # Fill uncovered minutes with same defaults as the no-OB branch
        ob_missing = df["data_from_orderbook"].isna()
        if ob_missing.any():
            for col, val in _ob_defaults.items():
                if col in df.columns:
                    df.loc[ob_missing, col] = val
            df.loc[ob_missing, "ob_microprice"] = df.loc[ob_missing, "close"]
            df.loc[ob_missing, "ob_mid_close"] = df.loc[ob_missing, "close"]
    else:
        for col, val in _ob_defaults.items():
            df[col] = val
        df["ob_microprice"] = df["close"]
        df["ob_mid_close"] = df["close"]

    df = df.fillna(0.0)
    df["exchange"] = "gate"

    logger.info(f"Merged DataFrame: {len(df)} rows, {len(df.columns)} columns")
    logger.info("Running add_features()...")

    # Import pipeline and add features + scanner events
    import json, sys
    sys.path.insert(0, str(Path(__file__).parent))
    from pipline_modified import add_features, build_scanner_events, add_forward_price_path

    df = add_features(df)
    logger.info(f"After add_features: {len(df)} rows, {len(df.columns)} columns")

    # Fill NaN and inf in OB-derived features that are undefined when no orderbook data
    ob_derived = ["obi_zscore", "quote_intensity_change", "spread_bps_ratio",
                  "depth_ratio_5", "depth_delta_5", "ob_mid_close_dist",
                  "depth_absorption_rate", "depth_divergence"]
    for col in ob_derived:
        if col in df.columns:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Add forward price path columns (required for labeling in train_xgb.py)
    df = add_forward_price_path(df)
    logger.info(f"After add_forward_price_path: {len(df)} rows, {len(df.columns)} columns")

    # Build scanner events using best_config.json scanner variant
    cfg_path = Path(__file__).parent / "best_config.json"
    if cfg_path.exists():
        cfg = json.load(open(cfg_path))
        scanner_name = cfg.get("scanner_name", "flow_reversal")
        scanner_variant = cfg.get("scanner_variant", "")
        # Parse variant params
        _param_str = scanner_variant.split("|", 1)[1] if "|" in scanner_variant else ""
        params = {}
        for kv in _param_str.split(","):
            if "=" in kv:
                k, v = kv.split("=", 1)
                params[k.strip()] = float(v.strip())
        # If no/partial orderbook data, disable spread and obi filters
        # "partial" = OB covers <50% of candle minutes (e.g. download still in progress)
        has_ob = ob is not None and not ob.empty
        if has_ob:
            ob_coverage = (df["data_from_orderbook"] > 0).mean() if "data_from_orderbook" in df.columns else 0.0
            logger.info(f"OB coverage: {ob_coverage:.1%} of minutes")
            if ob_coverage < 0.5:
                logger.info("OB coverage <50% — bypassing spread and obi filters")
                has_ob = False
                # Fill uncovered minutes with neutral non-zero defaults
                df["ob_spread_bps"] = df["ob_spread_bps"].replace(0.0, 1.0)
                df["obi"] = df["obi"].fillna(0.0)
        if not has_ob:
            logger.info("No orderbook data — bypassing spread and obi filters")
            if "spread_mult" in params:
                params["spread_mult"] = 0.0
                # Set dummy non-zero spread so (spread_bps > spread_ref * 0) = (1 > 0) = True
                df["ob_spread_bps"] = 1.0
            if "obi_abs" in params:
                params["obi_abs"] = -1.0  # obi > -1.0 always true
            # Rebuild variant string with updated params
            scanner_variant = scanner_name + "|" + ",".join(f"{k}={v}" for k, v in params.items())
        scanner_specs = [{"scanner_name": scanner_name, "params": params, "scanner_variant": scanner_variant}]
        logger.info(f"Building scanner events: {scanner_name} variant={scanner_variant}")
        events = build_scanner_events(df, scanner_specs)
        if events.empty:
            logger.warning("No scanner events found — check scanner params and data quality")
        else:
            logger.info(f"Scanner events: {len(events)} rows, long={(events['event_dir']==1).sum()}, short={(events['event_dir']==-1).sum()}")
            df = events
    else:
        logger.warning("best_config.json not found — skipping scanner events, saving raw features")

    output.parent.mkdir(parents=True, exist_ok=True)
    # Final cleanup: replace any remaining inf values with 0
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].replace([np.inf, -np.inf], 0.0)
    df.to_parquet(output)
    logger.info(f"Saved: {output} ({output.stat().st_size // 1024 // 1024} MB)")


def main():
    parser = argparse.ArgumentParser(description="Build Gate.io dataset parquet")
    parser.add_argument("--input-dir", default="/Volumes/TU280Pro/quant/raw_data")
    parser.add_argument("--output", default="dataset_gate_enhanced.parquet")
    parser.add_argument("--market", default="ETH_USDT")
    parser.add_argument("--start-month", required=True)
    parser.add_argument("--end-month", required=True)
    args = parser.parse_args()

    months = list(iter_months(args.start_month, args.end_month))
    logger.info(f"Building dataset: {args.market} {months[0]}..{months[-1]}")

    build_dataset(
        input_dir=Path(args.input_dir),
        market=args.market,
        months=months,
        output=Path(args.output),
    )


if __name__ == "__main__":
    main()
