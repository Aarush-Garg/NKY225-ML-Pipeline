#!/usr/bin/env python3
"""
split_features.py
Pivot each column of nky225_features.parquet into a wide-format
(date × ticker) matrix and save as an individual parquet file.

Output location: <OneDrive>/features/<column_name>.parquet
Each file has date as index and ticker codes as columns.
"""

import argparse, logging
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DATA_DIR   = Path.home() / "Library/CloudStorage/OneDrive-Personal/Aarush-One Drive/Summer 2026/Quant Papa Internship"
FEATS_FILE = DATA_DIR / "nky225_features.parquet"
OUT_DIR    = DATA_DIR / "features"

# Columns to skip (0% coverage or raw OHLCV that belong in a separate price file)
SKIP_COLS = {
    "nikkei_vi", "nikkei_vi_ret_1m", "nikkei_vi_zscore",   # all NaN (^VNKY broken)
    "eps_growth_yoy",                                        # all NaN
    "open", "high", "low", "close", "volume", "yen_volume",  # raw price/volume
}


def main():
    parser = argparse.ArgumentParser(description="Split feature panel into per-feature parquets")
    parser.add_argument("--cols", nargs="+", metavar="COL",
                        help="Only split these columns (default: all)")
    parser.add_argument("--include-raw", action="store_true",
                        help="Also write OHLCV columns (skipped by default)")
    parser.add_argument("--min-coverage", type=float, default=0.001,
                        help="Skip columns with less than this fraction non-null (default 0.1%%)")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Loading panel from %s", FEATS_FILE)
    panel = pd.read_parquet(FEATS_FILE)
    log.info("Panel: %d rows × %d columns", *panel.shape)

    target_cols = args.cols if args.cols else panel.columns.tolist()

    skip = set(SKIP_COLS)
    if args.include_raw:
        skip -= {"open", "high", "low", "close", "volume", "yen_volume"}

    written, skipped = 0, 0
    total_bytes = 0

    for col in target_cols:
        if col not in panel.columns:
            log.warning("Column '%s' not in panel — skipping", col)
            continue

        if col in skip:
            log.debug("Skipping %s (in skip list)", col)
            skipped += 1
            continue

        coverage = panel[col].notna().mean()
        if coverage < args.min_coverage:
            log.debug("Skipping %s (coverage %.4f < threshold)", col, coverage)
            skipped += 1
            continue

        # Pivot to wide format: index=date, columns=ticker
        wide = panel[col].unstack(level="ticker")
        wide.index.name = "date"

        out_path = OUT_DIR / f"{col}.parquet"
        wide.to_parquet(out_path)

        size_kb = out_path.stat().st_size / 1024
        total_bytes += out_path.stat().st_size
        log.info("  %-40s  %5d dates × %3d tickers  %6.0f KB  coverage %.1f%%",
                 col, wide.shape[0], wide.shape[1], size_kb, coverage * 100)
        written += 1

    log.info("")
    log.info("Done.  Written: %d  Skipped: %d  Total size: %.1f MB",
             written, skipped, total_bytes / 1e6)
    log.info("Output directory: %s", OUT_DIR)


if __name__ == "__main__":
    main()
