#!/usr/bin/env python3
"""
normalize_features.py
Create features_norm/ by copying each per-feature parquet from features/
with its canonical name from the FULL TAXONOMY AUDIT dictionary.

  features/pe_ttm.parquet  →  features_norm/price_earnings_ttm.parquet
  features/vol_20d.parquet →  features_norm/realized_volatility_20d.parquet
  ...

Files with no taxonomy entry are copied with their original name + warning.
"""

import argparse, logging, shutil
from pathlib import Path

from feature_taxonomy import NORMALIZED_NAMES, TAXONOMY

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DATA_DIR  = Path.home() / "Library/CloudStorage/OneDrive-Personal/Aarush-One Drive/Summer 2026/Quant Papa Internship"
RAW_DIR   = DATA_DIR / "features"
NORM_DIR  = DATA_DIR / "features_norm"


def main():
    parser = argparse.ArgumentParser(description="Normalize feature filenames via taxonomy dictionary")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print mapping without copying files")
    args = parser.parse_args()

    if not RAW_DIR.exists():
        log.error("Raw features directory not found: %s", RAW_DIR)
        return

    NORM_DIR.mkdir(parents=True, exist_ok=True)

    raw_files = sorted(RAW_DIR.glob("*.parquet"))
    log.info("Raw features dir  : %s  (%d files)", RAW_DIR, len(raw_files))
    log.info("Norm features dir : %s", NORM_DIR)
    log.info("")

    mapped, unmapped = 0, 0
    for src in raw_files:
        col_name = src.stem
        if col_name in NORMALIZED_NAMES:
            canonical = NORMALIZED_NAMES[col_name]
            category  = TAXONOMY[col_name]["category"]
            dst = NORM_DIR / f"{canonical}.parquet"
            log.info("  %-40s → %s  [%s]", col_name, canonical, category)
            if not args.dry_run:
                shutil.copy2(src, dst)
            mapped += 1
        else:
            dst = NORM_DIR / src.name
            log.warning("  %-40s  (no taxonomy entry; kept original name)", col_name)
            if not args.dry_run:
                shutil.copy2(src, dst)
            unmapped += 1

    action = "Would write" if args.dry_run else "Written"
    log.info("")
    log.info("%s %d normalized + %d passthrough files to %s",
             action, mapped, unmapped, NORM_DIR)


if __name__ == "__main__":
    main()
