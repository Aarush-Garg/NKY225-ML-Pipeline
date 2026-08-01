#!/usr/bin/env python3
"""
build_normalized_parquets.py
Compute TS and CS z-score parquets for every feature using the
normalization config in feature_taxonomy.py.

Naming convention (canonical names, mirrors features/ with suffixes):
  Time-series z-score   :  features_norm/<canonical>_ts<window>.parquet
  Cross-sectional zscore:  features_norm/<canonical>_xs.parquet

Each file is wide-format: index=date, columns=ticker codes.

Processing pipeline (per feature):
  1. [optional] log1p(clip(x, 0))   — for right-skewed features (Amihud etc.)
  2. [optional] cross-sectional winsorize at [1st, 99th] percentile per date
     (Hou-Xue-Zhang 2020 RFS; Green-Hand-Zhang 2017 RFS)
  3a. TS z-score: per-ticker rolling z-score for each configured window
  3b. CS z-score: per-date z-score across the ticker cross-section

Literature:
  Hou, Xue, Zhang (2020 RFS) — winsorize at 1%–99% before CS z-score
  Green, Hand, Zhang (2017 RFS) — same; for 94 characteristics
  Moskowitz, Ooi, Pedersen (2012 JFE) — TS z-score adds info beyond CS
  Amihud (2002 JFM); Lou, Shu (2016 JFE) — log transform for illiquidity
  MSCI Barra USE4 methodology — clip at ±3σ (≈ quantile winsorize in practice)
"""

import argparse, logging
from pathlib import Path

import numpy as np
import pandas as pd

from feature_taxonomy import TAXONOMY, NORMALIZED_NAMES, get_norm_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DATA_DIR = Path.home() / "Library/CloudStorage/OneDrive-Personal/Aarush-One Drive/Summer 2026/Quant Papa Internship"
RAW_DIR  = DATA_DIR / "features"
NORM_DIR = DATA_DIR / "features_norm"


# ── Normalisation helpers ──────────────────────────────────────────────────────

def cross_winsorize(df: pd.DataFrame, lo: float = 0.01, hi: float = 0.99) -> pd.DataFrame:
    """
    Clip each value to the [lo, hi] cross-sectional quantile of its date-row.
    Operates row-wise so the clip bounds adapt to each date's distribution.
    """
    lower = df.quantile(lo, axis=1)
    upper = df.quantile(hi, axis=1)
    return df.clip(lower=lower, upper=upper, axis=0)


def ts_zscore(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Per-column (per-ticker) rolling z-score.
    min_periods = window // 2 so values appear once half the window is filled.
    """
    min_p = max(window // 2, 5)
    roll  = df.rolling(window, min_periods=min_p)
    mu    = roll.mean()
    sigma = roll.std().replace(0, np.nan)
    return (df - mu) / sigma


def cs_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-row (per-date) z-score across all ticker columns.
    Rows with fewer than 5 non-null values return NaN to avoid degenerate scores.
    """
    row_mean = df.mean(axis=1)
    row_std  = df.std(axis=1).replace(0, np.nan)
    valid    = df.notna().sum(axis=1)
    z = df.sub(row_mean, axis=0).div(row_std, axis=0)
    z[valid < 5] = np.nan
    return z


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build TS and CS z-score parquets for all features"
    )
    parser.add_argument("--cols", nargs="+", metavar="COL",
                        help="Process only these internal column names")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print plan without computing or writing")
    args = parser.parse_args()

    NORM_DIR.mkdir(parents=True, exist_ok=True)

    raw_files = {f.stem: f for f in RAW_DIR.glob("*.parquet")}
    target    = args.cols if args.cols else sorted(raw_files)

    log.info("Raw dir  : %s  (%d files)", RAW_DIR, len(raw_files))
    log.info("Norm dir : %s", NORM_DIR)
    log.info("Features : %d to process", len(target))
    log.info("")

    ts_written = xs_written = skipped = 0

    for col in target:
        if col not in raw_files:
            log.warning("  %-40s  not found in features/ — skipping", col)
            continue

        cfg        = get_norm_config(col)
        ts_wins    = cfg["ts_windows"]     # list[int], e.g. [60, 252] or []
        do_xs      = cfg["xs"]
        do_winsor  = cfg["winsorize"]
        do_log     = cfg["log_first"]
        canonical  = NORMALIZED_NAMES.get(col, col)

        if not ts_wins and not do_xs:
            log.info("  %-40s  SKIP  (binary/categorical/pre-normalised)", col)
            skipped += 1
            continue

        ts_label  = ",".join(f"ts{w}" for w in ts_wins) if ts_wins else "—"
        xs_label  = "xs" if do_xs else "—"
        pre_label = ("log+" if do_log else "") + ("winsor " if do_winsor else "")
        log.info("  %-40s  [%s]%s  %s  → %s", col, pre_label, ts_label, xs_label, canonical)

        if args.dry_run:
            ts_written += len(ts_wins)
            xs_written += int(do_xs)
            continue

        # Load wide-format parquet (date × ticker)
        df = pd.read_parquet(raw_files[col])

        # ── Step 1: log transform (before winsorize, for severely skewed data) ──
        if do_log:
            df = np.log1p(df.clip(lower=0))

        # ── Step 2: cross-sectional winsorize (applied to the base series) ──────
        df_pre = cross_winsorize(df) if do_winsor else df

        # ── Step 3a: TS z-score (one parquet per window) ────────────────────────
        for w in ts_wins:
            z_ts  = ts_zscore(df_pre, w)
            out_p = NORM_DIR / f"{canonical}_ts{w}.parquet"
            z_ts.to_parquet(out_p)
            ts_written += 1

        # ── Step 3b: CS z-score ─────────────────────────────────────────────────
        if do_xs:
            z_xs  = cs_zscore(df_pre)
            out_p = NORM_DIR / f"{canonical}_xs.parquet"
            z_xs.to_parquet(out_p)
            xs_written += 1

    log.info("")
    log.info("Done.  TS files: %d   XS files: %d   Skipped: %d",
             ts_written, xs_written, skipped)
    log.info("Output : %s", NORM_DIR)


if __name__ == "__main__":
    main()
