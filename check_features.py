#!/usr/bin/env python3
"""
check_features.py
Diagnostic tool for the NKY225 feature parquet store.

Checks per file:
  • Shape consistency  — all files should share the same date index and ticker columns
  • NaN coverage       — overall and over the most-recent N trading days
  • Stale data         — fraction of rows where a value did not change from the prior day
                         (high staleness is EXPECTED for LOCF fundamentals / calendar flags;
                          suspicious for daily price/return/technical data)

Usage:
  python3 check_features.py                      # audit features/ (raw)
  python3 check_features.py --norm               # audit features_norm/
  python3 check_features.py --both               # audit both
  python3 check_features.py --top 20             # show the 20 worst by coverage
  python3 check_features.py --col pe_ttm         # inspect one feature (partial name ok)
  python3 check_features.py --recent 30          # use 30-day recent window (default 60)
  python3 check_features.py --flagged            # show only rows with issues
"""

import argparse, logging, sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    from feature_taxonomy import TAXONOMY
except ImportError:
    TAXONOMY = {}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DATA_DIR = Path.home() / "Library/CloudStorage/OneDrive-Personal/Aarush-One Drive/Summer 2026/Quant Papa Internship"
RAW_DIR  = DATA_DIR / "features"
NORM_DIR = DATA_DIR / "features_norm"

# Categories where high staleness is BY DESIGN (LOCF fundamentals, slow-moving signals)
EXPECTED_STALE = {
    "valuation", "profitability", "quality", "growth",  # quarterly LOCF fundamentals
    "corporate",   # rare capital-action events
    "cs_rank",     # pre-built ranks of fundamentals → inherit their staleness
    "calendar",    # binary/ordinal flags (mostly 0 by construction)
    "composite",   # pre-built composite scores
    "index",       # NKY225 weights change slowly (PAF adjustments a few times/year)
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def _category(stem: str) -> str:
    """Look up category from taxonomy, stripping normalisation suffixes first."""
    base = stem
    for sfx in ("_xs", "_ts252", "_ts126", "_ts63", "_ts60"):
        if stem.endswith(sfx):
            base = stem[: -len(sfx)]
            break
    return TAXONOMY.get(base, {}).get("category", "?")


def _stale_pct(df: pd.DataFrame) -> float:
    """
    Median across tickers of: (# rows where value unchanged from prior row) / (# non-null rows).
    Expects a numeric DataFrame (NaN-only cells fine; object dtype will raise).
    """
    diff_zero   = (df.diff() == 0)
    count_stale = diff_zero.sum()
    count_valid = df.notna().sum().clip(1)
    return float(count_stale.div(count_valid).median() * 100)


def _max_stale_run(series: pd.Series) -> int:
    """Longest consecutive run of unchanged non-null values in a Series."""
    arr   = series.dropna().values
    if len(arr) < 2:
        return 0
    is_eq = np.diff(arr) == 0
    if not is_eq.any():
        return 0
    # group consecutive True/False runs
    run_lengths = np.diff(np.where(np.concatenate(([is_eq[0]], is_eq[:-1] != is_eq[1:], [True])))[0])
    equal_runs  = run_lengths[0::2] if is_eq[0] else run_lengths[1::2]
    return int(equal_runs.max()) if len(equal_runs) else 0


# ── Core audit ────────────────────────────────────────────────────────────────

def audit_dir(
    directory: Path,
    recent_days: int = 60,
    top_n: Optional[int] = None,
    col_filter: Optional[str] = None,
    flagged_only: bool = False,
) -> pd.DataFrame:

    files = sorted(directory.glob("*.parquet"))
    if col_filter:
        files = [f for f in files if col_filter in f.stem]
    if not files:
        log.error("No parquet files found in %s", directory)
        return pd.DataFrame()

    log.info("Scanning %d files in %s", len(files), directory)

    ref_index   = None
    ref_cols    = None
    shape_issues: list[str] = []
    rows: list[dict] = []

    for i, fpath in enumerate(files, 1):
        stem = fpath.stem
        log.info("  [%3d/%d] %s", i, len(files), stem)

        df = pd.read_parquet(fpath)
        n_dates, n_tickers = df.shape

        # ── Shape consistency ─────────────────────────────────────────────
        if ref_index is None:
            ref_index = df.index
            ref_cols  = set(df.columns)
        else:
            if not df.index.equals(ref_index):
                shape_issues.append(f"{stem}: date index differs (shape {df.shape})")
            elif set(df.columns) != ref_cols:
                missing = ref_cols - set(df.columns)
                extra   = set(df.columns) - ref_cols
                shape_issues.append(
                    f"{stem}: ticker mismatch — missing {len(missing)}, extra {len(extra)}"
                )

        # ── Coerce to numeric once (handles bool/object/None dtypes) ─────
        dfn = df.apply(pd.to_numeric, errors="coerce")

        # ── Coverage ──────────────────────────────────────────────────────
        total  = n_dates * n_tickers
        cov    = (1 - dfn.isna().sum().sum() / total) * 100 if total else 0.0

        recent    = dfn.tail(recent_days)
        r_total   = recent.shape[0] * recent.shape[1]
        recent_cov = (1 - recent.isna().sum().sum() / r_total) * 100 if r_total else 0.0

        # ── Staleness ─────────────────────────────────────────────────────
        stale  = _stale_pct(dfn)

        # Max consecutive stale run across a sample of tickers
        sample_cols = dfn.columns[:20]
        max_run = int(max((_max_stale_run(dfn[c]) for c in sample_cols), default=0))

        # ── Flags ─────────────────────────────────────────────────────────
        cat        = _category(stem)
        exp_stale  = cat in EXPECTED_STALE
        flags: list[str] = []

        if cov < 5:
            flags.append("NO_DATA")
        elif cov < 50:
            flags.append("LOW_COV")
        if recent_cov < cov - 20:     # recent coverage significantly worse than overall
            flags.append("RECENT_DROP")
        if recent_cov < 50 and "NO_DATA" not in flags:
            flags.append("RECENT_LOW")
        if not exp_stale:
            if stale > 90:
                flags.append("STALE!")
            elif stale > 60:
                flags.append("STALE?")

        rows.append({
            "file":        stem,
            "cat":         cat,
            "shape":       f"{n_dates}×{n_tickers}",
            "cov_%":       round(cov, 1),
            "recent_%":    round(recent_cov, 1),
            "stale_%":     round(stale, 1),
            "max_run_d":   max_run,
            "exp_stale":   "Y" if exp_stale else "",
            "date_min":    str(df.index.min().date()),
            "date_max":    str(df.index.max().date()),
            "flags":       " ".join(flags),
        })

    result = pd.DataFrame(rows).sort_values("cov_%")

    # ── Print report ──────────────────────────────────────────────────────

    pd.set_option("display.max_colwidth", 60)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 500)

    print()
    print("=" * 80)
    print(f"  FEATURE AUDIT  —  {directory.name}  ({len(files)} files)")
    print("=" * 80)

    # Shape
    print("\n── Shape consistency ──")
    if shape_issues:
        for issue in shape_issues:
            print(f"  !! {issue}")
    else:
        r0 = result.iloc[0]
        print(f"  All files identical: {r0['shape']}  dates {r0['date_min']} → {r0['date_max']}  ✓")

    # Coverage buckets
    print("\n── Coverage buckets ──")
    print(f"  100 %         : {(result['cov_%'] == 100).sum():>4}")
    print(f"  80 – 99 %     : {((result['cov_%'] >= 80) & (result['cov_%'] < 100)).sum():>4}")
    print(f"  50 – 79 %     : {((result['cov_%'] >= 50) & (result['cov_%'] < 80)).sum():>4}")
    print(f"  < 50 %        : {(result['cov_%'] < 50).sum():>4}")
    print(f"  Flagged issues: {(result['flags'] != '').sum():>4}")

    # Flagged
    flagged = result[result["flags"] != ""]
    if len(flagged):
        print(f"\n── Flagged features ({len(flagged)}) ──")
        print(flagged[["file", "cat", "cov_%", "recent_%", "stale_%", "max_run_d", "exp_stale", "flags"]].to_string(index=False))

    # Main table
    display = result if not flagged_only else flagged
    if top_n:
        display = display.head(top_n)
        label = f"── Bottom {top_n} by coverage ──"
    else:
        label = "── All features (sorted by coverage ↑) ──"
    if not flagged_only:
        print(f"\n{label}")
        print(display[["file", "cat", "shape", "cov_%", "recent_%", "stale_%",
                        "max_run_d", "exp_stale", "date_min", "date_max", "flags"]].to_string(index=False))

    print()
    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Audit NaN coverage and stale data in feature parquets")
    p.add_argument("--norm",    action="store_true", help="Check features_norm/ instead of features/")
    p.add_argument("--both",    action="store_true", help="Check both directories")
    p.add_argument("--recent",  type=int, default=60, metavar="N",
                   help="Window (trading days) for recent-coverage check (default: 60)")
    p.add_argument("--top",     type=int, default=None, metavar="N",
                   help="Show only the N worst-coverage features")
    p.add_argument("--col",     metavar="SUBSTR",
                   help="Filter to files whose name contains SUBSTR")
    p.add_argument("--flagged", action="store_true",
                   help="Print only flagged features in the main table")
    args = p.parse_args()

    dirs = []
    if args.both:
        dirs = [RAW_DIR, NORM_DIR]
    elif args.norm:
        dirs = [NORM_DIR]
    else:
        dirs = [RAW_DIR]

    for d in dirs:
        audit_dir(d, args.recent, args.top, args.col, args.flagged)


if __name__ == "__main__":
    main()
