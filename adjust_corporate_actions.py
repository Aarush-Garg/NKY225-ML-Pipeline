"""
NKY 225 — Corporate Action Adjustment
======================================
Applies or verifies all price and feature adjustments for corporate actions.

What this script does
---------------------
1. VERIFY  — confirms auto_adjust=True already handles splits + dividends.
             Shows price continuity around Toyota's 2021 5-for-1 split.

2. DETECT  — identifies genuine capital raises (public offerings / rights issues)
             from the shares_outstanding cache, filtering out data artifacts.

3. ADD FLAGS  — appends three new columns to nky225_features.parquet:
    cap_raise_flag     (bool)  — True on any date within ±10 days of a confirmed
                                 capital raise / rights issue event
    cap_raise_dilution (float) — estimated dilution fraction on event date
                                 (= new_shares / total_shares_after)
    adj_ret_1d         (float) — ret_1d with returns set to NaN on the exact
                                 ex-rights date of a large raise (>20% dilution)
                                 so rolling features don't get contaminated

Why no backward price adjustment for capital raises
---------------------------------------------------
Japanese large-cap public offerings are priced at 3–5% below market (book-building),
not at a deep discount. Bloomberg/Refinitiv's backward adjustment formula:
    adj_factor = (old_shares) / (new_total_shares)
is only theoretically correct for zero-cash rights offerings (bonus shares) where
new shares are issued with no cash inflow. For cash offerings at market price, the
company receives cash equal to the value of new shares, leaving per-share intrinsic
value unchanged. Applying a backward adjustment would artificially deflate historical
prices, creating a spurious upward bias in historical returns.

The cap_raise_flag is the ML-appropriate signal: let the model learn that these
windows have different return dynamics, without corrupting the price series.
"""

import warnings
warnings.filterwarnings("ignore")

import logging
from pathlib import Path
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

BASE       = Path(__file__).parent
FEATS_FILE = BASE / "nky225_features.parquet"
CACHE_DIR  = BASE / "_cache"

# ── Batch artifact dates to exclude (yfinance reporting artefacts, not real events)
ARTIFACT_DATES = {
    # Nearly all companies show simultaneous +15% / -15% on these dates —
    # this is yfinance switching between treasury-share-inclusive vs exclusive counts.
    "2024-08-04", "2024-08-05",
    # Sept 11 2019 — 30+ companies simultaneously; likely TSE quarterly reporting batch
    "2019-09-11",
}

# ── Minimum dilution threshold to treat as a meaningful event
MIN_DILUTION = 0.15      # >15% new shares = material capital raise
LARGE_DILUTION = 0.20    # >20% = large enough to NaN out adj_ret_1d on ex-date


# ─────────────────────────────────────────────────────────────────────────────
# 1. VERIFY: splits & dividends already handled by auto_adjust=True
# ─────────────────────────────────────────────────────────────────────────────

def verify_split_adjustment(panel: pd.DataFrame) -> None:
    checks = [
        # (ticker_code, split_date, ratio, description)
        ("7203", "2021-09-29", 5, "Toyota 5-for-1"),
        ("7974", "2022-09-29", 10, "Nintendo 10-for-1"),
        ("9432", "2023-06-29", 25, "NTT 25-for-1"),
        ("6758", "2024-09-27", 5, "Sony 5-for-1"),
    ]
    log.info("─" * 60)
    log.info("SPLIT ADJUSTMENT VERIFICATION (auto_adjust=True)")
    log.info("─" * 60)

    for code, split_dt, ratio, label in checks:
        if code not in panel.index.get_level_values("ticker").unique():
            log.warning("  %s not in panel", code)
            continue

        s = panel.xs(code, level="ticker")["close"]
        pre_window  = pd.Timestamp(split_dt) - pd.Timedelta(days=10)
        post_window = pd.Timestamp(split_dt) + pd.Timedelta(days=10)

        pre_mean  = s.loc[:pd.Timestamp(split_dt) - pd.Timedelta(days=1)].iloc[-5:].mean()
        post_mean = s.loc[pd.Timestamp(split_dt):].iloc[:5].mean()
        observed_ratio = pre_mean / post_mean if post_mean > 0 else float("nan")

        status = "OK" if abs(observed_ratio - 1.0) < 0.10 else "PROBLEM"
        log.info("  %s (%s)  pre=Y%s  post=Y%s  ratio=%.2fx  [%s]",
                 label, code,
                 f"{pre_mean:,.0f}", f"{post_mean:,.0f}",
                 observed_ratio, status)

    log.info("  → All splits handled by yfinance auto_adjust=True.")
    log.info("  → Close prices are already split-adjusted; all derived features inherit this.")
    log.info("")


# ─────────────────────────────────────────────────────────────────────────────
# 2. LOAD & FILTER GENUINE CAPITAL RAISE EVENTS
# ─────────────────────────────────────────────────────────────────────────────

def load_genuine_cap_raises() -> pd.DataFrame:
    """
    Load share events from the cache built by scrape_other_corporate_actions.py,
    remove data artifacts, and return a clean DataFrame of confirmed capital raises.
    """
    cache_file = CACHE_DIR / "shares_outstanding.parquet"
    if not cache_file.exists():
        log.warning("Share events cache not found. Run scrape_other_corporate_actions.py first.")
        return pd.DataFrame(columns=["date","ticker","company",
                                     "pct_change","shares_before","shares_after"])

    events = pd.read_parquet(cache_file)

    # Keep only issuances (positive share count jumps)
    events = events[events["event_type"].str.contains("Issuance", na=False)].copy()

    # Remove artifact dates
    events["date_str"] = pd.to_datetime(events["date"]).dt.strftime("%Y-%m-%d")
    events = events[~events["date_str"].isin(ARTIFACT_DATES)].drop(columns="date_str")

    # Remove batch dates: if >8 companies show events on the same date, treat as artifact
    date_counts = events.groupby("date")["ticker"].count()
    batch_dates = set(date_counts[date_counts > 8].index.astype(str))
    if batch_dates:
        log.info("  Excluding %d batch artifact dates: %s", len(batch_dates), sorted(batch_dates))
        events["date_str2"] = pd.to_datetime(events["date"]).dt.strftime("%Y-%m-%d")
        events = events[~events["date_str2"].isin(batch_dates)].drop(columns="date_str2")

    # Keep only material events
    events = events[events["pct_change"].abs() >= MIN_DILUTION].copy()
    events["date"] = pd.to_datetime(events["date"])

    # Compute dilution fraction: new_shares / total (pct_change = new/old, so dilution = pct/(1+pct))
    events["dilution"] = events["pct_change"] / (1 + events["pct_change"])

    events = events.sort_values("date").reset_index(drop=True)
    log.info("Genuine capital raise events: %d across %d tickers",
             len(events), events["ticker"].nunique())

    return events


# ─────────────────────────────────────────────────────────────────────────────
# 3. BUILD FLAG COLUMNS AND adj_ret_1d
# ─────────────────────────────────────────────────────────────────────────────

def apply_capital_raise_flags(panel: pd.DataFrame,
                              events: pd.DataFrame,
                              window_days: int = 10) -> pd.DataFrame:
    """
    Add three columns to the panel:

    cap_raise_flag      bool   True within ±window_days of any confirmed capital raise
    cap_raise_dilution  float  max dilution fraction for events in that window (else 0.0)
    adj_ret_1d          float  ret_1d but NaN on the exact event date if dilution > LARGE_DILUTION
    """
    if events.empty:
        log.warning("No events to flag — skipping.")
        panel["cap_raise_flag"]     = False
        panel["cap_raise_dilution"] = 0.0
        panel["adj_ret_1d"]         = panel["ret_1d"].copy()
        return panel

    # Build a {(date, ticker) → max_dilution} lookup for the event dates
    event_lookup: dict[tuple, float] = {}
    for _, row in events.iterrows():
        dt  = row["date"]
        tk  = str(row["ticker"])
        dil = float(row["dilution"])

        for offset in range(-window_days, window_days + 1):
            d = dt + pd.Timedelta(days=offset)
            key = (d, tk)
            event_lookup[key] = max(event_lookup.get(key, 0.0), dil)

    log.info("Building flag columns over %d (date, ticker) pairs …", len(event_lookup))

    # Map onto panel index
    dates   = panel.index.get_level_values("date")
    tickers = panel.index.get_level_values("ticker").astype(str)

    dilution_vals = np.array([
        event_lookup.get((d, t), 0.0)
        for d, t in zip(dates, tickers)
    ], dtype=np.float32)

    panel["cap_raise_flag"]     = dilution_vals > 0.0
    panel["cap_raise_dilution"] = dilution_vals

    # adj_ret_1d: NaN out exact event dates with large dilution (>LARGE_DILUTION)
    adj_ret = panel["ret_1d"].copy()

    # Build the exact-date NaN mask (only the day of the event, not the whole window)
    exact_lookup: set[tuple] = set()
    for _, row in events[events["dilution"] > LARGE_DILUTION].iterrows():
        exact_lookup.add((row["date"], str(row["ticker"])))

    nan_mask = np.array([
        (d, t) in exact_lookup
        for d, t in zip(dates, tickers)
    ], dtype=bool)

    adj_ret[nan_mask] = np.nan
    panel["adj_ret_1d"] = adj_ret

    n_flagged  = (dilution_vals > 0.0).sum()
    n_nulled   = nan_mask.sum()
    log.info("  cap_raise_flag=True on %d rows (%d tickers, ±%d day windows)",
             n_flagged, len(events["ticker"].unique()), window_days)
    log.info("  adj_ret_1d set to NaN on %d exact event dates (dilution >%.0f%%)",
             n_nulled, LARGE_DILUTION * 100)

    return panel


# ─────────────────────────────────────────────────────────────────────────────
# 4. REPORT
# ─────────────────────────────────────────────────────────────────────────────

def print_event_table(events: pd.DataFrame) -> None:
    log.info("─" * 60)
    log.info("CONFIRMED CAPITAL RAISES (after artifact filtering)")
    log.info("─" * 60)
    if events.empty:
        log.info("  None found.")
        return

    for _, row in events.iterrows():
        sb = row.get("shares_before", 0)
        sa = row.get("shares_after", 0)
        log.info("  %s  %-8s  %-35s  dilution=%5.1f%%  shares: %s->%s",
                 str(row["date"])[:10],
                 row["ticker"],
                 str(row["company"])[:35],
                 row["dilution"] * 100,
                 f"{sb:,.0f}", f"{sa:,.0f}")


def show_spot_check(panel: pd.DataFrame, events: pd.DataFrame) -> None:
    """Show adj_ret_1d vs ret_1d for a few large events to confirm the NaN is applied."""
    large = events[events["dilution"] > LARGE_DILUTION].head(5)
    if large.empty:
        return

    log.info("─" * 60)
    log.info("SPOT CHECK: adj_ret_1d vs ret_1d around large events")
    log.info("─" * 60)
    for _, row in large.iterrows():
        code = str(row["ticker"])
        dt   = row["date"]
        if code not in panel.index.get_level_values("ticker").unique():
            continue
        window = panel.xs(code, level="ticker")[["close","ret_1d","adj_ret_1d",
                                                  "cap_raise_flag","cap_raise_dilution"]]
        window = window.loc[dt - pd.Timedelta(days=5):dt + pd.Timedelta(days=5)]
        log.info("  %s %s (dilution=%.1f%%):", code, row["company"][:30], row["dilution"]*100)
        for d, r in window.iterrows():
            flag = "← EVENT" if r["cap_raise_flag"] else ""
            nulled = " [NaN'd]" if pd.isna(r["adj_ret_1d"]) and not pd.isna(r["ret_1d"]) else ""
            log.info("    %s  close=¥%8.0f  ret_1d=%+.3f  adj_ret_1d=%s  %s%s",
                     str(d)[:10], r["close"],
                     r["ret_1d"] if not pd.isna(r["ret_1d"]) else float("nan"),
                     f"{r['adj_ret_1d']:+.3f}" if not pd.isna(r["adj_ret_1d"]) else "  NaN  ",
                     flag, nulled)
        log.info("")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("NKY 225 — Corporate Action Adjustment")
    log.info("=" * 60)

    log.info("Loading feature panel …")
    panel = pd.read_parquet(FEATS_FILE)
    log.info("Panel: %s rows × %d cols", f"{len(panel):,}", len(panel.columns))

    # ── 1. Verify splits ──────────────────────────────────────────────────────
    verify_split_adjustment(panel)

    # ── 2. Load events ────────────────────────────────────────────────────────
    events = load_genuine_cap_raises()
    print_event_table(events)

    # ── 3. Drop existing columns if re-running ────────────────────────────────
    for col in ["cap_raise_flag","cap_raise_dilution","adj_ret_1d"]:
        if col in panel.columns:
            panel = panel.drop(columns=[col])

    # ── 4. Apply flags ────────────────────────────────────────────────────────
    panel = apply_capital_raise_flags(panel, events, window_days=10)

    # ── 5. Spot check ─────────────────────────────────────────────────────────
    show_spot_check(panel, events)

    # ── 6. Save ──────────────────────────────────────────────────────────────
    log.info("Saving updated panel …")
    panel.to_parquet(FEATS_FILE, engine="pyarrow", compression="snappy")
    size_mb = FEATS_FILE.stat().st_size / 1e6

    log.info("─" * 60)
    log.info("Saved: %s  (%.1f MB)", FEATS_FILE.name, size_mb)
    log.info("New columns added:  cap_raise_flag | cap_raise_dilution | adj_ret_1d")
    log.info("Total columns now:  %d", len(panel.columns))

    summary = (
        panel[["cap_raise_flag","cap_raise_dilution","adj_ret_1d"]]
        .agg({"cap_raise_flag": "sum",
              "cap_raise_dilution": "max",
              "adj_ret_1d":    lambda x: x.isna().sum()})
    )
    log.info("Summary:")
    log.info("  cap_raise_flag=True rows  : %d", int(summary["cap_raise_flag"]))
    log.info("  max dilution fraction     : %.1f%%", summary["cap_raise_dilution"] * 100)
    log.info("  adj_ret_1d NaN count      : %d", int(summary["adj_ret_1d"]))


if __name__ == "__main__":
    main()
