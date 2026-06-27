"""
NKY225 — Accurate Index Weights via J-Quants (JPX Official Data)
=================================================================
Replaces the yfinance-based nky225_weight with weights computed from
J-Quants' official AdjustmentClose prices, which are the exact prices
that Nikkei Inc. uses in the Nikkei 225 weight formula:

    w_i(t) = PAF_i × AdjClose_i(t) / Σ_j [ PAF_j × AdjClose_j(t) ]

AdjustmentClose from J-Quants handles stock splits the same way Nikkei Inc.
does (via the official JPX adjustment factor), so PAF × AdjClose is the
authoritative "adjusted component price" for the index calculation.

Setup
-----
1. Register (free) at https://jpx-jquants.com/
2. Either:
   a. Set env vars:   JQUANTS_MAIL=...  JQUANTS_PASSWORD=...
   b. Or create ~/.config/jquants-api.toml:
        [jquants-api]
        mail_address = "you@example.com"
        password     = "your_password"
   c. Or pass --mail / --password on the command line

Usage
-----
    python3 compute_index_weights_jquants.py
    python3 compute_index_weights_jquants.py --mail you@example.com --password secret
"""

import argparse
import logging
import os
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

BASE     = Path(__file__).parent
DATA_DIR = Path.home() / "Library/CloudStorage/OneDrive-Personal/Aarush-One Drive/Summer 2026/Quant Papa Internship"
FEATS    = DATA_DIR / "nky225_features.parquet"

# ── Par Value Adjustment Factors ──────────────────────────────────────────────
# PAF_i = 50 / historical_par_value_i  (base par = ¥50)
# Source: Nikkei 225 Calculation Methodology, Nikkei Inc.
# These are FIXED factors frozen when Japan abolished par value (2006).
# AdjustmentClose from J-Quants handles splits; PAF handles legacy par values.
#
# Format: "XXXX" (4-digit TSE code without .T suffix) → float PAF
PAF: dict[str, float] = {
    # ── PAF = 0.1  (historical par ¥500) ─────────────────────────────────────
    "4151": 0.1,   "4506": 0.1,   "5201": 0.1,   "5401": 0.1,   "5411": 0.1,
    "6301": 0.1,   "6326": 0.1,   "6501": 0.1,   "6503": 0.1,   "6504": 0.1,
    "6508": 0.1,   "6701": 0.1,   "6702": 0.1,   "6724": 0.1,   "6752": 0.1,
    "6753": 0.1,   "6954": 0.1,   "7003": 0.1,   "7004": 0.1,   "7011": 0.1,
    "7012": 0.1,   "7013": 0.1,   "7201": 0.1,   "7202": 0.1,   "7203": 0.1,
    "7205": 0.1,   "7211": 0.1,   "7261": 0.1,   "7267": 0.1,   "7270": 0.1,
    "7733": 0.1,   "7751": 0.1,   "7752": 0.1,   "8001": 0.1,   "8002": 0.1,
    "8031": 0.1,   "8053": 0.1,   "8058": 0.1,   "8306": 0.1,   "8308": 0.1,
    "8309": 0.1,   "8316": 0.1,   "8411": 0.1,   "9020": 0.1,   "9022": 0.1,
    "9101": 0.1,   "9104": 0.1,   "9107": 0.1,   "9202": 0.1,   "9432": 0.1,
    "9433": 0.1,   "9602": 0.1,

    # ── PAF = 0.25  (historical par ¥200) ────────────────────────────────────
    "1605": 0.25,  "3101": 0.25,  "3402": 0.25,  "3405": 0.25,  "3861": 0.25,
    "4004": 0.25,  "4005": 0.25,  "4021": 0.25,  "4041": 0.25,  "4042": 0.25,
    "4061": 0.25,  "4063": 0.25,  "4183": 0.25,  "4188": 0.25,  "4208": 0.25,
    "4631": 0.25,  "5101": 0.25,  "5108": 0.25,  "5202": 0.25,  "5233": 0.25,
    "5301": 0.25,  "5332": 0.25,  "5334": 0.25,  "5631": 0.25,  "5706": 0.25,
    "5711": 0.25,  "5713": 0.25,  "5714": 0.25,  "5801": 0.25,  "5802": 0.25,
    "5803": 0.25,  "6302": 0.25,  "6361": 0.25,  "6367": 0.25,  "6471": 0.25,
    "6472": 0.25,  "6473": 0.25,  "6481": 0.25,  "6645": 0.25,  "6674": 0.25,
    "6723": 0.25,  "6762": 0.25,  "6857": 0.25,  "6902": 0.25,  "7735": 0.25,
    "7741": 0.25,  "7762": 0.25,  "7911": 0.25,  "7912": 0.25,  "8233": 0.25,
    "8252": 0.25,  "8801": 0.25,  "8802": 0.25,  "8830": 0.25,  "9005": 0.25,
    "9007": 0.25,  "9008": 0.25,  "9009": 0.25,  "9021": 0.25,  "9064": 0.25,
    "9201": 0.25,  "9501": 0.25,  "9502": 0.25,  "9503": 0.25,  "9531": 0.25,
    "9532": 0.25,  "9735": 0.25,  "9766": 0.25,
}

# ── NKY225 index code in J-Quants ─────────────────────────────────────────────
# Nikkei 225 = index code "0028" in the JPX/J-Quants system
NKY225_INDEX_CODE = "0028"


# ─────────────────────────────────────────────────────────────────────────────
# 1. AUTHENTICATION
# ─────────────────────────────────────────────────────────────────────────────

def get_client(mail: str = "", password: str = ""):
    """
    Build an authenticated J-Quants client.
    Credential priority: CLI args → env vars → ~/.config/jquants-api.toml
    """
    try:
        import jquantsapi
    except ImportError:
        raise SystemExit(
            "jquants-api-client not installed.  Run:\n"
            "  pip3 install jquants-api-client"
        )

    mail     = mail     or os.environ.get("JQUANTS_MAIL", "")
    password = password or os.environ.get("JQUANTS_PASSWORD", "")

    if mail and password:
        log.info("Authenticating with J-Quants as %s …", mail)
        cli = jquantsapi.Client(mail_address=mail, password=password)
    else:
        log.info("Loading J-Quants credentials from config file …")
        cli = jquantsapi.Client()   # reads ~/.config/jquants-api.toml

    # Quick connectivity test
    try:
        _ = cli.get_id_token()
        log.info("J-Quants authentication successful.")
    except Exception as e:
        raise SystemExit(f"J-Quants auth failed: {e}\n"
                         "Register free at https://jpx-jquants.com/")
    return cli


# ─────────────────────────────────────────────────────────────────────────────
# 2. FETCH ADJUSTMENT PRICES FROM J-QUANTS
# ─────────────────────────────────────────────────────────────────────────────

def fetch_adj_close(
    cli,
    tickers: list[str],
    start: str = "2014-01-01",
    end: str   = "",
) -> pd.DataFrame:
    """
    Download AdjustmentClose from J-Quants for all NKY225 tickers.
    Returns a (date, ticker) wide DataFrame of AdjustmentClose prices.

    J-Quants AdjustmentClose = Close × AdjustmentFactor, where AdjustmentFactor
    captures all splits/reverse-splits using the official JPX factor — the same
    mechanism Nikkei Inc. uses when computing adjusted component prices.
    """
    import datetime
    if not end:
        end = datetime.date.today().strftime("%Y-%m-%d")

    log.info("Fetching J-Quants prices: %s → %s for %d tickers …",
             start, end, len(tickers))

    # J-Quants uses 5-digit codes (e.g. "72030") on the free tier.
    # Map our 4-digit codes → 5-digit by appending "0"
    code_map   = {t + "0": t for t in tickers}   # "72030" → "7203"
    jq_codes   = list(code_map.keys())

    # get_price_range fetches ALL listed stocks; we filter after
    log.info("  Downloading full price history from J-Quants …")
    prices_all = cli.get_price_range(start_dt=start, end_dt=end)

    if prices_all.empty:
        raise ValueError("J-Quants returned empty price DataFrame — check tier/auth.")

    log.info("  J-Quants returned %s rows.", f"{len(prices_all):,}")

    # Filter to our NKY225 universe
    prices = prices_all[prices_all["Code"].isin(jq_codes)].copy()
    prices["ticker"] = prices["Code"].map(code_map)
    prices["Date"]   = pd.to_datetime(prices["Date"])

    log.info("  NKY225 tickers matched: %d of %d",
             prices["ticker"].nunique(), len(tickers))

    # Check for AdjustmentClose column
    if "AdjustmentClose" not in prices.columns:
        log.warning("  AdjustmentClose not in columns — using Close instead.")
        prices["AdjustmentClose"] = prices["Close"]

    # Pivot to wide (date × ticker)
    adj = (
        prices
        .pivot_table(index="Date", columns="ticker", values="AdjustmentClose")
        .sort_index()
    )
    adj.index.name = "date"
    adj.columns.name = "ticker"

    return adj


# ─────────────────────────────────────────────────────────────────────────────
# 3. COMPUTE PAF-ADJUSTED INDEX WEIGHTS
# ─────────────────────────────────────────────────────────────────────────────

def compute_weights(
    adj_close: pd.DataFrame,
    in_index:  pd.DataFrame,
) -> pd.Series:
    """
    w_i(t) = PAF_i × AdjClose_i(t) / Σ_j [ PAF_j × AdjClose_j(t) ]
    where the sum is only over stocks that are in-index on date t.

    Returns a Series indexed (date, ticker) matching the panel.
    """
    log.info("Applying PAF factors and computing weights …")
    tickers = adj_close.columns.tolist()
    pafs    = pd.Series({t: PAF.get(t, 1.0) for t in tickers}, dtype="float64")

    # paf-adjusted price matrix
    adj_paf = adj_close.multiply(pafs, axis="columns")

    # Zero out non-index stocks using in_index wide matrix
    # in_index is (date, ticker) MultiIndex → pivot to wide
    in_wide = in_index.unstack("ticker").fillna(False)
    in_wide.columns = in_wide.columns.droplevel(0)   # drop 'in_index' level

    # Align columns
    common  = sorted(set(adj_paf.columns) & set(in_wide.columns))
    adj_paf = adj_paf[common].reindex(in_wide.index)
    in_mask = in_wide[common].reindex(adj_paf.index).fillna(False)

    adj_paf = adj_paf.where(in_mask, other=0.0)

    # Row total → weights
    row_sum = adj_paf.sum(axis=1).replace(0, np.nan)
    wt_wide = adj_paf.div(row_sum, axis=0).fillna(0.0)

    # Sanity check
    daily_sum = wt_wide.sum(axis=1)
    bad = ((daily_sum - 1.0).abs() > 0.005).sum()
    if bad > 0:
        log.warning("  %d dates where weights don't sum to 1.000 ± 0.005", bad)
    else:
        log.info("  ✓ Weights sum to 1.000 on all dates.")

    wt_long = wt_wide.stack()
    wt_long.index.names = ["date", "ticker"]
    return wt_long.astype("float32")


# ─────────────────────────────────────────────────────────────────────────────
# 4. COMPARE WITH EXISTING WEIGHTS (SANITY CHECK)
# ─────────────────────────────────────────────────────────────────────────────

def compare_weights(panel: pd.DataFrame, new_weights: pd.Series) -> None:
    """Print top-10 diff vs old nky225_weight to confirm improvement."""
    if "nky225_weight" not in panel.columns:
        return

    latest = panel.index.get_level_values("date").max()
    old = panel.loc[panel.index.get_level_values("date") == latest, "nky225_weight"]
    new = new_weights.loc[new_weights.index.get_level_values("date") == latest]

    # Align
    combined = pd.DataFrame({"old": old, "new": new}).dropna()
    combined["diff_bp"] = (combined["new"] - combined["old"]) * 10_000

    top10 = combined.reindex(combined["new"].nlargest(10).index)

    log.info("─" * 64)
    log.info("Top 10 by NEW weight vs previous (latest date %s):", latest.date())
    log.info("  %-8s  %-12s  %-12s  %s", "ticker", "new_wt", "old_wt", "diff_bp")
    for tk, r in top10.iterrows():
        tk_code = tk[1] if isinstance(tk, tuple) else tk
        new_pct = f"{r['new']*100:.3f}%"
        old_pct = f"{r['old']*100:.3f}%"
        log.info("  %-8s  %-12s  %-12s  %+.1f", tk_code, new_pct, old_pct, r["diff_bp"])

    log.info("─" * 64)
    log.info("Mean absolute diff (all in-index): %.2f bp",
             combined["diff_bp"].abs().mean())


# ─────────────────────────────────────────────────────────────────────────────
# 5. UPDATE PANEL WITH DERIVED FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def update_panel_weights(panel: pd.DataFrame, new_weights: pd.Series) -> pd.DataFrame:
    """
    Replace nky225_weight and recompute all derived columns.
    Derived columns (cs_rank, excess_weight, log, chg_1m, chg_3m) are
    recalculated from the new base weight.
    """
    log.info("Updating panel with J-Quants-based weights …")

    # Replace core weight
    panel["nky225_weight"] = new_weights.reindex(panel.index).fillna(0.0).astype("float32")

    # ── cs_rank ──────────────────────────────────────────────────────────────
    w_in = panel["nky225_weight"].where(panel["in_index"] == True)
    panel["cs_rank_nky225_weight"] = (
        w_in.groupby(level="date")
            .rank(pct=True, na_option="keep")
            .astype("float32")
    )

    # ── excess_weight  (deviation from equal weight 1/N) ─────────────────────
    n_stocks = (panel["in_index"] == True).groupby(level="date").sum()
    n_ser    = panel.index.get_level_values("date").map(n_stocks)
    equal_w  = pd.Series(1.0 / n_ser.values, index=panel.index)
    panel["excess_weight"] = (
        (panel["nky225_weight"] - equal_w)
        .where(panel["in_index"] == True, other=0.0)
        .astype("float32")
    )

    # ── log weight ────────────────────────────────────────────────────────────
    w_pos = panel["nky225_weight"].replace(0, np.nan)
    panel["log_nky225_weight"] = np.log(w_pos).astype("float32")

    # ── weight change 1m / 3m ─────────────────────────────────────────────────
    w_wide = panel["nky225_weight"].unstack("ticker")
    for lag, name in [(21, "nky225_weight_chg_1m"), (63, "nky225_weight_chg_3m")]:
        chg = (w_wide - w_wide.shift(lag)).stack()
        chg.index.names = ["date", "ticker"]
        panel[name] = chg.reindex(panel.index).astype("float32")

    return panel


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Compute NKY225 weights via J-Quants")
    parser.add_argument("--mail",     default="", help="J-Quants registered email")
    parser.add_argument("--password", default="", help="J-Quants password")
    parser.add_argument("--start",    default="2014-01-01", help="Data start date")
    args = parser.parse_args()

    log.info("=" * 64)
    log.info("NKY225 Index Weights — J-Quants Official Data")
    log.info("=" * 64)

    # ── Load panel ────────────────────────────────────────────────────────────
    log.info("Loading panel …")
    panel = pd.read_parquet(FEATS)
    log.info("  %s rows × %d cols", f"{len(panel):,}", panel.columns.size)

    # All tickers in universe
    tickers = panel.index.get_level_values("ticker").unique().tolist()
    log.info("  Universe: %d tickers", len(tickers))

    # in_index series for masking
    in_index_series = panel["in_index"].copy()

    # ── Authenticate ──────────────────────────────────────────────────────────
    cli = get_client(args.mail, args.password)

    # ── Fetch adjusted prices ─────────────────────────────────────────────────
    end = panel.index.get_level_values("date").max().strftime("%Y-%m-%d")
    adj_close = fetch_adj_close(cli, tickers, start=args.start, end=end)

    # ── Compute weights ───────────────────────────────────────────────────────
    new_weights = compute_weights(adj_close, in_index_series.to_frame())

    # ── Compare against current weights ──────────────────────────────────────
    compare_weights(panel, new_weights)

    # ── Update panel ──────────────────────────────────────────────────────────
    panel = update_panel_weights(panel, new_weights)

    # ── Save ─────────────────────────────────────────────────────────────────
    log.info("Saving updated panel …")
    panel.to_parquet(FEATS, engine="pyarrow", compression="snappy")
    size_mb = FEATS.stat().st_size / 1e6
    log.info("Saved %s  (%.1f MB, %d cols)", FEATS.name, size_mb, panel.columns.size)
    log.info("Run complete. nky225_weight now uses J-Quants AdjustmentClose × PAF.")


if __name__ == "__main__":
    main()
