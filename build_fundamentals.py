#!/usr/bin/env python3
"""
build_fundamentals.py
Fetch quarterly + annual financial statements from yfinance for all NKY225 tickers,
compute point-in-time fundamental features (60-day reporting lag), and
merge them into nky225_features.parquet.

Coverage strategy:
  - Balance sheet metrics (P/B, D/E, current ratio): quarterly BS snapshots, LOCF
  - Profitability metrics (P/E, ROE, margins): quarterly TTM where available,
    annual data as fallback → extends history back ~4 years
  - Growth metrics (rev/EPS growth): annual YoY comparison
"""

import logging, warnings, sys, pickle, time, argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DATA_DIR   = Path.home() / "Library/CloudStorage/OneDrive-Personal/Aarush-One Drive/Summer 2026/Quant Papa Internship"
FEATS_FILE = DATA_DIR / "nky225_features.parquet"
CACHE_DIR  = Path("_cache/fins_raw")

REPORT_LAG = 60   # calendar days after period-end before data assumed public
SLEEP_SEC  = 0.3  # courtesy pause between yfinance calls

# ── Field maps ───────────────────────────────────────────────────────────────
QFIN_FIELDS = {
    "q_revenue":      "Total Revenue",
    "q_gross_profit": "Gross Profit",
    "q_op_income":    "Operating Income",
    "q_net_income":   "Net Income",
    "q_ebitda":       "EBITDA",
}
QCF_FIELDS = {
    "q_ocf": "Operating Cash Flow",
    "q_fcf": "Free Cash Flow",
}
ANN_FIN_FIELDS = {
    "a_revenue":      "Total Revenue",
    "a_gross_profit": "Gross Profit",
    "a_op_income":    "Operating Income",
    "a_net_income":   "Net Income",
    "a_ebitda":       "EBITDA",
}
ANN_CF_FIELDS = {
    "a_ocf": "Operating Cash Flow",
    "a_fcf": "Free Cash Flow",
}
BS_FIELDS = {
    "total_assets":        "Total Assets",
    "current_assets":      "Current Assets",
    "current_liabilities": "Current Liabilities",
    "common_equity":       "Common Stock Equity",
    "total_debt":          "Total Debt",
    "net_debt":            "Net Debt",
    "shares":              "Ordinary Shares Number",
}

# Fundamental features that get cross-sectional ranks
CS_RANK_COLS = [
    "pe_ttm", "pb_ratio", "ps_ratio", "ev_ebitda",
    "roe_ttm", "roa_ttm", "net_margin_ttm", "op_margin_ttm",
    "rev_growth_yoy", "netinc_growth_yoy",
    "debt_to_equity", "current_ratio", "accruals_ratio", "fcf_yield",
]


# ─────────────────────────────────────────────────────────────────────────────
#  Raw download + caching
# ─────────────────────────────────────────────────────────────────────────────

CACHE_VERSION = 2  # bump when cache schema changes

def download_ticker(ticker_code: str) -> dict:
    """
    Download quarterly + annual statements from yfinance; cache to _cache/fins_raw/.
    If cache exists but predates CACHE_VERSION, upgrades it by fetching annual data.
    """
    cache_file = CACHE_DIR / f"{ticker_code}.pkl"

    if cache_file.exists():
        with open(cache_file, "rb") as f:
            data = pickle.load(f)
        if data.get("_version", 1) >= CACHE_VERSION:
            return data
        # Upgrade: add annual data to existing cache
        yf_sym = f"{ticker_code}.T"
        tk = yf.Ticker(yf_sym)
        data["fin_ann"] = tk.financials
        data["cf_ann"]  = tk.cashflow
        data["_version"] = CACHE_VERSION
        with open(cache_file, "wb") as f:
            pickle.dump(data, f)
        time.sleep(SLEEP_SEC)
        return data

    yf_sym = f"{ticker_code}.T"
    tk = yf.Ticker(yf_sym)
    data = {
        "fin":     tk.quarterly_financials,
        "bs":      tk.quarterly_balance_sheet,
        "cf":      tk.quarterly_cashflow,
        "fin_ann": tk.financials,
        "cf_ann":  tk.cashflow,
        "_version": CACHE_VERSION,
    }
    with open(cache_file, "wb") as f:
        pickle.dump(data, f)
    time.sleep(SLEEP_SEC)
    return data


# ─────────────────────────────────────────────────────────────────────────────
#  Build point-in-time fundamental time series for one ticker
# ─────────────────────────────────────────────────────────────────────────────

def _extract_rows(df, field_map: dict) -> pd.DataFrame:
    """Extract named rows from a yfinance statement DataFrame → date-indexed."""
    if df is None or df.empty:
        return pd.DataFrame()
    frames = {}
    for dest, src in field_map.items():
        if src in df.index:
            frames[dest] = df.loc[src]
    if not frames:
        return pd.DataFrame()
    out = pd.DataFrame(frames)
    out.index = pd.DatetimeIndex(out.index)
    return out.sort_index()


def _locf_to_panel(df: pd.DataFrame, panel_dates: pd.DatetimeIndex,
                   lag_days: int = REPORT_LAG) -> pd.DataFrame:
    """Apply reporting lag then forward-fill df to panel dates."""
    if df.empty:
        return pd.DataFrame(index=panel_dates)
    df = df.copy()
    df.index = df.index + pd.Timedelta(days=lag_days)
    full_idx = df.index.union(panel_dates)
    return df.reindex(full_idx).sort_index().ffill().reindex(panel_dates)


def build_fundamental_ts(ticker_code: str, panel_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Return a panel_dates-indexed DataFrame of fundamental metrics, ready for ratio
    computation. Uses quarterly data where available, annual as fallback/extension.
    """
    raw = download_ticker(ticker_code)

    # ── Extract raw statements ───────────────────────────────────────────────
    fin_q = _extract_rows(raw.get("fin"),     QFIN_FIELDS)
    cf_q  = _extract_rows(raw.get("cf"),      QCF_FIELDS)
    fin_a = _extract_rows(raw.get("fin_ann"), ANN_FIN_FIELDS)
    cf_a  = _extract_rows(raw.get("cf_ann"),  ANN_CF_FIELDS)
    bs_df = _extract_rows(raw.get("bs"),      BS_FIELDS)

    if fin_q.empty and fin_a.empty and bs_df.empty:
        return pd.DataFrame(index=panel_dates)

    # ── Build TTM income/CF series ────────────────────────────────────────────
    # Quarterly: rolling 4-quarter sum (min 2 quarters required)
    ttm_q = pd.DataFrame()
    if not fin_q.empty:
        all_q = fin_q.join(cf_q, how="outer") if not cf_q.empty else fin_q
        q_cols = [c for c in all_q.columns if c.startswith("q_")]
        rolled = all_q[q_cols].rolling(4, min_periods=2).sum()
        ttm_q = rolled.rename(columns=lambda c: c.replace("q_", "ttm_"))

    # Annual: full-year figure IS the TTM for that year
    ttm_a = pd.DataFrame()
    if not fin_a.empty:
        all_a = fin_a.join(cf_a, how="outer") if not cf_a.empty else fin_a
        ttm_a = all_a.rename(columns=lambda c: c.replace("a_", "ttm_"))

    # Annual YoY growth (compare this year vs last year)
    growth = pd.DataFrame()
    if not fin_a.empty:
        g = pd.DataFrame(index=fin_a.index)
        if "a_revenue" in fin_a.columns:
            r = fin_a["a_revenue"]
            g["g_revenue"] = (r / r.shift(1) - 1).clip(-3, 10)
        if "a_net_income" in fin_a.columns:
            ni = fin_a["a_net_income"]
            g["g_net_income"] = (ni / ni.abs().shift(1) - 1).clip(-3, 10)
        growth = g.dropna(how="all")

    # ── Combine quarterly and annual TTM (quarterly takes priority) ───────────
    if not ttm_q.empty and not ttm_a.empty:
        all_dates = ttm_q.index.union(ttm_a.index)
        tq = ttm_q.reindex(all_dates)
        ta = ttm_a.reindex(all_dates)
        ttm_merged = tq.combine_first(ta)   # quarterly preferred, annual fills gaps
    elif not ttm_q.empty:
        ttm_merged = ttm_q
    elif not ttm_a.empty:
        ttm_merged = ttm_a
    else:
        ttm_merged = pd.DataFrame()

    # ── Assemble all report-date-indexed series ────────────────────────────────
    parts = [p for p in [ttm_merged, growth, bs_df] if not p.empty]
    if not parts:
        return pd.DataFrame(index=panel_dates)

    # Outer-join all parts, keeping the finest-grained date index per column
    combined = parts[0]
    for p in parts[1:]:
        combined = combined.join(p, how="outer")

    # ── Apply reporting lag and LOCF to panel dates ──────────────────────────
    return _locf_to_panel(combined, panel_dates)


# ─────────────────────────────────────────────────────────────────────────────
#  Compute ratios from fundamentals + price
# ─────────────────────────────────────────────────────────────────────────────

def _safe_div(a, b, clip_lo=None, clip_hi=None) -> pd.Series:
    result = (a / b).replace([np.inf, -np.inf], np.nan)
    if clip_lo is not None or clip_hi is not None:
        result = result.clip(clip_lo, clip_hi)
    return result


def compute_ratios(fund: pd.DataFrame, price: pd.DataFrame) -> pd.DataFrame:
    close = price["close"]

    shares     = fund.get("shares")
    mkt_cap    = close * shares if shares is not None else None

    equity     = fund.get("common_equity")
    assets     = fund.get("total_assets")
    total_debt = fund.get("total_debt")
    net_debt   = fund.get("net_debt")
    cur_assets = fund.get("current_assets")
    cur_liab   = fund.get("current_liabilities")

    rev  = fund.get("ttm_revenue")
    gp   = fund.get("ttm_gross_profit")
    opi  = fund.get("ttm_op_income")
    ni   = fund.get("ttm_net_income")
    ebt  = fund.get("ttm_ebitda")
    ocf  = fund.get("ttm_ocf")
    fcf  = fund.get("ttm_fcf")

    g_rev = fund.get("g_revenue")
    g_ni  = fund.get("g_net_income")

    r = pd.DataFrame(index=fund.index)

    # ── Valuation ────────────────────────────────────────────────────────────
    # P/E: derived from TTM net income / shares (avoids sparse yfinance EPS field)
    if ni is not None and shares is not None:
        eps = _safe_div(ni, shares)
        r["pe_ttm"] = _safe_div(close, eps).where(lambda x: x > 0).clip(0, 500)

    if equity is not None and shares is not None:
        bvps = _safe_div(equity, shares)
        r["pb_ratio"] = _safe_div(close, bvps).where(lambda x: x > 0).clip(0, 100)

    if rev is not None and shares is not None:
        r["ps_ratio"] = _safe_div(close, _safe_div(rev, shares)).where(lambda x: x > 0).clip(0, 200)

    if ebt is not None and net_debt is not None and mkt_cap is not None:
        ev = mkt_cap + net_debt
        r["ev_ebitda"] = _safe_div(ev, ebt).where(lambda x: x > 0).clip(0, 200)

    if fcf is not None and mkt_cap is not None:
        r["fcf_yield"] = _safe_div(fcf, mkt_cap).clip(-0.5, 0.5)

    # ── Profitability ────────────────────────────────────────────────────────
    if ni is not None and equity is not None:
        r["roe_ttm"] = _safe_div(ni, equity).clip(-2, 5)

    if ni is not None and assets is not None:
        r["roa_ttm"] = _safe_div(ni, assets).clip(-1, 2)

    if ni is not None and rev is not None:
        r["net_margin_ttm"] = _safe_div(ni, rev).clip(-2, 2)

    if opi is not None and rev is not None:
        r["op_margin_ttm"] = _safe_div(opi, rev).clip(-2, 2)

    if gp is not None and rev is not None:
        r["gross_margin_ttm"] = _safe_div(gp, rev).clip(-1, 2)

    if fcf is not None and rev is not None:
        r["fcf_margin_ttm"] = _safe_div(fcf, rev).clip(-2, 2)

    # ── Growth ───────────────────────────────────────────────────────────────
    if g_rev is not None:
        r["rev_growth_yoy"] = g_rev.clip(-1, 5)

    if g_ni is not None:
        r["netinc_growth_yoy"] = g_ni.clip(-3, 10)

    # ── Quality / Balance sheet ──────────────────────────────────────────────
    if total_debt is not None and equity is not None:
        r["debt_to_equity"] = _safe_div(total_debt, equity).clip(0, 20)

    if net_debt is not None and ebt is not None:
        r["net_debt_to_ebitda"] = _safe_div(net_debt, ebt).clip(-5, 30)

    if cur_assets is not None and cur_liab is not None:
        r["current_ratio"] = _safe_div(cur_assets, cur_liab).clip(0, 20)

    if ni is not None and ocf is not None and assets is not None:
        r["accruals_ratio"] = _safe_div(ni - ocf, assets).clip(-1, 1)

    if assets is not None:
        assets_1y = assets.shift(252)
        r["asset_growth"] = (_safe_div(assets, assets_1y) - 1).clip(-1, 5)

    if fcf is not None and ni is not None:
        r["fcf_to_earnings"] = _safe_div(fcf, ni.abs()).clip(-5, 10)

    return r


# ─────────────────────────────────────────────────────────────────────────────
#  Cross-sectional ranks
# ─────────────────────────────────────────────────────────────────────────────

def add_cs_ranks(panel: pd.DataFrame, fund_cols: list) -> pd.DataFrame:
    rank_cols = [c for c in CS_RANK_COLS if c in fund_cols]
    if not rank_cols:
        return panel
    log.info("Computing cross-sectional ranks for %d fundamental features…", len(rank_cols))
    dates = panel.index.get_level_values("date").unique()
    chunks = []
    for dt in dates:
        slice_ = panel.xs(dt, level="date")[rank_cols]
        ranked = slice_.rank(pct=True)
        ranked.columns = [f"cs_rank_{c}" for c in ranked.columns]
        ranked.index = pd.MultiIndex.from_tuples(
            [(dt, t) for t in slice_.index], names=["date", "ticker"]
        )
        chunks.append(ranked)
    ranks = pd.concat(chunks)
    return panel.join(ranks)


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Add fundamental features via yfinance")
    parser.add_argument("--refresh", action="store_true",
                        help="Delete cached downloads and re-fetch everything")
    parser.add_argument("--tickers", nargs="+", metavar="CODE",
                        help="Process only these ticker codes (space-separated)")
    args = parser.parse_args()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if args.refresh:
        log.info("--refresh: clearing %s", CACHE_DIR)
        for f in CACHE_DIR.glob("*.pkl"):
            f.unlink()

    log.info("Loading panel from %s", FEATS_FILE)
    panel = pd.read_parquet(FEATS_FILE)
    panel_dates = panel.index.get_level_values("date").unique().sort_values()
    all_tickers = panel.index.get_level_values("ticker").unique().tolist()

    target_tickers = args.tickers if args.tickers else all_tickers
    log.info("Processing %d tickers, %d panel dates", len(target_tickers), len(panel_dates))

    # ── Download + process each ticker ──────────────────────────────────────
    fund_parts = {}
    n = len(target_tickers)
    for i, tk in enumerate(target_tickers, 1):
        if i % 25 == 0 or i == n:
            log.info("  Fetching/upgrading %d/%d  (%s)", i, n, tk)
        try:
            fund_ts = build_fundamental_ts(tk, panel_dates)
            if fund_ts.empty or fund_ts.isna().all().all():
                continue
            price_tk = panel.xs(tk, level="ticker")[["close"]]
            ratios   = compute_ratios(fund_ts, price_tk)
            if not ratios.empty:
                fund_parts[tk] = ratios
        except Exception as exc:
            log.warning("  Skipped %s: %s", tk, exc)

    if not fund_parts:
        log.error("No fundamental data retrieved; exiting.")
        sys.exit(1)

    log.info("Built fundamental features for %d/%d tickers", len(fund_parts), len(target_tickers))

    # ── Assemble into panel MultiIndex ────────────────────────────────────────
    fund_panel = pd.concat(fund_parts, names=["ticker"])
    fund_panel.index = fund_panel.index.swaplevel(0, 1)
    fund_panel.index.names = ["date", "ticker"]
    fund_panel = fund_panel.sort_index()

    fund_cols = fund_panel.columns.tolist()
    log.info("Fundamental columns (%d): %s", len(fund_cols), fund_cols)

    # ── Drop stale fundamental columns if re-running ──────────────────────────
    existing_fund = [c for c in panel.columns
                     if c in fund_cols or any(c == f"cs_rank_{f}" for f in fund_cols)]
    if existing_fund:
        log.info("Dropping %d stale fundamental columns before re-merge", len(existing_fund))
        panel = panel.drop(columns=existing_fund)

    # ── Merge ────────────────────────────────────────────────────────────────
    panel = panel.join(fund_panel, how="left")

    # ── Cross-sectional ranks ────────────────────────────────────────────────
    panel = add_cs_ranks(panel, fund_cols)

    # ── Coverage report ───────────────────────────────────────────────────────
    log.info("Final panel: %d rows × %d columns", *panel.shape)
    recent_dates = panel_dates[-250:]
    recent = panel[panel.index.get_level_values("date").isin(recent_dates)]
    log.info("Coverage on last 250 trading days:")
    for col in fund_cols:
        if col in panel.columns:
            pct_all    = panel[col].notna().mean() * 100
            pct_recent = recent[col].notna().mean() * 100
            log.info("  %-28s  all-time %4.1f%%  recent %5.1f%%", col, pct_all, pct_recent)

    # ── Save ─────────────────────────────────────────────────────────────────
    log.info("Saving to %s  (%.0f MB)", FEATS_FILE, FEATS_FILE.stat().st_size / 1e6)
    panel.to_parquet(FEATS_FILE)
    log.info("Done.  Panel now %d columns.", panel.shape[1])


if __name__ == "__main__":
    main()
