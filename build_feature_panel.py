"""
NKY 225 Feature Panel Builder
==============================
Fetches 10 years of daily OHLCV data for all 225 NKY constituents,
computes ~45 features per (stock, date) pair, and writes to parquet.

Output: nky225_features.parquet
Shape:  MultiIndex (date, ticker) × N_features columns
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from bs4 import BeautifulSoup
from tqdm import tqdm

# ── logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── constants ─────────────────────────────────────────────────────────────────
START       = "2014-06-01"
END         = datetime.today().strftime("%Y-%m-%d")
CACHE_DIR   = Path(__file__).parent / "_cache"
OUT_FILE    = Path(__file__).parent / "nky225_features.parquet"

CACHE_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. NKY 225 CONSTITUENT TICKERS
# ─────────────────────────────────────────────────────────────────────────────

HARDCODED_NKY225 = [
    "1332","1333","1605","1721","1801","1802","1803","1808","1812","1925",
    "1928","1963","2002","2269","2282","2413","2432","2501","2502","2503",
    "2531","2702","2801","2802","2871","2914","3086","3099","3289","3382",
    "3401","3402","3405","3407","3436","3659","3861","3863","3893","3941",
    "4004","4005","4021","4042","4043","4061","4063","4151","4183","4188",
    "4208","4272","4324","4452","4502","4503","4519","4523","4543","4568",
    "4578","4631","4689","4704","4751","4755","4901","4902","4911","5019",
    "5020","5101","5108","5201","5202","5214","5232","5233","5301","5332",
    "5333","5401","5406","5411","5541","5631","5703","5706","5707","5711",
    "5713","5714","5715","5741","5802","5803","5901","6088","6098","6103",
    "6113","6146","6178","6273","6301","6302","6305","6326","6361","6367",
    "6376","6383","6412","6479","6501","6503","6504","6506","6645","6674",
    "6701","6702","6703","6724","6752","6753","6758","6762","6770","6841",
    "6857","6861","6902","6952","6954","6971","6976","6988","7003","7004",
    "7011","7013","7012","7186","7201","7202","7203","7205","7211","7261",
    "7267","7269","7270","7272","7731","7733","7735","7741","7751","7752",
    "7762","7832","7911","7912","7951","7974","8001","8002","8003","8015",
    "8028","8031","8035","8053","8058","8233","8252","8267","8306","8308",
    "8309","8316","8331","8354","8355","8411","8601","8604","8628","8630",
    "8697","8725","8729","8750","8766","8795","8802","8804","8830","9001",
    "9005","9007","9008","9009","9020","9021","9022","9062","9064","9101",
    "9104","9107","9202","9301","9432","9433","9434","9531","9532","9602",
    "9613","9681","9697","9719","9735","9766","9983","9984",
]


def get_nky225_tickers() -> list[str]:
    """Scrape Wikipedia for current NKY 225 codes; fall back to hardcoded list."""
    cache_file = CACHE_DIR / "nky225_codes.txt"
    if cache_file.exists() and (time.time() - cache_file.stat().st_mtime < 86400):
        codes = cache_file.read_text().split()
        log.info("Loaded %d tickers from cache", len(codes))
        return codes

    log.info("Scraping Wikipedia for NKY 225 constituents …")
    try:
        url  = "https://en.wikipedia.org/wiki/Nikkei_225"
        resp = requests.get(url, timeout=15,
                            headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(resp.text, "html.parser")
        codes = []
        for table in soup.find_all("table", {"class": "wikitable"}):
            for row in table.find_all("tr")[1:]:
                cells = row.find_all("td")
                if cells:
                    code = cells[0].get_text(strip=True).replace("\xa0", "")
                    if code.isdigit() and len(code) == 4:
                        codes.append(code)

        if len(codes) < 200:
            raise ValueError(f"Only found {len(codes)} codes — using hardcoded list")

        cache_file.write_text("\n".join(codes))
        log.info("Found %d NKY 225 constituents from Wikipedia", len(codes))
        return codes

    except Exception as e:
        log.warning("Wikipedia scrape failed (%s); using hardcoded list", e)
        return HARDCODED_NKY225


# ─────────────────────────────────────────────────────────────────────────────
# 2. DOWNLOAD PRICE DATA
# ─────────────────────────────────────────────────────────────────────────────

def load_ohlcv(codes: list[str]) -> pd.DataFrame:
    """Download adjusted OHLCV for all NKY stocks + NKY index + USD/JPY."""
    cache_file = CACHE_DIR / "raw_ohlcv.parquet"

    if cache_file.exists():
        age_hours = (time.time() - cache_file.stat().st_mtime) / 3600
        if age_hours < 23:
            log.info("Loading raw OHLCV from cache (%.1f h old) …", age_hours)
            return pd.read_parquet(cache_file)

    tickers_yf  = [f"{c}.T" for c in codes]
    aux_tickers = ["^N225", "JPY=X"]   # NKY index, USD/JPY
    all_tickers = tickers_yf + aux_tickers

    log.info("Downloading %d tickers from yfinance (%s → %s) …",
             len(all_tickers), START, END)

    # yfinance batch download — one API call
    raw = yf.download(
        all_tickers,
        start=START,
        end=END,
        auto_adjust=True,
        group_by="ticker",
        threads=True,
        progress=True,
    )

    log.info("Download complete. Shape: %s", raw.shape)
    raw.to_parquet(cache_file)
    return raw


def extract_series(raw: pd.DataFrame, ticker: str, field: str) -> pd.Series:
    """Pull a single field for a single ticker from the multi-level DataFrame."""
    try:
        s = raw[(field, ticker)]
        return s.dropna(how="all")
    except KeyError:
        return pd.Series(dtype=float)


def build_stock_ohlcv(raw: pd.DataFrame, codes: list[str]) -> dict[str, pd.DataFrame]:
    """Reshape raw multi-ticker download into {ticker: OHLCV_df}.

    yfinance ≥1.2 uses (ticker, field) column order; earlier versions used (field, ticker).
    We detect which layout is present by checking the first level of the MultiIndex.
    """
    stock_data = {}
    fields = ["Open", "High", "Low", "Close", "Volume"]

    # Detect column layout
    sample_lvl0 = raw.columns.get_level_values(0)[0]
    ticker_first = sample_lvl0 not in fields   # True for yfinance ≥1.2

    for code in tqdm(codes, desc="Reshaping OHLCV"):
        ticker = f"{code}.T"
        try:
            if ticker_first:
                # yfinance ≥1.2: (ticker, field)
                df = raw[ticker].copy()
                df.columns = [c.lower() for c in df.columns]
            else:
                # old yfinance: (field, ticker)
                df = pd.concat(
                    {f.lower(): raw[(f, ticker)] for f in fields}, axis=1
                )
            df = df.dropna(how="all")
            if len(df) < 100:
                continue
            df.index = pd.to_datetime(df.index)
            stock_data[code] = df
        except Exception:
            pass
    log.info("Valid stocks with data: %d / %d", len(stock_data), len(codes))
    return stock_data


# ─────────────────────────────────────────────────────────────────────────────
# 3. FEATURE COMPUTATION (per stock)
# ─────────────────────────────────────────────────────────────────────────────

def compute_features(
    df: pd.DataFrame,
    nky_ret: pd.Series,
    usdjpy_ret: pd.Series,
) -> pd.DataFrame:
    """
    Given a stock's OHLCV DataFrame, compute all features.
    Returns a DataFrame with one row per trading day.
    """
    out = pd.DataFrame(index=df.index)

    c = df["close"]
    h = df["high"]
    l = df["low"]
    v = df["volume"]

    # ── raw price / volume ──────────────────────────────────────────────────
    out["close"]       = c
    out["high"]        = h
    out["low"]         = l
    out["open"]        = df["open"]
    out["volume"]      = v
    out["yen_volume"]  = c * v

    # ── returns ─────────────────────────────────────────────────────────────
    log_ret = np.log(c).diff()

    out["ret_1d"]  = c.pct_change(1)
    out["ret_1w"]  = c.pct_change(5)
    out["ret_1m"]  = c.pct_change(21)
    out["ret_3m"]  = c.pct_change(63)
    out["ret_6m"]  = c.pct_change(126)
    out["ret_12m"] = c.pct_change(252)

    # ── reversal & momentum ─────────────────────────────────────────────────
    out["reversal_1w"]    = -out["ret_1w"]
    out["reversal_1m"]    = -out["ret_1m"]
    # momentum = 12-month total EXCLUDING most-recent month (12-1)
    out["momentum_12_1"]  = c.shift(21).pct_change(231)

    # ── volatility ──────────────────────────────────────────────────────────
    out["vol_20d"]  = log_ret.rolling(20).std()  * np.sqrt(252)
    out["vol_60d"]  = log_ret.rolling(60).std()  * np.sqrt(252)
    out["vol_120d"] = log_ret.rolling(120).std() * np.sqrt(252)

    # Parkinson range-based volatility
    log_hl = np.log(h / l)
    out["vol_parkinson"] = np.sqrt(
        (1 / (4 * np.log(2))) * log_hl.pow(2).rolling(20).mean() * 252
    )

    # ── RSI (14-day) ────────────────────────────────────────────────────────
    delta     = c.diff()
    gain      = delta.clip(lower=0).rolling(14).mean()
    loss      = (-delta.clip(upper=0)).rolling(14).mean()
    rs        = gain / loss.replace(0, np.nan)
    out["rsi_14"] = 100 - (100 / (1 + rs))

    # ── MACD (12-26-9) ──────────────────────────────────────────────────────
    ema12             = c.ewm(span=12, adjust=False).mean()
    ema26             = c.ewm(span=26, adjust=False).mean()
    macd_line         = ema12 - ema26
    macd_signal_line  = macd_line.ewm(span=9, adjust=False).mean()
    out["macd_line"]   = macd_line
    out["macd_signal"] = macd_signal_line
    out["macd_hist"]   = macd_line - macd_signal_line

    # ── Bollinger Bands (20-day, 2σ) ────────────────────────────────────────
    bb_mid   = c.rolling(20).mean()
    bb_std   = c.rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    bb_range = (bb_upper - bb_lower).replace(0, np.nan)
    out["bb_pos"]     = (c - bb_lower) / bb_range      # 0=lower band, 1=upper
    out["bb_width"]   = bb_range / bb_mid               # relative bandwidth

    # ── EMA crossovers ──────────────────────────────────────────────────────
    ema5   = c.ewm(span=5,   adjust=False).mean()
    ema20  = c.ewm(span=20,  adjust=False).mean()
    ema50  = c.ewm(span=50,  adjust=False).mean()
    ema200 = c.ewm(span=200, adjust=False).mean()
    out["ema_cross_5_20"]   = ema5 / ema20.replace(0, np.nan) - 1
    out["ema_cross_50_200"] = ema50 / ema200.replace(0, np.nan) - 1

    # ── 52-week high proximity ───────────────────────────────────────────────
    high_52w          = h.rolling(252, min_periods=126).max()
    out["high_to_price"] = high_52w / c.replace(0, np.nan)  # >1 means below 52w-high

    # ── Amihud illiquidity (60-day rolling) ─────────────────────────────────
    yen_vol_nonzero   = out["yen_volume"].replace(0, np.nan)
    amihud_daily      = out["ret_1d"].abs() / yen_vol_nonzero
    out["amihud_60d"] = amihud_daily.rolling(60).mean()

    # ── Relative volume ──────────────────────────────────────────────────────
    vol_nonzero = v.replace(0, np.nan)
    out["rel_volume_5_60"] = (
        vol_nonzero.rolling(5).mean() / vol_nonzero.rolling(60).mean()
    )
    out["log_yen_volume"] = np.log(yen_vol_nonzero)

    # ── Market beta (60-day rolling OLS beta to NKY 225) ────────────────────
    stock_ret_aligned  = out["ret_1d"].reindex(nky_ret.index).ffill()

    def rolling_beta(s_ret: pd.Series, m_ret: pd.Series, w: int = 60) -> pd.Series:
        cov = s_ret.rolling(w).cov(m_ret)
        var = m_ret.rolling(w).var().replace(0, np.nan)
        return cov / var

    out["beta_nky_60d"]    = rolling_beta(out["ret_1d"], nky_ret,    60)
    out["beta_nky_252d"]   = rolling_beta(out["ret_1d"], nky_ret,   252)
    out["beta_usdjpy_60d"] = rolling_beta(out["ret_1d"], usdjpy_ret, 60)

    # ── USD/JPY macro features (same for all stocks, cross-sectional) ─────
    out["usdjpy_ret_1m"] = usdjpy_ret.rolling(21).sum()
    out["usdjpy_ret_3m"] = usdjpy_ret.rolling(63).sum()

    # ── Idiosyncratic volatility ─────────────────────────────────────────────
    # vol of OLS residuals after removing market exposure
    resid = out["ret_1d"] - out["beta_nky_60d"] * nky_ret
    out["idio_vol_60d"] = resid.rolling(60).std() * np.sqrt(252)

    # ── Calendar features ────────────────────────────────────────────────────
    idx = out.index
    out["fiscal_year_end_flag"] = (idx.month == 3).astype(np.int8)
    out["pre_fiscal_year_end"]  = (idx.month == 2).astype(np.int8)
    out["ex_div_season"]        = idx.month.isin([6, 7, 8, 9]).astype(np.int8)
    out["golden_week"]          = (
        ((idx.month == 4) & (idx.day >= 29)) |
        ((idx.month == 5) & (idx.day <= 6))
    ).astype(np.int8)
    out["obon_holiday"]         = (
        (idx.month == 8) & (idx.day >= 13) & (idx.day <= 16)
    ).astype(np.int8)
    out["calendar_year_end"]    = (idx.month == 12).astype(np.int8)
    out["day_of_week"]          = idx.dayofweek.astype(np.int8)
    out["is_monday"]            = (idx.dayofweek == 0).astype(np.int8)
    out["is_friday"]            = (idx.dayofweek == 4).astype(np.int8)

    # Japan fiscal quarter (Apr-Jun=Q1, Jul-Sep=Q2, Oct-Dec=Q3, Jan-Mar=Q4)
    fiscal_month     = ((idx.month - 4) % 12) + 1
    out["fiscal_quarter"] = np.ceil(fiscal_month / 3).astype(np.int8)

    return out


# ─────────────────────────────────────────────────────────────────────────────
# 4. BUILD FULL PANEL
# ─────────────────────────────────────────────────────────────────────────────

def build_panel(stock_data: dict, raw: pd.DataFrame) -> pd.DataFrame:
    """Stack all per-stock feature DataFrames into a long-format MultiIndex panel."""

    # --- auxiliary series ---
    log.info("Extracting NKY 225 index and USD/JPY …")
    # Handle both yfinance ≥1.2 (ticker, field) and older (field, ticker) layouts
    sample_lvl0 = raw.columns.get_level_values(0)[0]
    fields_known = {"Open", "High", "Low", "Close", "Volume"}
    ticker_first = sample_lvl0 not in fields_known

    if ticker_first:
        nky_close    = raw["^N225"]["Close"].dropna()
        usdjpy_close = raw["JPY=X"]["Close"].dropna()
    else:
        nky_close    = raw[("Close", "^N225")].dropna()
        usdjpy_close = raw[("Close", "JPY=X")].dropna()

    nky_ret    = np.log(nky_close).diff().rename("nky_ret")
    usdjpy_ret = np.log(usdjpy_close).diff().rename("usdjpy_ret")

    # align to a common trading-day index (union of NKY + stock data)
    nky_ret    = nky_ret.ffill()
    usdjpy_ret = usdjpy_ret.ffill()

    # --- compute per-stock features ---
    frames = []
    log.info("Computing features for %d stocks …", len(stock_data))

    for code, df in tqdm(stock_data.items(), desc="Features"):
        try:
            feat = compute_features(df, nky_ret, usdjpy_ret)
            feat.index = pd.to_datetime(feat.index)
            feat["ticker"] = code
            feat["ticker"] = feat["ticker"].astype("category")
            frames.append(feat.reset_index().rename(columns={"index": "date", "Date": "date"}))
        except Exception as e:
            log.warning("Skipped %s: %s", code, e)
            continue

    log.info("Concatenating %d stock panels …", len(frames))
    panel = pd.concat(frames, ignore_index=True)
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.set_index(["date", "ticker"]).sort_index()

    return panel


# ─────────────────────────────────────────────────────────────────────────────
# 5. MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("NKY 225 Feature Panel Builder")
    log.info("Date range: %s → %s", START, END)
    log.info("=" * 60)

    # 1. Tickers
    codes = get_nky225_tickers()
    log.info("Universe: %d stocks", len(codes))

    # 2. Download
    raw = load_ohlcv(codes)

    # 3. Reshape into per-stock OHLCV dicts
    stock_data = build_stock_ohlcv(raw, codes)

    # 4. Build feature panel
    panel = build_panel(stock_data, raw)

    # 5. Report
    n_dates   = panel.index.get_level_values("date").nunique()
    n_stocks  = panel.index.get_level_values("ticker").nunique()
    n_feats   = len(panel.columns)

    log.info("─" * 60)
    log.info("Panel shape: %d dates × %d stocks × %d features",
             n_dates, n_stocks, n_feats)
    log.info("Total rows : %d", len(panel))
    log.info("Date range : %s → %s",
             panel.index.get_level_values("date").min().date(),
             panel.index.get_level_values("date").max().date())
    log.info("Columns    : %s", list(panel.columns))

    # 6. Save
    log.info("Writing to %s …", OUT_FILE)
    panel.to_parquet(OUT_FILE, engine="pyarrow", compression="snappy")
    size_mb = OUT_FILE.stat().st_size / 1e6
    log.info("Saved. File size: %.1f MB", size_mb)
    log.info("Done.")

    return panel


if __name__ == "__main__":
    panel = main()
