"""
Extended Feature Engineering — NKY 225 Panel
=============================================
Adds ~65 new features to nky225_features.parquet across 6 groups:

  A. Advanced technical oscillators     (14 features)
  B. Distribution / tail-risk           (10 features)
  C. Volume & price-volume interaction  ( 8 features)
  D. Cross-sectional rank signals       (18 features)
  E. Macro regime (Nikkei VI + rates)   (13 features)
  F. Composite factor scores            ( 6 features)
  + Fix beta_usdjpy_60d (FX alignment bug)

Research basis
--------------
Feature selection guided by:
  • NKY225_Feature_Research.md  — Tier 1/2 signal evidence from Goldman, SSGA,
    GMO, SuMi Trust, Verdad, OQ Funds, arXiv papers
  • NKY225_Techniques_Research.md — IC/ICIR ranking, SHAP importance literature
  • Key Japan-specific findings:
      - Short-term reversal > price momentum in Japan (Lo & MacKinlay, Iwanaga 2024)
      - Amihud illiquidity among highest-IC factors on TSE (Iwanaga et al.)
      - Idiosyncratic skewness negatively priced (Harvey & Siddique 2000)
      - Maximum 1-month return (MAX) negatively priced in Japan (Bali et al. 2011)
      - Volume trend as flow proxy (foreign investor imbalance signal)
      - USD/JPY beta strongest macro signal for Japan equities (Goldman AM 2024)
      - Low-volatility premium robust in Japan (Ang et al. 2006, Baker et al. 2011)
      - Cross-sectional rank normalisation reduces IC noise (Lopez de Prado 2018)
"""

import warnings
warnings.filterwarnings("ignore")

import time
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DATA_DIR    = Path.home() / "Library/CloudStorage/OneDrive-Personal/Aarush-One Drive/Summer 2026/Quant Papa Internship"
FEATS_FILE  = DATA_DIR / "nky225_features.parquet"
CACHE_DIR   = Path(__file__).parent / "_cache"
MACRO_CACHE = CACHE_DIR / "macro_aux.parquet"
START = "2014-01-01"


# ─────────────────────────────────────────────────────────────────────────────
# GROUP A — Advanced technical oscillators
# Research: RSI variants, stochastic (George & Hwang 2004), ATR-normalised
# momentum (Baltas & Kosowski 2012), CCI, MFI
# ─────────────────────────────────────────────────────────────────────────────

def add_oscillators(out: pd.DataFrame, c, h, l, v) -> pd.DataFrame:
    """
    Stochastic, ATR, Williams %R, CCI, MFI, ROC, OBV, Donchian, Keltner, ADX.
    All computed from OHLCV — no extra data download required.
    """
    # ── Stochastic oscillator K (14-period), D (3-period SMA of K) ──────────
    # Basis: George & Hwang (2004) 52-week high; Stochastic is the rolling version
    lo14 = l.rolling(14).min()
    hi14 = h.rolling(14).max()
    denom = (hi14 - lo14).replace(0, np.nan)
    stoch_k = 100 * (c - lo14) / denom
    out["stoch_k"] = stoch_k
    out["stoch_d"] = stoch_k.rolling(3).mean()
    out["stoch_kd_cross"] = stoch_k - out["stoch_d"]  # positive = bullish crossover

    # ── Average True Range (14D) — volatility normalisation baseline ─────────
    # Used in ATR-normalised momentum (Baltas & Kosowski 2012)
    prev_c   = c.shift(1)
    tr       = pd.concat([h - l,
                           (h - prev_c).abs(),
                           (l - prev_c).abs()], axis=1).max(axis=1)
    atr14    = tr.rolling(14).mean()
    out["atr_14"]       = atr14
    out["atr_pct"]      = atr14 / c.replace(0, np.nan)        # ATR as % of price
    out["atr_norm_ret"] = out.get("ret_1m", c.pct_change(21)) / atr14  # ATR-normalised momentum

    # ── Williams %R (14D) — overbought/oversold ───────────────────────────────
    out["williams_r"] = -100 * (hi14 - c) / denom              # range [-100, 0]

    # ── Commodity Channel Index (20D) ────────────────────────────────────────
    # CCI measures deviation of price from its mean relative to MAD
    tp       = (h + l + c) / 3                                 # typical price
    tp_ma    = tp.rolling(20).mean()
    tp_mad   = tp.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    out["cci_20"] = (tp - tp_ma) / (0.015 * tp_mad.replace(0, np.nan))

    # ── Money Flow Index (14D) — volume-weighted RSI ─────────────────────────
    # Basis: Chaikin Money Flow / MFI literature for order-flow imbalance
    mf     = tp * v
    pos_mf = mf.where(tp > tp.shift(1), 0.0)
    neg_mf = mf.where(tp < tp.shift(1), 0.0)
    pos_sum = pos_mf.rolling(14).sum()
    neg_sum = neg_mf.rolling(14).sum().replace(0, np.nan)
    mfr     = pos_sum / neg_sum
    out["mfi_14"] = 100 - (100 / (1 + mfr))

    # ── Rate of Change (ROC) — various periods ────────────────────────────────
    # Simple momentum at additional horizons
    out["roc_5"]  = c.pct_change(5)   * 100
    out["roc_10"] = c.pct_change(10)  * 100
    out["roc_20"] = c.pct_change(20)  * 100

    # ── On-Balance Volume (OBV) ───────────────────────────────────────────────
    # Basis: OBV momentum captures institutional accumulation (Granville 1963)
    direction = np.sign(c.diff())
    obv       = (direction * v).cumsum()
    out["obv"]        = obv
    out["obv_ret_1m"] = obv.pct_change(21)    # OBV momentum
    out["obv_ret_3m"] = obv.pct_change(63)

    # ── Donchian Channel position (20D) ──────────────────────────────────────
    # Related to 52-week high momentum — George & Hwang (2004)
    don_hi = h.rolling(20).max()
    don_lo = l.rolling(20).min()
    don_rng = (don_hi - don_lo).replace(0, np.nan)
    out["donchian_pos"] = (c - don_lo) / don_rng            # 0=bottom, 1=top

    # ── Keltner Channel position (20D) ───────────────────────────────────────
    ema20      = c.ewm(span=20, adjust=False).mean()
    kelt_upper = ema20 + 2 * atr14
    kelt_lower = ema20 - 2 * atr14
    kelt_rng   = (kelt_upper - kelt_lower).replace(0, np.nan)
    out["keltner_pos"] = (c - kelt_lower) / kelt_rng

    # ── ADX — Average Directional Index (14D, Wilder 1978) ───────────────────
    # ADX measures trend STRENGTH (0–100), not direction.
    #   ADX > 25 → strong trend (up or down); ADX < 20 → range-bound / choppy.
    # di_diff_14 = +DI − −DI: positive means bullish trend dominates.
    #   This directional component is a momentum signal (added to RANK_COLS).
    # Wilder smoothing = EMA with alpha = 1/N (heavier weight on recent bars
    # than a simple rolling mean, lighter than standard EMA span=N).
    prev_h   = h.shift(1)
    prev_l   = l.shift(1)
    dm_plus  = (h - prev_h).clip(lower=0)
    dm_minus = (prev_l - l).clip(lower=0)
    # When both DMs are equal, both zero out; otherwise the smaller is zeroed
    dm_plus  = dm_plus.where(dm_plus >= dm_minus, 0.0)
    dm_minus = dm_minus.where(dm_minus > dm_plus,  0.0)

    atr_w    = tr.ewm(alpha=1/14, adjust=False).mean().replace(0, np.nan)
    di_plus  = 100 * dm_plus.ewm(alpha=1/14,  adjust=False).mean() / atr_w
    di_minus = 100 * dm_minus.ewm(alpha=1/14, adjust=False).mean() / atr_w
    di_sum   = (di_plus + di_minus).replace(0, np.nan)
    dx       = 100 * (di_plus - di_minus).abs() / di_sum
    adx      = dx.ewm(alpha=1/14, adjust=False).mean()

    out["adx_14"]      = adx
    out["di_plus_14"]  = di_plus
    out["di_minus_14"] = di_minus
    out["di_diff_14"]  = di_plus - di_minus   # momentum: positive = bullish

    return out


# ─────────────────────────────────────────────────────────────────────────────
# GROUP B — Distribution / tail-risk features
# Research: Idiosyncratic skewness negatively priced (Harvey & Siddique 2000);
# MAX return anomaly (Bali et al. 2011) — tested on Japan by Iwanaga (2024);
# Historical VaR as risk measure (Rockafellar & Uryasev 2000)
# ─────────────────────────────────────────────────────────────────────────────

def add_tail_risk(out: pd.DataFrame, log_ret: pd.Series) -> pd.DataFrame:

    # ── Realised skewness (20D, 60D) ─────────────────────────────────────────
    # Negative skew premium: investors pay for insurance against left tails
    out["skew_20d"]  = log_ret.rolling(20).skew()
    out["skew_60d"]  = log_ret.rolling(60).skew()

    # ── Realised excess kurtosis (20D, 60D) ──────────────────────────────────
    out["kurt_20d"]  = log_ret.rolling(20).kurt()
    out["kurt_60d"]  = log_ret.rolling(60).kurt()

    # ── MAX return (highest 1D return in past month) ─────────────────────────
    # Bali, Cakici & Whitelaw (2011): MAX negatively predicts returns
    # Replicated on Japan by Iwanaga (2024) — strong Tier 2 signal
    out["max_ret_1m"]  = log_ret.rolling(21).max()
    out["min_ret_1m"]  = log_ret.rolling(21).min()   # crash risk complement

    # ── Historical Value-at-Risk 95% (20D) ───────────────────────────────────
    out["var_95_20d"] = log_ret.rolling(20).quantile(0.05)

    # ── Expected Shortfall / CVaR 95% (20D) ──────────────────────────────────
    def cvar_95(x):
        q = np.quantile(x, 0.05)
        tail = x[x <= q]
        return tail.mean() if len(tail) > 0 else np.nan
    out["cvar_95_20d"] = log_ret.rolling(20).apply(cvar_95, raw=True)

    # ── Maximum drawdown (60D, 252D) ─────────────────────────────────────────
    # Drawdown momentum (stocks that fell hard tend to recover or continue)
    def max_dd(x):
        peaks = pd.Series(x).cummax()
        dd    = (pd.Series(x) - peaks) / peaks.replace(0, np.nan)
        return dd.min()

    out["max_dd_60d"]  = log_ret.rolling(60).apply(max_dd,  raw=True)
    out["max_dd_252d"] = log_ret.rolling(252).apply(max_dd, raw=True)

    # ── Volatility-of-volatility (20D) ───────────────────────────────────────
    # Basis: Corsi et al. (2013) — VoV as proxy for uncertainty-of-uncertainty
    vol5d = log_ret.rolling(5).std()
    out["vol_of_vol_20d"] = vol5d.rolling(20).std()

    # ── Gain-Loss ratio (20D) ─────────────────────────────────────────────────
    gains  = log_ret.clip(lower=0).rolling(20).mean()
    losses = log_ret.clip(upper=0).abs().rolling(20).mean().replace(0, np.nan)
    out["gain_loss_ratio_20d"] = gains / losses

    return out


# ─────────────────────────────────────────────────────────────────────────────
# GROUP C — Volume and price-volume interaction
# Research: Order imbalance (Chordia & Subrahmanyam 2004); share turnover
# anomaly (Datar et al. 1998); Amihud (2002); foreign flow proxy via volume
# ─────────────────────────────────────────────────────────────────────────────

def add_volume_features(out: pd.DataFrame, c, v, log_ret: pd.Series) -> pd.DataFrame:

    yen_vol    = c * v
    yen_vol_nz = yen_vol.replace(0, np.nan)
    v_nz       = v.replace(0, np.nan)

    # ── Volume trend (5D vs 252D) ─────────────────────────────────────────────
    # High volume relative to history signals institutional activity
    out["vol_ratio_5_252"]  = v_nz.rolling(5).mean()  / v_nz.rolling(252).mean()
    out["vol_ratio_20_252"] = v_nz.rolling(20).mean() / v_nz.rolling(252).mean()

    # ── Price-volume divergence ────────────────────────────────────────────────
    # Price up + volume down = weak signal (bearish divergence)
    ret_sign    = np.sign(log_ret)
    vol_chg     = v_nz.pct_change(5)
    vol_chg_sgn = np.sign(vol_chg)
    out["price_vol_diverge"] = (ret_sign != vol_chg_sgn).astype(np.int8)

    # ── Dollar volume Z-score (252D) ──────────────────────────────────────────
    # Proxy for size factor (market cap) and liquidity tier
    dv_mu  = yen_vol_nz.rolling(252).mean()
    dv_std = yen_vol_nz.rolling(252).std().replace(0, np.nan)
    out["dv_zscore_252d"] = (yen_vol_nz - dv_mu) / dv_std

    # ── Turnover deceleration (volume momentum reversal) ─────────────────────
    # Basis: Lee & Swaminathan (2000) — volume as momentum life-cycle signal
    out["turnover_decel"] = (
        v_nz.rolling(5).mean() / v_nz.rolling(20).mean() - 1
    )

    # ── Signed volume (Tick-direction volume proxy) ────────────────────────────
    # Up-volume minus down-volume over 20D — order flow imbalance
    up_vol   = v.where(log_ret > 0, 0.0).rolling(20).sum()
    down_vol = v.where(log_ret < 0, 0.0).rolling(20).sum()
    total_vol_20 = (up_vol + down_vol).replace(0, np.nan)
    out["vol_imbalance_20d"] = (up_vol - down_vol) / total_vol_20

    # ── Chaikin Money Flow (20D) ───────────────────────────────────────────────
    # Measures buying/selling pressure using close position within H-L range
    h = out.get("high", c)
    l = out.get("low",  c)
    hl_rng = (h - l).replace(0, np.nan)
    mfm    = ((c - l) - (h - c)) / hl_rng              # money flow multiplier [-1,1]
    mfv    = mfm * v
    out["cmf_20"] = mfv.rolling(20).sum() / v_nz.rolling(20).sum().replace(0, np.nan)

    # ── VWAP deviation (approximate — daily OHLCV) ────────────────────────────
    # Typical price as VWAP proxy; deviation signals institutional vs retail
    tp      = (h + l + c) / 3
    vwap_20 = (tp * v).rolling(20).sum() / v_nz.rolling(20).sum().replace(0, np.nan)
    out["vwap_dev_20d"] = (c - vwap_20) / vwap_20.replace(0, np.nan)

    return out


# ─────────────────────────────────────────────────────────────────────────────
# GROUP D — Cross-sectional rank features (computed AFTER stacking all stocks)
# Research: Lopez de Prado (2018) §4 — rank transformation reduces IC noise
# and makes features more comparable across stocks with different price scales.
# Computed within the in-index universe to match portfolio construction scope.
# ─────────────────────────────────────────────────────────────────────────────

RANK_COLS = [
    "ret_1d",
    "ret_1w",
    "ret_1m",
    "ret_3m",
    "ret_12m",
    "reversal_1m",
    "momentum_12_1",
    "vol_20d",
    "vol_60d",
    "amihud_60d",
    "rsi_14",
    "bb_pos",
    "high_to_price",
    "log_yen_volume",
    "max_ret_1m",
    "skew_20d",
    "stoch_k",
    "obv_ret_1m",
    "di_diff_14",   # ADX directional: +DI−−DI, positive=bullish trend
    "adx_14",       # ADX trend strength: >25 strong trend, <20 choppy
]

def add_cross_sectional_ranks(panel: pd.DataFrame) -> pd.DataFrame:
    """
    For each feature in RANK_COLS, compute the percentile rank within the
    in-index universe on each date. Result is in [0, 1].

    Also adds cross-sectional z-score for vol and return dispersion.
    """
    log.info("  Computing cross-sectional ranks for %d signals …", len(RANK_COLS))

    # Restrict to existing columns
    cols_present = [c for c in RANK_COLS if c in panel.columns]

    for col in tqdm(cols_present, desc="CS ranks", leave=False):
        rank_col = f"cs_rank_{col}"
        # rank pct=True gives [0,1]; na_option='keep' preserves NaN
        panel[rank_col] = (
            panel.groupby(level="date")[col]
            .rank(pct=True, na_option="keep")
        )

    # ── Cross-sectional z-scores for key signals ──────────────────────────────
    for col in ["vol_20d", "ret_1m", "ret_12m", "amihud_60d"]:
        if col not in panel.columns:
            continue
        grp  = panel.groupby(level="date")[col]
        mu   = grp.transform("mean")
        std  = grp.transform("std").replace(0, np.nan)
        panel[f"cs_z_{col}"] = (panel[col] - mu) / std

    # ── Quantile decile (1–10) for top signals ────────────────────────────────
    for col in ["ret_1m", "momentum_12_1", "vol_20d"]:
        if col not in panel.columns:
            continue
        panel[f"cs_decile_{col}"] = (
            panel.groupby(level="date")[col]
            .transform(lambda x: pd.qcut(x.rank(method="first"), 10,
                                          labels=False, duplicates="drop") + 1)
        ).astype("Int8")

    return panel


# ─────────────────────────────────────────────────────────────────────────────
# GROUP E — Macro regime features
# Research: Goldman Sachs AM (2024/2025) — USD/JPY beta #1 Japan signal;
# BOJ VIX (Nikkei VI) as risk-on/off regime; JGB slope as rate sensitivity;
# global risk (VIX) drives Japan beta exposures
# ─────────────────────────────────────────────────────────────────────────────

def download_macro_aux() -> pd.DataFrame:
    """
    Download additional macro series and cache to parquet.
    Sources: yfinance (Nikkei VI, US markets), FRED (rates, VIX)
    """
    if MACRO_CACHE.exists():
        age_h = (time.time() - MACRO_CACHE.stat().st_mtime) / 3600
        if age_h < 23:
            log.info("  Loading macro aux from cache (%.1f h old)", age_h)
            return pd.read_parquet(MACRO_CACHE)

    log.info("  Downloading macro auxiliary series from yfinance …")
    tickers = {
        "^VNKY":  "nikkei_vi",      # Nikkei Volatility Index (Japan's VIX)
        "^VIX":   "vix",            # CBOE VIX
        "^N225":  "nky",            # NKY 225 index level
        "^GSPC":  "spx",            # S&P 500
        "JPY=X":  "usdjpy",         # USD/JPY spot
        "^TNX":   "us10y",          # US 10Y Treasury yield (×10 = %)
        "^TYX":   "us30y",          # US 30Y
        "^FVX":   "us5y",           # US 5Y
    }

    raw = yf.download(
        list(tickers.keys()),
        start=START,
        auto_adjust=True,
        group_by="ticker",
        threads=True,
        progress=False,
    )

    macro = pd.DataFrame(index=raw.index)
    for yf_tk, col_name in tickers.items():
        try:
            macro[col_name] = raw[yf_tk]["Close"]
        except Exception:
            log.warning("    Could not extract %s (%s)", yf_tk, col_name)

    macro.index = pd.to_datetime(macro.index)
    macro = macro.dropna(how="all")
    macro.to_parquet(MACRO_CACHE)
    log.info("  Saved macro aux cache (%d rows × %d cols)", len(macro), len(macro.columns))
    return macro


def build_macro_panel(macro: pd.DataFrame, trading_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Derive macro features from raw macro series and forward-fill to
    match the panel's trading-day index.
    """
    m = macro.reindex(trading_dates).ffill()

    out = pd.DataFrame(index=trading_dates)

    # ── Nikkei VI level and change ────────────────────────────────────────────
    if "nikkei_vi" in m:
        out["nikkei_vi"]        = m["nikkei_vi"]
        out["nikkei_vi_ret_1m"] = m["nikkei_vi"].pct_change(21)
        out["nikkei_vi_zscore"] = (
            (m["nikkei_vi"] - m["nikkei_vi"].rolling(252).mean()) /
            m["nikkei_vi"].rolling(252).std()
        )

    # ── CBOE VIX ─────────────────────────────────────────────────────────────
    if "vix" in m:
        out["vix"]           = m["vix"]
        out["vix_ret_1m"]    = m["vix"].pct_change(21)
        out["vix_zscore"]    = (
            (m["vix"] - m["vix"].rolling(252).mean()) /
            m["vix"].rolling(252).std()
        )

    # ── USD/JPY — level, momentum, z-score, trend ─────────────────────────────
    if "usdjpy" in m:
        usdjpy = m["usdjpy"]
        usdjpy_log = np.log(usdjpy)
        out["usdjpy_level"]   = usdjpy
        out["usdjpy_zscore"]  = (
            (usdjpy - usdjpy.rolling(252).mean()) /
            usdjpy.rolling(252).std()
        )
        out["usdjpy_ret_1w"]  = usdjpy_log.diff(5)
        out["usdjpy_trend"]   = (                          # EMA50 vs EMA200
            usdjpy.ewm(span=50,  adjust=False).mean() /
            usdjpy.ewm(span=200, adjust=False).mean() - 1
        )
        # Rolling USD/JPY volatility (regime indicator)
        out["usdjpy_vol_20d"] = usdjpy_log.diff().rolling(20).std() * np.sqrt(252)

    # ── US 10Y yield and curve ─────────────────────────────────────────────────
    if "us10y" in m and "us5y" in m:
        us10y = m["us10y"]
        us5y  = m["us5y"]
        out["us10y"]          = us10y
        out["us10y_ret_1m"]   = us10y.diff(21)             # yield change (bps-like)
        out["us_curve_5_10"]  = us10y - us5y               # yield curve slope

    # ── S&P 500 momentum (overnight signal for next-day Japan open) ───────────
    if "spx" in m:
        spx_log = np.log(m["spx"])
        out["spx_ret_1d"]  = spx_log.diff(1)
        out["spx_ret_1m"]  = spx_log.diff(21)
        out["spx_trend"]   = (
            m["spx"].ewm(span=50, adjust=False).mean() /
            m["spx"].ewm(span=200, adjust=False).mean() - 1
        )

    # ── NKY 225 index momentum (market timing signals) ───────────────────────
    if "nky" in m:
        nky_log = np.log(m["nky"])
        out["nky_ret_1m"]   = nky_log.diff(21)
        out["nky_ret_3m"]   = nky_log.diff(63)
        out["nky_ret_12m"]  = nky_log.diff(252)
        out["nky_trend"]    = (
            m["nky"].ewm(span=50,  adjust=False).mean() /
            m["nky"].ewm(span=200, adjust=False).mean() - 1
        )
        # Market volatility (realised 20D)
        out["nky_vol_20d"]  = nky_log.diff().rolling(20).std() * np.sqrt(252)

    # ── Risk-on / Risk-off regime flag ───────────────────────────────────────
    # Risk-off = VIX above 3M moving average AND NKY below 50D EMA
    if "vix" in m and "nky" in m:
        vix_above_ma  = m["vix"] > m["vix"].rolling(63).mean()
        nky_below_ema = m["nky"] < m["nky"].ewm(span=50, adjust=False).mean()
        out["risk_off_flag"] = (vix_above_ma & nky_below_ema).astype(np.int8)

    return out


def fix_usdjpy_beta(panel: pd.DataFrame, macro: pd.DataFrame,
                    trading_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Replace the 91%-NaN beta_usdjpy_60d with a correctly aligned version.

    Bug in original: rolling_cov(stock_ret, usdjpy_ret) failed because
    JPY=X trades on weekends while stocks don't — pandas couldn't align.

    Fix: forward-fill USD/JPY to trading-day calendar BEFORE computing returns,
    then compute rolling beta stock-by-stock.
    """
    log.info("  Fixing beta_usdjpy_60d (FX alignment bug) …")

    usdjpy_daily = (
        macro["usdjpy"]
        .reindex(trading_dates)
        .ffill()
    )
    usdjpy_ret = np.log(usdjpy_daily).diff().rename("usdjpy_ret")

    def rolling_beta(s_ret: pd.Series, m_ret: pd.Series, w: int = 60) -> pd.Series:
        cov = s_ret.rolling(w).cov(m_ret)
        var = m_ret.rolling(w).var().replace(0, np.nan)
        return cov / var

    tickers = panel.index.get_level_values("ticker").unique()
    new_betas = []

    for ticker in tqdm(tickers, desc="USD/JPY beta", leave=False):
        s = panel.xs(ticker, level="ticker")["ret_1d"]
        beta = rolling_beta(s, usdjpy_ret, 60).rename("beta_usdjpy_60d")
        beta.index = pd.MultiIndex.from_arrays(
            [beta.index, [ticker] * len(beta)], names=["date", "ticker"]
        )
        new_betas.append(beta)

    panel["beta_usdjpy_60d"] = pd.concat(new_betas).sort_index()
    nan_pct = panel["beta_usdjpy_60d"].isna().mean() * 100
    log.info("  beta_usdjpy_60d NaN after fix: %.1f%% (was 91%%)", nan_pct)
    return panel


# ─────────────────────────────────────────────────────────────────────────────
# GROUP F — Composite factor scores
# Research: Grinold-Kahn (2000) α = IC × σ × score; composite signals
# blend IC-weighted sub-signals (Asness et al. 2013, AQR multi-factor)
# ─────────────────────────────────────────────────────────────────────────────

def add_composite_scores(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Build composite factor scores by averaging rank-normalised sub-signals.
    Each composite is in [0,1]; higher = more attractive on that factor.

    Weights reflect Japan-specific IC evidence from Feature Research doc:
      Reversal (Tier 1), Low-vol (Tier 1), Amihud (Tier 1), Momentum (Tier 2)
    """

    def safe(col):
        return panel.get(col, pd.Series(np.nan, index=panel.index))

    # ── Reversal composite (Tier 1 Japan signal) ──────────────────────────────
    # Short-term reversal dominates price momentum in Japan
    rev_comps = [
        ("cs_rank_reversal_1m", 0.50),
        ("cs_rank_ret_1w",      0.30),   # 1W reversal
        ("cs_rank_stoch_k",     0.20),   # stochastic (overbought → mean-revert)
    ]
    panel["score_reversal"] = sum(
        safe(c) * w for c, w in rev_comps
        if c in panel.columns
    ) / sum(w for c, w in rev_comps if c in panel.columns)

    # ── Momentum composite (Tier 2 — weaker in Japan, use with caution) ───────
    # 12-1 month momentum with volume confirmation
    mom_comps = [
        ("cs_rank_momentum_12_1", 0.60),
        ("cs_rank_obv_ret_1m",    0.25),   # OBV confirms price momentum
        ("cs_rank_roc_20",        0.15) if "cs_rank_roc_20" not in panel.columns
            else ("cs_rank_ret_3m", 0.15),
    ]
    panel["score_momentum"] = sum(
        safe(c) * w for c, w in mom_comps
        if c in panel.columns
    ) / sum(w for c, w in mom_comps if c in panel.columns)

    # ── Low-volatility composite (Tier 1 Japan signal) ────────────────────────
    # Low vol / low beta premium is strong and persistent in Japan
    # Inverted ranks: lower vol = higher score
    low_vol_comps = [
        ("cs_rank_vol_20d",    -1.0),   # negate: low vol = high score
        ("cs_rank_vol_60d",    -0.5),
        ("cs_rank_amihud_60d", -0.3),   # liquid = low friction
        ("cs_rank_max_ret_1m", -0.2),   # low MAX = low lottery premium
    ]
    raw_scores = []
    weights    = []
    for c, w in low_vol_comps:
        if c in panel.columns:
            raw_scores.append(panel[c] * np.sign(w))
            weights.append(abs(w))
    if raw_scores:
        composite = sum(s * w for s, w in zip(raw_scores, weights)) / sum(weights)
        # Re-rank composite to [0,1]
        panel["score_low_vol"] = panel.groupby(level="date")[composite.name
            if hasattr(composite, "name") else "dummy"].transform(
            lambda x: x.rank(pct=True, na_option="keep")
        ) if False else composite.groupby(level="date").rank(pct=True, na_option="keep") / len(raw_scores)
        # Simpler: just normalise
        panel["score_low_vol"] = (
            sum(safe(c) * (1 - panel[c]) * abs(w) if w < 0 else safe(c) * w
                for c, w in low_vol_comps if c in panel.columns)
            / sum(abs(w) for c, w in low_vol_comps if c in panel.columns)
        )

    # ── Liquidity composite (Amihud + Volume) ────────────────────────────────
    liq_comps = [
        ("cs_rank_amihud_60d",    -0.50),  # low amihud = liquid = higher score
        ("cs_rank_log_yen_volume",  0.30),  # high turnover = liquid
        ("cs_rank_vol_ratio_5_252", 0.20) if "cs_rank_vol_ratio_5_252" in panel.columns
            else ("cs_rank_rel_volume_5_60", 0.20),
    ]
    panel["score_liquidity"] = sum(
        safe(c) * (1 - safe(c)) * abs(w) if w < 0
        else safe(c) * w
        for c, w in liq_comps if c in panel.columns
    ) / sum(abs(w) for c, w in liq_comps if c in panel.columns)

    # ── Technical sentiment composite ─────────────────────────────────────────
    tech_comps = [
        ("cs_rank_bb_pos",     0.25),
        ("cs_rank_stoch_k",    0.25),
        ("cs_rank_rsi_14",     0.25),
        ("cs_rank_obv_ret_1m", 0.25),
    ]
    panel["score_technical"] = sum(
        safe(c) * w for c, w in tech_comps
        if c in panel.columns
    ) / sum(w for c, w in tech_comps if c in panel.columns)

    return panel


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("Extended Feature Engineering")
    log.info("=" * 60)

    # ── 1. Load existing panel + OHLCV cache ─────────────────────────────────
    log.info("Loading feature panel …")
    panel = pd.read_parquet(FEATS_FILE)
    n_cols_before = len(panel.columns)

    log.info("Loading raw OHLCV cache …")
    raw = pd.read_parquet(CACHE_DIR / "raw_ohlcv.parquet")

    trading_dates = panel.index.get_level_values("date").unique().sort_values()
    tickers       = panel.index.get_level_values("ticker").unique()

    # ── 2. Per-stock features (groups A, B, C) ────────────────────────────────
    # Detect column layout (yfinance ≥1.2 = ticker first)
    sample_lvl0  = raw.columns.get_level_values(0)[0]
    ticker_first = sample_lvl0 not in {"Open","High","Low","Close","Volume"}

    log.info("Computing per-stock features (Groups A/B/C) for %d stocks …", len(tickers))
    new_stock_frames = []

    for ticker in tqdm(tickers, desc="Per-stock"):
        yf_tk = f"{ticker}.T"
        try:
            if ticker_first:
                df = raw[yf_tk].copy()
                df.columns = [c.lower() for c in df.columns]
            else:
                fields = ["Open","High","Low","Close","Volume"]
                df = pd.concat({f.lower(): raw[(f, yf_tk)] for f in fields}, axis=1)

            df = df.dropna(how="all")
            # yfinance inconsistently represents Japanese national holidays:
            # some years → all-NaN rows (removed by dropna above)
            # other years → volume=0 with close=prev_close (must remove separately)
            # With these holiday zeros left in, v.replace(0,NaN).rolling(252) has
            # max consecutive non-NaN streak of only ~55 days → rolling always NaN.
            df = df[df["volume"] > 0]
            if len(df) < 60:
                continue

            c       = df["close"]
            h       = df["high"]
            l       = df["low"]
            v       = df["volume"]
            log_ret = np.log(c).diff()

            out = pd.DataFrame(index=df.index)
            out["high"] = h
            out["low"]  = l

            # Group A: oscillators
            out = add_oscillators(out, c, h, l, v)
            # Group B: tail risk
            out = add_tail_risk(out, log_ret)
            # Group C: volume
            out = add_volume_features(out, c, v, log_ret)

            # Drop helper columns (high/low already in main panel)
            out = out.drop(columns=["high","low"], errors="ignore")

            out.index    = pd.to_datetime(out.index)
            out["ticker"] = ticker
            new_stock_frames.append(
                out.reset_index().rename(columns={"index":"date","Date":"date"})
            )
        except Exception as e:
            log.debug("Skipped %s: %s", ticker, e)

    log.info("Concatenating %d per-stock frames …", len(new_stock_frames))
    new_stock = pd.concat(new_stock_frames, ignore_index=True)
    new_stock["date"] = pd.to_datetime(new_stock["date"])
    new_stock = new_stock.set_index(["date","ticker"]).sort_index()

    # Drop any column that already exists in panel (avoid duplicates on re-run)
    existing_overlap = [c for c in new_stock.columns if c in panel.columns]
    if existing_overlap:
        new_stock = new_stock.drop(columns=existing_overlap)

    log.info("Merging %d new per-stock columns …", len(new_stock.columns))
    panel = panel.join(new_stock, how="left")

    # ── 3. Macro features (Group E) ───────────────────────────────────────────
    log.info("Group E — Macro features …")
    macro = download_macro_aux()
    macro_panel = build_macro_panel(macro, trading_dates)

    # Broadcast macro (date-level) to full (date, ticker) panel
    macro_long = macro_panel.stack().rename("_val").reset_index()
    macro_long.columns = ["date","macro_col","_val"]

    for col in macro_panel.columns:
        col_series = macro_panel[col].reindex(trading_dates).ffill()
        # Map date → value for each (date, ticker) row
        date_idx = panel.index.get_level_values("date")
        panel[col] = col_series.reindex(date_idx).values

    log.info("Added %d macro columns", len(macro_panel.columns))

    # ── 4. Fix beta_usdjpy_60d ────────────────────────────────────────────────
    log.info("Group E — Fix beta_usdjpy_60d …")
    panel = fix_usdjpy_beta(panel, macro, trading_dates)

    # ── 5. Cross-sectional rank features (Group D) ────────────────────────────
    log.info("Group D — Cross-sectional ranks …")
    panel = add_cross_sectional_ranks(panel)

    # ── 6. Composite scores (Group F) ─────────────────────────────────────────
    log.info("Group F — Composite scores …")
    panel = add_composite_scores(panel)

    # ── 7. Report and save ────────────────────────────────────────────────────
    n_cols_after = len(panel.columns)
    # column count comparison done via n_cols_before / n_cols_after

    log.info("─" * 60)
    log.info("Features before : %d", n_cols_before)
    log.info("Features after  : %d", n_cols_after)
    log.info("New features    : %d", n_cols_after - n_cols_before)
    log.info("Shape           : %s", panel.shape)
    log.info("─" * 60)

    # Show NaN % for new columns
    new_col_names = panel.columns[n_cols_before:]
    nan_pct = panel[new_col_names].isnull().mean() * 100
    log.info("NaN %% for new features (top 10):\n%s",
             nan_pct.sort_values(ascending=False).head(10).round(1).to_string())

    log.info("Writing to %s …", FEATS_FILE.name)
    panel.to_parquet(FEATS_FILE, engine="pyarrow", compression="snappy")
    size_mb = FEATS_FILE.stat().st_size / 1e6
    log.info("Saved. File size: %.1f MB", size_mb)

    log.info("\nFull column list (%d total):", len(panel.columns))
    for i, col in enumerate(panel.columns):
        log.info("  [%3d] %s", i+1, col)

    return panel


if __name__ == "__main__":
    panel = main()
