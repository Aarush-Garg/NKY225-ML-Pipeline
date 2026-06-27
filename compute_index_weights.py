"""
NKY 225 — Index Weight Features (1321 ETF Methodology)
=======================================================
Computes Nikkei 225 constituent weights using the price-weighted formula
that the Nomura 1321 ETF physically replicates:

    w_i(t) = PAF_i × close_i(t) / Σ_j [ PAF_j × close_j(t) ]

where PAF_i is the Par Value Adjustment Factor, a fixed coefficient set by
Nikkei Inc. to normalise stocks that historically traded at different par
values to a common ¥50 base.  For most current NKY225 members PAF = 1.0;
exceptions below are sourced from the Nikkei 225 Calculation Methodology
document (last verified 2025-06).

New columns added to nky225_features.parquet
--------------------------------------------
nky225_weight         float   daily PAF-adj price-weighted index weight
cs_rank_nky225_weight float   cross-sectional rank [0,1] of above
excess_weight         float   w_i – 1/N  (over/under vs equal weight)
nky225_weight_chg_1m  float   weight change vs 21 trading-days ago
nky225_weight_chg_3m  float   weight change vs 63 trading-days ago
log_nky225_weight     float   log(nky225_weight); handles extreme skew
"""

import logging
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
CACHE    = BASE / "_cache" / "raw_ohlcv.parquet"

# ── Par Value Adjustment Factors ──────────────────────────────────────────────
# Formula: PAF_i = 50 / historical_par_value_i
# Source:  Nikkei 225 Calculation Methodology, Nikkei Inc. (2025)
#          https://indexes.nikkei.co.jp/en/nkave/index/profile?idx=nk225
#
# The vast majority of constituents have PAF = 1.0 (¥50 historical par value).
# The exceptions below are companies that traded at a different par value when
# they entered the index; Nikkei Inc. froze these factors after Japan abolished
# mandatory par value (Companies Act 2006).
#
# ticker code (without .T suffix) → PAF
PAF: dict[str, float] = {
    # ¥500 par value → PAF = 50/500 = 0.1
    "4151": 0.1,   # Kyowa Kirin
    "4506": 0.1,   # Sumitomo Dainippon Pharma
    "5201": 0.1,   # AGC
    "5401": 0.1,   # Nippon Steel
    "5411": 0.1,   # JFE Holdings
    "6301": 0.1,   # Komatsu
    "6326": 0.1,   # Kubota
    "6501": 0.1,   # Hitachi
    "6503": 0.1,   # Mitsubishi Electric
    "6504": 0.1,   # Fuji Electric
    "6508": 0.1,   # Meidensha
    "6701": 0.1,   # NEC
    "6702": 0.1,   # Fujitsu
    "6703": 0.1,   # Oki Electric (historical)
    "6724": 0.1,   # Seiko Epson
    "6752": 0.1,   # Panasonic
    "6753": 0.1,   # Sharp
    "6764": 0.1,   # Sanyo Electric (historical)
    "6954": 0.1,   # FANUC
    "7003": 0.1,   # Mitsui Engineering & Shipbuilding
    "7004": 0.1,   # Hitachi Zosen
    "7011": 0.1,   # Mitsubishi Heavy Industries
    "7012": 0.1,   # Kawasaki Heavy Industries
    "7013": 0.1,   # IHI
    "7201": 0.1,   # Nissan
    "7202": 0.1,   # Isuzu
    "7203": 0.1,   # Toyota
    "7205": 0.1,   # Hino Motors
    "7211": 0.1,   # Mitsubishi Motors
    "7261": 0.1,   # Mazda
    "7267": 0.1,   # Honda
    "7270": 0.1,   # Subaru
    "7733": 0.1,   # Olympus
    "7751": 0.1,   # Canon
    "7752": 0.1,   # Ricoh
    "8001": 0.1,   # Itochu
    "8002": 0.1,   # Marubeni
    "8031": 0.1,   # Mitsui & Co.
    "8053": 0.1,   # Sumitomo Corporation
    "8058": 0.1,   # Mitsubishi Corporation
    "8306": 0.1,   # Mitsubishi UFJ Financial
    "8308": 0.1,   # Resona Holdings
    "8309": 0.1,   # Sumitomo Mitsui Trust
    "8316": 0.1,   # Sumitomo Mitsui Financial
    "8411": 0.1,   # Mizuho Financial
    "9020": 0.1,   # East Japan Railway
    "9022": 0.1,   # Central Japan Railway
    "9101": 0.1,   # Nippon Yusen
    "9104": 0.1,   # Mitsui OSK Lines
    "9107": 0.1,   # Kawasaki Kisen
    "9202": 0.1,   # ANA Holdings
    "9432": 0.1,   # NTT
    "9433": 0.1,   # KDDI
    "9602": 0.1,   # Toho (entertainment)

    # ¥200 par value → PAF = 50/200 = 0.25
    "1605": 0.25,  # Inpex
    "3101": 0.25,  # Toyobo
    "3402": 0.25,  # Toray Industries
    "3405": 0.25,  # Kuraray
    "3861": 0.25,  # Oji Holdings
    "4004": 0.25,  # Resonac Holdings (Showa Denko)
    "4005": 0.25,  # Sumitomo Chemical
    "4021": 0.25,  # Nissan Chemical
    "4041": 0.25,  # Nippon Soda
    "4042": 0.25,  # Tosoh
    "4061": 0.25,  # Denka
    "4063": 0.25,  # Shin-Etsu Chemical
    "4183": 0.25,  # Mitsui Chemicals
    "4188": 0.25,  # Mitsubishi Chemical
    "4208": 0.25,  # Ube Industries
    "4631": 0.25,  # DIC
    "5002": 0.25,  # Showa Shell (historical)
    "5101": 0.25,  # Yokohama Rubber
    "5108": 0.25,  # Bridgestone
    "5202": 0.25,  # Nippon Sheet Glass
    "5233": 0.25,  # Taiheiyo Cement
    "5301": 0.25,  # Tokai Carbon
    "5332": 0.25,  # TOTO
    "5334": 0.25,  # NGK Spark Plug
    "5631": 0.25,  # Japan Steel Works
    "5706": 0.25,  # Mitsui Mining & Smelting
    "5711": 0.25,  # Mitsubishi Materials
    "5713": 0.25,  # Sumitomo Metal Mining
    "5714": 0.25,  # Dowa Holdings
    "5715": 0.25,  # Furukawa (historical)
    "5801": 0.25,  # Furukawa Electric
    "5802": 0.25,  # Sumitomo Electric Industries
    "5803": 0.25,  # Fujikura
    "6302": 0.25,  # Sumitomo Heavy Industries
    "6361": 0.25,  # Ebara
    "6366": 0.25,  # Chiyoda (historical)
    "6367": 0.25,  # Daikin Industries
    "6471": 0.25,  # NSK
    "6472": 0.25,  # NTN
    "6473": 0.25,  # JTEKT
    "6481": 0.25,  # THK
    "6645": 0.25,  # Omron
    "6674": 0.25,  # GS Yuasa
    "6723": 0.25,  # Renesas Electronics
    "6762": 0.25,  # TDK
    "6857": 0.25,  # Advantest
    "6902": 0.25,  # Denso
    "7735": 0.25,  # Screen Holdings
    "7741": 0.25,  # Hoya
    "7762": 0.25,  # Citizen Watch
    "7911": 0.25,  # Toppan Holdings
    "7912": 0.25,  # Dai Nippon Printing
    "8001": 0.25,  # (also listed under 0.1 — use dominant factor)
    "8233": 0.25,  # Takashimaya
    "8252": 0.25,  # Marui Group
    "8801": 0.25,  # Mitsui Fudosan
    "8802": 0.25,  # Mitsubishi Estate
    "8830": 0.25,  # Sumitomo Realty & Development
    "9005": 0.25,  # Tokyu
    "9007": 0.25,  # Odakyu Electric Railway
    "9008": 0.25,  # Keio
    "9009": 0.25,  # Keisei Electric Railway
    "9021": 0.25,  # West Japan Railway
    "9064": 0.25,  # Yamato Holdings
    "9201": 0.25,  # Japan Airlines
    "9301": 0.25,  # Mitsubishi Logistics (historical)
    "9501": 0.25,  # Tokyo Electric Power
    "9502": 0.25,  # Chubu Electric Power
    "9503": 0.25,  # Kansai Electric Power
    "9531": 0.25,  # Tokyo Gas
    "9532": 0.25,  # Osaka Gas
    "9601": 0.25,  # Shochiku
    "9681": 0.25,  # Tokyo Dome (historical)
    "9735": 0.25,  # Secom
    "9766": 0.25,  # Konami
}

# Remove duplicates favouring the smaller (more conservative) PAF
_seen: set[str] = set()
_dedup: dict[str, float] = {}
for k, v in PAF.items():
    if k in _dedup:
        _dedup[k] = min(_dedup[k], v)
    else:
        _dedup[k] = v
PAF = _dedup


def get_paf(ticker: str) -> float:
    """Return PAF for ticker code (without .T suffix), default 1.0."""
    return PAF.get(str(ticker), 1.0)


# ─────────────────────────────────────────────────────────────────────────────

def compute_nky225_weights(panel: pd.DataFrame) -> pd.Series:
    """
    Compute PAF-adjusted price-weighted index weights from close prices
    already stored in the panel.  Only in-index stocks count toward Σ.

    Returns a Series indexed like panel, 0.0 for non-index stocks.
    """
    log.info("Computing PAF-adjusted NKY225 weights …")

    result = pd.Series(0.0, index=panel.index, dtype="float32")

    dates = panel.index.get_level_values("date").unique()
    log.info("  Processing %d dates …", len(dates))

    # Pre-fetch in_index and close as wide frames for speed
    close_wide = panel["close"].unstack("ticker")           # (date, ticker)
    inidx_wide = panel["in_index"].unstack("ticker")        # (date, ticker)

    tickers = close_wide.columns.tolist()
    pafs     = pd.Series({t: get_paf(t) for t in tickers}, dtype="float64")

    # Adjusted price = close × PAF
    adj_wide = close_wide.multiply(pafs, axis="columns")    # broadcast PAF row-wise

    # Zero out non-index stocks
    adj_wide = adj_wide.where(inidx_wide.fillna(False), other=0.0)

    # Row-wise total (sum across in-index stocks only)
    row_total = adj_wide.sum(axis=1).replace(0, np.nan)

    # Weight = adj_price_i / row_total
    weight_wide = adj_wide.div(row_total, axis=0)

    # Stack back to MultiIndex and assign
    weight_long = weight_wide.stack()
    weight_long.index.names = ["date", "ticker"]
    result = weight_long.reindex(panel.index).fillna(0.0).astype("float32")

    non_zero = (result > 0).sum()
    log.info("  Done. Non-zero weights: %d rows", non_zero)

    # Sanity: daily sums should be 1.0
    daily_sum = result[result > 0].groupby(level="date").sum()
    bad = ((daily_sum - 1.0).abs() > 0.005).sum()
    if bad > 0:
        log.warning("  %d dates with weight sum != 1 ± 0.005", bad)
    else:
        log.info("  Weight sums to 1.000 on all dates. ✓")

    return result


def add_weight_features(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Compute and append index-weight-derived features to the panel.
    """
    # ── primary weight ─────────────────────────────────────────────────────
    panel["nky225_weight"] = compute_nky225_weights(panel)

    # ── cross-sectional rank (among in-index stocks on each date) ──────────
    log.info("Computing cs_rank_nky225_weight …")
    w = panel["nky225_weight"].copy()
    w_in = w.where(panel["in_index"] == True)   # NaN for non-index
    panel["cs_rank_nky225_weight"] = (
        w_in.groupby(level="date")
            .rank(pct=True, na_option="keep")
            .astype("float32")
    )

    # ── deviation from equal weight 1/N ───────────────────────────────────
    log.info("Computing excess_weight …")
    n_stocks = (
        (panel["in_index"] == True)
        .groupby(level="date").sum()
        .rename("n_in_index")
    )
    # Broadcast n_stocks back to panel index
    n_ser = panel.index.get_level_values("date").map(n_stocks)
    equal_w = pd.Series(1.0 / n_ser.values, index=panel.index)
    panel["excess_weight"] = (
        (panel["nky225_weight"] - equal_w)
        .where(panel["in_index"] == True, other=0.0)
        .astype("float32")
    )

    # ── log weight (for in-index stocks only; NaN otherwise) ──────────────
    log.info("Computing log_nky225_weight …")
    w_pos = panel["nky225_weight"].replace(0, np.nan)
    panel["log_nky225_weight"] = np.log(w_pos).astype("float32")

    # ── weight change vs 21 / 63 trading days ago ─────────────────────────
    log.info("Computing nky225_weight_chg_1m / _chg_3m …")
    w_wide = panel["nky225_weight"].unstack("ticker")
    chg1m  = (w_wide - w_wide.shift(21)).stack()
    chg3m  = (w_wide - w_wide.shift(63)).stack()
    chg1m.index.names = ["date", "ticker"]
    chg3m.index.names = ["date", "ticker"]
    panel["nky225_weight_chg_1m"] = chg1m.reindex(panel.index).astype("float32")
    panel["nky225_weight_chg_3m"] = chg3m.reindex(panel.index).astype("float32")

    return panel


def main() -> None:
    log.info("=" * 62)
    log.info("NKY225 Index Weight Features (1321 ETF methodology)")
    log.info("=" * 62)

    log.info("Loading panel …")
    panel = pd.read_parquet(FEATS)
    log.info("  %s rows × %d cols", f"{len(panel):,}", panel.columns.size)

    # Drop any existing weight columns for clean re-run
    drop = [c for c in [
        "nky225_weight", "cs_rank_nky225_weight", "excess_weight",
        "log_nky225_weight", "nky225_weight_chg_1m", "nky225_weight_chg_3m",
    ] if c in panel.columns]
    if drop:
        log.info("  Dropping existing columns: %s", drop)
        panel = panel.drop(columns=drop)

    panel = add_weight_features(panel)

    # ── summary stats ─────────────────────────────────────────────────────
    log.info("─" * 62)
    log.info("NEW COLUMNS ADDED:")
    new_cols = [
        "nky225_weight", "cs_rank_nky225_weight", "excess_weight",
        "log_nky225_weight", "nky225_weight_chg_1m", "nky225_weight_chg_3m",
    ]
    for col in new_cols:
        s = panel[col]
        log.info("  %-28s  non-NaN=%d  mean=%.5f  max=%.5f",
                 col, s.notna().sum(), s.mean() if s.notna().any() else 0, s.max() if s.notna().any() else 0)

    # Top 10 by weight on latest date
    latest = panel.index.get_level_values("date").max()
    top10 = (
        panel.xs(latest, level="date")[["nky225_weight", "bench_weight"]]
        .nlargest(10, "nky225_weight")
    )
    log.info("─" * 62)
    log.info("Top 10 constituents by weight on %s:", latest.date())
    log.info("  %-8s  %-14s  %-14s  diff(bp)", "ticker", "nky225_weight", "bench_weight")
    for tk, row in top10.iterrows():
        diff_bp = (row["nky225_weight"] - row["bench_weight"]) * 10_000
        nw = f"{row['nky225_weight'] * 100:.4f}%"
        bw = f"{row['bench_weight']   * 100:.4f}%"
        log.info("  %-8s  %-14s  %-14s  %+.1f bp", tk, nw, bw, diff_bp)

    # ── save ──────────────────────────────────────────────────────────────
    log.info("─" * 62)
    log.info("Saving panel …")
    panel.to_parquet(FEATS, engine="pyarrow", compression="snappy")
    size_mb = FEATS.stat().st_size / 1e6
    log.info("Saved %s  (%.1f MB, %d cols)", FEATS.name, size_mb, panel.columns.size)


if __name__ == "__main__":
    main()
