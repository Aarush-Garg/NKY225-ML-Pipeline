"""
NKY 225 Feature Panel — Visual Dashboard
==========================================
Produces NKY225_Panel_Dashboard.pdf  (6 pages, dark theme)

Page 1 — Dataset overview & NaN coverage
Page 2 — Data coverage heatmap  (stocks × time)
Page 3 — Constituent history & benchmark weights
Page 4 — Return distributions across horizons
Page 5 — Feature correlation matrix
Page 6 — Volatility regime & cross-sectional spread
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

# ── style ─────────────────────────────────────────────────────────────────────
BG      = "#0f1117"
PANEL   = "#1a1d2e"
ACCENT  = "#4f8ef7"
GREEN   = "#3ecf8e"
AMBER   = "#f59e0b"
RED     = "#ef4444"
PURPLE  = "#a855f7"
TEXT    = "#e2e8f0"
SUBTEXT = "#94a3b8"
GRID    = "#2d3748"

plt.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    PANEL,
    "axes.edgecolor":    GRID,
    "axes.labelcolor":   TEXT,
    "axes.titlecolor":   TEXT,
    "xtick.color":       SUBTEXT,
    "ytick.color":       SUBTEXT,
    "text.color":        TEXT,
    "grid.color":        GRID,
    "grid.linewidth":    0.5,
    "font.family":       "monospace",
    "font.size":         9,
})

FIGSIZE = (16, 10)
OUT     = Path(__file__).parent / "NKY225_Panel_Dashboard.pdf"

# ─────────────────────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────────────────────
print("Loading parquet files …")
_DATA = Path.home() / "Library/CloudStorage/OneDrive-Personal/Aarush-One Drive/Summer 2026/Quant Papa Internship"
feats = pd.read_parquet(_DATA / "nky225_features.parquet")
const = pd.read_parquet(_DATA / "nky225_constituents.parquet")

dates   = feats.index.get_level_values("date")
tickers = feats.index.get_level_values("ticker")

# Wide-format close prices  (date × ticker)
close_wide = feats["close"].unstack("ticker")

# In-index panel (date × ticker)
in_idx_wide = const["in_index"].unstack("ticker").reindex(close_wide.index)

# Macro: USD/JPY 1m return — single value per date (same for all stocks)
usdjpy_1m = feats["usdjpy_ret_1m"].groupby(level="date").first()

print(f"Panel: {feats.shape[0]:,} rows × {feats.shape[1]} columns")
print(f"Dates: {dates.min().date()} → {dates.max().date()}")
print(f"Stocks: {tickers.nunique()}")


# ─────────────────────────────────────────────────────────────────────────────
# Helper — thin wrapper so every page gets a clean dark figure
# ─────────────────────────────────────────────────────────────────────────────
def new_fig(title=""):
    fig = plt.figure(figsize=FIGSIZE, facecolor=BG)
    if title:
        fig.suptitle(title, color=TEXT, fontsize=13, fontweight="bold",
                     x=0.5, y=0.98)
    return fig

def style_ax(ax, xlabel="", ylabel="", title=""):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=SUBTEXT, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)
    if xlabel: ax.set_xlabel(xlabel, color=SUBTEXT, fontsize=8)
    if ylabel: ax.set_ylabel(ylabel, color=SUBTEXT, fontsize=8)
    if title:  ax.set_title(title,  color=TEXT,    fontsize=9, pad=6)
    ax.grid(True, color=GRID, linewidth=0.5, alpha=0.6)
    return ax


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1 — Dataset overview
# ─────────────────────────────────────────────────────────────────────────────
def page_overview(pdf):
    fig = new_fig("NKY 225 Feature Panel — Dataset Overview")
    gs  = gridspec.GridSpec(2, 3, figure=fig,
                            hspace=0.45, wspace=0.35,
                            left=0.07, right=0.97, top=0.91, bottom=0.07)

    # ── Summary stat boxes (top-left) ────────────────────────────────────────
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.axis("off")
    stats = [
        ("Total rows",       f"{len(feats):,}"),
        ("Features",         str(feats.shape[1])),
        ("Stocks",           str(tickers.nunique())),
        ("Trading days",     str(dates.nunique())),
        ("Start date",       str(dates.min().date())),
        ("End date",         str(dates.max().date())),
        ("Parquet size",     "178.8 MB"),
        ("Constituents file","5.5 MB"),
    ]
    for i, (k, v) in enumerate(stats):
        y = 0.93 - i * 0.125
        ax0.text(0.05, y, k, color=SUBTEXT, fontsize=9, transform=ax0.transAxes)
        ax0.text(0.62, y, v, color=ACCENT,  fontsize=9, transform=ax0.transAxes,
                 fontweight="bold")
    ax0.set_title("Summary", color=TEXT, fontsize=9, pad=6)

    # ── NaN % per column (top-middle + top-right spanning) ──────────────────
    ax1 = fig.add_subplot(gs[0, 1:])
    nan_pct = (feats.isnull().mean() * 100).sort_values(ascending=True)
    colours = [RED if v > 50 else AMBER if v > 10 else GREEN for v in nan_pct]
    bars = ax1.barh(nan_pct.index, nan_pct.values, color=colours, height=0.7)
    ax1.axvline(0, color=GRID, linewidth=0.8)
    ax1.set_xlim(0, 100)
    for bar, val in zip(bars, nan_pct.values):
        if val > 2:
            ax1.text(val + 0.5, bar.get_y() + bar.get_height()/2,
                     f"{val:.1f}%", va="center", color=TEXT, fontsize=7)
    style_ax(ax1, xlabel="NaN %", title="Missing Data by Feature")
    ax1.tick_params(axis="y", labelsize=7)

    # ── Rows per stock (bottom-left) ─────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    rows_per_stock = feats.groupby(level="ticker").size().sort_values()
    ax2.barh(range(len(rows_per_stock)), rows_per_stock.values,
             color=ACCENT, alpha=0.8, height=1.0)
    ax2.set_yticks([])
    ax2.axvline(rows_per_stock.median(), color=GREEN, linewidth=1.2,
                linestyle="--", label=f"Median {rows_per_stock.median():.0f}")
    ax2.legend(fontsize=7, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)
    style_ax(ax2, xlabel="Trading days with data", title="Data Depth per Stock")

    # ── Close price range per stock (bottom-middle) ──────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    price_stats = feats.groupby(level="ticker")["close"].agg(["min","max"])
    order = price_stats["max"].sort_values().index
    price_stats = price_stats.loc[order]
    ax3.barh(range(len(price_stats)),
             price_stats["max"].values - price_stats["min"].values,
             left=price_stats["min"].values,
             color=PURPLE, alpha=0.6, height=1.0)
    ax3.set_yticks([])
    ax3.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"¥{x:,.0f}"))
    style_ax(ax3, xlabel="Price range (JPY)", title="Stock Price Ranges")

    # ── Feature group coverage (bottom-right) ────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 2])
    groups = {
        "OHLCV":      ["close","high","low","open","volume","yen_volume"],
        "Returns":    ["ret_1d","ret_1w","ret_1m","ret_3m","ret_6m","ret_12m"],
        "Reversal/Mom":["reversal_1w","reversal_1m","momentum_12_1"],
        "Volatility": ["vol_20d","vol_60d","vol_120d","vol_parkinson"],
        "Technical":  ["rsi_14","macd_line","macd_signal","macd_hist","bb_pos",
                       "bb_width","ema_cross_5_20","ema_cross_50_200","high_to_price"],
        "Microstructure":["amihud_60d","rel_volume_5_60","log_yen_volume"],
        "Market Beta":["beta_nky_60d","beta_nky_252d","beta_usdjpy_60d","idio_vol_60d"],
        "Macro":      ["usdjpy_ret_1m","usdjpy_ret_3m"],
        "Calendar":   ["fiscal_year_end_flag","pre_fiscal_year_end","ex_div_season",
                       "golden_week","obon_holiday","calendar_year_end",
                       "day_of_week","is_monday","is_friday","fiscal_quarter"],
        "Universe":   ["in_index","bench_weight"],
    }
    grp_names   = list(groups.keys())
    grp_counts  = [len(v) for v in groups.values()]
    grp_colours = [ACCENT,GREEN,PURPLE,AMBER,RED,"#06b6d4","#f97316","#10b981","#8b5cf6","#ec4899"]
    wedges, texts, autotexts = ax4.pie(
        grp_counts, labels=grp_names, colors=grp_colours,
        autopct="%d", startangle=140,
        textprops={"color": TEXT, "fontsize": 7},
        pctdistance=0.75, wedgeprops={"linewidth": 0.5, "edgecolor": BG},
    )
    for at in autotexts:
        at.set_fontsize(6)
        at.set_color(BG)
    ax4.set_title(f"Feature Groups ({feats.shape[1]} total)", color=TEXT, fontsize=9, pad=6)

    pdf.savefig(fig, facecolor=BG)
    plt.close(fig)
    print("  Page 1 done")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2 — Data coverage heatmap
# ─────────────────────────────────────────────────────────────────────────────
def page_coverage(pdf):
    fig = new_fig("Data Coverage Heatmap  (1 = close price present)")
    ax  = fig.add_axes([0.10, 0.06, 0.87, 0.86])

    # Sample ~80 evenly spaced dates to keep chart readable
    all_dates  = close_wide.index
    step       = max(1, len(all_dates) // 80)
    sampled_dates = all_dates[::step]
    coverage   = close_wide.loc[sampled_dates].notna().astype(float)

    # Sort stocks by first non-NaN date (= entry into our dataset)
    first_date = close_wide.notna().idxmax()
    stock_order = first_date.sort_values().index
    coverage   = coverage[stock_order]

    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "cov", [PANEL, ACCENT])
    im = ax.imshow(coverage.T.values, aspect="auto", cmap=cmap,
                   interpolation="nearest", vmin=0, vmax=1)

    # X-axis: sampled years
    year_ticks = []
    year_labels = []
    for i, d in enumerate(sampled_dates):
        if i == 0 or d.year != sampled_dates[i-1].year:
            year_ticks.append(i)
            year_labels.append(str(d.year))
    ax.set_xticks(year_ticks)
    ax.set_xticklabels(year_labels, color=SUBTEXT, fontsize=8)

    # Y-axis: ticker labels (every 5th)
    step_y = max(1, len(stock_order) // 30)
    ax.set_yticks(range(0, len(stock_order), step_y))
    ax.set_yticklabels(stock_order[::step_y], color=SUBTEXT, fontsize=6)

    ax.set_xlabel("Date (sampled)", color=SUBTEXT, fontsize=8)
    ax.set_ylabel("Ticker", color=SUBTEXT, fontsize=8)
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)

    cbar = fig.colorbar(im, ax=ax, fraction=0.015, pad=0.01)
    cbar.ax.tick_params(colors=SUBTEXT, labelsize=7)
    cbar.set_label("Data present", color=SUBTEXT, fontsize=7)

    # Annotate change events
    for entry in [
        ("2020-10-01", "2020 review"),
        ("2022-10-03", "2022 review"),
        ("2024-10-01", "2024 review"),
    ]:
        t = pd.Timestamp(entry[0])
        idx = int(np.argmin(np.abs((sampled_dates - t).total_seconds().values)))
        ax.axvline(idx, color=AMBER, linewidth=0.8, linestyle="--", alpha=0.7)
        ax.text(idx + 0.3, 2, entry[1], color=AMBER, fontsize=6, rotation=90, va="top")

    pdf.savefig(fig, facecolor=BG)
    plt.close(fig)
    print("  Page 2 done")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3 — Constituent history & benchmark weights
# ─────────────────────────────────────────────────────────────────────────────
def page_constituents(pdf):
    fig = new_fig("Constituent Membership & Benchmark Weights")
    gs  = gridspec.GridSpec(2, 2, figure=fig,
                            hspace=0.40, wspace=0.32,
                            left=0.07, right=0.97, top=0.91, bottom=0.07)

    # ── Count in-index per day ───────────────────────────────────────────────
    ax0 = fig.add_subplot(gs[0, :])
    count_per_day = in_idx_wide.sum(axis=1)
    ax0.fill_between(count_per_day.index, count_per_day.values,
                     color=ACCENT, alpha=0.25)
    ax0.plot(count_per_day.index, count_per_day.values,
             color=ACCENT, linewidth=0.8)
    ax0.axhline(225, color=GREEN, linewidth=1.0, linestyle="--",
                label="Target: 225")
    ax0.set_ylim(0, 240)
    ax0.yaxis.set_major_locator(mticker.MultipleLocator(25))
    style_ax(ax0, ylabel="Stocks in-index", title="NKY 225 Membership Count Over Time")
    ax0.legend(fontsize=8, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)

    # Annotate annual reviews
    for yr in range(2015, 2026):
        ax0.axvline(pd.Timestamp(f"{yr}-10-01"), color=AMBER,
                    linewidth=0.5, linestyle=":", alpha=0.6)
    ax0.text(pd.Timestamp("2016-01-01"), 232,
             "↑ Vertical dashes = annual review dates (Oct)", color=AMBER, fontsize=7)

    # ── Benchmark weight: top 20 stocks most recent date ─────────────────────
    ax1 = fig.add_subplot(gs[1, 0])
    latest_date = const.index.get_level_values("date").max()
    latest_weights = (
        const.loc[latest_date, "bench_weight"]
        .sort_values(ascending=False)
        .head(20)
    )
    colours = [ACCENT if i < 5 else GREEN if i < 10 else PURPLE
               for i in range(len(latest_weights))]
    ax1.barh(latest_weights.index[::-1], latest_weights.values[::-1],
             color=colours[::-1], height=0.7)
    ax1.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{x*100:.1f}%"))
    style_ax(ax1, xlabel="Benchmark weight",
             title=f"Top 20 Stocks by Weight\n({latest_date.date()})")
    ax1.tick_params(axis="y", labelsize=7)

    # ── Weight concentration (Lorenz-style) ──────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 1])
    # compute Lorenz curve on multiple sample dates
    sample_dates = pd.date_range(
        const.index.get_level_values("date").min(),
        const.index.get_level_values("date").max(),
        periods=6,
    )
    unique_dates = const.index.get_level_values("date").unique().sort_values()
    pallete = [ACCENT, GREEN, AMBER, PURPLE, RED, "#06b6d4"]
    for d, col in zip(sample_dates, pallete):
        d_actual = unique_dates[unique_dates.get_indexer([d], method="pad")[0]]
        w = const.loc[d_actual, "bench_weight"].sort_values(ascending=False)
        w = w[w > 0]
        lorenz_y = w.cumsum() / w.sum()
        ax2.plot(range(1, len(lorenz_y)+1), lorenz_y.values * 100,
                 color=col, linewidth=1.2, label=str(d_actual.year))
    ax2.axhline(50, color=GRID, linewidth=0.7, linestyle="--")
    ax2.set_xlim(0)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    style_ax(ax2, xlabel="Rank (highest weight first)",
             ylabel="Cumulative weight %",
             title="Weight Concentration (Lorenz Curve)")
    ax2.legend(fontsize=7, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT,
               title="Year", title_fontsize=7)

    pdf.savefig(fig, facecolor=BG)
    plt.close(fig)
    print("  Page 3 done")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 4 — Return distributions
# ─────────────────────────────────────────────────────────────────────────────
def page_returns(pdf):
    fig = new_fig("Return Distributions Across Horizons")
    gs  = gridspec.GridSpec(2, 3, figure=fig,
                            hspace=0.45, wspace=0.35,
                            left=0.07, right=0.97, top=0.91, bottom=0.07)

    horizons = [
        ("ret_1d",  "Daily (1D)",   ACCENT),
        ("ret_1w",  "Weekly (1W)",  GREEN),
        ("ret_1m",  "Monthly (1M)", AMBER),
        ("ret_3m",  "Quarterly (3M)", PURPLE),
        ("ret_6m",  "Semi-annual (6M)", RED),
        ("ret_12m", "Annual (12M)", "#06b6d4"),
    ]
    for idx, (col, label, colour) in enumerate(horizons):
        row, c = divmod(idx, 3)
        ax = fig.add_subplot(gs[row, c])

        data = feats[col].dropna()
        # clip extreme tails at 1/99 percentile for display
        lo, hi = data.quantile(0.01), data.quantile(0.99)
        data_clipped = data.clip(lo, hi)

        ax.hist(data_clipped * 100, bins=80, color=colour,
                alpha=0.8, edgecolor="none", density=True)

        mu  = data.mean() * 100
        med = data.median() * 100
        std = data.std() * 100

        ax.axvline(mu,  color="white",  linewidth=1.2, linestyle="--",
                   label=f"μ {mu:+.2f}%")
        ax.axvline(med, color=AMBER,    linewidth=1.0, linestyle=":",
                   label=f"m {med:+.2f}%")
        ax.axvline(0,   color=GRID,     linewidth=0.8)

        ax.legend(fontsize=6.5, facecolor=PANEL, edgecolor=GRID,
                  labelcolor=TEXT, handlelength=1)

        # Stat box
        skew = float(data.skew())
        kurt = float(data.kurt())
        ax.text(0.97, 0.95,
                f"σ={std:.2f}%\nSkew={skew:+.2f}\nKurt={kurt:+.1f}",
                transform=ax.transAxes, fontsize=6.5, color=SUBTEXT,
                ha="right", va="top", linespacing=1.6)

        style_ax(ax, xlabel="Return (%)", title=label)
        ax.yaxis.set_visible(False)

    pdf.savefig(fig, facecolor=BG)
    plt.close(fig)
    print("  Page 4 done")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 5 — Feature correlation matrix
# ─────────────────────────────────────────────────────────────────────────────
def page_correlations(pdf):
    fig = new_fig("Feature Correlation Matrix  (sample of 50k rows)")
    ax  = fig.add_axes([0.12, 0.04, 0.85, 0.86])

    FEAT_COLS = [
        "ret_1d","ret_1m","ret_3m","ret_12m",
        "reversal_1m","momentum_12_1",
        "vol_20d","vol_60d","vol_parkinson",
        "rsi_14","macd_hist","bb_pos","ema_cross_5_20","ema_cross_50_200",
        "high_to_price","amihud_60d","rel_volume_5_60","log_yen_volume",
        "beta_nky_60d","idio_vol_60d",
        "usdjpy_ret_1m","usdjpy_ret_3m",
        "bench_weight",
    ]
    existing = [c for c in FEAT_COLS if c in feats.columns]

    sample = feats[existing].dropna(how="all").sample(
        n=min(50_000, len(feats)), random_state=42)
    corr = sample.corr()

    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)  # show lower triangle

    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "div", [RED, PANEL, ACCENT])

    im = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1,
                   interpolation="nearest", aspect="auto")

    n = len(existing)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(existing, rotation=45, ha="right",
                       color=SUBTEXT, fontsize=7)
    ax.set_yticklabels(existing, color=SUBTEXT, fontsize=7)
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)

    # Annotate cells with correlation value (only if |r| > 0.2)
    for i in range(n):
        for j in range(n):
            v = corr.values[i, j]
            if abs(v) > 0.20 and i != j:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=5.5,
                        color="white" if abs(v) > 0.5 else SUBTEXT)

    cbar = fig.colorbar(im, ax=ax, fraction=0.020, pad=0.01)
    cbar.ax.tick_params(colors=SUBTEXT, labelsize=7)
    cbar.set_label("Pearson r", color=SUBTEXT, fontsize=7)

    pdf.savefig(fig, facecolor=BG)
    plt.close(fig)
    print("  Page 5 done")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 6 — Volatility regime & cross-sectional spread
# ─────────────────────────────────────────────────────────────────────────────
def page_volatility(pdf):
    fig = new_fig("Volatility Regime & Cross-Sectional Return Spread")
    gs  = gridspec.GridSpec(3, 2, figure=fig,
                            hspace=0.55, wspace=0.32,
                            left=0.07, right=0.97, top=0.91, bottom=0.07)

    # ── Median cross-sectional vol_20d over time ─────────────────────────────
    ax0 = fig.add_subplot(gs[0, :])
    cs_vol = feats["vol_20d"].groupby(level="date").agg(
        ["median","quantile"])

    cs_vol_med  = feats["vol_20d"].groupby(level="date").median()
    cs_vol_p10  = feats["vol_20d"].groupby(level="date").quantile(0.10)
    cs_vol_p90  = feats["vol_20d"].groupby(level="date").quantile(0.90)

    ax0.fill_between(cs_vol_p10.index, cs_vol_p10 * 100, cs_vol_p90 * 100,
                     color=ACCENT, alpha=0.15, label="P10–P90 band")
    ax0.plot(cs_vol_med.index, cs_vol_med * 100,
             color=ACCENT, linewidth=1.0, label="Median vol_20d")

    # USD/JPY volatility proxy — annualised 20d std of usdjpy_1m diffs
    usd_vol = usdjpy_1m.diff().rolling(20).std() * 100
    ax0.plot(usd_vol.index, usd_vol * 15,   # scale to same axis
             color=AMBER, linewidth=0.8, linestyle="--", alpha=0.8,
             label="USD/JPY vol (×15 scaled)")

    style_ax(ax0, ylabel="Annualised vol (%)",
             title="Cross-Sectional Realised Volatility (20D)")
    ax0.legend(fontsize=7.5, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)

    # ── Cross-sectional dispersion of 1M returns ─────────────────────────────
    ax1 = fig.add_subplot(gs[1, :])
    cs_ret = feats["ret_1m"].groupby(level="date")
    cs_q25 = cs_ret.quantile(0.25) * 100
    cs_q75 = cs_ret.quantile(0.75) * 100
    cs_med = cs_ret.median() * 100

    ax1.fill_between(cs_q25.index, cs_q25, cs_q75,
                     color=GREEN, alpha=0.15, label="IQR (P25–P75)")
    ax1.plot(cs_med.index, cs_med, color=GREEN, linewidth=1.0,
             label="Median 1M return")
    ax1.axhline(0, color=GRID, linewidth=0.8, linestyle="--")
    style_ax(ax1, ylabel="1M return (%)",
             title="Cross-Sectional 1M Return Dispersion")
    ax1.legend(fontsize=7.5, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)

    # ── RSI distribution over time ────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[2, 0])
    cs_rsi_med = feats["rsi_14"].groupby(level="date").median()
    ax2.plot(cs_rsi_med.index, cs_rsi_med, color=PURPLE, linewidth=0.8)
    ax2.axhline(70, color=RED,   linewidth=0.7, linestyle="--", label="Overbought 70")
    ax2.axhline(30, color=GREEN, linewidth=0.7, linestyle="--", label="Oversold 30")
    ax2.axhline(50, color=GRID,  linewidth=0.7, linestyle=":")
    ax2.set_ylim(20, 80)
    style_ax(ax2, ylabel="RSI", title="Median Cross-Sectional RSI (14D)")
    ax2.legend(fontsize=7, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)

    # ── Amihud illiquidity trend ───────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2, 1])
    cs_illiq_med = feats["amihud_60d"].groupby(level="date").median()
    ax3.plot(cs_illiq_med.index, cs_illiq_med * 1e9,
             color=AMBER, linewidth=0.8)
    ax3.fill_between(cs_illiq_med.index, 0, cs_illiq_med * 1e9,
                     color=AMBER, alpha=0.15)
    style_ax(ax3, ylabel="Amihud ILLIQ (×10⁻⁹)",
             title="Median Cross-Sectional Amihud Illiquidity")

    pdf.savefig(fig, facecolor=BG)
    plt.close(fig)
    print("  Page 6 done")


# ─────────────────────────────────────────────────────────────────────────────
# RENDER
# ─────────────────────────────────────────────────────────────────────────────
print(f"\nRendering dashboard → {OUT.name}")
with PdfPages(OUT) as pdf:
    page_overview(pdf)
    page_coverage(pdf)
    page_constituents(pdf)
    page_returns(pdf)
    page_correlations(pdf)
    page_volatility(pdf)

    d = pdf.infodict()
    d["Title"]   = "NKY 225 Feature Panel Dashboard"
    d["Subject"] = "Quant Internship — NKY 225 ML Pipeline"

size_mb = OUT.stat().st_size / 1e6
print(f"\nSaved: {OUT.name}  ({size_mb:.1f} MB)")
print("Open with: open NKY225_Panel_Dashboard.pdf")
