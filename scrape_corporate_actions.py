"""
NKY 225 — Corporate Actions Scraper
=====================================
Fetches dividends and stock splits for all 262 NKY 225 tickers
(current + historical constituents) from Yahoo Finance for 2014-present.

Output: corporate_actions_NKY225.pdf
"""

import time
import warnings
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch
import matplotlib.ticker as mticker

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# ── paths ─────────────────────────────────────────────────────────────────────
BASE     = Path(__file__).parent
OUT_PDF  = BASE / "corporate_actions_NKY225.pdf"
CACHE    = BASE / "_cache" / "corporate_actions.parquet"
START_DT = pd.Timestamp("2014-06-01", tz="Asia/Tokyo")

# ── style ─────────────────────────────────────────────────────────────────────
BG      = "#0f1117"
PANEL   = "#1a1d2e"
ACCENT  = "#4f8ef7"
GREEN   = "#2ecc71"
ORANGE  = "#f39c12"
RED     = "#e74c3c"
TEXT    = "#e8eaf6"
SUBTEXT = "#8892b0"
WHITE   = "#ffffff"


def style_ax(ax, title="", subtitle=""):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=SUBTEXT, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(PANEL)
    if title:
        ax.set_title(title, color=TEXT, fontsize=11, fontweight="bold", pad=8)
    if subtitle:
        ax.text(0.5, 1.01, subtitle, transform=ax.transAxes,
                ha="center", color=SUBTEXT, fontsize=8)


def fig_bg(fig):
    fig.patch.set_facecolor(BG)


# ─────────────────────────────────────────────────────────────────────────────
# 1. FETCH CORPORATE ACTIONS
# ─────────────────────────────────────────────────────────────────────────────

def fetch_all_actions(tickers_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return (dividends_df, splits_df) for all tickers since START_DT.
    Uses a local cache — re-fetches only if cache is missing or >23h old.
    """
    if CACHE.exists():
        age_h = (time.time() - CACHE.stat().st_mtime) / 3600
        if age_h < 23:
            log.info("Loading corporate actions from cache (%.1f h old)", age_h)
            stored = pd.read_parquet(CACHE)
            divs   = stored[stored["action_type"] == "Dividend"]
            splits = stored[stored["action_type"] == "Split"]
            return divs, splits

    all_rows = []
    n = len(tickers_df)

    log.info("Fetching corporate actions for %d tickers …", n)
    for i, (_, row) in enumerate(tickers_df.iterrows()):
        code    = str(row["ticker"])
        name    = row["company_name"]
        yf_tick = f"{code}.T"

        try:
            tk      = yf.Ticker(yf_tick)
            actions = tk.actions  # DataFrame with Dividends + Stock Splits columns

            if actions is None or actions.empty:
                log.debug("  [%3d/%d] %s — no actions", i+1, n, yf_tick)
                time.sleep(0.2)
                continue

            actions.index = pd.to_datetime(actions.index, utc=True).tz_convert("Asia/Tokyo")
            actions = actions[actions.index >= START_DT]

            # Dividends
            div_rows = actions[actions["Dividends"] > 0]
            for dt, dr in div_rows.iterrows():
                all_rows.append({
                    "date":        dt.normalize(),
                    "ticker":      code,
                    "company":     name,
                    "action_type": "Dividend",
                    "value":       dr["Dividends"],
                    "description": f"¥{dr['Dividends']:.2f} per share",
                })

            # Splits
            split_rows = actions[actions["Stock Splits"] > 0]
            for dt, sr in split_rows.iterrows():
                ratio = sr["Stock Splits"]
                all_rows.append({
                    "date":        dt.normalize(),
                    "ticker":      code,
                    "company":     name,
                    "action_type": "Split",
                    "value":       ratio,
                    "description": f"{int(ratio)}-for-1 split" if ratio == int(ratio)
                                   else f"{ratio:.2f}-for-1 split",
                })

            n_div   = len(div_rows)
            n_split = len(split_rows)
            log.info("  [%3d/%d] %-8s  dividends=%d  splits=%d",
                     i+1, n, yf_tick, n_div, n_split)

        except Exception as e:
            log.warning("  [%3d/%d] %s failed: %s", i+1, n, yf_tick, e)

        time.sleep(0.3)

    if not all_rows:
        log.warning("No corporate actions found!")
        return pd.DataFrame(), pd.DataFrame()

    all_df = pd.DataFrame(all_rows)
    all_df["date"] = pd.to_datetime(all_df["date"]).dt.tz_localize(None)
    all_df = all_df.sort_values(["date", "ticker"]).reset_index(drop=True)
    all_df.to_parquet(CACHE)

    divs   = all_df[all_df["action_type"] == "Dividend"].copy()
    splits = all_df[all_df["action_type"] == "Split"].copy()
    return divs, splits


# ─────────────────────────────────────────────────────────────────────────────
# 2. PDF GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def draw_table(ax, df_display, col_widths, header_color=ACCENT,
               row_colors=None, fontsize=7.5, header_fontsize=8.5):
    """Draw a styled table on ax using ax.table()."""
    ax.axis("off")
    if df_display.empty:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                ha="center", va="center", color=SUBTEXT, fontsize=10)
        return

    if row_colors is None:
        n = len(df_display)
        row_colors = [[PANEL if i % 2 == 0 else "#21263a"] * len(df_display.columns)
                      for i in range(n)]

    tbl = ax.table(
        cellText=df_display.values,
        colLabels=df_display.columns,
        cellLoc="center",
        loc="center",
        colWidths=col_widths,
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(fontsize)

    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor(BG)
        if row == 0:
            cell.set_facecolor(header_color)
            cell.set_text_props(color=WHITE, fontweight="bold",
                                fontsize=header_fontsize)
        else:
            cell.set_facecolor(row_colors[row-1][col])
            cell.set_text_props(color=TEXT)


# ── Page 1: Cover ─────────────────────────────────────────────────────────────

def page_cover(pdf, divs, splits, run_date):
    fig = plt.figure(figsize=(11.69, 8.27))
    fig_bg(fig)

    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_facecolor(BG)

    # Title block
    ax.text(0.5, 0.82, "NKY 225 — Corporate Actions Report",
            ha="center", color=WHITE, fontsize=22, fontweight="bold")
    ax.text(0.5, 0.75, "Dividends & Stock Splits  |  June 2014 – Present",
            ha="center", color=ACCENT, fontsize=14)
    ax.text(0.5, 0.70, f"Generated {run_date}  ·  Universe: {divs['ticker'].nunique() + splits['ticker'].nunique()} unique tickers with actions",
            ha="center", color=SUBTEXT, fontsize=10)

    ax.axhline(0.66, xmin=0.05, xmax=0.95, color=ACCENT, linewidth=1.5, alpha=0.6)

    # Stats boxes
    boxes = [
        ("Total Dividend Events",     f"{len(divs):,}",    GREEN),
        ("Unique Payers",             f"{divs['ticker'].nunique()}",  GREEN),
        ("Total Split Events",        f"{len(splits):,}",  ORANGE),
        ("Tickers with Splits",       f"{splits['ticker'].nunique()}", ORANGE),
        ("Avg Dividend / Event",      f"¥{divs['value'].mean():.1f}", ACCENT),
        ("Largest Split Ratio",       f"{splits['value'].max():.0f}:1", RED),
    ]

    for i, (label, val, color) in enumerate(boxes):
        x = 0.08 + (i % 3) * 0.30
        y = 0.54 if i < 3 else 0.38
        rect = FancyBboxPatch((x, y), 0.24, 0.11,
                               boxstyle="round,pad=0.01",
                               facecolor=PANEL, edgecolor=color, linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x+0.12, y+0.077, val,   ha="center", color=color,
                fontsize=18, fontweight="bold")
        ax.text(x+0.12, y+0.025, label, ha="center", color=SUBTEXT, fontsize=8.5)

    # Adjustment note
    note = (
        "Note on price adjustment:  All OHLCV data in the feature panel was downloaded with  auto_adjust=True.\n"
        "This means historical prices are backward-adjusted for every dividend and split below —\n"
        "returns, volatilities, and all derived features are computed on total-return adjusted prices.\n"
        "Dividends are baked into price continuity; splits do not cause artificial return spikes."
    )
    ax.text(0.5, 0.20, note, ha="center", color=SUBTEXT, fontsize=9,
            linespacing=1.7, style="italic",
            bbox=dict(facecolor=PANEL, edgecolor=ACCENT, boxstyle="round,pad=0.5", alpha=0.8))

    ax.text(0.5, 0.04, "Source: Yahoo Finance via yfinance  ·  Universe: NKY 225 constituents 2014–2025",
            ha="center", color=SUBTEXT, fontsize=8, alpha=0.7)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ── Page 2: Stock Splits (full list) ─────────────────────────────────────────

def page_splits(pdf, splits):
    fig = plt.figure(figsize=(11.69, 8.27))
    fig_bg(fig)

    fig.text(0.5, 0.96, "Stock Splits — NKY 225 Universe  (2014 – present)",
             ha="center", color=WHITE, fontsize=14, fontweight="bold")
    fig.text(0.5, 0.93, f"{len(splits)} split events across {splits['ticker'].nunique()} tickers",
             ha="center", color=SUBTEXT, fontsize=9)

    ax = fig.add_axes([0.03, 0.05, 0.94, 0.85])

    disp = splits.copy()
    disp["date"] = disp["date"].dt.strftime("%Y-%m-%d")
    disp = disp.sort_values("date", ascending=False).reset_index(drop=True)
    disp = disp[["date", "ticker", "company", "description", "value"]]
    disp.columns = ["Date", "Code", "Company", "Split Ratio", "Ratio"]
    disp["Ratio"] = disp["Ratio"].apply(lambda x: f"{x:.0f}:1")
    disp = disp.drop(columns=["Split Ratio"])

    # Colour rows by ratio magnitude
    def split_color(ratio_str):
        r = float(ratio_str.replace(":1",""))
        if r >= 5:  return RED
        if r >= 3:  return ORANGE
        return GREEN

    row_colors = [
        [split_color(row["Ratio"]) if col == 3 else (PANEL if i%2==0 else "#21263a")
         for col in range(len(disp.columns))]
        for i, (_, row) in enumerate(disp.iterrows())
    ]

    draw_table(ax, disp,
               col_widths=[0.12, 0.08, 0.52, 0.14],
               row_colors=row_colors, fontsize=8)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ── Page 3: Splits timeline + annual split count bar ─────────────────────────

def page_splits_analysis(pdf, splits):
    fig, axes = plt.subplots(1, 2, figsize=(11.69, 8.27))
    fig_bg(fig)
    fig.text(0.5, 0.96, "Stock Splits — Analysis", ha="center",
             color=WHITE, fontsize=13, fontweight="bold")

    # Left: splits per year
    ax1 = axes[0]
    splits2 = splits.copy()
    splits2["year"] = pd.to_datetime(splits2["date"]).dt.year
    yearly = splits2.groupby("year").size()
    bars = ax1.bar(yearly.index.astype(str), yearly.values,
                   color=ORANGE, alpha=0.85, edgecolor=BG, linewidth=0.5)
    for bar, val in zip(bars, yearly.values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                 str(val), ha="center", color=TEXT, fontsize=9)
    style_ax(ax1, "Stock Splits per Year")
    ax1.set_facecolor(PANEL)
    ax1.tick_params(axis="x", rotation=45)
    ax1.set_ylabel("Count", color=SUBTEXT, fontsize=9)
    ax1.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax1.grid(axis="y", color=BG, alpha=0.5)

    # Right: split ratio distribution
    ax2 = axes[1]
    ratio_counts = splits2["value"].value_counts().sort_index()
    bars2 = ax2.barh(
        [f"{int(r)}:1" if r == int(r) else f"{r:.1f}:1" for r in ratio_counts.index],
        ratio_counts.values,
        color=[RED if r >= 5 else ORANGE if r >= 3 else GREEN for r in ratio_counts.index],
        alpha=0.85, edgecolor=BG, linewidth=0.5
    )
    for bar, val in zip(bars2, ratio_counts.values):
        ax2.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                 str(val), va="center", color=TEXT, fontsize=9)
    style_ax(ax2, "Distribution of Split Ratios")
    ax2.set_facecolor(PANEL)
    ax2.set_xlabel("Number of events", color=SUBTEXT, fontsize=9)
    ax2.grid(axis="x", color=BG, alpha=0.5)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ── Pages 4+: Dividends by sector / summary table ────────────────────────────

def page_dividends_overview(pdf, divs):
    fig, axes = plt.subplots(1, 2, figsize=(11.69, 8.27))
    fig_bg(fig)
    fig.text(0.5, 0.96, "Dividend Events — Overview", ha="center",
             color=WHITE, fontsize=13, fontweight="bold")

    divs2 = divs.copy()
    divs2["year"] = pd.to_datetime(divs2["date"]).dt.year

    # Left: events per year
    ax1 = axes[0]
    yearly = divs2.groupby("year").size()
    bars = ax1.bar(yearly.index.astype(str), yearly.values,
                   color=GREEN, alpha=0.85, edgecolor=BG, linewidth=0.5)
    for bar, val in zip(bars, yearly.values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 str(val), ha="center", color=TEXT, fontsize=8.5)
    style_ax(ax1, "Dividend Events per Year")
    ax1.set_facecolor(PANEL)
    ax1.tick_params(axis="x", rotation=45)
    ax1.set_ylabel("Event count", color=SUBTEXT, fontsize=9)
    ax1.grid(axis="y", color=BG, alpha=0.5)

    # Right: avg dividend per year
    ax2 = axes[1]
    avg_div = divs2.groupby("year")["value"].mean()
    ax2.plot(avg_div.index.astype(str), avg_div.values,
             color=ACCENT, linewidth=2, marker="o", markersize=6)
    ax2.fill_between(range(len(avg_div)), avg_div.values,
                     alpha=0.15, color=ACCENT)
    style_ax(ax2, "Average Dividend per Event (¥)")
    ax2.set_facecolor(PANEL)
    ax2.tick_params(axis="x", rotation=45)
    ax2.set_xticks(range(len(avg_div)))
    ax2.set_xticklabels(avg_div.index.astype(str))
    ax2.set_ylabel("¥ per share", color=SUBTEXT, fontsize=9)
    ax2.grid(axis="y", color=BG, alpha=0.3, linestyle="--")
    for i, (x, y) in enumerate(zip(range(len(avg_div)), avg_div.values)):
        ax2.annotate(f"¥{y:.0f}", (x, y), textcoords="offset points",
                     xytext=(0, 8), ha="center", color=ACCENT, fontsize=7.5)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_top_dividend_payers(pdf, divs):
    """Top 30 tickers by total dividends paid."""
    fig = plt.figure(figsize=(11.69, 8.27))
    fig_bg(fig)
    fig.text(0.5, 0.96, "Top 30 — Total Dividends Paid per Share (2014–present)",
             ha="center", color=WHITE, fontsize=13, fontweight="bold")

    ax = fig.add_axes([0.03, 0.04, 0.94, 0.88])

    ticker_totals = (
        divs.groupby(["ticker", "company"])["value"]
        .sum()
        .reset_index()
        .sort_values("value", ascending=False)
        .head(30)
    )
    ticker_totals["label"] = ticker_totals["ticker"] + " " + ticker_totals["company"].str[:25]

    bars = ax.barh(
        ticker_totals["label"][::-1],
        ticker_totals["value"][::-1],
        color=GREEN, alpha=0.8, edgecolor=BG, linewidth=0.4
    )
    for bar, val in zip(bars, ticker_totals["value"][::-1]):
        ax.text(bar.get_width() + max(ticker_totals["value"])*0.005,
                bar.get_y() + bar.get_height()/2,
                f"¥{val:,.0f}", va="center", color=TEXT, fontsize=7.5)

    style_ax(ax, "")
    ax.set_facecolor(PANEL)
    ax.set_xlabel("Total dividends paid per share (JPY, sum of events)", color=SUBTEXT)
    ax.tick_params(axis="y", labelsize=7.5)
    ax.grid(axis="x", color=BG, alpha=0.5)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_dividends_detail(pdf, divs, page_num, total_pages):
    """One page of the full dividend table — paginated."""
    ROWS_PER_PAGE = 42

    fig = plt.figure(figsize=(11.69, 8.27))
    fig_bg(fig)
    fig.text(0.5, 0.96, f"Full Dividend Listing  (page {page_num} of {total_pages})",
             ha="center", color=WHITE, fontsize=13, fontweight="bold")

    ax = fig.add_axes([0.02, 0.02, 0.96, 0.91])

    disp = divs.copy()
    disp = disp.sort_values("date", ascending=False).reset_index(drop=True)
    disp["date"] = disp["date"].dt.strftime("%Y-%m-%d")
    disp["value"] = disp["value"].apply(lambda x: f"¥{x:.2f}")
    disp = disp[["date", "ticker", "company", "value"]].copy()
    disp.columns = ["Ex-Date", "Code", "Company", "Dividend/Share"]

    start = (page_num - 1) * ROWS_PER_PAGE
    end   = start + ROWS_PER_PAGE
    chunk = disp.iloc[start:end]

    draw_table(ax, chunk,
               col_widths=[0.13, 0.08, 0.60, 0.15],
               fontsize=7.5)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_per_ticker_summary(pdf, divs, splits):
    """One page per group of 30 tickers — their dividend + split summary."""
    ROWS_PER_PAGE = 35

    summary = []
    all_tickers = sorted(
        set(divs["ticker"].unique()) | set(splits["ticker"].unique())
    )

    for code in all_tickers:
        d = divs[divs["ticker"] == code]
        s = splits[splits["ticker"] == code]
        name = d["company"].iloc[0] if not d.empty else (
               s["company"].iloc[0] if not s.empty else "")

        first_div = d["date"].min().strftime("%Y-%m") if not d.empty else "—"
        last_div  = d["date"].max().strftime("%Y-%m") if not d.empty else "—"
        n_divs    = len(d)
        total_div = f"¥{d['value'].sum():,.0f}" if not d.empty else "—"
        n_splits  = len(s)
        split_str = ", ".join(s["description"].tolist()) if not s.empty else "—"

        summary.append({
            "Code":        code,
            "Company":     name[:28],
            "Div Events":  n_divs,
            "Total Div":   total_div,
            "First Div":   first_div,
            "Last Div":    last_div,
            "Splits":      n_splits,
            "Split Detail": split_str[:30],
        })

    summary_df = pd.DataFrame(summary)
    total_pages = int(np.ceil(len(summary_df) / ROWS_PER_PAGE))

    for pg in range(total_pages):
        fig = plt.figure(figsize=(11.69, 8.27))
        fig_bg(fig)
        fig.text(0.5, 0.96,
                 f"Per-Ticker Summary — Dividends & Splits  (page {pg+1} of {total_pages})",
                 ha="center", color=WHITE, fontsize=12, fontweight="bold")

        ax = fig.add_axes([0.01, 0.02, 0.98, 0.91])
        chunk = summary_df.iloc[pg*ROWS_PER_PAGE : (pg+1)*ROWS_PER_PAGE]

        draw_table(ax, chunk,
                   col_widths=[0.06, 0.19, 0.07, 0.09, 0.09, 0.09, 0.06, 0.25],
                   fontsize=7.2, header_fontsize=8)

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    tickers_df = pd.read_csv(BASE / "nky225_ticker_names.csv")
    log.info("Universe: %d tickers", len(tickers_df))

    divs, splits = fetch_all_actions(tickers_df)

    log.info("Dividends: %d events, %d tickers", len(divs), divs["ticker"].nunique())
    log.info("Splits:    %d events, %d tickers", len(splits), splits["ticker"].nunique())

    run_date = datetime.today().strftime("%B %d, %Y")
    ROWS_PER_PAGE = 42
    n_div_pages = max(1, int(np.ceil(len(divs) / ROWS_PER_PAGE)))

    log.info("Generating PDF …")
    with PdfPages(OUT_PDF) as pdf:
        # P1: Cover
        page_cover(pdf, divs, splits, run_date)
        log.info("  Page 1: Cover done")

        # P2: Splits table
        if not splits.empty:
            page_splits(pdf, splits)
            log.info("  Page 2: Splits table done")

        # P3: Splits analysis
        if len(splits) >= 2:
            page_splits_analysis(pdf, splits)
            log.info("  Page 3: Splits analysis done")

        # P4: Dividend overview
        if not divs.empty:
            page_dividends_overview(pdf, divs)
            log.info("  Page 4: Dividend overview done")

        # P5: Top dividend payers
        if not divs.empty:
            page_top_dividend_payers(pdf, divs)
            log.info("  Page 5: Top payers done")

        # P6+: Full dividend listing (paginated)
        for pg in range(1, n_div_pages + 1):
            page_dividends_detail(pdf, divs, pg, n_div_pages)
        log.info("  Pages 6–%d: Full dividend listing done", 5 + n_div_pages)

        # Final pages: per-ticker summary
        page_per_ticker_summary(pdf, divs, splits)
        log.info("  Per-ticker summary pages done")

        meta = pdf.infodict()
        meta["Title"]   = "NKY 225 Corporate Actions Report"
        meta["Author"]  = "NKY 225 ML Pipeline"
        meta["Subject"] = "Dividends & Splits 2014–present"
        meta["Keywords"] = "NKY225 Japan dividends splits TSE"

    size_mb = OUT_PDF.stat().st_size / 1e6
    log.info("Saved: %s  (%.1f MB)", OUT_PDF.name, size_mb)

    # Quick console summary
    print("\n" + "="*60)
    print("SPLITS SUMMARY")
    print("="*60)
    print(splits[["date","ticker","company","description"]].to_string(index=False))
    print()
    print("="*60)
    print(f"DIVIDEND SUMMARY: {len(divs)} events, {divs['ticker'].nunique()} tickers")
    print("="*60)
    top = (divs.groupby(["ticker","company"])["value"].sum()
              .sort_values(ascending=False).head(20))
    print(top.to_string())


if __name__ == "__main__":
    main()
