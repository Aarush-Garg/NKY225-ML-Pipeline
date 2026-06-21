from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.colors import HexColor, white, black
from reportlab.pdfgen import canvas
from reportlab.lib.utils import simpleSplit

W, H = A4  # 210 x 297 mm

# ── Palette ───────────────────────────────────────────────────────────────────
BG          = HexColor("#0f1117")
CARD_BG     = HexColor("#161b22")
BORDER_DATA = HexColor("#1d4ed8")
BORDER_CLEA = HexColor("#0891b2")
BORDER_P1   = HexColor("#16a34a")
BORDER_P2   = HexColor("#7c3aed")
BORDER_P3   = HexColor("#d97706")
BORDER_MN   = HexColor("#dc2626")
BORDER_OUT  = HexColor("#059669")

HEAD_DATA   = HexColor("#1e3a5f")
HEAD_CLEA   = HexColor("#164e63")
HEAD_P1     = HexColor("#14532d")
HEAD_P2     = HexColor("#3b0764")
HEAD_P3     = HexColor("#451a03")
HEAD_MN     = HexColor("#450a0a")
HEAD_OUT    = HexColor("#064e3b")

TEXT_DATA   = HexColor("#93c5fd")
TEXT_CLEA   = HexColor("#67e8f9")
TEXT_P1     = HexColor("#86efac")
TEXT_P2     = HexColor("#c4b5fd")
TEXT_P3     = HexColor("#fcd34d")
TEXT_MN     = HexColor("#fca5a5")
TEXT_OUT    = HexColor("#6ee7b7")

BODY_DATA   = HexColor("#172554")
BODY_CLEA   = HexColor("#083344")
BODY_P1     = HexColor("#052e16")
BODY_P2     = HexColor("#2e1065")
BODY_P3     = HexColor("#292204")
BODY_MN     = HexColor("#300a0a")
BODY_OUT    = HexColor("#022c22")

MUTED       = HexColor("#64748b")
SUBTLE      = HexColor("#334155")
ARROW_COL   = HexColor("#475569")

# ── Helpers ───────────────────────────────────────────────────────────────────
def draw_bg(c, w, h):
    c.setFillColor(BG)
    c.rect(0, 0, w, h, fill=1, stroke=0)

def rounded_rect(c, x, y, w, h, r, fill_color, stroke_color=None, stroke_width=1):
    c.setFillColor(fill_color)
    if stroke_color:
        c.setStrokeColor(stroke_color)
        c.setLineWidth(stroke_width)
    c.roundRect(x, y, w, h, r, fill=1, stroke=1 if stroke_color else 0)

def draw_arrow(c, x, top_y, length=14):
    mid = x
    c.setStrokeColor(ARROW_COL)
    c.setLineWidth(1.5)
    # line
    c.line(mid, top_y, mid, top_y - length + 5)
    # arrowhead
    c.setFillColor(ARROW_COL)
    p = c.beginPath()
    p.moveTo(mid - 4, top_y - length + 5)
    p.lineTo(mid + 4, top_y - length + 5)
    p.lineTo(mid, top_y - length)
    p.close()
    c.drawPath(p, fill=1, stroke=0)

def wrap_text(c, text, x, y, max_width, font, size, color, line_gap=2, align="left"):
    c.setFont(font, size)
    c.setFillColor(color)
    lines = simpleSplit(text, font, size, max_width)
    cur_y = y
    for line in lines:
        if align == "center":
            c.drawCentredString(x + max_width / 2, cur_y, line)
        else:
            c.drawString(x, cur_y, line)
        cur_y -= (size + line_gap)
    return cur_y  # bottom y after last line

def section_block(c, x, y_top, width,
                  phase_label, phase_num,
                  head_bg, head_text_col,
                  body_bg, border_col,
                  items,          # list of (bullet, text) or plain strings
                  sub_cols=None): # optional list of sub-columns [(title, [lines])]
    """Draws a full section card. Returns the y coordinate of its bottom."""
    MARGIN = 4 * mm
    HEAD_H = 9 * mm
    r = 3

    # measure body height
    body_lines = 0
    if sub_cols:
        max_col_lines = 0
        for col_title, col_lines in sub_cols:
            max_col_lines = max(max_col_lines, 1 + len(col_lines))
        body_lines = max_col_lines
    else:
        for item in items:
            body_lines += 1

    LINE_H = 5 * mm
    body_h = MARGIN + body_lines * LINE_H + MARGIN
    total_h = HEAD_H + body_h

    body_y = y_top - total_h

    # border + body bg
    rounded_rect(c, x, body_y, width, total_h, r, body_bg, border_col, 1.2)

    # header bg (top rounded only — draw full rect then cover bottom)
    c.setFillColor(head_bg)
    c.roundRect(x, body_y + total_h - HEAD_H, width, HEAD_H, r, fill=1, stroke=0)
    c.rect(x, body_y + total_h - HEAD_H, width, HEAD_H / 2, fill=1, stroke=0)

    # header label
    c.setFillColor(head_text_col)
    c.setFont("Helvetica-Bold", 7.5)
    label = f"  {phase_num}  {phase_label.upper()}"
    c.drawString(x + MARGIN, body_y + total_h - HEAD_H + 3 * mm, label)

    # body content
    if sub_cols:
        col_w = (width - 2 * MARGIN) / len(sub_cols)
        for ci, (col_title, col_lines) in enumerate(sub_cols):
            cx = x + MARGIN + ci * col_w
            cy = body_y + body_h - MARGIN
            c.setFont("Helvetica-Bold", 6.8)
            c.setFillColor(head_text_col)
            c.drawString(cx, cy, col_title)
            cy -= LINE_H
            for line in col_lines:
                c.setFont("Helvetica", 6.2)
                c.setFillColor(white)
                c.drawString(cx, cy, line)
                cy -= LINE_H
    else:
        cy = body_y + body_h - MARGIN
        for item in items:
            if isinstance(item, tuple):
                bullet, text = item
                c.setFont("Helvetica-Bold", 6.5)
                c.setFillColor(head_text_col)
                c.drawString(x + MARGIN, cy, bullet)
                c.setFont("Helvetica", 6.2)
                c.setFillColor(white)
                c.drawString(x + MARGIN + 12, cy, text)
            else:
                c.setFont("Helvetica", 6.2)
                c.setFillColor(white)
                c.drawString(x + MARGIN, cy, item)
            cy -= LINE_H

    return body_y  # return top of the bottom edge


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 1 — Title + Data Collection
# ══════════════════════════════════════════════════════════════════════════════
def page1(c):
    draw_bg(c, W, H)
    M = 14 * mm
    cw = W - 2 * M   # content width
    cx = M            # content x

    # ── Title block ──────────────────────────────────────────────────────────
    title_h = 38 * mm
    ty = H - M - title_h
    rounded_rect(c, cx, ty, cw, title_h, 4, HexColor("#0d2a4a"), BORDER_DATA, 1.5)

    c.setFillColor(TEXT_DATA)
    c.setFont("Helvetica-Bold", 16)
    c.drawCentredString(W / 2, ty + title_h - 14 * mm, "NKY 225 Enhanced Index")
    c.setFont("Helvetica-Bold", 11)
    c.setFillColor(white)
    c.drawCentredString(W / 2, ty + title_h - 22 * mm, "ML Alpha Generation & Portfolio Optimisation")

    c.setFillColor(MUTED)
    c.setFont("Helvetica", 7.5)
    c.drawCentredString(W / 2, ty + title_h - 29 * mm,
        "Market-neutral active weight strategy  ·  Data window: 2014-01-01 → present")
    c.drawCentredString(W / 2, ty + title_h - 33.5 * mm,
        "Benchmark weights & constituent membership sourced from JPY121 ETF (1321.T)")

    # page indicator
    c.setFont("Helvetica", 6.5)
    c.setFillColor(MUTED)
    c.drawRightString(W - M, M / 2, "Page 1 of 4")

    # ── Step numbers legend strip ─────────────────────────────────────────────
    ARROW_X = W / 2
    y = ty - 8 * mm

    # ── SECTION 1: Data Collection ───────────────────────────────────────────
    items_dc = [
        ("①", "NKY 225 Index Price — Yahoo Finance (^N225), J-Quants, Stooq  |  From: 2014-01-01"),
        ("②", "JPY121 ETF (1321.T) Holdings — daily constituent weights & in/out membership events"),
        ("③", "Constituent OHLCV — yfinance (.T suffix), J-Quants, Alpha Vantage  |  ALL ever-members"),
        ("④", "Fundamentals — EDINET (FSA), J-Quants /fins/statements, SimFin"),
        ("⑤", "Macro & Regime — FRED (fredapi), BOJ API, OECD, Cabinet Office Japan"),
        ("⑥", "Sentiment — Nikkei VI (JPX), CBOE VIX via FRED, TSE short interest"),
        ("⑦", "Calendar & Cross-Section — derived from price data & dates"),
    ]
    y = section_block(c, cx, y, cw,
                      "Data Collection", "STEP 1",
                      HEAD_DATA, TEXT_DATA, BODY_DATA, BORDER_DATA, items_dc)

    draw_arrow(c, ARROW_X, y)
    y -= 14 * mm

    # ── SECTION 2: Raw Factor Store ──────────────────────────────────────────
    items_fs = [
        ("→", "~300 raw factors across: Momentum (41) · Volatility (37) · Volume/Liquidity (25)"),
        ("→", "Fundamental (44) · Macro & Regime (44) · Sentiment (19) · Cross-Section (14) · Calendar (21)"),
    ]
    y = section_block(c, cx, y, cw,
                      "Raw Factor Store", "STEP 2",
                      HEAD_CLEA, TEXT_CLEA, BODY_CLEA, BORDER_CLEA, items_fs)

    draw_arrow(c, ARROW_X, y)
    y -= 14 * mm

    # ── SECTION 3: Survivorship + Weights ────────────────────────────────────
    items_sv = [
        ("①", "Download 1321.T (JPY121 ETF) daily holdings from 2014-01-01"),
        ("②", "Build constituent panel: date × stock → {in_index, w_bench_i (price-weighted)}"),
        ("③", "Flag ADDED events (entry date) and REMOVED events (exit date + delisting return)"),
        ("④", "Universe = all stocks ever in ETF holdings → ~270-300 unique stocks (bias-free)"),
    ]
    y = section_block(c, cx, y, cw,
                      "Survivorship-Bias-Free Universe & Benchmark Weights (JPY121 ETF)",
                      "STEP 3",
                      HEAD_CLEA, TEXT_CLEA, BODY_CLEA, BORDER_CLEA, items_sv)

    c.showPage()


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 2 — Data Cleaning & Normalisation → Feature Panel
# ══════════════════════════════════════════════════════════════════════════════
def page2(c):
    draw_bg(c, W, H)
    M = 14 * mm
    cw = W - 2 * M
    cx = M
    ARROW_X = W / 2

    c.setFillColor(MUTED)
    c.setFont("Helvetica-Bold", 8)
    c.drawCentredString(W / 2, H - M - 5 * mm, "CONTINUED — Data Cleaning, Normalisation & Feature Panel")
    c.setFont("Helvetica", 6.5)
    c.setFillColor(MUTED)
    c.drawRightString(W - M, M / 2, "Page 2 of 4")

    y = H - M - 14 * mm

    # ── SECTION 4: PIT Alignment ──────────────────────────────────────────────
    items_pit = [
        ("①", "Fundamental data stamped at announcement date (not period end) — prevents look-ahead"),
        ("②", "Use J-Quants announcement_date field or EDINET disclosure timestamps"),
        ("③", "Stocks removed from index: features retained up to removal date only"),
    ]
    y = section_block(c, cx, y, cw,
                      "Point-in-Time (PIT) Alignment", "STEP 4",
                      HEAD_CLEA, TEXT_CLEA, BODY_CLEA, BORDER_CLEA, items_pit)

    draw_arrow(c, ARROW_X, y); y -= 14 * mm

    # ── SECTION 5: Outlier Winsorisation ─────────────────────────────────────
    items_win = [
        ("①", "Winsorise each raw factor at 1st and 99th percentile — cross-sectional, per date"),
        ("②", "Prevents extreme single observations from dominating the normalised panel"),
    ]
    y = section_block(c, cx, y, cw,
                      "Outlier Winsorisation", "STEP 5",
                      HEAD_CLEA, TEXT_CLEA, BODY_CLEA, BORDER_CLEA, items_win)

    draw_arrow(c, ARROW_X, y); y -= 14 * mm

    # ── SECTION 6: Imputation ─────────────────────────────────────────────────
    items_imp = [
        ("①", "Price data: forward-fill up to 5 trading days (covers halted / suspended stocks)"),
        ("②", "Fundamental data: Last Observation Carried Forward (LOCF) from prior report period"),
        ("③", "If missing > 20 consecutive days: set NaN; exclude from IC calculation for that period"),
    ]
    y = section_block(c, cx, y, cw,
                      "Missing Value Imputation", "STEP 6",
                      HEAD_CLEA, TEXT_CLEA, BODY_CLEA, BORDER_CLEA, items_imp)

    draw_arrow(c, ARROW_X, y); y -= 14 * mm

    # ── SECTION 7: Corporate Actions ─────────────────────────────────────────
    items_ca = [
        ("①", "Apply split & dividend adjustment factors from J-Quants or yfinance adjusted close"),
        ("②", "All price series on total-return-equivalent basis — critical for momentum factors"),
    ]
    y = section_block(c, cx, y, cw,
                      "Corporate Action Adjustment", "STEP 7",
                      HEAD_CLEA, TEXT_CLEA, BODY_CLEA, BORDER_CLEA, items_ca)

    draw_arrow(c, ARROW_X, y); y -= 14 * mm

    # ── SECTION 8: Normalisation ──────────────────────────────────────────────
    items_norm = [
        ("①", "Cross-sectional z-score per date:  z_{i,t} = (f_{i,t} − mean_t) / std_t"),
        ("②", "Optional: time-series z-score per stock (stabilises long-run factor mean shifts)"),
        ("③", "Hard clip to ±7σ — removes any remaining extreme outliers post-normalisation"),
    ]
    y = section_block(c, cx, y, cw,
                      "Normalisation  (CS z-score → optional TS z-score → clip ±7σ)", "STEP 8",
                      HEAD_CLEA, TEXT_CLEA, BODY_CLEA, BORDER_CLEA, items_norm)

    draw_arrow(c, ARROW_X, y); y -= 14 * mm

    # ── SECTION 9: Clean Panel ────────────────────────────────────────────────
    items_panel = [
        ("→", "~480,000 observations:  Daily × ~305 stocks × ~270 normalised features"),
        ("→", "Each row carries:  w_bench_{i,t}  (from JPY121 ETF)  and  in_index_{i,t}  ∈ {0, 1}"),
        ("→", "Factor groups: Momentum · Volatility · Liquidity · Fundamental · Macro · Sentiment · Calendar"),
    ]
    y = section_block(c, cx, y, cw,
                      "Clean Feature Panel Output", "OUTPUT",
                      HEAD_P1, TEXT_P1, BODY_P1, BORDER_P1, items_panel)

    c.showPage()


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 3 — Phase 1 Feature Selection + Phase 2 Signal Models
# ══════════════════════════════════════════════════════════════════════════════
def page3(c):
    draw_bg(c, W, H)
    M = 14 * mm
    cw = W - 2 * M
    cx = M
    ARROW_X = W / 2

    c.setFillColor(MUTED)
    c.setFont("Helvetica-Bold", 8)
    c.drawCentredString(W / 2, H - M - 5 * mm, "CONTINUED — Feature Selection & Signal Models")
    c.setFont("Helvetica", 6.5)
    c.setFillColor(MUTED)
    c.drawRightString(W - M, M / 2, "Page 3 of 4")

    y = H - M - 14 * mm

    # ── PHASE 1 Header ────────────────────────────────────────────────────────
    c.setFillColor(HEAD_P1)
    c.roundRect(cx, y - 7 * mm, cw, 7 * mm, 3, fill=1, stroke=0)
    c.setFillColor(TEXT_P1)
    c.setFont("Helvetica-Bold", 9)
    c.drawCentredString(W / 2, y - 5 * mm, "⬛  PHASE 1 — FEATURE SELECTION")
    y -= 11 * mm

    items_p1a = [
        ("①", "Rank every feature by standalone cross-sectional Spearman IC vs forward returns"),
        ("②", "Use purged walk-forward panel — no overlapping return windows in IC estimation"),
        ("③", "Keep top K = 50 features with highest mean |IC| across history"),
        ("④", "Stability filter: IC sign must be consistent ≥ 70% of months (discard noisy features)"),
        ("⑤", "Output: ONE locked 50-feature set — identical input to ALL downstream models"),
    ]
    y = section_block(c, cx, y, cw,
                      "Univariate IC Ranking → Locked 50-Feature Set", "STEP 9",
                      HEAD_P1, TEXT_P1, BODY_P1, BORDER_P1, items_p1a)

    draw_arrow(c, ARROW_X, y); y -= 14 * mm

    # ── PHASE 2 Header ────────────────────────────────────────────────────────
    c.setFillColor(HEAD_P2)
    c.roundRect(cx, y - 7 * mm, cw, 7 * mm, 3, fill=1, stroke=0)
    c.setFillColor(TEXT_P2)
    c.setFont("Helvetica-Bold", 9)
    c.drawCentredString(W / 2, y - 5 * mm, "⬛  PHASE 2 — SIGNAL MODELS")
    y -= 11 * mm

    # Two sub-columns for the two models
    sub2 = [
        ("MODEL A — LGBMRanker", [
            "Loss: LambdaRank (pairwise ranking)",
            "Learns relative ORDER of stocks",
            "Target: which names beat the index",
            "Covered IC ≈ 0.025 (in blend)",
            "Robust to return-scale noise",
        ]),
        ("MODEL B — LightGBM-Huber", [
            "Loss: Huber (L2 small + L1 large)",
            "Predicts return SIZE, outlier-robust",
            "Target: forward return magnitude",
            "Covered IC ≈ 0.045",
            "Uncovered IC ≈ 0.12",
        ]),
    ]
    y = section_block(c, cx, y, cw,
                      "Two Complementary Gradient-Boosted Models", "STEP 10",
                      HEAD_P2, TEXT_P2, BODY_P2, BORDER_P2,
                      [], sub_cols=sub2)

    draw_arrow(c, ARROW_X, y); y -= 14 * mm

    items_route = [
        ("→", "Covered stocks (full index members): Blend of Huber + Ranker (e.g. 60% / 40%)"),
        ("→", "Uncovered stocks (extended universe, partial data): Huber only (more robust)"),
    ]
    y = section_block(c, cx, y, cw,
                      "Signal Routing", "STEP 11",
                      HEAD_P2, TEXT_P2, BODY_P2, BORDER_P2, items_route)

    draw_arrow(c, ARROW_X, y); y -= 14 * mm

    items_train = [
        ("①", "Walk-forward expanding window — training t₀ → t_end; purge gap; test t_end+gap onward"),
        ("②", "Covered model training: 2014-01 → 2017-06 (expanding); Uncovered: 2014-01 → 2018-06"),
        ("③", "Retrain monthly / quarterly — no future data ever enters training features"),
    ]
    y = section_block(c, cx, y, cw,
                      "Training Protocol", "STEP 12",
                      HEAD_P2, TEXT_P2, BODY_P2, BORDER_P2, items_train)

    draw_arrow(c, ARROW_X, y); y -= 14 * mm

    items_gk = [
        ("→", "Grinold-Kahn:  α_i  =  IC  ×  σ_i  ×  score_i"),
        ("→", "IC = realised signal IC  ·  σ_i = annualised vol of stock i  ·  score_i = z-scored ML output"),
        ("→", "Converts dimensionless ML score → expected annual return in portfolio-optimiser units"),
    ]
    y = section_block(c, cx, y, cw,
                      "Alpha Construction — Grinold-Kahn", "STEP 13",
                      HEAD_P3, TEXT_P3, BODY_P3, BORDER_P3, items_gk)

    c.showPage()


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 4 — Phase 3 Optimisation + Market-Neutral + Output
# ══════════════════════════════════════════════════════════════════════════════
def page4(c):
    draw_bg(c, W, H)
    M = 14 * mm
    cw = W - 2 * M
    cx = M
    ARROW_X = W / 2

    c.setFillColor(MUTED)
    c.setFont("Helvetica-Bold", 8)
    c.drawCentredString(W / 2, H - M - 5 * mm, "CONTINUED — Portfolio Optimisation, Market-Neutral Book & Output")
    c.setFont("Helvetica", 6.5)
    c.setFillColor(MUTED)
    c.drawRightString(W - M, M / 2, "Page 4 of 4")

    y = H - M - 14 * mm

    # ── PHASE 3 Header ────────────────────────────────────────────────────────
    c.setFillColor(HEAD_P3)
    c.roundRect(cx, y - 7 * mm, cw, 7 * mm, 3, fill=1, stroke=0)
    c.setFillColor(TEXT_P3)
    c.setFont("Helvetica-Bold", 9)
    c.drawCentredString(W / 2, y - 5 * mm, "⬛  PHASE 3 — PORTFOLIO OPTIMISATION  (QP · CLARABEL)")
    y -= 11 * mm

    items_obj = [
        ("→", "Objective:  maximise  αᵀδw  −  λ · δwᵀΩδw"),
        ("→", "α = Grinold-Kahn alpha vector  ·  Ω = covariance matrix of active returns  ·  λ = risk aversion"),
    ]
    y = section_block(c, cx, y, cw,
                      "QP Objective Function", "STEP 14",
                      HEAD_P3, TEXT_P3, BODY_P3, BORDER_P3, items_obj)

    draw_arrow(c, ARROW_X, y); y -= 14 * mm

    # Constraints — 2 col layout
    sub_c = [
        ("Portfolio Constraints", [
            "C1:  Σw = 1            (fully invested)",
            "C2:  w_bench + δw ≥ 0  (long-only, no short single stocks)",
            "C4:  δw = 0 if no signal  (passive fallback)",
        ]),
        ("Risk & Cost Constraints", [
            "C3:  β caps by cap tier  (large / mid / small)",
            "C5:  TE ≤ budget  (annualised tracking error)",
            "C6:  |sector Δw| ≤ cap  (sector exposure limit)",
            "C7:  Turnover ≤ budget  (transaction cost control)",
        ]),
    ]
    y = section_block(c, cx, y, cw,
                      "Constraints  C1 – C7", "STEP 15",
                      HEAD_P3, TEXT_P3, BODY_P3, BORDER_P3,
                      [], sub_cols=sub_c)

    draw_arrow(c, ARROW_X, y); y -= 14 * mm

    items_aw = [
        ("→", "Active weight:  δw_i  =  w_i  −  w_bench_i  (deviation from JPY121 ETF benchmark weight)"),
        ("→", "Constraint:  Σδw_i = 0  (active-neutral; overweights cancel underweights exactly)"),
        ("→", "TE definition:  √(δwᵀ Ω δw) × √252  |  Target: TE > 2% annualised (5-day)"),
    ]
    y = section_block(c, cx, y, cw,
                      "Active Weight Output", "STEP 16",
                      HEAD_P3, TEXT_P3, BODY_P3, BORDER_P3, items_aw)

    draw_arrow(c, ARROW_X, y); y -= 14 * mm

    # ── Market Neutral ────────────────────────────────────────────────────────
    c.setFillColor(HEAD_MN)
    c.roundRect(cx, y - 7 * mm, cw, 7 * mm, 3, fill=1, stroke=0)
    c.setFillColor(TEXT_MN)
    c.setFont("Helvetica-Bold", 9)
    c.drawCentredString(W / 2, y - 5 * mm, "⬛  MARKET-NEUTRAL ACTIVE BOOK  (NET = 0%)")
    y -= 11 * mm

    sub_mn = [
        ("📗 LONG LEG", [
            "~225 NKY 225 constituents",
            "Fully invested, long-only",
            "Overweight high-alpha stocks",
            "δw_i > 0  →  above benchmark",
        ]),
        ("📕 SHORT LEG", [
            "NKY 225 index ~100%",
            "via futures / ETF (1321.T)",
            "Hedges all market beta",
            "NET market exposure = 0%",
        ]),
        ("Active P&L", [
            "P&L = Σ δw_i · (r_i − r_index)",
            "Purely from stock selection",
            "Delta-neutral to NKY 225",
            "No directional index exposure",
        ]),
    ]
    y = section_block(c, cx, y, cw,
                      "Long + Short Construction", "STEP 17",
                      HEAD_MN, TEXT_MN, BODY_MN, BORDER_MN,
                      [], sub_cols=sub_mn)

    draw_arrow(c, ARROW_X, y); y -= 14 * mm

    # ── Output metrics ────────────────────────────────────────────────────────
    items_out = [
        ("→", "Rebalance: monthly  ·  Live window: Jun 2018 → present"),
        ("→", "Target CAGR: ~11.3%  ·  Sharpe: ~5.76  ·  Max Drawdown: < 15%"),
        ("→", "IC target: > 0.03  ·  ICIR > 0.5  ·  Tracking Error: 2–5% annualised"),
        ("→", "Turnover: < 20% one-way per month  ·  Sector Δw within cap"),
    ]
    y = section_block(c, cx, y, cw,
                      "Live Portfolio — Target Performance", "OUTPUT",
                      HEAD_OUT, TEXT_OUT, BODY_OUT, BORDER_OUT, items_out)

    # ── Implementation stack footer ───────────────────────────────────────────
    y -= 8 * mm
    stack = [
        "yfinance · jquantsapi · fredapi · pandas_datareader",
        "pandas · numpy · pandas_ta · talib  (feature engineering)",
        "lightgbm  (LGBMRanker + LGBMRegressor Huber)",
        "clarabel · cvxpy  (QP optimisation)",
        "scipy.stats  (IC / Spearman evaluation)",
    ]
    rounded_rect(c, cx, y - len(stack) * 5 * mm - 8 * mm, cw,
                 len(stack) * 5 * mm + 12 * mm, 3, HexColor("#0d1117"), SUBTLE, 0.8)
    c.setFillColor(MUTED)
    c.setFont("Helvetica-Bold", 6.5)
    c.drawString(cx + 4 * mm, y - 5 * mm, "IMPLEMENTATION STACK")
    iy = y - 10 * mm
    for s in stack:
        c.setFont("Helvetica", 6)
        c.setFillColor(HexColor("#94a3b8"))
        c.drawString(cx + 4 * mm, iy, "▸  " + s)
        iy -= 5 * mm

    c.showPage()


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
OUT = "/Users/ags/Developer Aarush/Finance Quant Internship/NKY225_Pipeline.pdf"
c = canvas.Canvas(OUT, pagesize=A4)
c.setTitle("NKY 225 ML Alpha Pipeline")
c.setAuthor("Aarush Garg")
c.setSubject("NKY 225 Enhanced Index — ML Alpha Generation & Portfolio Optimisation")

page1(c)
page2(c)
page3(c)
page4(c)

c.save()
print(f"PDF saved → {OUT}")
