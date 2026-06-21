# NKY 225 — Feature Classification & Public Data Sources

> Every source listed here is **freely and publicly accessible** (free tier, open API, or government publication). No Bloomberg, Refinitiv, or paid data required to build the core pipeline.

---

## Category Map

| # | Category | Features Inside | Primary Public Source |
|---|----------|-----------------|-----------------------|
| 1 | [Price & OHLCV](#1-price--ohlcv) | Raw prices, volume, returns | `yfinance`, J-Quants, Stooq |
| 2 | [Technical / Price-Derived](#2-technical--price-derived) | Momentum, reversal, vol, MA | Calculated from OHLCV |
| 3 | [Fundamental](#3-fundamental) | Value, quality, growth, size | J-Quants, EDINET, SimFin |
| 4 | [Macro & Regime](#4-macro--regime) | FX, rates, inflation, PMI | FRED, BOJ API, Cabinet Office Japan |
| 5 | [Sentiment & Flows](#5-sentiment--flows) | Short interest, foreign flows, options | JPX, FRED, TSE investor data |
| 6 | [Japan-Specific / Governance](#6-japan-specific--governance) | TSE reform, cross-shareholding, BOJ ETF | EDINET, JPX, BOJ |
| 7 | [Alternative / NLP](#7-alternative--nlp) | News sentiment, Yūhō NLP | EDINET, public corpora |
| 8 | [Calendar & Structural](#8-calendar--structural) | Fiscal year-end, ex-div, holidays | Derived from dates |
| 9 | [Universe & Weights](#9-universe--weights) | Constituent membership, bench weights | JPY121 ETF (1321.T), JPX |

---

## 1. Price & OHLCV

**Features:** Raw daily Open, High, Low, Close, Volume, Adjusted Close. Foundation for all technical factors.

### Sources

#### 1A. yfinance (Yahoo Finance)
- **Access:** `pip install yfinance` — no API key needed
- **Coverage:** All TSE-listed stocks (`.T` suffix), `^N225` index, `1321.T` ETF, `JPY=X` FX rate
- **History:** ~20+ years daily for major stocks; 1 min intraday (60 days)
- **Cost:** Free, unlimited

```python
import yfinance as yf

# Single stock — Toyota
toyota = yf.download("7203.T", start="2014-01-01", auto_adjust=True)

# NKY 225 index level
nky = yf.download("^N225", start="2014-01-01", auto_adjust=True)

# Batch download all constituents
tickers = ["7203.T", "6758.T", "9984.T", "6501.T"]   # add all 225+
data = yf.download(tickers, start="2014-01-01", auto_adjust=True, group_by="ticker")

# USD/JPY
usdjpy = yf.download("JPY=X", start="2014-01-01")
```

**Limitations:** Adjusted close only (not total return for Japanese dividends); no official constituent history; rate-limited for large batch downloads.

---

#### 1B. J-Quants API (Japan Exchange Group — official)
- **Access:** Register free at [jpx-jquants.com](https://jpx-jquants.com) → API key → `pip install jquantsapi`
- **Coverage:** All TSE-listed stocks; OHLCV + trading value (yen); dividend/split adjusted; constituent history
- **History:** Daily from 2013-01-01
- **Cost:** Free tier (standard plan); paid tiers for intraday

```python
import jquantsapi as jq

client = jq.Client(mail_address="your@email.com", password="yourpw")

# Daily OHLCV for a stock (TSE code, no .T suffix)
prices = client.get_prices_daily_quotes(code="7203")

# All prices for a date (cross-section)
all_prices = client.get_prices_daily_quotes(date="2024-01-05")

# Dividend-adjusted close
adj = client.get_prices_daily_quotes(code="7203", from_yyyymmdd="20140101")

# Trading calendar
calendar = client.get_market_trading_calendar()
```

**Why preferred over yfinance for Japan:** PIT-clean splits/dividends; official TSE data; constituent history built-in; yen trading value for Amihud illiquidity.

---

#### 1C. Stooq (via pandas_datareader)
- **Access:** `pip install pandas-datareader` — no API key
- **Coverage:** Japanese stocks (format: `NNNN.JP`), Nikkei 225
- **History:** 20+ years daily
- **Cost:** Free

```python
import pandas_datareader.data as web
from datetime import datetime

# Nikkei 225 index
nky = web.DataReader("^NKX", "stooq", start="2014-01-01")

# Individual stock (Toyota)
toyota = web.DataReader("7203.JP", "stooq", start="2014-01-01")
```

---

#### 1D. Alpha Vantage
- **Access:** Free API key at [alphavantage.co](https://alphavantage.co) (500 requests/day free)
- **Coverage:** Japanese stocks, FX, intraday
- **Cost:** Free tier (5 req/min, 500 req/day); premium plans available

```python
from alpha_vantage.timeseries import TimeSeries

ts = TimeSeries(key="YOUR_API_KEY", output_format="pandas")
data, meta = ts.get_daily_adjusted(symbol="7203.T", outputsize="full")
```

---

## 2. Technical / Price-Derived

**Features:** All calculated from OHLCV data — no additional data source needed beyond Section 1.

### Feature List & Calculation

| Feature | Formula / Library | Category |
|---------|-------------------|----------|
| Short-term reversal (1M) | `-1 × prior_month_return` | **Tier 1 — strongest Japan signal** |
| Short-term reversal (1W) | `-1 × prior_week_return` | Tier 1 |
| Price momentum (12-1M) | `cumret(t-12, t-2)` | Tier 2 (weak in Japan — use with caution) |
| 52-week high proximity | `close / rolling_52w_max` | Tier 2 |
| RSI (14-day) | `pandas_ta.rsi(close, 14)` | Tier 2 |
| MACD signal | `pandas_ta.macd(close)` | Tier 2 |
| Bollinger Band position | `(close − BB_lower) / (BB_upper − BB_lower)` | Tier 2 |
| EMA crossover (5/20) | `EMA_5 / EMA_20 − 1` | Tier 2 |
| EMA crossover (50/200) | `EMA_50 / EMA_200 − 1` | Tier 2 |
| Realised volatility (20D) | `std(log_returns, 20) × √252` | Tier 1 (low-vol premium in Japan) |
| Realised volatility (60D) | `std(log_returns, 60) × √252` | Tier 1 |
| EWMA volatility | RiskMetrics λ=0.94 | Tier 2 |
| Beta (market, 60D) | `OLS(stock_ret, nky_ret, 60D)` | Tier 2 |
| Idiosyncratic vol | `std(residuals from market OLS)` | Tier 2 |
| High-Low Parkinson vol | `1/(4 ln2) × mean((ln H/L)²)` | Tier 2 |
| Relative volume (5D/60D) | `vol_5d_avg / vol_60d_avg` | Tier 2 |
| Intraday momentum | `AM_session_return (09:00–11:30 JST)` | Tier 3 — needs intraday data |
| Amihud illiquidity (60D) | `mean(|ret| / yen_volume, 60D)` | **Tier 1 — highest Japan premium** |
| Turnover rate | `volume / shares_outstanding` | Tier 2 |

### Calculation libraries

```python
import pandas_ta as ta     # pip install pandas_ta   — 130+ indicators
import talib               # pip install TA-Lib      — C-backed, fastest
import numpy as np

df = prices[["Open","High","Low","Close","Volume"]]

# Short-term reversal (1M)
df["ret_1m"]     = df["Close"].pct_change(21)          # 21 trading days
df["reversal_1m"] = -df["ret_1m"]                      # negate = contrarian

# Price momentum (12-1M)
df["momentum_12_1"] = df["Close"].pct_change(252) / df["Close"].pct_change(21)

# Realised volatility
df["vol_20d"] = np.log(df["Close"]).diff().rolling(20).std() * np.sqrt(252)

# Amihud illiquidity
df["yen_volume"] = df["Close"] * df["Volume"]
df["illiq_60d"]  = (df["ret_1m"].abs() / df["yen_volume"]).rolling(60).mean()

# RSI
df["rsi_14"] = ta.rsi(df["Close"], length=14)

# Bollinger Bands
bb = ta.bbands(df["Close"], length=20)
df["bb_pos"] = (df["Close"] - bb["BBL_20_2.0"]) / (bb["BBU_20_2.0"] - bb["BBL_20_2.0"])

# 52-week high proximity (high-to-price — Iwanaga 2024)
df["high_52w"]      = df["High"].rolling(252).max()
df["high_to_price"] = df["high_52w"] / df["Close"]

# Rolling beta to NKY 225
def rolling_beta(stock_ret, index_ret, window=60):
    cov = stock_ret.rolling(window).cov(index_ret)
    var = index_ret.rolling(window).var()
    return cov / var
```

---

## 3. Fundamental

**Features:** Value ratios, profitability, quality, growth, size. All require financial statements.

### Feature List

| Feature | Formula | Tier | Category |
|---------|---------|------|----------|
| Book-to-Price (B/P) | `book_equity / market_cap` | **1** | Value |
| EBITDA/EV | `ebitda / (market_cap + debt − cash)` | **1** | Value |
| Earnings yield (E/P) | `eps_ttm / price` | 2 | Value |
| Dividend yield | `dps_ttm / price` | 2 | Value |
| Total shareholder yield | `(dps + buyback_per_share) / price` | 2 | Value |
| Gross profit-to-assets (GPA) | `(revenue − cogs) / total_assets` | **1** | Quality |
| ROE | `net_income / book_equity` | 2 | Quality |
| ROE improvement (YoY) | `roe_t / roe_t−1 − 1` | 2 | Quality |
| ROA | `net_income / total_assets` | 2 | Quality |
| ROIC | `ebit × (1−tax) / invested_capital` | 2 | Quality |
| Operating margin | `operating_income / revenue` | 2 | Quality |
| Accruals | `(net_income − op_cashflow) / total_assets` (negated) | 2 | Quality |
| Asset growth | `total_assets_t / total_assets_t−1 − 1` (negated) | 3 | Growth |
| Revenue growth (YoY) | `revenue_t / revenue_t−1 − 1` | 2 | Growth |
| EPS growth (YoY) | `eps_t / eps_t−1 − 1` | 2 | Growth |
| Net cash / EV | `(cash − total_debt) / ev` | 2 | Financial strength |
| Debt/EV | `total_debt / ev` | 2 | Leverage |
| Current ratio | `current_assets / current_liabilities` | 3 | Solvency |
| Altman Z-score | Standard formula | 3 | Distress |
| Log market cap (Size) | `log(market_cap_jpy)` | 2 | Size |

---

### Sources

#### 3A. J-Quants API — Financial Statements (Best for Japan)
- **Access:** Same free account as Section 1B
- **Endpoint:** `/fins/statements` — quarterly P&L, balance sheet, cash flow
- **Coverage:** All TSE-listed stocks; PIT announcement dates included
- **History:** From ~2012

```python
import jquantsapi as jq

client = jq.Client(mail_address="your@email.com", password="yourpw")

# Quarterly financial statements (PIT: uses announcement_date, not fiscal period-end)
fins = client.get_fins_statements(code="7203")
# Columns: AnnouncementDate, NetSales, OperatingProfit, OrdinaryProfit,
#          NetIncome, TotalAssets, NetAssets, EPS, DividendPerShare, ...

# All companies for a given announcement date window
fins_all = client.get_fins_statements(date="2024-01-05")

# Shares outstanding (for market cap calculation)
shares = client.get_fins_dividend(code="7203")
```

**Key advantage:** `AnnouncementDate` field = exact PIT timestamp — no look-ahead bias. This is the most important feature for avoiding data leakage in fundamental factors.

---

#### 3B. EDINET (FSA — Financial Services Agency)
- **Access:** [edinet-api.fsa.go.jp](https://disclosure2.edinet-api.fsa.go.jp) — free, no key required for public documents
- **Coverage:** All TSE/Osaka-listed companies; Yūhō (annual report), quarterly reports, large shareholder filings
- **Format:** XBRL or PDF; requires parsing for structured data
- **History:** 2008 → present

```python
import requests, json

BASE = "https://disclosure2.edinet-api.fsa.go.jp/api/v2"

# List filings for a date
docs = requests.get(f"{BASE}/documents.json",
                    params={"date": "2024-01-15", "type": 2}).json()
# type=2 → Yūhō (annual securities report)

# Download XBRL document (financial statements)
doc_id = docs["results"][0]["docID"]
xbrl   = requests.get(f"{BASE}/documents/{doc_id}",
                       params={"type": 5})   # type=5 → XBRL ZIP
```

**Use case for NKY 225:**
- Cross-shareholding reduction data (大量保有報告書, type=3)
- PIT fundamental data cross-check
- Yūhō text for NLP sentiment (Section 7)

---

#### 3C. SimFin (Standardised Fundamentals)
- **Access:** [simfin.com](https://simfin.com) — free API key; `pip install simfin`
- **Coverage:** ~3,000 companies globally; Japan coverage limited (~200 companies, mainly large-caps)
- **History:** From ~2008; standardised income statement / balance sheet / cash flow
- **Cost:** Free tier (bulk download); plus plan for daily updates

```python
import simfin as sf

sf.set_api_key("YOUR_FREE_KEY")
sf.set_data_dir("~/simfin_data/")

# Income statements
income = sf.load_income(variant="annual", market="jp")

# Balance sheets
balance = sf.load_balance(variant="annual", market="jp")

# Cash flow
cashflow = sf.load_cashflow(variant="annual", market="jp")
```

**Limitation for NKY 225:** Japan coverage is sparser than J-Quants. Use as a cross-check or for companies not well-covered by J-Quants. J-Quants should be the primary source.

---

#### 3D. Yahoo Finance (yfinance) — Quick Fundamentals
- **Access:** Already installed (Section 1A)
- **Coverage:** P/E, EPS, book value, shares outstanding for major Japanese stocks
- **Limitation:** Not PIT-clean; limited historical depth; use only for screening, not rigorous backtesting

```python
import yfinance as yf

ticker = yf.Ticker("7203.T")

# Balance sheet
balance_sheet = ticker.balance_sheet         # annual
quarterly_bs  = ticker.quarterly_balance_sheet

# Income statement
income_stmt   = ticker.income_stmt
cashflow      = ticker.cashflow

# Fast info (market cap, shares, P/E, EPS)
info = ticker.info
market_cap = info.get("marketCap")
book_value = info.get("bookValue")
pe_ratio   = info.get("trailingPE")
```

---

## 4. Macro & Regime

**Features:** USD/JPY, JGB yields, CPI, PMI, BOJ policy, global risk indicators.

### Feature List

| Feature | Definition | Tier |
|---------|-----------|------|
| USD/JPY level | Spot rate daily close | **1** |
| USD/JPY 1M change | `usdjpy_t / usdjpy_t−21 − 1` | **1** |
| USD/JPY 3M change | Monthly returns over 3 months | **1** |
| Stock-level USD/JPY beta (24M) | `rolling_OLS(stock_ret, usdjpy_ret, 504D)` | **1** |
| JGB 2Y yield | Bank of Japan published | 2 |
| JGB 10Y yield | Bank of Japan published | 2 |
| JGB yield curve slope | `jgb_10y − jgb_2y` | 2 |
| JGB 2Y 1M change | Monthly change in 2Y yield | 2 |
| Japan CPI YoY | Cabinet Office Japan | 2 |
| Japan core CPI | Ex-fresh food CPI | 2 |
| Japan wage growth | Ministry of Health, Labour & Welfare | 2 |
| Japan manufacturing PMI | S&P Global / au Jibun Bank | 2 |
| Japan services PMI | S&P Global | 2 |
| US 10Y Treasury yield | FRED `DGS10` | 2 |
| VIX | CBOE via FRED `VIXCLS` | 2 |
| US-Japan 10Y spread | `us_10y − jgb_10y` | 2 |
| Reflation regime flag | Binary: CPI YoY > 1% AND wages > 2% | 2 |
| Overnight S&P 500 return | `^GSPC` prev-day close to open | 3 |

---

### Sources

#### 4A. FRED (Federal Reserve Bank of St. Louis)
- **Access:** [fred.stlouisfed.org](https://fred.stlouisfed.org) — free API key; `pip install fredapi`
- **Coverage:** 800,000+ economic time series; USD/JPY, VIX, US rates, global macro
- **History:** Decades of daily/monthly data

```python
from fredapi import Fred

fred = Fred(api_key="YOUR_FREE_FRED_KEY")   # free at fred.stlouisfed.org/api/

# USD/JPY exchange rate (daily)
usdjpy = fred.get_series("DEXJPUS", observation_start="2014-01-01")

# CBOE VIX
vix = fred.get_series("VIXCLS", observation_start="2014-01-01")

# US 10-Year Treasury yield
us10y = fred.get_series("DGS10", observation_start="2014-01-01")

# Japan CPI (monthly)
japan_cpi = fred.get_series("CPALTT01JPM657N", observation_start="2014-01-01")

# Japan manufacturing PMI
japan_pmi = fred.get_series("JPNPMIOUT", observation_start="2014-01-01")
```

**Key FRED series for NKY 225:**

| Series ID | Description |
|-----------|-------------|
| `DEXJPUS` | USD/JPY daily spot rate |
| `VIXCLS` | CBOE VIX daily |
| `DGS10` | US 10Y Treasury yield |
| `DGS2` | US 2Y Treasury yield |
| `CPALTT01JPM657N` | Japan CPI YoY |
| `JPNPMIOUT` | Japan Manufacturing PMI |
| `JTOV` | Japan job openings-to-vacancies ratio |

---

#### 4B. Bank of Japan (BOJ) Statistics
- **Access:** [stat.boj.or.jp](https://www.stat.boj.or.jp/en/index.htm) — free, no key; also via `pandas_datareader`
- **Coverage:** JGB yields (1Y, 2Y, 5Y, 10Y, 20Y, 30Y), monetary base, overnight call rate, BOJ balance sheet

```python
import requests
import pandas as pd

# BOJ offers CSV downloads directly
# JGB yields (10Y):
url = "https://www.stat.boj.or.jp/data/ir/ryutan.csv"
jgb = pd.read_csv(url, encoding="shift_jis", skiprows=2, index_col=0, parse_dates=True)

# Via pandas_datareader (BOJ)
import pandas_datareader.data as web
jgb_2y  = web.DataReader("IR01'MAIP'S50N@IR",  "boj", start="2014-01-01")
jgb_10y = web.DataReader("IR01'MAIP'S316N@IR", "boj", start="2014-01-01")
```

---

#### 4C. Cabinet Office Japan (CPI, GDP, Consumer Confidence)
- **Access:** [esri.cao.go.jp](https://www.esri.cao.go.jp/en/sna/data/sokuhou/files/2023/toukei_e.html) — free CSV downloads; no API
- **Coverage:** GDP, household consumption, consumer confidence index, average wages

```python
import pandas as pd

# Consumer Confidence Index (monthly, CSV direct download)
cc_url = "https://www.esri.cao.go.jp/en/stat/shouhi/shouhi-e.html"
# Download CSV from above page → read as:
cc = pd.read_csv("consumer_confidence.csv", parse_dates=["Date"])

# Alternative: FRED mirrors many Japanese macro series
# Japan consumer confidence: FRED series CSCICP02JPM460S
consumer_conf = fred.get_series("CSCICP02JPM460S")
```

---

#### 4D. OECD Data API
- **Access:** [stats.oecd.org](https://stats.oecd.org) — free; `pip install pandasdmx`
- **Coverage:** Japan leading indicators, industrial production, trade balances

```python
import pandasdmx as sdmx

oecd = sdmx.Request("OECD")

# Japan CLI (Composite Leading Indicator)
resp = oecd.data("MEI_CLI", key={"LOCATION": "JPN", "SUBJECT": "LOLITOAA"})
cli  = resp.to_pandas()

# Industrial production
ip_resp = oecd.data("MEI", key={"LOCATION": "JPN", "SUBJECT": "PRINTO01"})
```

---

#### 4E. Ministry of Finance Japan (Trade Data)
- **Access:** [customs.mof.go.jp](https://www.customs.mof.go.jp/toukei/info/tsdl_e.htm) — free CSV downloads
- **Coverage:** Japan monthly trade balance (exports, imports by category)

```python
# Download from MOF trade statistics portal
# https://www.customs.mof.go.jp/toukei/info/tsdl_e.htm
import pandas as pd
trade = pd.read_csv("japan_trade.csv")   # after manual download
```

---

## 5. Sentiment & Flows

**Features:** Short interest, foreign investor flows, options put-call ratio, analyst consensus.

### Feature List

| Feature | Definition | Tier |
|---------|-----------|------|
| Short interest ratio | `short_shares / float` | 2 |
| Short interest change | MoM change in short ratio | 2 |
| Foreign investor net flow (weekly) | Net foreign equity purchases (¥bn) | 2 |
| Put-Call ratio (NKY 225 options) | `NKY_puts_OI / NKY_calls_OI` | 3 |
| Nikkei VI | Japan's VIX equivalent | 2 |
| Analyst EPS revision (1M) | `(consensus_t − consensus_t−21) / |consensus_t−21|` | **1** |
| Analyst EPS revision (3M) | 3-month version of above | **1** |
| Earnings surprise (SUE) | `(actual_eps − consensus) / std(surprise)` | 2 |
| Analyst upgrade/downgrade | Net upgrades minus downgrades (1M) | 2 |
| Number of analysts (coverage) | Count of analysts covering stock | 3 |

---

### Sources

#### 5A. JPX / TSE — Short Interest & Investor-Type Data (Free)
- **Access:** [jpx.co.jp/english/markets/statistics-equities](https://www.jpx.co.jp/english/markets/statistics-equities/short-selling/index.html) — free CSV downloads, no registration
- **Coverage:**
  - **Short selling:** Bi-weekly short interest by stock (shares sold short, balance)
  - **Investor-type trading:** Weekly net purchase/sale by investor category (foreign, trust banks, individuals, securities firms, etc.)

```python
import pandas as pd
import requests

# === SHORT INTEREST ===
# JPX publishes bi-weekly CSV of short positions by stock
# URL pattern: https://www.jpx.co.jp/english/markets/statistics-equities/short-selling/nlsgeu000003ygwk-att/shortbysec_20240101.csv
# (Date varies — scrape the index page for latest file links)

# Manual download example:
short_df = pd.read_csv("shortbysec_20240101.csv", encoding="shift_jis")
# Columns: Code, Name, Outstanding_Short_Shares, Float_Shares

# Compute short interest ratio
short_df["short_ratio"] = short_df["Outstanding_Short_Shares"] / short_df["Float_Shares"]

# === FOREIGN INVESTOR FLOWS ===
# Weekly investor-type trading data (net buy/sell in ¥mn by investor type)
# Download from: https://www.jpx.co.jp/english/markets/statistics-equities/investor-type/index.html
flows_df = pd.read_csv("investor_trading_weekly.csv")
# Columns: Date, Foreign_Net, Trust_Bank_Net, Individual_Net, ...
```

---

#### 5B. Nikkei Volatility Index (Nikkei VI)
- **Access:** Via yfinance ticker `^VNKY` or JPX direct download
- **Coverage:** Daily Nikkei VI level; Japan's equivalent of the VIX
- **History:** From 2012

```python
import yfinance as yf

nikkei_vi = yf.download("^VNKY", start="2014-01-01")
```

---

#### 5C. Analyst Consensus (Free Options)

Paid sources (Bloomberg, Refinitiv I/B/E/S) have the best coverage, but **free alternatives** exist:

| Source | Access | Coverage | Python |
|--------|--------|---------|--------|
| Yahoo Finance `yf.Ticker().analyst_price_targets` | Free | Major stocks | `yfinance` |
| Macrotrends | Scraping | Historical EPS estimates for large-caps | `requests`/`bs4` |
| Wisesheets | Free tier | EPS consensus for some Japan stocks | API |
| J-Quants `/fins/forecast` | Free (registered) | Company-issued guidance (not analyst consensus) | `jquantsapi` |

```python
import yfinance as yf

ticker = yf.Ticker("7203.T")

# Analyst recommendations history
recs  = ticker.recommendations          # upgrade/downgrade history
info  = ticker.info
eps_estimate = info.get("epsForward")   # forward EPS estimate
```

**Note on analyst revisions for backtesting:** True point-in-time analyst consensus (I/B/E/S history) requires Bloomberg or Refinitiv for rigorous backtesting. As a free approximation: use company-issued guidance revisions from J-Quants `/fins/forecast` endpoint, which is PIT and publicly available.

```python
# Company earnings guidance (PIT — best free substitute)
guidance = client.get_fins_statements(code="7203")
# Use ForecastEPS, ForecastNetSales columns for guidance momentum
guidance["guidance_revision"] = guidance["ForecastEPS"].pct_change()
```

---

#### 5D. Osaka Exchange — Options Data (JPX)
- **Access:** [jpx.co.jp/english/derivatives/statistics](https://www.jpx.co.jp/english/derivatives/statistics/index.html) — free CSV downloads
- **Coverage:** Nikkei 225 options OI and volume by strike/expiry; monthly historical

```python
# Download options statistics CSV from JPX derivatives statistics page
# Compute put-call ratio from OI data
import pandas as pd

options = pd.read_csv("nky_options_stats.csv")
pcr = options["Put_OI"].sum() / options["Call_OI"].sum()
```

---

## 6. Japan-Specific / Governance

**Features:** TSE PBR reform signal, cross-shareholding reduction, constituent membership, BOJ ETF ownership, JPY121 ETF weights.

### Feature List

| Feature | Definition | Tier |
|---------|-----------|------|
| In-index flag | Binary: stock is NKY 225 constituent on date t | **1** |
| Benchmark weight `w_bench_i` | Price-weighted index weight from JPY121 ETF | **1** |
| P/B < 1.0 flag | Binary: market price / book < 1 | **1** |
| TSE improvement plan flag | Binary: disclosed TSE PBR improvement plan | **1** |
| Buyback ratio (TTM) | `buyback_jpy_ttm / market_cap` | 2 |
| Cross-shareholding delta | Change in cross-held shares / float | 2 |
| BOJ ETF eligible % | `boj_etf_holdings / market_cap` (historical) | 3 (historical) |
| Price-weight rank (NKY) | Rank of nominal share price within NKY 225 | 3 |
| TSE market segment | Prime / Standard / Growth (post-2022 reform) | 3 |

---

### Sources

#### 6A. JPY121 ETF (1321.T) — Constituent Weights & Membership
- **Access:** Via yfinance + Nomura ETF holdings page
- **Primary method:** Download 1321.T daily holdings PDF/CSV from Nomura AM website
- **URL:** [nextfunds.jp/lineup/1321/composition.html](https://nextfunds.jp/lineup/1321/composition.html) (Nomura NEXT FUNDS Nikkei 225)

```python
import yfinance as yf
import pandas as pd
import requests

# Method 1: yfinance for 1321.T price (index-level tracking)
etf = yf.download("1321.T", start="2014-01-01", auto_adjust=True)

# Method 2: Nomura publishes current holdings as CSV/PDF
# Scrape the holdings page to get per-stock weights
# The holdings CSV lists: Code, Stock Name, Shares, Weight(%)
holdings_url = "https://nextfunds.jp/lineup/1321/composition.html"
# Parse and extract table with requests + BeautifulSoup

# Method 3: Build historical weights from Nikkei 225 price-weighting formula
# w_i = price_i / divisor   (published by Nikkei Inc.)
# Divisor available at: https://indexes.nikkei.co.jp/en/nkave/index/profile?idx=nk225

# Reconstruct weight from constituent prices (yfinance) + divisor
```

**Building the historical constituent panel:**

```python
# Step 1: Get Nikkei Inc. constituent history
# Nikkei publishes constituent changes at:
# https://indexes.nikkei.co.jp/en/nkave/index/profile?idx=nk225
# (Annual review in late September, effective October)
# Scrape or manually maintain a change log

# Step 2: For each constituent on each date, pull closing price from yfinance/J-Quants
# Step 3: w_bench_i_t = price_i_t / sum(price_j_t for j in constituents_t)
# This reconstructs the exact price-weighted benchmark weight
```

---

#### 6B. EDINET — Cross-Shareholding & Large Shareholder Filings
- **Access:** [edinet-api.fsa.go.jp](https://edinet-api.fsa.go.jp) — free, no key (public documents)
- **Coverage:** 大量保有報告書 (large shareholding reports, type=3) filed when crossing ±5% threshold;変更報告書 for subsequent changes

```python
import requests

BASE = "https://disclosure2.edinet-api.fsa.go.jp/api/v2"

# Search for large shareholder filing type for a company
# type=3 → 大量保有報告書 (large shareholder reports)
docs = requests.get(f"{BASE}/documents.json",
                    params={"date": "2024-03-01", "type": 3}).json()

for doc in docs["results"]:
    print(doc["filerName"], doc["docDescription"], doc["submitDateTime"])
```

**Building the cross-shareholding delta signal:**
1. Pull all 大量保有報告書 (type=3) for a given period.
2. Filter for corporate filers (vs. institutional/index funds) selling NKY 225 stocks.
3. Compute `Δ(cross-held shares) / float` for each stock per quarter.
4. Negative delta (reduction in cross-held shares) = corporate governance improvement signal.

---

#### 6C. JPX — TSE PBR Improvement Disclosures (Post-2023)
- **Access:** [jpx.co.jp/listing/market-enhancement/market-segment/index.html](https://www.jpx.co.jp/english/listing/market-enhancement/index.html)
- **Coverage:** TSE Prime market companies' disclosure of capital cost / PBR improvement plans; updated monthly
- **Format:** PDF list or searchable portal

```python
# TSE publishes a list of companies that have disclosed PBR/ROE improvement plans
# Download from: https://www.jpx.co.jp/english/listing/market-enhancement/
# Cross-reference with P/B ratios from yfinance/J-Quants

# Create binary signal:
df["pbr_below_1"]     = (df["price"] / df["book_per_share"]) < 1.0
df["tse_plan_filed"]  = df["code"].isin(tse_disclosure_list)  # from JPX list
df["governance_flag"] = df["pbr_below_1"] & df["tse_plan_filed"]
```

---

#### 6D. BOJ — ETF Holdings (Historical)
- **Access:** [boj.or.jp/en/statistics/boj/other/etf](https://www.boj.or.jp/en/statistics/boj/other/etf/index.htm)
- **Coverage:** BOJ's total ETF purchases by ETF type (Nikkei 225 ETF, TOPIX ETF) — aggregate level, not per-stock
- **Status:** New purchases ceased March 2024; existing holdings being maintained

```python
import pandas as pd

# BOJ publishes aggregate ETF purchase history
# Download from: https://www.boj.or.jp/en/statistics/boj/other/etf/index.htm
boj_etf = pd.read_csv("boj_etf_purchases.csv")

# Per-stock BOJ exposure:
# Nikkei 225 ETF (e.g., 1321.T) holds stocks proportional to their price weights
# BOJ_exposure_i = BOJ_total_1321T_holdings × w_bench_i_t
```

---

## 7. Alternative / NLP

**Features:** Japanese news sentiment, earnings call tone, Yūhō (annual report) NLP.

### Feature List

| Feature | Source | Method | Tier |
|---------|--------|--------|------|
| News sentiment (30D rolling) | Nikkei / financial news | Japanese NLP (MeCab + FinBERT-ja) | 3 |
| Yūhō sentiment | EDINET annual reports | NLP on XBRL text blocks | 3 |
| Management tone (risk words) | EDINET / company disclosures | Keyword frequency in Japanese | 3 |
| Social media sentiment | Twitter/X Japanese finance | Japanese NLP | 3 |

---

### Sources

#### 7A. EDINET — Yūhō (Annual Reports) for NLP
- **Access:** Same as Section 6B — free, no key
- **Coverage:** Full text of annual securities reports (有価証券報告書) for all listed companies

```python
import requests, zipfile, io

BASE = "https://disclosure2.edinet-api.fsa.go.jp/api/v2"

# Get filing metadata (type=2 = Yūhō annual report)
docs = requests.get(f"{BASE}/documents.json",
                    params={"date": "2024-06-01", "type": 2}).json()

doc_id = docs["results"][0]["docID"]

# Download ZIP containing XBRL + inline HTML
resp = requests.get(f"{BASE}/documents/{doc_id}", params={"type": 5})
z    = zipfile.ZipFile(io.BytesIO(resp.content))
# Extract Japanese text from XBRL/HTML files for NLP processing
```

**NLP pipeline for Japanese text:**

```python
# pip install fugashi ipadic transformers torch
import fugashi

tagger = fugashi.Tagger()          # MeCab tokeniser (morphological analysis)

def tokenise_japanese(text):
    return [word.surface for word in tagger(text)]

# Optional: fine-tuned Japanese FinBERT for sentiment
# Model: "uer/roberta-base-finetuned-jd-binary" or "cl-tohoku/bert-base-japanese"
from transformers import pipeline
sentiment = pipeline("sentiment-analysis",
                     model="lxyuan/distilbert-base-multilingual-cased-sentiments-student")
```

---

#### 7B. Free Japanese News Corpora
- **ABEMA News API:** Limited free tier for Japanese news headlines
- **NHK News Web Easy API:** [www3.nhk.or.jp/news/easy](https://www3.nhk.or.jp/news/easy/) — free, Japanese news summaries
- **RSS feeds:** Nikkei, Reuters Japan, Bloomberg Japan all publish Japanese RSS

```python
import feedparser

# Reuters Japan RSS
rss_url = "https://feeds.reuters.com/reuters/JPjpMarketNews"
feed    = feedparser.parse(rss_url)

for entry in feed.entries[:10]:
    print(entry.title, entry.published)
```

---

## 8. Calendar & Structural

**Features:** All derived from the date — no data download needed.

```python
import pandas as pd
import numpy as np
import jpholiday            # pip install jpholiday

def add_calendar_features(df):
    """Add Japan-specific calendar flags to a DataFrame with DatetimeIndex."""
    df = df.copy()
    idx = df.index

    # Fiscal year-end proximity (Japan: March 31)
    df["fiscal_year_end_flag"]  = (idx.month == 3).astype(int)
    df["pre_fiscal_year_end"]   = (idx.month == 2).astype(int)   # window dressing

    # Ex-dividend season (Japanese companies cluster June–September)
    df["ex_div_season"]         = idx.month.isin([6, 7, 8, 9]).astype(int)

    # Golden Week (late April – early May)
    df["golden_week"]           = ((idx.month == 4) & (idx.day >= 29) |
                                   (idx.month == 5) & (idx.day <= 6)).astype(int)

    # Obon holidays (mid-August)
    df["obon_holiday"]          = ((idx.month == 8) & idx.day.isin(range(13, 17))).astype(int)

    # Calendar year-end (domestic tax selling)
    df["calendar_year_end"]     = (idx.month == 12).astype(int)

    # Japan public holiday flag
    df["japan_holiday"]         = [jpholiday.is_holiday(d) for d in idx.date]

    # Quarter of fiscal year (Japan fiscal: Apr–Mar)
    fiscal_month = ((idx.month - 4) % 12) + 1
    df["fiscal_quarter"]        = np.ceil(fiscal_month / 3).astype(int)

    # Day-of-week effects
    df["day_of_week"]           = idx.dayofweek    # 0=Mon, 4=Fri
    df["is_monday"]             = (idx.dayofweek == 0).astype(int)
    df["is_friday"]             = (idx.dayofweek == 4).astype(int)

    return df
```

---

## 9. Universe & Weights

**Features:** Constituent membership and benchmark weights — the spine of the entire pipeline.

### Sources

#### 9A. JPY121 ETF (1321.T) — Primary Universe & Weight Source

> See Section 6A for full implementation. This is the **single most important data source** for the pipeline.

| What you get | How |
|-------------|-----|
| Current constituent list | Scrape Nomura NEXT FUNDS holdings page daily |
| Historical entry/exit dates | Nikkei Inc. constituent change announcements |
| Daily benchmark weights `w_bench_i_t` | Reconstruct from price-weighting formula using constituent close prices |
| In-index flag per stock per date | Build panel from constituent history |

#### 9B. Nikkei Inc. — Index Constituent Announcements
- **Access:** [indexes.nikkei.co.jp/en/nkave](https://indexes.nikkei.co.jp/en/nkave/index/profile?idx=nk225) — free
- **Coverage:** Annual constituent review results (typically announced September, effective October); ad-hoc changes for delistings/mergers

#### 9C. J-Quants — Listed Issues Master
- **Access:** Same free account; endpoint `/listed/info`
- **Coverage:** All TSE-listed stocks with listing/delisting dates; sector codes; market segment

```python
# All listed companies (living and delisted)
listed = client.get_listed_info()
# Columns: Code, CompanyName, MarketCode, Sector33Code, ScaleCategory,
#          ListingDate, DelistingDate (if applicable)

# Filter for Nikkei 225 members on a given date
# (Cross-reference with JPY121 ETF holdings list)
```

---

## Full Source Summary

| Source | URL / Install | Free? | Key Data | Auth |
|--------|--------------|-------|----------|------|
| **yfinance** | `pip install yfinance` | Yes | OHLCV all TSE stocks, FX, index | None |
| **J-Quants API** | `pip install jquantsapi` · jpx-jquants.com | Yes (free tier) | OHLCV, fundamentals (PIT), constituent info | Email registration |
| **Stooq** | `pip install pandas-datareader` | Yes | OHLCV historical | None |
| **Alpha Vantage** | alphavantage.co | Yes (500/day) | OHLCV, intraday, FX | Free API key |
| **EDINET (FSA)** | disclosure2.edinet-api.fsa.go.jp | Yes | Yūhō filings, large-shareholder reports | None |
| **SimFin** | `pip install simfin` · simfin.com | Yes (free tier) | Standardised fundamentals | Free API key |
| **FRED** | `pip install fredapi` · fred.stlouisfed.org | Yes | USD/JPY, VIX, US rates, Japan CPI | Free API key |
| **BOJ API** | stat.boj.or.jp | Yes | JGB yields, monetary base, call rate | None |
| **Cabinet Office Japan** | esri.cao.go.jp | Yes | GDP, CPI, consumer confidence | None |
| **OECD** | `pip install pandasdmx` | Yes | Leading indicators, industrial production | None |
| **MOF Japan** | customs.mof.go.jp | Yes | Trade balance | None |
| **JPX Short Interest** | jpx.co.jp/markets/statistics | Yes | Bi-weekly short positions by stock | None |
| **JPX Investor Flows** | jpx.co.jp/markets/statistics | Yes | Weekly investor-type trading data | None |
| **JPX Options Stats** | jpx.co.jp/derivatives/statistics | Yes | NKY 225 put-call OI | None |
| **Nikkei VI** | `yfinance` `^VNKY` | Yes | Japan volatility index | None |
| **Nomura ETF (1321.T)** | nextfunds.jp/lineup/1321 | Yes | Current constituent weights | None |
| **Nikkei Inc.** | indexes.nikkei.co.jp | Yes | Constituent change announcements | None |
| **pandas_ta** | `pip install pandas_ta` | Yes | 130+ technical indicators | None |
| **TA-Lib** | `pip install TA-Lib` | Yes | Technical indicators (C-backed) | None |
| **jpholiday** | `pip install jpholiday` | Yes | Japan public holiday calendar | None |
| **feedparser** | `pip install feedparser` | Yes | RSS news feeds | None |
| **fugashi / MeCab** | `pip install fugashi ipadic` | Yes | Japanese morphological analysis | None |

---

## Pip Install All

```bash
# Core data
pip install yfinance jquantsapi pandas-datareader fredapi alpha_vantage simfin

# Technical indicators
pip install pandas_ta TA-Lib

# ML / modelling
pip install lightgbm xgboost scikit-learn torch torch_geometric

# Portfolio optimisation
pip install cvxpy clarabel riskfolio-lib

# NLP
pip install feedparser fugashi ipadic transformers

# Utilities
pip install jpholiday pandasdmx requests beautifulsoup4 optuna shap
```
