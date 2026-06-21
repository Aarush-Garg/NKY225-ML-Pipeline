# NKY 225 — Feature & Indicator Research
## Deep Literature Review: Equity Research + ArXiv + Academic Sources

> Sources: Goldman Sachs AM, SuMi Trust AM, State Street SSGA, GMO, Verdad Capital, OQ Funds, Eastspring, Matthews Asia · ArXiv papers · Pacific-Basin Finance Journal · NBER · RIETI

---

## Priority Table — Features Ranked by Evidence Strength

| Rank | Feature | Category | Definition | Evidence | Japan IC / Note | Source |
|------|---------|----------|------------|----------|-----------------|--------|
| 1 | Short-Term Reversal (1M) | Momentum/Reversal | Prior 1-month return, negated | **Very Strong** — strongest anomaly in Japan | Dominant over price momentum | Chou 2023; Miwa 2019; Iwanaga 2024 |
| 2 | Book-to-Price (B/P) | Value | Book equity / market cap | **Strong** — most consistent value signal | EBITDA/EV t-stat = 6.69 | Kubota 2018; Verdad 2022; SuMi Trust |
| 3 | Gross Profit-to-Assets (GPA) | Quality | (Revenue − COGS) / Total Assets | **Strong** — preferred quality metric in Japan | ~0.16%/month anomaly returns | Ng & Shen 2020; Chou 2023 |
| 4 | Analyst EPS Revision (1M/3M) | Sentiment | % change in 12M FWD consensus EPS | **Strong** — recommended Japan momentum substitute | Ranked "most effective" by practitioners | SuMi Trust; SSGA 2024; GSAM 2025 |
| 5 | USD/JPY Beta (24M rolling) | Macro | Stock's rolling beta to monthly USD/JPY change | **Very Strong** — bifurcates exporters vs. domestics | 3–5% spread per 1% yen move | Matthews Asia 2024; Eastspring 2025 |
| 6 | Amihud Illiquidity (60D) | Liquidity | |return| / yen volume, daily average | **Strong** — higher premium than US | Dominant premium in Japan | Kazumori 2015; Iwanaga 2022 |
| 7 | BOJ ETF Ownership % | Japan-Specific | % market cap in BOJ-eligible ETF holdings | **Strong (historical)** — ceased March 2024 | IR = 1.2; 97% vs 58% benchmark (7yr) | Barbon 2019; Harada 2021 |
| 8 | P/B < 1 + TSE Reform Signal | Corporate Gov. | Binary: P/B<1 AND disclosed improvement plan | **Strong (2023+)** — TSE structural reform | 65% of PBR<1 firms announced buybacks | GMO 2023; Morikawa 2024 |
| 9 | Buyback / Repurchase Ratio | Corporate Action | Announced buyback amount / market cap (TTM) | **Moderate–Strong** | Significant post-announcement drift | Sakawa 2025; Uchiyama 2022 |
| 10 | Earnings Surprise (SUE) | Sentiment | (Actual EPS − consensus) / std(forecast errors) | **Moderate** — PEAD confirmed in Japan | Smaller magnitude than US | Jinushi 2023 |
| 11 | Price Momentum (12-1M) | Momentum | Cumulative return months t-12 to t-2 | **Weak–Moderate** — controversial in Japan | Weak/negative in many periods | Chou 2023; Cheema 2018; Hofmann 2024 |
| 12 | Low Volatility / Low Beta | Quality/Risk | 12M daily return std dev, negated | **Moderate** — confirmed in Japan | Low-vol anomaly survives | Kohvakka 2023; Maguire 2017 |
| 13 | ROE / ROE Improvement | Quality | Net income / book equity; YoY change | **Moderate** — rising vs. TSE 8% threshold | Structural theme 2023–2025 | GSAM 2025; GMO 2023 |
| 14 | Accruals | Quality | (Net income − Op. cash flow) / assets, negated | **Moderate** — confirmed in Japan | Accruals anomaly documented | Isoyama 2024; Lu 2017 |
| 15 | Dividend Yield | Value | Annual DPS / price | **Moderate** — growing with reform cycle | Combine with buyback for total shareholder yield | Liang 2019; Matthews Asia 2024 |
| 16 | Size (Market Cap) | Size | Log market capitalisation | **Weak raw** — strong after profitability control | Size × profitability combo is strong | Cheema 2021; Kubota 2018 |
| 17 | Intraday Momentum | Microstructure | First-half-hour return (same day signal) | **Moderate** — confirmed in Japan | Confirmed across Japan and China | Limkriangkrai 2023 |
| 18 | EBITDA/EV | Value | EBITDA / Enterprise Value | **Strong** — preferred over P/E for Japan | t-stat = 6.69; net cash normalised | Verdad 2022 |
| 19 | JGB 2Y Yield Change (1M) | Macro | Monthly change in 2yr JGB yield | **Moderate** | BOJ policy sensitivity; banking sector strongest | Kubota 2025; Zhong 2024 |
| 20 | News Sentiment (NLP) | Alt Data | 30D rolling NLP score on Nikkei text / Yūhō | **Moderate** — language edge | Japanese-text NLP has information premium | Souma 2019; Nakano 2023 |
| 21 | Short Interest Ratio | Sentiment | Short shares / float | **Moderate** | Aggregate spikes predictive of market direction | Gorbenko 2023 |
| 22 | Cross-Shareholding Reduction | Japan-Specific | Change in cross-held shares / float | **Moderate** | Structural reform alpha 2023+ | Miyajima 2025; RIETI |
| 23 | Supply-Chain GNN Feature | Alternative | Graph neural network on supplier–customer links | **Moderate** — novel | 2.2× Sharpe vs. benchmark in backtest | Matsunaga 2019 (arXiv:1909.10660) |
| 24 | Foreign Investor Net Flow | Sentiment | Weekly net foreign equity purchases (TSE data) | **Moderate** | Foreigners = marginal price setters (54% volume) | GSAM 2026; Yamamoto 2022 |
| 25 | 52-Week High Proximity | Momentum | Price / 52-week high | **Mixed** — regime-dependent | Works in bull markets only | Iwanaga 2024; Avramov 2021 |

---

## Section 1 — Momentum Signals

### 1.1 Price Momentum (12-1 Month)
**Definition:** Cumulative total return from month t-12 to t-2.

**Japan Evidence:**
- **Weak and contested.** Chou et al. (2023, *Pacific-Basin Finance Journal*): momentum "cannot explain the cross-section of stock returns" in Japan.
- Cheema et al. (2018): large during market continuations, turns negative during reversals.
- Hofmann et al. (2024): "Momentum is not followed by reversal in Japan" — distinguishes Japan from all other developed markets.
- Roy & Shijin (2019): Japan has lowest momentum factor among markets studied; MOM is "insignificant at < 0.1%" in Japan.
- MSCI Japan Momentum Index Sharpe: 0.55 (5yr), 0.56 (10yr) — below market.

**Why it fails in Japan:**
- BOJ Nikkei 225 ETF purchases artificially supported high-priced (often growth/momentum) stocks from 2013–2024.
- Domestic investors are contrarian; foreign investors are momentum-driven — they cancel each other (GSAM 2026).
- Post-TSE reform: State Street notes a "sturdy resurgence in Momentum since March 2023" — monitor.

**Recommendation:** De-emphasise raw price momentum. Use earnings revision momentum instead.

---

### 1.2 Short-Term Reversal (1-Week / 1-Month)
**Definition:** Prior 1-month (or 1-week) raw return, negated — long prior losers, short prior winners.

**Japan Evidence:**
- **The strongest momentum-type anomaly in Japan.** Confirmed across multiple independent studies.
- Miwa (2019, *Quarterly Journal of Finance*): short-term contrarian strategy profitable on TSE first section.
- Chou et al. (2023): "Short-term reversal is prevalent" while price momentum is absent.
- Iwanaga et al. (2024): Decomposed momentum confirms 1-month reversal as dominant component.
- Iwanaga & Hirose (2022): Illiquidity amplifies reversal effects — small, illiquid stocks show strongest reversal.

**Recommendation:** Include 1-month and 1-week reversal as a **Tier 1 core feature**.

---

### 1.3 Analyst EPS Revision Momentum
**Definition:** Percentage change in 12M forward consensus EPS estimate over the past 1 month and 3 months.

**Japan Evidence:**
- **The recommended Japan-specific substitute for price momentum.**
- SuMi Trust AM: explicitly uses earnings momentum (not price momentum) to capture the momentum premium in Japan.
- SSGA (2024): Sentiment (analyst revision) and Value are two of four core themes; "resurgence of Sentiment and Momentum measures" since 2023 reform.
- Jinushi (2023): Post-earnings announcement drift (PEAD) confirmed in Japan — prices drift in direction of earnings surprise.
- Suzuki et al. (2022): Analyst forecast reports predict Japanese stock prices; serial correlation in Japanese analyst revisions is exploitable (Jung 2019).

**Japan-Specific Nuance:** Only 54% of Japanese companies report in English (GSAM 2024). This means analyst revision signals carry more information content in Japan than in US/EU markets — especially in small/mid-cap.

**Recommendation:** Use both 1M and 3M analyst revision as **Tier 1** signals.

---

### 1.4 Intraday Momentum
**Definition:** First half-hour return predicts last half-hour return of the same trading day.

**Japan Evidence:**
- Limkriangkrai et al. (2023): Intraday momentum confirmed in Japan and China. "The first half-hour return predicts the last half-hour return."
- Primarily useful for daily signal generation in higher-frequency variants.

---

### 1.5 52-Week High Proximity (High-to-Price)
**Definition:** Iwanaga et al. (2024) decompose momentum into: price-to-high (current price / prior peak) and high-to-price (52-week high / current price — distance from peak).

**Japan Evidence:**
- "High-to-price has a lower downside risk and higher Sharpe ratio than price-to-high in Japan" (Iwanaga 2024).
- During bull markets, works like conventional momentum; during bear markets, pure price momentum has poor downside.
- **Recommended variant:** Use high-to-price (=52-week high / current price) as the risk-adjusted momentum proxy in Japan.

---

## Section 2 — Value / Fundamental Signals

### 2.1 Book-to-Price (B/P)
**Definition:** Book value of equity per share / price.

**Japan Evidence:**
- **Most consistently predictive value factor in Japan across all studies.**
- SuMi Trust AM: "Value factors have been effective in both Japanese and global markets since 2020."
- Chou et al. (2023): "Value characteristics can explain the cross-section of stock returns" in Japan.
- Kubota & Takehara (2018): Value (B/M) remains a key predictor despite Fama-French 5F model failing overall in Japan.
- Over 50% of TOPIX stocks had P/B < 1x as of 2023 — value signal has outsized opportunity set.
- TSE reform (March 2023): P/B < 1x companies must disclose improvement plans. Value + governance reform = compound alpha source.

**Japan-Specific Nuances:**
- **Net cash distortion:** Over 40% of liquid Japanese companies hold net cash. Standard B/P understates relative attractiveness. Always compute EV/EBITDA as a companion metric.
- **Real estate undervaluation:** Japanese companies carry real estate at historical cost; true book value understated by estimated ¥22 trillion in aggregate.
- **Cross-shareholding distortion:** Keiretsu cross-held shares trade below book. Adjust or use EV-normalised metrics.

---

### 2.2 EBITDA / Enterprise Value
**Definition:** EBITDA / (Market Cap + Total Debt − Cash).

**Japan Evidence:**
- **Recommended primary value metric for Japan** given pervasive cash holdings.
- Verdad Capital (2022): EBITDA/EV coefficient = 0.93, t-stat = **6.69** — the highest statistical significance of any value metric tested in Japan.
- Debt/EV also significantly positive: coefficient = 0.38, t-stat = 4.40 — firms willing to use leverage earn higher returns.

**Recommendation:** Use EBITDA/EV as the primary value signal; B/P as secondary. Both normalised cross-sectionally per date.

---

### 2.3 Dividend Yield + Buyback Yield (Total Shareholder Yield)
**Definition:** (Annual DPS + Buyback per share) / price. Combine dividend yield and buyback yield into total shareholder return yield.

**Japan Evidence:**
- Liang (2019): Dividend yield and earnings yield identified as Japan return drivers.
- Matthews Asia (2024): "Healthy dividend yields supplemented by accelerating share buybacks adding 2–3% to returns."
- Total shareholder yield accelerating: buybacks hit a record ¥22.4 trillion in FY2024 (GSAM 2025).
- Combine into single "total shareholder yield" feature for maximum signal.

---

### 2.4 Forward P/E (Earnings Yield)
**Definition:** Consensus FWD EPS / price (inverse of forward P/E).

**Japan Evidence:**
- Musallam (2018): Earnings per share, earnings yield, and dividend yield predict market performance in Japan.
- GMO (2023): Japan trades at ~15x forward earnings (historical average) — reasonable entry for value.
- GSAM (2025): Forward earnings yield interacts with analyst revision to create a "quality-at-reasonable-price" composite.

---

## Section 3 — Quality Signals

### 3.1 Gross Profit-to-Assets (GPA)
**Definition:** (Revenue − COGS) / Total Assets. Novy-Marx (2013) profitability measure.

**Japan Evidence:**
- **Preferred quality metric in Japan (SuMi Trust AM)**. Specifically chosen because it "resists manipulation through leverage adjustments and better reflects actual business stability" versus ROE.
- Ng & Shen (2020): Operating cash flow and ROA anomaly returns ~0.16%/month in Japan.
- Lu et al. (2017, NBER): Gross profitability anomalies confirmed across international markets including Japan.
- Chou et al. (2023): "Value and operating profitability anomalies are prevalent in Japan."

---

### 3.2 Return on Equity (ROE) and ROE Improvement
**Definition:** Net income / Book equity. YoY change in ROE as a separate signal.

**Japan Evidence:**
- Moderate predictor; significantly increased importance post-2023.
- Yanagi & Kawakami (2018): "Equity Spread" (ROE minus cost of capital) predicts stock returns.
- GSAM (2025): Rising ROE is a structural alpha theme; stocks crossing 8% ROE threshold (= cost of equity) outperform.
- Matthews Asia (2024): Quality stocks with ROE above market average outperform.
- TSE guidance asks companies to target ROE > 8% — binary signal: crossing threshold = outperformance.

---

### 3.3 Accruals
**Definition:** (Net income − Operating cash flow) / Total assets. Low accruals (= high cash earnings quality) → higher returns.

**Japan Evidence:**
- Isoyama (2024): "Adverse selection risk induced by the information structure" explains accrual anomalies in Japan.
- Kubota & Takehara (2019): Accruals-based trading strategy profitable when investors cannot measure earnings persistency.
- Lu et al. (2017): Accruals anomaly confirmed in Japan.

---

### 3.4 Low Volatility / Low Beta
**Definition:** Trailing 12-month daily return standard deviation or CAPM beta. Long low-vol.

**Japan Evidence:**
- Kohvakka (2023): "Stock portfolios with lower variance in monthly returns demonstrate superior performance" in Japan.
- Maguire et al. (2017): "Lowest volatility decile had both lower risk and higher return" in Japan.
- MAX effect does not work in Japan — Cheon & Lee (2018): MAX effect only −0.49% in Japan. Low-vol premium does survive.
- Downside beta adds explanatory power beyond standard beta in Japan (Atilgan 2019).

---

### 3.5 Net Cash / EV (Financial Strength)
**Definition:** (Cash − Total Debt) / Enterprise Value. Positive = net cash position.

**Japan Evidence:**
- **Japan-specific: uniquely high net cash holdings create a distinct signal.**
- OQ Funds (2024): "Over 40% of liquid companies hold net cash positions."
- Verdad (2022): Higher Debt/EV (more levered firms) → higher returns in Japan — firms willing to deploy capital are rewarded.
- Use as feature to distinguish cash-rich compounders from underdeployed value traps.

---

## Section 4 — Macro / Regime Signals

### 4.1 USD/JPY Sensitivity (Cross-Sectional Beta)
**Definition:** Each stock's rolling 24-month beta to monthly USD/JPY return. Also: 1M and 3M change in USD/JPY level.

**Japan Evidence:**
- **#1 macro cross-sectional differentiator in Japan.** Creates a 3–5% return spread per 1% yen move.
- Exporters (electronics, automotive, precision instruments): strong positive beta to JPY weakness.
- Domestic stocks (banking, retail, construction, consumer staples): zero or negative beta.
- Matthews Asia (2024): USD/JPY currency sensitivity "has almost halved over the past two decades" as manufacturers moved offshore — but differential remains highly significant.
- CEPR and Eastspring (2025): USD/JPY is a central macro driver of the NKY 225.
- Kurihara (2006); Salimi (2020): USD/JPY movements are the primary driver of Japan equity cross-section.

**Implementation:**
```python
# Estimate per-stock beta to USD/JPY monthly change
usdjpy_beta = rolling_OLS(stock_return_monthly, usdjpy_pct_change, window=24)
# Also use level signal: 1M and 3M change in USD/JPY
usdjpy_mom_1m = usdjpy_spot_t / usdjpy_spot_t-1 - 1
```

**Sector mapping (beta sign):**
| Positive USD/JPY beta | Negative / Zero USD/JPY beta |
|---|---|
| Toyota, Honda, Subaru | AEON, Seven & i |
| Sony, Panasonic | Mitsubishi UFJ Bank |
| Fanuc, Keyence | NTT, KDDI |
| Shin-Etsu Chemical | Tokyo Gas, Kansai Electric |

---

### 4.2 JGB Yield Curve
**Definition:** Level and change of 1Y, 2Y, 5Y, 10Y JGB yields. Yield curve slope (10Y − 2Y).

**Japan Evidence:**
- Kubota & Shintani (2025): Interest rate surprises via JGB futures predict cross-sectional stock returns.
- Zhong (2024): BOJ yield targeting policy interacts with Nikkei 225 returns.
- Wang et al. (2020): Yield spreads in low interest rate environment interact with Nikkei 225.
- BOJ's YCC abandonment (2024): banking sector surged ~40% on the rate cycle turn. 2Y JGB yield change is the most sensitive to BOJ policy.

**Implementation:** Use 1M change in 2Y JGB yield as primary signal; interact with bank sector membership.

---

### 4.3 US Market (S&P 500 Overnight Return)
**Definition:** US S&P 500 return from prior US close to Japan open (overnight).

**Japan Evidence:**
- Bathia et al. (2016): "US variables are helpful in forecasting Japanese stock returns."
- Imai & Kim (2024): Foreign investor rebalancing between US and Japan drives daily correlation.
- Ishijima et al. (2015): Market sentiment index based on US/global markets predicts Nikkei 225 direction.

---

### 4.4 Reflation Regime Signal (CPI / Wage Growth)
**Definition:** Binary flag: Japan CPI YoY > 1% AND average wage growth > 2%. Monthly Japan CPI (ex-fresh food) change.

**Japan Evidence:**
- Eastspring (2025): "Record average pay hike, highest in over three decades" drives domestic consumption stocks.
- GMO (2023): "Transition from deflation to inflation explains 77% of total return performance of MSCI Japan in 2023."
- Matthews Asia (2024): "Emerging positive real wage growth after 30 years of deflation" is a structural theme.
- Domestic sectors (retail, restaurants, services) benefit most from reflation; exporters are mixed.

---

### 4.5 VIX / Global Risk-Off Regime
**Definition:** CBOE VIX level and 1M change; binary risk-off flag (VIX > 25).

**Japan Evidence:**
- Nikkei VI (Japan's VIX equivalent, from JPX) is the Japan-specific equivalent.
- BOJ historically purchased ETFs when market weakness was detected — VIX interacted directly with BOJ intervention probability (now historical signal).
- High VIX environments tend to benefit defensive / domestic stocks in Japan over exporters.

---

## Section 5 — Sentiment Signals

### 5.1 News Sentiment (NLP on Japanese Text)
**Definition:** 30-day rolling NLP sentiment score derived from Nikkei newspaper articles or corporate Yūhō filings.

**Japan Evidence:**
- Souma et al. (2019, 205 citations): "Predictive power of historical news sentiments" via machine learning confirmed for Japanese equities.
- Seki & Ikuta (2020, arXiv:2003.02973): RNN-GRU on Nikkei newspaper articles (2013–2018) predicts Nikkei 225 direction; morphological analysis (MeCab tokeniser) required for Japanese text.
- Nakano & Yamaoka (2023): LLM-enhanced Japanese sentiment improves large-cap return prediction.
- Okada et al. (2025): Advanced NLP on Japanese Yūhō (10-K equivalent) predicts cross-sectional returns.
- Ikeda et al. (2025): Twitter-sourced Japanese text correlated with Nikkei 225 closing prices.

**Japan Competitive Advantage:** Only 54% of Japanese companies report in English. Japanese-language NLP has significant information edge vs. English-only models. Use MeCab morphological analyser as tokeniser.

---

### 5.2 Short Interest Ratio
**Definition:** Shares sold short / total float. Monthly update from TSE short-sales data.

**Japan Evidence:**
- Gorbenko (2023, *Review of Asset Pricing Studies*): "Aggregate short interest in Japan exhibits periodical spikes" that predict cross-section and market returns.
- Khan et al. (2019): Short-sale constraints affect return behaviour on TSE and JASDAQ.
- TSE publishes bi-weekly short-selling statistics (freely available from JPX).

---

### 5.3 Foreign Investor Net Flow
**Definition:** Net weekly equity purchases by foreign investors (available from JPX investor-type trading data).

**Japan Evidence:**
- GSAM (2026): "Foreign institutional investors account for over 54% of trading volume and act as the marginal price setters."
- Yamamoto (2022): 11 investor types on TSE; foreign institutional price impact is highest.
- Imai & Kim (2024): Foreign rebalancing between US and Japan drives cross-asset correlations.
- **Data source:** JPX publishes weekly "Investor Behaviour" trading statistics by investor type (free).

---

### 5.4 Put-Call Ratio (Nikkei 225 Options)
**Definition:** Put-Call ratio on Nikkei 225 options (Osaka Exchange).

**Japan Evidence:**
- Bathia & Bredin (2016): Put-Call ratio is a component of composite sentiment indices for Japan.
- Elevated put-call ratio historically preceded BOJ ETF purchase days (now historical).
- Available from Osaka Exchange (JPX) options statistics.

---

## Section 6 — Liquidity / Microstructure

### 6.1 Amihud Illiquidity (ILLIQ)
**Definition:** Daily average of |daily return| / (daily yen trading volume), averaged over 60 days.

**Japan Evidence:**
- **Higher liquidity premium in Japan than in US** — confirmed across multiple studies.
- Kazumori (2015): "Higher premium for liquidity risks in the Japanese market" vs. US equity markets.
- Iwanaga & Hirose (2022): "Illiquidity strongly contributes to price underreaction effects" in Japan — links liquidity to reversal signals.
- Iwanaga (2026, *Asia-Pacific Financial Markets*): Short-term and long-term liquidity changes both priced in Japan.
- Zhong & Takehara (2019): Multiple liquidity measures (Amihud, bid-ask spread) priced in cross-section.
- Yang & Tamaki (SSRN 4503466): Duration premium via Amihud-sorted portfolios in Japan.

**Implementation:**
```python
ILLIQ_i_t = mean(|r_i,d| / VolumeYen_i,d for d in past 60 days)
```

---

### 6.2 Relative Trading Volume
**Definition:** Current 5-day average volume / 60-day average volume.

**Japan Evidence:**
- Used as a feature in Abe & Nakayama (2018): 100+ feature deep learning model for Japan.
- Ausloos & Ivanova (2002): Volume as multiplicative factor in momentum-style indicators.
- Volume spikes associated with BOJ ETF purchase days (historically predictive; now historical).

---

### 6.3 Bid-Ask Spread
**Definition:** (Ask − Bid) / midpoint, or (High − Low) / Close as a proxy.

**Japan Evidence:**
- OQ Funds (2024): Japan has "exceptionally low bid-ask spreads (~1bp brokerage cost)" and very high liquidity. Short-term reversal strategies especially benefit from Japan's low transaction costs.
- Low transaction costs make higher-frequency signals (weekly reversal) more viable in Japan than in emerging markets.

---

## Section 7 — Japan-Specific Signals

### 7.1 BOJ ETF Ownership (AuM Eligible%)
**Definition:** Percentage of each stock's market capitalisation owned via BOJ ETF holdings.

**Japan Evidence:**
- **Was the highest-IC proprietary signal for Nikkei 225 constituents from 2013–2024.**
- Barbon & Gianinazzi (2019, *Review of Asset Pricing Studies*): BOJ ETF purchases create pricing distortions; elasticity = 1 → 22bps market return per ¥1trn purchased.
- Harada & Okimoto (2021): "Nikkei 225 stocks' afternoon returns significantly higher than non-Nikkei 225 stocks when BOJ purchases ETFs."
- NBER Charoenwong et al. (2019): Cumulative treatment effect on Nikkei 225 ≈ 20% by October 2017.
- Informaconnect (2024): "AuM Eligible%" → 97% total return vs. 58% benchmark over 7 years; Information Ratio = 1.2.
- **Status:** BOJ announced end of new ETF purchases March 2024. This signal no longer generates alpha but legacy ownership concentration still explains valuation anomalies in high-weight Nikkei 225 names.

---

### 7.2 TSE PBR Improvement Signal (Post-2023)
**Definition:** Composite: (P/B < 1.0) AND (company disclosed TSE PBR improvement plan) AND (buyback / dividend announcement in 12M). Continuous version: distance of P/B below 1.0 × disclosure score.

**Japan Evidence:**
- GMO (2023): "65% of companies with PBR < 1x announced buybacks or dividend increases" following TSE March 2023 initiative.
- Morikawa et al. (2024): Measurable abnormal returns around TSE PBR disclosure dates.
- Small/mid-cap especially impacted: "280 mid-cap companies trade below 0.7x PBR vs. only 18 large-cap companies" (GMO).
- **Horizon:** 6–18 month alpha signal post-disclosure.

---

### 7.3 Cross-Shareholding Reduction
**Definition:** Change in % of shares held by reported corporate cross-shareholders. Event signal when large institutional cross-shareholder files a sales notification.

**Japan Evidence:**
- RIETI: Cross-shareholdings halved from 60%+ (1990) to ~25% (end 2023); unwinding accelerating.
- Miyajima & Arikawa (2025): "Reforms accelerated the unwinding of cross-shareholdings, with stock repurchases serving signaling functions."
- Ishizuka (2026): Unwinding cross-shareholdings signals "receptivity to shareholder frustrations" — precedes capital return announcements.
- Franks et al. (2018): Cross-shareholding unwinding creates conditions for activist investor responses.
- **Data source:** Large shareholder filings (大量保有報告書, Tōshisha) — filed when crossing 5% threshold; available via EDINET.

---

### 7.4 Nikkei 225 Price-Weighting Bias
**Definition:** Rank of stock price within Nikkei 225 constituents (price-weighted index, not market-cap weighted). High-priced stocks have disproportionate index weight.

**Japan Evidence:**
- The Nikkei 225 is price-weighted like the DJIA. A ¥100,000 stock has 100× the index impact of a ¥1,000 stock regardless of market cap.
- BOJ ETF purchases amplified this — Fast Retailing, SoftBank, Fanuc received disproportionate BOJ flows.
- Harada & Okimoto (2021): Documents the mechanism explicitly.
- **Current relevance:** BOJ no longer purchasing, but residual overvaluation of high-priced stocks vs. index contribution remains.

---

### 7.5 Calendar / Fiscal Year-End Effects
**Definition:** Binary flags for calendar-specific Japan effects.

| Flag | Dates | Effect |
|------|-------|--------|
| `fiscal_year_end` | March 31 | Window dressing; tax-loss selling; ex-dividend clustering |
| `ex_dividend_season` | Mid-June → September | Largest dividend ex-dates; Kato & Loewenstein effect |
| `golden_week` | Late April – early May | Low volume precursor; gap risk |
| `bon_holidays` | Mid-August | Reduced liquidity; gap risk |
| `calendar_year_end` | Late December | Domestic retail tax selling |

**Japan Evidence:**
- Kato & Loewenstein (1995): "Prices rise on ex-day" in Japan; dividend tax effects.
- Dhatt et al. (1994): Largest distributions yield highest excess returns around ex-date.
- Coakley et al. (2007): Confirmed clustering of ex-rights and ex-dividend dates June–September.
- Chia et al. (2016); Woo et al. (2020): Calendar anomalies confirmed in Japan across multiple studies.

---

### 7.6 Supply-Chain / Keiretsu Graph Signal
**Definition:** GNN-derived features from supplier–customer network edges among TSE-listed companies.

**Japan Evidence:**
- Matsunaga et al. (2019, **arXiv:1909.10660**): GNN on company knowledge graphs including supplier relations and Nikkei 225 companies; 20-year backtest.
- **Result: 29.5% increase in return ratio; 2.2× Sharpe ratio vs. market benchmark.**
- Japan's dense keiretsu supply-chain relationships make this signal more predictive here than in other markets.
- **Data source:** Ministry of Economy trade databases; commercial data from Teikoku Databank or TDB (Japanese credit bureau).

---

## Section 8 — ML Research on Japan (ArXiv Papers)

### 8.1 Deep Factor Model — Nakagawa et al. (2018, 2019)
- **Papers:** arXiv:1810.01278 (Deep Factor Model); arXiv:1901.11493 (Deep Recurrent Factor Model)
- **Method:** LSTM + Layer-wise Relevance Propagation (LRP) for interpretable non-linear multi-factor model on Japanese equities.
- **Key finding:** Deep neural networks outperform shallow models. LRP reveals: **value factor effective; momentum factor less effective** in Japan.
- **Performance:** Outperforms CAPM, FF3, FF5 in out-of-sample R².

### 8.2 100+ Feature Cross-Sectional DNN — Abe & Nakayama (2018)
- **Paper:** arXiv:1801.01777; PAKDD 2018.
- **Method:** Deep learning for 1-month-ahead stock return prediction in Japan; 100+ predictive factors.
- **Key finding:** Deep neural networks outperform shallow neural networks in cross-section prediction for Japan.
- **Features used:** Fundamental, technical, and composite factors (full list in paper).

### 8.3 Graph Neural Network — Matsunaga et al. (2019)
- **Paper:** arXiv:1909.10660
- **Method:** GNN on company knowledge graphs (supplier–customer, Nikkei 225); rolling window backtest; 20 years.
- **Key finding:** 29.5% improvement in return ratio; 2.2× Sharpe ratio vs. benchmark. Japan's dense keiretsu relationships make GNN particularly effective here.

### 8.4 News Sentiment — Seki & Ikuta (2020)
- **Paper:** arXiv:2003.02973
- **Method:** RNN-GRU on Nikkei Newspaper articles 2013–2018; one-class SVM filtering; domain adaptation.
- **Key finding:** Temporal factor analysis confirms news sentiment predicts Nikkei 225 direction.

### 8.5 Quantum Neural Network Cross-Section — Kobayashi et al. (2023)
- **Method:** Quantum neural network applied to ~10 features for Japanese equity cross-section prediction.
- **Notes:** Benchmarks recent literature noting "over a hundred" proposed factor sets for Japan.

---

## Section 9 — Factor Model Findings for Japan

### What Standard Factor Models Say

| Model | Japan Performance | Key Failure |
|-------|------------------|-------------|
| CAPM | Beta weak cross-section predictor on TSE (Ichiue 2026) | Beta not priced |
| Fama-French 3F | Value (B/M) works; market beta weak | Size mixed |
| Carhart 4F (+MOM) | MOM insignificant; model fails (Roy & Shijin 2019) | Momentum not priced |
| Fama-French 5F | "Not best benchmark for Japan" (Kubota 2018); RMW and CMA weak as factors | Profitability/investment factors weak |
| q-Factor (Hou et al.) | Fails to generate significant premia in Japan (Chou 2023) | Investment not priced |
| IEC + PCA (Morimoto 2026) | Industry equi-correlation structures substantially improve explanatory power | Best current framework |

**Critical Insight (Chou et al. 2023):** In Japan, the **characteristics approach outperforms the factor approach**. Use firm characteristics (B/P, GPA) directly in cross-sectional prediction models rather than constructing factor-mimicking portfolios. This validates the LightGBM characteristics-based approach over a traditional factor model.

---

## Recommended Feature Set for ML Pipeline

### Tier 1 — Core Alpha Signals (Build First)

| Feature | Definition | Data Source |
|---------|-----------|-------------|
| `short_term_reversal_1m` | Prior 1-month return, negated | yfinance / J-Quants price |
| `book_to_price` | Book equity / market cap | J-Quants fundamentals |
| `ebitda_ev` | EBITDA / Enterprise Value | J-Quants / EDINET |
| `gross_profit_to_assets` | (Revenue − COGS) / Total Assets | J-Quants fundamentals |
| `analyst_revision_1m` | % change in 12M FWD consensus EPS (1M) | Bloomberg / Refinitiv / I/B/E/S |
| `analyst_revision_3m` | % change in 12M FWD consensus EPS (3M) | Bloomberg / Refinitiv |
| `usdjpy_beta_24m` | 24M rolling OLS beta to monthly USD/JPY change | yfinance `JPY=X` |
| `amihud_illiq_60d` | 60D mean(|ret| / yen volume) | J-Quants / yfinance |

### Tier 2 — Strong Supporting Signals

| Feature | Definition | Data Source |
|---------|-----------|-------------|
| `total_shareholder_yield` | (DPS + Buyback/share) / price | J-Quants / TSE filings |
| `pbr_below_1_flag` | Binary: P/B < 1.0 | J-Quants fundamentals |
| `roe_improvement` | YoY change in ROE | J-Quants fundamentals |
| `accruals` | (Net income − Op. CF) / assets, negated | J-Quants fundamentals |
| `low_volatility_12m` | 12M daily return std dev, negated | Calculated from prices |
| `high_to_price` | 52-week high / current price | J-Quants / yfinance |
| `jgb_2yr_change_1m` | 1M change in 2Y JGB yield | BOJ API / FRED |
| `short_interest_ratio` | Short shares / float | TSE JPX bi-weekly data |
| `cpi_yoy` | Japan CPI YoY (reflation regime) | Cabinet Office / FRED |

### Tier 3 — Differentiated / Alternative Signals

| Feature | Definition | Data Source |
|---------|-----------|-------------|
| `earnings_surprise_sue` | (Actual EPS − consensus) / std(errors) | J-Quants / Bloomberg |
| `cross_shareholding_delta` | Change in cross-held shares / float | EDINET large shareholder filings |
| `news_sentiment_30d` | 30D rolling NLP score on Japanese text | Nikkei / in-house NLP |
| `foreign_flow_weekly` | Net foreign investor equity purchases | JPX investor-type data |
| `intraday_momentum_am` | AM session return (09:00–11:30 JST) | Intraday price data |
| `supply_chain_gnn` | GNN-derived supplier-network momentum | Teikoku Databank |
| `net_cash_ratio` | (Cash − Debt) / Enterprise Value | J-Quants fundamentals |
| `vix_1m_change` | 1M change in CBOE VIX | FRED `VIXCLS` |

### Tier 4 — Calendar / Regime Flags

| Feature | Definition | Data Source |
|---------|-----------|-------------|
| `fiscal_year_end_flag` | Binary: March | Calendar |
| `ex_dividend_season` | Binary: June–September | Calendar |
| `golden_week_flag` | Binary: late April – early May | Calendar |
| `reflation_regime_flag` | Binary: CPI YoY > 1% AND wages > 2% | Cabinet Office / MOL Japan |
| `boj_etf_eligible_pct` | % market cap in BOJ ETF holdings (historical) | BOJ disclosure / RIETI |

---

## Key Japan-Specific Nuances — Master List

1. **Momentum Reversal:** Price momentum weak/absent; short-term reversal is the dominant pattern. Use earnings revision momentum as the momentum proxy.

2. **Value Premium Timing:** Suppressed by BOJ ETF purchases (2013–2024) and Abenomics FX effects. Strongly recovered since 2020. Currently at peak opportunity with P/B < 1x widespread.

3. **USD/JPY is the #1 Macro Factor:** Creates 3–5% cross-sectional spread per 1% yen move. Estimate per-stock beta; separate exporters from domestics.

4. **BOJ ETF Distortions (Historical):** ¥37 trillion of ETF purchases (2010–2024) inflated Nikkei 225 high-priced stocks. Artificial price support now gone. Residual overvaluation in price-weighted Nikkei 225 names may be slowly correcting.

5. **Governance Reform Alpha (2023+):** TSE P/B < 1x initiative + cross-shareholding unwind + ROE > 8% threshold = structural alpha for 5+ years. Most important new signal source since Abenomics.

6. **Net Cash Distortion:** Over 40% of Japanese companies hold net cash. Always use EV-based valuation metrics (EBITDA/EV, EV/EBIT). Standard P/E overstates expensiveness.

7. **Size + Profitability Combo:** Raw size effect is weak in Japan. Small-cap stocks with improving profitability is a strong combination (Cheema 2021).

8. **Language Barrier Premium:** Only 54% of Japanese companies report in English. Japanese-language NLP on Nikkei newspaper / Yūhō filings has material information edge in small/mid-cap.

9. **Keiretsu Supply Chains:** Dense supplier–customer relationships make GNN-based supply chain signals more predictive than in other markets (arXiv:1909.10660).

10. **Calendar Effects:** Japan's March fiscal year-end, June–September ex-dividend clustering, and Golden Week create exploitable seasonal patterns not present in other major markets.

11. **Shareholder Perks (Yūtai):** Japan's unique perks program makes perk-offering companies appear expensive on standard metrics. Maintain adjustment or exclusion list.

12. **Characteristics > Factors:** In Japan, using firm characteristics directly (as ML features) outperforms factor-mimicking portfolio approaches. Validates LightGBM characteristics-based approach.

---

## Full Source Citations

### Institutional / Practitioner
1. OQ Funds Management (2024). *Quant Factor Investing in Japan: Opportunities and Challenges.*
2. SuMi Trust Asset Management (2024). *Factor Investment in the Japanese Equity Market.*
3. State Street SSGA (2024). *A Turn to Japan* — Systematic Active Monthly.
4. Goldman Sachs Asset Management (2025/2026). *The Japanese Paradox: A Systematic Path to Alpha.*
5. GMO (2023). *The Four 4s Behind the Compelling Opportunity in Japan Equities.*
6. Verdad Capital (2022). *Explaining Japanese Leveraged Value Equity Returns.*
7. Eastspring Investments (2025). *Japan – A Structural Alpha Opportunity.*
8. Matthews Asia (2024). *Investing in Japan Equities: Resilience Amid Market Volatility.*
9. Informaconnect (2024). *A Rising Tide Lifts Some Japanese Boats: BoJ ETF Purchases and Market Signals.*
10. MSCI (2023/2026). *MSCI Japan Equity Factor Model Factsheet.*
11. J.P. Morgan AM (Q2 2026). *Factor Views.*

### Academic / ArXiv
12. Chou, P-H. et al. (2023). "Comparing competing factor and characteristics models: Evidence in Japan." *Pacific-Basin Finance Journal.*
13. Cheema, M.A. et al. (2021). "Resurrecting the size effect in Japan." *Pacific-Basin Finance Journal*, 69.
14. Kubota, K. & Takehara, H. (2018). "Does the Fama and French Five-Factor Model Work Well in Japan?" *International Review of Finance.*
15. Nakagawa, K. et al. (2018). "Deep Factor Model." arXiv:1810.01278.
16. Nakagawa, K. et al. (2019). "Deep Recurrent Factor Model." arXiv:1901.11493.
17. Abe, M. & Nakayama, H. (2018). "Deep Learning for Forecasting Stock Returns in the Cross-Section." arXiv:1801.01777.
18. Matsunaga, D. et al. (2019). "Exploring Graph Neural Networks for Stock Market Predictions." **arXiv:1909.10660.**
19. Seki, K. & Ikuta, Y. (2020). "S-APIR: News-based Business Sentiment Index." arXiv:2003.02973.
20. Barbon, A. & Gianinazzi, V. (2019). "Quantitative Easing and Equity Prices: Evidence from the ETF Program of the Bank of Japan." *Review of Asset Pricing Studies*, 9(2), 210–255.
21. Harada, K. & Okimoto, T. (2021). "The BOJ's ETF purchases and effects on Nikkei 225 stocks." *Finance Research: Risk Analysis.*
22. Charoenwong, B. et al. (2019). "Asset Prices, Corporate Actions, and Bank of Japan Equity Purchases." NBER WP 25525.
23. Iwanaga, Y. & Hirose, T. (2022). "Liquidity shock and stock returns in the Japanese equity market." *Pacific-Basin Finance Journal*, 75.
24. Iwanaga, Y. et al. (2024). "Decomposing the Momentum in the Japanese Stock Market." *Asia-Pacific Financial Markets*, 31(2).
25. Kazumori, E. (2015). "Asset Pricing with Liquidity Risk: US vs. Japanese Equity Markets."
26. Jinushi, J. (2023). "Post-Earnings Announcement Drift and Ownership Structure in the Modern Japanese Stock Market." *Japanese Accounting Review.*
27. Isoyama, H. (2024). "Accruals Anomalies and Adverse Selection Risk in Japan." *Cogent Economics & Finance.*
28. Lu, X. et al. (2017). "Anomalies Abroad: Beyond Data Mining." NBER Working Paper.
29. Ng, C.C.A. & Shen, J. (2020). "Quality Investing in Asian Stock Markets." *Accounting & Finance.*
30. Miwa, K. (2019). "Short-term return reversals and intraday transactions." *Quarterly Journal of Finance.*
31. Kohvakka, P. (2023). "Low volatility anomaly in the Japanese stock market."
32. Gorbenko, A. (2023). "Short interest and aggregate stock returns: International evidence." *Review of Asset Pricing Studies.*
33. Liang, S.X. (2019). "What drives stock returns in Japan?"
34. Hofmann, D. et al. (2024). "On the Linkage of Momentum and Reversal." *Journal of Economics and Finance.*
35. Roy, R. & Shijin, S. (2019). "The nexus of anomalies-stock returns-asset pricing models."
36. Limkriangkrai, M. et al. (2023). "Market Intraday Momentum: APAC Evidence."
37. Morimoto, T. et al. (2026). "Explaining the Cross-Section of Japanese Equity Returns Using Equi-Correlation Structures."
38. Nakano, M. & Yamaoka, T. (2023). "Enhancing Sentiment Analysis Based Investment by LLMs in Japanese Stock Market."
39. Souma, W. et al. (2019). "Predictive power of historical news sentiments." *arXiv.*
40. Morikawa, M. et al. (2024). "Market Response to Non-Mandatory Governance Initiatives: TSE 2023 PBR Improvement Request."
41. Miyajima, H. & Arikawa, Y. (2025). "The Evolution of Corporate Governance in Japan." RIETI.
42. Yamamoto, S. (2022). "Investor price impact on the TSE: 11 investor types." *Pacific-Basin Finance Journal.*
43. Yanagi, R. & Kawakami, H. (2018). "Equity Spread as a Value Creation Metric for Japanese Firms."
