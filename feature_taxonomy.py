"""
feature_taxonomy.py  —  FULL TAXONOMY AUDIT  (stable; do not rename keys)

Normalization dictionary mapping every internal column name in
nky225_features.parquet to its canonical academic / industry name,
factor category, direction, and primary literature sources.

Taxonomy compiled from:
  Academic  : Fama-French (1992, 1993, 2015), Jegadeesh-Titman (1993),
               Carhart (1997), Novy-Marx (2013), Sloan (1996),
               Pontiff-Woodgate (2008), Asness-Frazzini-Pedersen (2019 QMJ),
               Amihud (2002), Ang et al. (2006), Harvey-Liu-Zhu (2016),
               George-Hwang (2004), Bali et al. (2011 MAX)
  Industry  : MSCI Barra FaCS (2018), AQR Factor Taxonomy (2024),
               Citadel EQR, Jane Street alpha-signal framework,
               FTSE Russell factor definitions (2025)
  Technical : Wilder (RSI, ATR, ADX 1978), Bollinger (1983),
               Appel (MACD 1979), Lane (Stochastic 1950s),
               Lambert (CCI 1980), Granville (OBV 1963)

Usage
-----
  from feature_taxonomy import TAXONOMY, NORMALIZED_NAMES

  # get canonical name for a column
  NORMALIZED_NAMES["pe_ttm"]          # → "price_earnings_ttm"

  # get full metadata
  TAXONOMY["pe_ttm"]["category"]      # → "valuation"
  TAXONOMY["pe_ttm"]["msci_factor"]   # → "Value"

Normalized-folder convention
-----------------------------
  Raw folder    :  .../features/<col_name>.parquet
  Norm folder   :  .../features_norm/<canonical_name>.parquet
  (mirrors raw with _norm suffix on the directory)
"""

# ─── top-level categories ────────────────────────────────────────────────────
# return       – raw price-based return series
# momentum     – medium-term trend and price-level signals
# reversal     – short-term mean-reversion signals
# volatility   – realized dispersion measures
# risk         – tail risk and drawdown metrics
# technical    – chart-pattern and oscillator signals
# liquidity    – market-microstructure and trading-activity signals
# valuation    – price relative to fundamental anchor
# profitability– income-statement efficiency ratios
# quality      – balance-sheet and accruals-based signals
# growth       – year-over-year fundamental change rates
# macro        – economy-wide and cross-asset signals
# calendar     – seasonal / event-window dummies
# corporate    – dilution, restructuring, capital-action flags
# index        – index-weight and membership signals
# composite    – multi-signal score aggregations
# cs_rank      – cross-sectional percentile ranks of raw signals

TAXONOMY: dict[str, dict] = {

    # ── RAW RETURNS ──────────────────────────────────────────────────────────

    "ret_1d": {
        "canonical_name": "daily_return",
        "category": "return", "sub_category": "short_term",
        "direction": 0,
        "ff_factor": "Mkt-RF", "msci_factor": None, "aqr_factor": None,
        "sources": ["Fama-French (1993)"],
        "notes": "Raw 1-day log or arithmetic return.",
    },
    "ret_1w": {
        "canonical_name": "weekly_return",
        "category": "return", "sub_category": "short_term",
        "direction": 0,
        "ff_factor": None, "msci_factor": None, "aqr_factor": None,
        "sources": [],
    },
    "ret_1m": {
        "canonical_name": "monthly_return",
        "category": "return", "sub_category": "short_term",
        "direction": 0,
        "ff_factor": None, "msci_factor": "Momentum", "aqr_factor": "Momentum",
        "sources": ["Jegadeesh (1990) — used as reversal baseline"],
    },
    "ret_3m": {
        "canonical_name": "quarterly_return",
        "category": "return", "sub_category": "medium_term",
        "direction": 0,
        "ff_factor": None, "msci_factor": None, "aqr_factor": None,
        "sources": [],
    },
    "ret_6m": {
        "canonical_name": "semi_annual_return",
        "category": "return", "sub_category": "medium_term",
        "direction": 0,
        "ff_factor": None, "msci_factor": "Momentum", "aqr_factor": "Momentum",
        "sources": ["Jegadeesh-Titman (1993)"],
    },
    "ret_12m": {
        "canonical_name": "annual_return",
        "category": "return", "sub_category": "long_term",
        "direction": 0,
        "ff_factor": "Mom", "msci_factor": "Momentum", "aqr_factor": "Momentum",
        "sources": ["Jegadeesh-Titman (1993)", "Carhart (1997)"],
    },
    "adj_ret_1d": {
        "canonical_name": "adjusted_daily_return",
        "category": "return", "sub_category": "short_term",
        "direction": 0,
        "ff_factor": None, "msci_factor": None, "aqr_factor": None,
        "sources": [],
        "notes": "Corporate-action adjusted return.",
    },

    # ── MOMENTUM ─────────────────────────────────────────────────────────────

    "momentum_12_1": {
        "canonical_name": "price_momentum_12_1",
        "category": "momentum", "sub_category": "intermediate_term",
        "direction": +1,
        "ff_factor": "Mom", "msci_factor": "Momentum", "aqr_factor": "Momentum",
        "sources": ["Jegadeesh-Titman (1993)", "Carhart (1997)",
                    "Asness-Moskowitz-Pedersen (2013)"],
        "notes": "12-month return skipping most-recent month (t-12 to t-1).",
    },
    "high_to_price": {
        "canonical_name": "nearness_to_52wk_high",
        "category": "momentum", "sub_category": "price_level",
        "direction": +1,
        "ff_factor": None, "msci_factor": "Momentum", "aqr_factor": None,
        "sources": ["George-Hwang (2004)"],
        "notes": "Close / 52-week high; higher = closer to peak (bullish signal).",
    },

    # ── REVERSAL ─────────────────────────────────────────────────────────────

    "reversal_1w": {
        "canonical_name": "short_term_reversal_1w",
        "category": "reversal", "sub_category": "weekly",
        "direction": -1,
        "ff_factor": "ST_Rev", "msci_factor": "Momentum", "aqr_factor": "Momentum",
        "sources": ["Jegadeesh (1990)"],
        "notes": "Negative sign: prior losers expected to rebound.",
    },
    "reversal_1m": {
        "canonical_name": "short_term_reversal_1m",
        "category": "reversal", "sub_category": "monthly",
        "direction": -1,
        "ff_factor": "ST_Rev", "msci_factor": "Momentum", "aqr_factor": "Momentum",
        "sources": ["Jegadeesh (1990)", "Lehmann (1990)"],
    },

    # ── VOLATILITY ───────────────────────────────────────────────────────────

    "vol_20d": {
        "canonical_name": "realized_volatility_20d",
        "category": "volatility", "sub_category": "short_term",
        "direction": -1,
        "ff_factor": None, "msci_factor": "Low Volatility", "aqr_factor": "Defensive",
        "sources": ["Ang et al. (2006)", "Baker-Bradley-Wurgler (2011)"],
    },
    "vol_60d": {
        "canonical_name": "realized_volatility_60d",
        "category": "volatility", "sub_category": "medium_term",
        "direction": -1,
        "ff_factor": None, "msci_factor": "Low Volatility", "aqr_factor": "Defensive",
        "sources": ["Ang et al. (2006)"],
    },
    "vol_120d": {
        "canonical_name": "realized_volatility_120d",
        "category": "volatility", "sub_category": "long_term",
        "direction": -1,
        "ff_factor": None, "msci_factor": "Low Volatility", "aqr_factor": "Defensive",
        "sources": [],
    },
    "vol_parkinson": {
        "canonical_name": "parkinson_high_low_volatility",
        "category": "volatility", "sub_category": "intraday_estimate",
        "direction": -1,
        "ff_factor": None, "msci_factor": "Low Volatility", "aqr_factor": None,
        "sources": ["Parkinson (1980)"],
        "notes": "Uses OHLC range; more efficient estimator than close-to-close vol.",
    },
    "idio_vol_60d": {
        "canonical_name": "idiosyncratic_volatility_60d",
        "category": "volatility", "sub_category": "idiosyncratic",
        "direction": -1,
        "ff_factor": None, "msci_factor": "Low Volatility", "aqr_factor": None,
        "sources": ["Ang et al. (2006)", "Fu (2009)"],
        "notes": "Residual vol from FF3 or market regression. Low-vol anomaly.",
    },
    "vol_of_vol_20d": {
        "canonical_name": "volatility_of_volatility_20d",
        "category": "volatility", "sub_category": "vol_uncertainty",
        "direction": -1,
        "ff_factor": None, "msci_factor": None, "aqr_factor": None,
        "sources": ["Baltussen-Van Bekkum-Van der Grient (2018)"],
    },
    "vol_ratio_5_252": {
        "canonical_name": "short_to_long_vol_ratio",
        "category": "volatility", "sub_category": "vol_term_structure",
        "direction": -1,
        "ff_factor": None, "msci_factor": None, "aqr_factor": None,
        "sources": [],
    },
    "vol_ratio_20_252": {
        "canonical_name": "medium_to_long_vol_ratio",
        "category": "volatility", "sub_category": "vol_term_structure",
        "direction": -1,
        "ff_factor": None, "msci_factor": None, "aqr_factor": None,
        "sources": [],
    },

    # ── RISK ─────────────────────────────────────────────────────────────────

    "var_95_20d": {
        "canonical_name": "value_at_risk_95_20d",
        "category": "risk", "sub_category": "tail_risk",
        "direction": -1,
        "ff_factor": None, "msci_factor": "Low Volatility", "aqr_factor": "Defensive",
        "sources": [],
    },
    "cvar_95_20d": {
        "canonical_name": "conditional_var_95_20d",
        "category": "risk", "sub_category": "tail_risk",
        "direction": -1,
        "ff_factor": None, "msci_factor": "Low Volatility", "aqr_factor": None,
        "sources": ["Rockafellar-Uryasev (2000)"],
        "notes": "Expected Shortfall (ES); average loss beyond VaR threshold.",
    },
    "max_dd_60d": {
        "canonical_name": "maximum_drawdown_60d",
        "category": "risk", "sub_category": "drawdown",
        "direction": -1,
        "ff_factor": None, "msci_factor": "Low Volatility", "aqr_factor": None,
        "sources": [],
    },
    "max_dd_252d": {
        "canonical_name": "maximum_drawdown_252d",
        "category": "risk", "sub_category": "drawdown",
        "direction": -1,
        "ff_factor": None, "msci_factor": None, "aqr_factor": None,
        "sources": [],
    },
    "beta_nky_252d": {
        "canonical_name": "nikkei_market_beta_252d",
        "category": "risk", "sub_category": "systematic_risk",
        "direction": -1,
        "ff_factor": "Mkt-RF", "msci_factor": "Low Volatility", "aqr_factor": "Defensive",
        "sources": ["Frazzini-Pedersen (2014) BAB"],
        "notes": "Betting-Against-Beta (BAB) factor exploits low-beta premium.",
    },
    "beta_nky_60d": {
        "canonical_name": "nikkei_market_beta_60d",
        "category": "risk", "sub_category": "systematic_risk",
        "direction": -1,
        "ff_factor": "Mkt-RF", "msci_factor": "Low Volatility", "aqr_factor": "Defensive",
        "sources": ["Frazzini-Pedersen (2014) BAB"],
    },
    "beta_usdjpy_60d": {
        "canonical_name": "usdjpy_beta_60d",
        "category": "risk", "sub_category": "fx_sensitivity",
        "direction": 0,
        "ff_factor": None, "msci_factor": None, "aqr_factor": "Carry",
        "sources": [],
    },

    # ── HIGHER MOMENTS ───────────────────────────────────────────────────────

    "skew_20d": {
        "canonical_name": "return_skewness_20d",
        "category": "risk", "sub_category": "higher_moments",
        "direction": +1,
        "ff_factor": None, "msci_factor": None, "aqr_factor": None,
        "sources": ["Harvey-Siddique (2000)", "Boyer-Mitton-Vorkink (2010)"],
        "notes": "+1: positive skew preferred by lottery-ticket investors.",
    },
    "skew_60d": {
        "canonical_name": "return_skewness_60d",
        "category": "risk", "sub_category": "higher_moments",
        "direction": +1,
        "ff_factor": None, "msci_factor": None, "aqr_factor": None,
        "sources": ["Harvey-Siddique (2000)"],
    },
    "kurt_20d": {
        "canonical_name": "return_kurtosis_20d",
        "category": "risk", "sub_category": "higher_moments",
        "direction": -1,
        "ff_factor": None, "msci_factor": None, "aqr_factor": None,
        "sources": [],
    },
    "kurt_60d": {
        "canonical_name": "return_kurtosis_60d",
        "category": "risk", "sub_category": "higher_moments",
        "direction": -1,
        "ff_factor": None, "msci_factor": None, "aqr_factor": None,
        "sources": [],
    },
    "max_ret_1m": {
        "canonical_name": "max_daily_return_1m",
        "category": "risk", "sub_category": "extreme_return",
        "direction": -1,
        "ff_factor": None, "msci_factor": None, "aqr_factor": None,
        "sources": ["Bali-Cakici-Whitelaw (2011) MAX effect"],
        "notes": "MAX anomaly: stocks with high max returns are overpriced.",
    },
    "min_ret_1m": {
        "canonical_name": "min_daily_return_1m",
        "category": "risk", "sub_category": "extreme_return",
        "direction": +1,
        "ff_factor": None, "msci_factor": None, "aqr_factor": None,
        "sources": [],
    },

    # ── TECHNICAL ────────────────────────────────────────────────────────────

    "rsi_14": {
        "canonical_name": "relative_strength_index_14",
        "category": "technical", "sub_category": "oscillator",
        "direction": 0,
        "ff_factor": None, "msci_factor": None, "aqr_factor": None,
        "sources": ["Wilder (1978) New Concepts in Technical Trading Systems"],
        "notes": "Overbought >70, oversold <30. Non-linear mean-reversion signal.",
    },
    "macd_line": {
        "canonical_name": "macd_line_12_26",
        "category": "technical", "sub_category": "trend",
        "direction": +1,
        "ff_factor": None, "msci_factor": None, "aqr_factor": None,
        "sources": ["Appel (1979)"],
        "notes": "EMA(12) - EMA(26); positive = bullish.",
    },
    "macd_signal": {
        "canonical_name": "macd_signal_9",
        "category": "technical", "sub_category": "trend",
        "direction": +1,
        "ff_factor": None, "msci_factor": None, "aqr_factor": None,
        "sources": ["Appel (1979)"],
        "notes": "9-day EMA of MACD line.",
    },
    "macd_hist": {
        "canonical_name": "macd_histogram",
        "category": "technical", "sub_category": "trend",
        "direction": +1,
        "ff_factor": None, "msci_factor": None, "aqr_factor": None,
        "sources": ["Appel (1979)"],
        "notes": "MACD line - signal line; sign change = trend reversal.",
    },
    "bb_pos": {
        "canonical_name": "bollinger_band_position",
        "category": "technical", "sub_category": "volatility_band",
        "direction": 0,
        "ff_factor": None, "msci_factor": None, "aqr_factor": None,
        "sources": ["Bollinger (1983)"],
        "notes": "(close - lower_band) / (upper_band - lower_band). 0=lower, 1=upper.",
    },
    "bb_width": {
        "canonical_name": "bollinger_band_width",
        "category": "technical", "sub_category": "volatility_band",
        "direction": 0,
        "ff_factor": None, "msci_factor": None, "aqr_factor": None,
        "sources": ["Bollinger (1983)"],
        "notes": "(upper - lower) / mid. Low width = volatility contraction.",
    },
    "keltner_pos": {
        "canonical_name": "keltner_channel_position",
        "category": "technical", "sub_category": "volatility_band",
        "direction": 0,
        "ff_factor": None, "msci_factor": None, "aqr_factor": None,
        "sources": ["Keltner (1960)", "Chester Keltner"],
    },
    "stoch_k": {
        "canonical_name": "stochastic_oscillator_k_14",
        "category": "technical", "sub_category": "oscillator",
        "direction": 0,
        "ff_factor": None, "msci_factor": None, "aqr_factor": None,
        "sources": ["Lane (1950s)"],
        "notes": "(close - low_14) / (high_14 - low_14) × 100.",
    },
    "stoch_d": {
        "canonical_name": "stochastic_oscillator_d_3",
        "category": "technical", "sub_category": "oscillator",
        "direction": 0,
        "ff_factor": None, "msci_factor": None, "aqr_factor": None,
        "sources": ["Lane (1950s)"],
        "notes": "3-period SMA of %K; signal line for crossover.",
    },
    "stoch_kd_cross": {
        "canonical_name": "stochastic_kd_crossover_signal",
        "category": "technical", "sub_category": "crossover_signal",
        "direction": +1,
        "ff_factor": None, "msci_factor": None, "aqr_factor": None,
        "sources": [],
    },
    "roc_5": {
        "canonical_name": "rate_of_change_5d",
        "category": "technical", "sub_category": "momentum_oscillator",
        "direction": +1,
        "ff_factor": None, "msci_factor": None, "aqr_factor": None,
        "sources": [],
        "notes": "(close_t / close_{t-5}) - 1.",
    },
    "roc_10": {
        "canonical_name": "rate_of_change_10d",
        "category": "technical", "sub_category": "momentum_oscillator",
        "direction": +1,
        "ff_factor": None, "msci_factor": None, "aqr_factor": None,
        "sources": [],
    },
    "roc_20": {
        "canonical_name": "rate_of_change_20d",
        "category": "technical", "sub_category": "momentum_oscillator",
        "direction": +1,
        "ff_factor": None, "msci_factor": None, "aqr_factor": None,
        "sources": [],
    },
    "cci_20": {
        "canonical_name": "commodity_channel_index_20",
        "category": "technical", "sub_category": "oscillator",
        "direction": 0,
        "ff_factor": None, "msci_factor": None, "aqr_factor": None,
        "sources": ["Lambert (1980)"],
    },
    "williams_r": {
        "canonical_name": "williams_percent_r_14",
        "category": "technical", "sub_category": "oscillator",
        "direction": 0,
        "ff_factor": None, "msci_factor": None, "aqr_factor": None,
        "sources": ["Williams (1973)"],
        "notes": "Similar to %K stochastic; -100 to 0 scale.",
    },
    "adx_14": {
        "canonical_name": "average_directional_index_14",
        "category": "technical", "sub_category": "trend_strength",
        "direction": +1,
        "ff_factor": None, "msci_factor": None, "aqr_factor": None,
        "sources": ["Wilder (1978)"],
        "notes": ">25 = trending market. Direction-agnostic trend strength.",
    },
    "di_plus_14": {
        "canonical_name": "directional_indicator_plus_14",
        "category": "technical", "sub_category": "trend_direction",
        "direction": +1,
        "ff_factor": None, "msci_factor": None, "aqr_factor": None,
        "sources": ["Wilder (1978)"],
    },
    "di_minus_14": {
        "canonical_name": "directional_indicator_minus_14",
        "category": "technical", "sub_category": "trend_direction",
        "direction": -1,
        "ff_factor": None, "msci_factor": None, "aqr_factor": None,
        "sources": ["Wilder (1978)"],
    },
    "di_diff_14": {
        "canonical_name": "directional_indicator_diff_14",
        "category": "technical", "sub_category": "trend_direction",
        "direction": +1,
        "ff_factor": None, "msci_factor": None, "aqr_factor": None,
        "sources": ["Wilder (1978)"],
        "notes": "DI+ - DI-; positive = bullish trend.",
    },
    "donchian_pos": {
        "canonical_name": "donchian_channel_position",
        "category": "technical", "sub_category": "breakout",
        "direction": +1,
        "ff_factor": None, "msci_factor": None, "aqr_factor": None,
        "sources": ["Donchian (1960)"],
        "notes": "Position within N-day high/low channel; basis of turtle trading.",
    },
    "ema_cross_5_20": {
        "canonical_name": "ema_crossover_5_20",
        "category": "technical", "sub_category": "crossover_signal",
        "direction": +1,
        "ff_factor": None, "msci_factor": None, "aqr_factor": None,
        "sources": [],
        "notes": "Binary: 1 if EMA(5) > EMA(20), else -1 or 0.",
    },
    "ema_cross_50_200": {
        "canonical_name": "golden_death_cross_50_200",
        "category": "technical", "sub_category": "crossover_signal",
        "direction": +1,
        "ff_factor": None, "msci_factor": "Momentum", "aqr_factor": None,
        "sources": [],
        "notes": "Golden cross (50>200) / death cross (50<200).",
    },
    "atr_14": {
        "canonical_name": "average_true_range_14",
        "category": "volatility", "sub_category": "intraday_range",
        "direction": 0,
        "ff_factor": None, "msci_factor": None, "aqr_factor": None,
        "sources": ["Wilder (1978)"],
        "notes": "Absolute volatility measure; used for position sizing.",
    },
    "atr_pct": {
        "canonical_name": "atr_percent_of_price",
        "category": "volatility", "sub_category": "normalized_range",
        "direction": -1,
        "ff_factor": None, "msci_factor": "Low Volatility", "aqr_factor": None,
        "sources": ["Wilder (1978)"],
    },
    "atr_norm_ret": {
        "canonical_name": "atr_normalized_return",
        "category": "technical", "sub_category": "risk_adjusted_return",
        "direction": +1,
        "ff_factor": None, "msci_factor": None, "aqr_factor": None,
        "sources": [],
        "notes": "Return / ATR; reward-to-range ratio.",
    },

    # ── VOLUME / MICROSTRUCTURE ───────────────────────────────────────────────

    "log_yen_volume": {
        "canonical_name": "log_trading_volume_jpy",
        "category": "liquidity", "sub_category": "volume",
        "direction": +1,
        "ff_factor": None, "msci_factor": "Low Size", "aqr_factor": None,
        "sources": ["Datar-Naik-Radcliffe (1998)"],
    },
    "obv": {
        "canonical_name": "on_balance_volume",
        "category": "technical", "sub_category": "volume_accumulation",
        "direction": +1,
        "ff_factor": None, "msci_factor": None, "aqr_factor": None,
        "sources": ["Granville (1963)"],
        "notes": "Cumulative volume signed by daily return direction.",
    },
    "obv_ret_1m": {
        "canonical_name": "obv_monthly_return",
        "category": "technical", "sub_category": "volume_momentum",
        "direction": +1,
        "ff_factor": None, "msci_factor": None, "aqr_factor": None,
        "sources": ["Granville (1963)"],
    },
    "obv_ret_3m": {
        "canonical_name": "obv_quarterly_return",
        "category": "technical", "sub_category": "volume_momentum",
        "direction": +1,
        "ff_factor": None, "msci_factor": None, "aqr_factor": None,
        "sources": [],
    },
    "mfi_14": {
        "canonical_name": "money_flow_index_14",
        "category": "technical", "sub_category": "volume_oscillator",
        "direction": 0,
        "ff_factor": None, "msci_factor": None, "aqr_factor": None,
        "sources": ["Quong-Soudack"],
        "notes": "Volume-weighted RSI; 0–100 scale.",
    },
    "cmf_20": {
        "canonical_name": "chaikin_money_flow_20",
        "category": "technical", "sub_category": "volume_oscillator",
        "direction": +1,
        "ff_factor": None, "msci_factor": None, "aqr_factor": None,
        "sources": ["Chaikin (1980s)"],
        "notes": "Measures buying/selling pressure via volume × close-location.",
    },
    "amihud_60d": {
        "canonical_name": "amihud_illiquidity_60d",
        "category": "liquidity", "sub_category": "price_impact",
        "direction": -1,
        "ff_factor": None, "msci_factor": None, "aqr_factor": None,
        "sources": ["Amihud (2002) JFM"],
        "notes": "Mean(|ret|/volume). High ratio = illiquid. Liquidity premium.",
    },
    "rel_volume_5_60": {
        "canonical_name": "relative_volume_5_60d",
        "category": "liquidity", "sub_category": "volume_ratio",
        "direction": 0,
        "ff_factor": None, "msci_factor": None, "aqr_factor": None,
        "sources": [],
        "notes": "5-day avg volume / 60-day avg volume.",
    },
    "vol_imbalance_20d": {
        "canonical_name": "volume_imbalance_20d",
        "category": "liquidity", "sub_category": "order_flow",
        "direction": +1,
        "ff_factor": None, "msci_factor": None, "aqr_factor": None,
        "sources": [],
    },
    "dv_zscore_252d": {
        "canonical_name": "dollar_volume_zscore_252d",
        "category": "liquidity", "sub_category": "volume_z_score",
        "direction": +1,
        "ff_factor": None, "msci_factor": None, "aqr_factor": None,
        "sources": [],
    },
    "price_vol_diverge": {
        "canonical_name": "price_volume_divergence",
        "category": "technical", "sub_category": "divergence",
        "direction": 0,
        "ff_factor": None, "msci_factor": None, "aqr_factor": None,
        "sources": [],
    },
    "vwap_dev_20d": {
        "canonical_name": "vwap_deviation_20d",
        "category": "liquidity", "sub_category": "price_dislocation",
        "direction": 0,
        "ff_factor": None, "msci_factor": None, "aqr_factor": None,
        "sources": [],
    },
    "turnover_decel": {
        "canonical_name": "turnover_deceleration",
        "category": "liquidity", "sub_category": "trading_activity_change",
        "direction": -1,
        "ff_factor": None, "msci_factor": None, "aqr_factor": None,
        "sources": [],
    },
    "gain_loss_ratio_20d": {
        "canonical_name": "gain_loss_ratio_20d",
        "category": "technical", "sub_category": "oscillator",
        "direction": +1,
        "ff_factor": None, "msci_factor": None, "aqr_factor": None,
        "sources": [],
    },

    # ── VALUATION (FUNDAMENTAL) ───────────────────────────────────────────────

    "pe_ttm": {
        "canonical_name": "price_earnings_ttm",
        "category": "valuation", "sub_category": "earnings_multiple",
        "direction": -1,
        "ff_factor": "HML", "msci_factor": "Value", "aqr_factor": "Value",
        "sources": ["Basu (1977)", "Fama-French (1992)", "AQR Value and Momentum (2013)"],
        "notes": "Lower P/E = cheaper. Inverse (E/P = earnings yield) used in many models.",
    },
    "pb_ratio": {
        "canonical_name": "price_book_ratio",
        "category": "valuation", "sub_category": "book_multiple",
        "direction": -1,
        "ff_factor": "HML", "msci_factor": "Value", "aqr_factor": "Value",
        "sources": ["Fama-French (1992, 1993)", "Rosenberg-Reid-Lanstein (1985)"],
        "notes": "Core HML signal. B/P (inverse) loads positively on value factor.",
    },
    "ps_ratio": {
        "canonical_name": "price_sales_ttm",
        "category": "valuation", "sub_category": "sales_multiple",
        "direction": -1,
        "ff_factor": "HML", "msci_factor": "Value", "aqr_factor": "Value",
        "sources": ["Barbee-Mukherji-Raines (1996)"],
    },
    "ev_ebitda": {
        "canonical_name": "ev_ebitda_ttm",
        "category": "valuation", "sub_category": "enterprise_multiple",
        "direction": -1,
        "ff_factor": "HML", "msci_factor": "Value", "aqr_factor": "Value",
        "sources": ["Loughran-Wellman (2011)", "AQR Value Everywhere"],
        "notes": "More leverage-neutral than P/E. Key in AQR value composite.",
    },
    "fcf_yield": {
        "canonical_name": "free_cash_flow_yield",
        "category": "valuation", "sub_category": "cash_yield",
        "direction": +1,
        "ff_factor": "HML", "msci_factor": "Value", "aqr_factor": "Value",
        "sources": ["Lakonishok-Shleifer-Vishny (1994)"],
        "notes": "FCF / market cap. Higher = cheaper on cash basis.",
    },

    # ── PROFITABILITY (FUNDAMENTAL) ───────────────────────────────────────────

    "roe_ttm": {
        "canonical_name": "return_on_equity_ttm",
        "category": "profitability", "sub_category": "equity_efficiency",
        "direction": +1,
        "ff_factor": "RMW", "msci_factor": "Quality", "aqr_factor": "Quality",
        "sources": ["Haugen-Baker (1996)", "Asness-Frazzini-Pedersen (2019) QMJ",
                    "MSCI Quality Index"],
        "notes": "Core MSCI Quality descriptor. RMW loads on operating profitability.",
    },
    "roa_ttm": {
        "canonical_name": "return_on_assets_ttm",
        "category": "profitability", "sub_category": "asset_efficiency",
        "direction": +1,
        "ff_factor": "RMW", "msci_factor": "Quality", "aqr_factor": "Quality",
        "sources": ["Fama-French (2015)", "Novy-Marx (2013)"],
    },
    "net_margin_ttm": {
        "canonical_name": "net_profit_margin_ttm",
        "category": "profitability", "sub_category": "margins",
        "direction": +1,
        "ff_factor": "RMW", "msci_factor": "Quality", "aqr_factor": "Quality",
        "sources": ["Novy-Marx (2013)", "Asness-Frazzini-Pedersen (2019) QMJ"],
    },
    "op_margin_ttm": {
        "canonical_name": "operating_profit_margin_ttm",
        "category": "profitability", "sub_category": "margins",
        "direction": +1,
        "ff_factor": "RMW", "msci_factor": "Quality", "aqr_factor": "Quality",
        "sources": ["Fama-French (2015) — operating profitability"],
        "notes": "FF5 RMW factor uses operating profitability / book equity.",
    },
    "gross_margin_ttm": {
        "canonical_name": "gross_profit_margin_ttm",
        "category": "profitability", "sub_category": "margins",
        "direction": +1,
        "ff_factor": "RMW", "msci_factor": "Quality", "aqr_factor": "Quality",
        "sources": ["Novy-Marx (2013) — gross profitability premium"],
        "notes": "Novy-Marx: gross profit / total assets predicts returns better than net income.",
    },
    "fcf_margin_ttm": {
        "canonical_name": "free_cash_flow_margin_ttm",
        "category": "profitability", "sub_category": "cash_margins",
        "direction": +1,
        "ff_factor": "RMW", "msci_factor": "Quality", "aqr_factor": "Quality",
        "sources": ["Asness-Frazzini-Pedersen (2019) QMJ — cash flow payout"],
    },

    # ── QUALITY / BALANCE SHEET ───────────────────────────────────────────────

    "debt_to_equity": {
        "canonical_name": "book_leverage_ratio",
        "category": "quality", "sub_category": "leverage",
        "direction": -1,
        "ff_factor": None, "msci_factor": "Quality", "aqr_factor": "Quality",
        "sources": ["Fama-French (1992)", "Asness-Frazzini-Pedersen (2019) QMJ — safety"],
        "notes": "MSCI Quality uses D/E as one of three descriptors.",
    },
    "net_debt_to_ebitda": {
        "canonical_name": "net_leverage_ebitda",
        "category": "quality", "sub_category": "leverage",
        "direction": -1,
        "ff_factor": None, "msci_factor": "Quality", "aqr_factor": "Quality",
        "sources": [],
    },
    "current_ratio": {
        "canonical_name": "current_ratio",
        "category": "quality", "sub_category": "liquidity_coverage",
        "direction": +1,
        "ff_factor": None, "msci_factor": "Quality", "aqr_factor": "Quality",
        "sources": ["Asness-Frazzini-Pedersen (2019) QMJ — safety component"],
        "notes": "Current assets / current liabilities. Proxies short-term solvency.",
    },
    "accruals_ratio": {
        "canonical_name": "accruals_to_assets",
        "category": "quality", "sub_category": "earnings_quality",
        "direction": -1,
        "ff_factor": "RMW", "msci_factor": "Quality", "aqr_factor": "Quality",
        "sources": ["Sloan (1996)", "Richardson et al. (2005)"],
        "notes": "(Net income - OCF) / avg assets. Low accruals = high earnings quality.",
    },
    "fcf_to_earnings": {
        "canonical_name": "fcf_to_net_income",
        "category": "quality", "sub_category": "earnings_quality",
        "direction": +1,
        "ff_factor": "RMW", "msci_factor": "Quality", "aqr_factor": "Quality",
        "sources": ["Asness-Frazzini-Pedersen (2019) QMJ — cash flow / earnings"],
    },
    "asset_growth": {
        "canonical_name": "total_asset_growth_yoy",
        "category": "quality", "sub_category": "investment_aggression",
        "direction": -1,
        "ff_factor": "CMA", "msci_factor": None, "aqr_factor": None,
        "sources": ["Pontiff-Woodgate (2008)", "Cooper-Gulen-Schill (2008)",
                    "Fama-French (2015) CMA"],
        "notes": "High asset growth = aggressive investment → negative alpha.",
    },

    # ── GROWTH (FUNDAMENTAL) ─────────────────────────────────────────────────

    "rev_growth_yoy": {
        "canonical_name": "revenue_growth_yoy",
        "category": "growth", "sub_category": "top_line_growth",
        "direction": +1,
        "ff_factor": None, "msci_factor": None, "aqr_factor": "Quality",
        "sources": ["Asness-Frazzini-Pedersen (2019) QMJ — growth component",
                    "Mohanram (2005) G-score"],
    },
    "netinc_growth_yoy": {
        "canonical_name": "net_income_growth_yoy",
        "category": "growth", "sub_category": "bottom_line_growth",
        "direction": +1,
        "ff_factor": None, "msci_factor": None, "aqr_factor": "Quality",
        "sources": ["Asness-Frazzini-Pedersen (2019) QMJ — growth component"],
    },

    # ── MACRO ─────────────────────────────────────────────────────────────────

    "usdjpy_level": {
        "canonical_name": "usdjpy_spot_rate",
        "category": "macro", "sub_category": "fx",
        "direction": 0, "ff_factor": None, "msci_factor": None, "aqr_factor": "Carry",
        "sources": [],
    },
    "usdjpy_ret_1m": {
        "canonical_name": "usdjpy_monthly_return",
        "category": "macro", "sub_category": "fx_momentum",
        "direction": 0, "ff_factor": None, "msci_factor": None, "aqr_factor": "Carry",
        "sources": ["Menkhoff et al. (2012) — FX momentum"],
    },
    "usdjpy_ret_1w": {
        "canonical_name": "usdjpy_weekly_return",
        "category": "macro", "sub_category": "fx_momentum",
        "direction": 0, "ff_factor": None, "msci_factor": None, "aqr_factor": None,
        "sources": [],
    },
    "usdjpy_ret_3m": {
        "canonical_name": "usdjpy_quarterly_return",
        "category": "macro", "sub_category": "fx_momentum",
        "direction": 0, "ff_factor": None, "msci_factor": None, "aqr_factor": "Carry",
        "sources": [],
    },
    "usdjpy_trend": {
        "canonical_name": "usdjpy_trend_signal",
        "category": "macro", "sub_category": "fx_trend",
        "direction": 0, "ff_factor": None, "msci_factor": None, "aqr_factor": None,
        "sources": [],
    },
    "usdjpy_vol_20d": {
        "canonical_name": "usdjpy_realized_vol_20d",
        "category": "macro", "sub_category": "fx_volatility",
        "direction": -1, "ff_factor": None, "msci_factor": None, "aqr_factor": None,
        "sources": [],
    },
    "usdjpy_zscore": {
        "canonical_name": "usdjpy_z_score_252d",
        "category": "macro", "sub_category": "fx_positioning",
        "direction": 0, "ff_factor": None, "msci_factor": None, "aqr_factor": None,
        "sources": [],
    },
    "vix": {
        "canonical_name": "cboe_vix_level",
        "category": "macro", "sub_category": "market_fear",
        "direction": -1, "ff_factor": None, "msci_factor": None, "aqr_factor": "Defensive",
        "sources": ["Whaley (1993)", "CBOE VIX white paper (2003)"],
        "notes": "Implied vol of S&P 500 options; fear gauge.",
    },
    "vix_ret_1m": {
        "canonical_name": "vix_monthly_change",
        "category": "macro", "sub_category": "fear_momentum",
        "direction": -1, "ff_factor": None, "msci_factor": None, "aqr_factor": None,
        "sources": [],
    },
    "vix_zscore": {
        "canonical_name": "vix_z_score_252d",
        "category": "macro", "sub_category": "fear_regime",
        "direction": -1, "ff_factor": None, "msci_factor": None, "aqr_factor": None,
        "sources": [],
    },
    "spx_ret_1d": {
        "canonical_name": "sp500_daily_return",
        "category": "macro", "sub_category": "us_equity",
        "direction": +1, "ff_factor": None, "msci_factor": None, "aqr_factor": None,
        "sources": [],
    },
    "spx_ret_1m": {
        "canonical_name": "sp500_monthly_return",
        "category": "macro", "sub_category": "us_equity_momentum",
        "direction": +1, "ff_factor": None, "msci_factor": None, "aqr_factor": None,
        "sources": [],
    },
    "spx_trend": {
        "canonical_name": "sp500_trend_signal",
        "category": "macro", "sub_category": "us_equity_trend",
        "direction": +1, "ff_factor": None, "msci_factor": None, "aqr_factor": None,
        "sources": [],
    },
    "nky_ret_1m": {
        "canonical_name": "nikkei225_monthly_return",
        "category": "macro", "sub_category": "jp_index_momentum",
        "direction": +1, "ff_factor": None, "msci_factor": None, "aqr_factor": None,
        "sources": [],
    },
    "nky_ret_3m": {
        "canonical_name": "nikkei225_quarterly_return",
        "category": "macro", "sub_category": "jp_index_momentum",
        "direction": +1, "ff_factor": None, "msci_factor": None, "aqr_factor": None,
        "sources": [],
    },
    "nky_ret_12m": {
        "canonical_name": "nikkei225_annual_return",
        "category": "macro", "sub_category": "jp_index_momentum",
        "direction": +1, "ff_factor": None, "msci_factor": None, "aqr_factor": None,
        "sources": [],
    },
    "nky_trend": {
        "canonical_name": "nikkei225_trend_signal",
        "category": "macro", "sub_category": "jp_index_trend",
        "direction": +1, "ff_factor": None, "msci_factor": None, "aqr_factor": None,
        "sources": [],
    },
    "nky_vol_20d": {
        "canonical_name": "nikkei225_realized_vol_20d",
        "category": "macro", "sub_category": "jp_index_volatility",
        "direction": -1, "ff_factor": None, "msci_factor": None, "aqr_factor": None,
        "sources": [],
    },
    "us10y": {
        "canonical_name": "us_10yr_treasury_yield",
        "category": "macro", "sub_category": "rates",
        "direction": 0, "ff_factor": None, "msci_factor": None, "aqr_factor": "Carry",
        "sources": [],
    },
    "us10y_ret_1m": {
        "canonical_name": "us_10yr_yield_monthly_change",
        "category": "macro", "sub_category": "rates_momentum",
        "direction": 0, "ff_factor": None, "msci_factor": None, "aqr_factor": None,
        "sources": [],
    },
    "us_curve_5_10": {
        "canonical_name": "us_yield_curve_5_10",
        "category": "macro", "sub_category": "curve_slope",
        "direction": +1, "ff_factor": None, "msci_factor": None, "aqr_factor": None,
        "sources": [],
        "notes": "10Y - 5Y spread; steeper = risk-on.",
    },
    "risk_off_flag": {
        "canonical_name": "risk_off_regime_flag",
        "category": "macro", "sub_category": "regime",
        "direction": -1, "ff_factor": None, "msci_factor": None, "aqr_factor": "Defensive",
        "sources": [],
    },

    # ── CALENDAR ─────────────────────────────────────────────────────────────

    "day_of_week": {
        "canonical_name": "day_of_week",
        "category": "calendar", "sub_category": "intraweek_seasonality",
        "direction": 0, "ff_factor": None, "msci_factor": None, "aqr_factor": None,
        "sources": ["Cross (1973)", "French (1980) — weekend effect"],
    },
    "is_monday": {
        "canonical_name": "monday_indicator",
        "category": "calendar", "sub_category": "day_of_week_effect",
        "direction": -1, "ff_factor": None, "msci_factor": None, "aqr_factor": None,
        "sources": ["French (1980) — weekend effect"],
    },
    "is_friday": {
        "canonical_name": "friday_indicator",
        "category": "calendar", "sub_category": "day_of_week_effect",
        "direction": +1, "ff_factor": None, "msci_factor": None, "aqr_factor": None,
        "sources": [],
    },
    "golden_week": {
        "canonical_name": "golden_week_pre_holiday",
        "category": "calendar", "sub_category": "japan_holiday",
        "direction": 0, "ff_factor": None, "msci_factor": None, "aqr_factor": None,
        "sources": [],
    },
    "obon_holiday": {
        "canonical_name": "obon_period_indicator",
        "category": "calendar", "sub_category": "japan_holiday",
        "direction": 0, "ff_factor": None, "msci_factor": None, "aqr_factor": None,
        "sources": [],
    },
    "fiscal_year_end_flag": {
        "canonical_name": "fiscal_year_end_flag",
        "category": "calendar", "sub_category": "earnings_season",
        "direction": 0, "ff_factor": None, "msci_factor": None, "aqr_factor": None,
        "sources": [],
    },
    "pre_fiscal_year_end": {
        "canonical_name": "pre_fiscal_year_end_flag",
        "category": "calendar", "sub_category": "earnings_season",
        "direction": 0, "ff_factor": None, "msci_factor": None, "aqr_factor": None,
        "sources": [],
    },
    "ex_div_season": {
        "canonical_name": "ex_dividend_season_flag",
        "category": "calendar", "sub_category": "dividend_event",
        "direction": 0, "ff_factor": None, "msci_factor": "High Yield", "aqr_factor": None,
        "sources": [],
    },
    "calendar_year_end": {
        "canonical_name": "calendar_year_end_flag",
        "category": "calendar", "sub_category": "year_end_effect",
        "direction": 0, "ff_factor": None, "msci_factor": None, "aqr_factor": None,
        "sources": ["Haugen-Lakonishok (1988) — January effect"],
    },
    "fiscal_quarter": {
        "canonical_name": "fiscal_quarter_number",
        "category": "calendar", "sub_category": "reporting_period",
        "direction": 0, "ff_factor": None, "msci_factor": None, "aqr_factor": None,
        "sources": [],
    },

    # ── CORPORATE ACTIONS ────────────────────────────────────────────────────

    "cap_raise_flag": {
        "canonical_name": "capital_raise_flag",
        "category": "corporate", "sub_category": "dilution_event",
        "direction": -1, "ff_factor": None, "msci_factor": None, "aqr_factor": None,
        "sources": ["Loughran-Ritter (1995) — SEO underperformance",
                    "Pontiff-Woodgate (2008) — share issuance"],
    },
    "cap_raise_dilution": {
        "canonical_name": "capital_raise_dilution_pct",
        "category": "corporate", "sub_category": "dilution_magnitude",
        "direction": -1, "ff_factor": None, "msci_factor": None, "aqr_factor": None,
        "sources": ["Loughran-Ritter (1995)"],
    },

    # ── INDEX / WEIGHT ────────────────────────────────────────────────────────

    "nky225_weight": {
        "canonical_name": "nikkei225_index_weight",
        "category": "index", "sub_category": "price_weighted_share",
        "direction": 0, "ff_factor": None, "msci_factor": None, "aqr_factor": None,
        "sources": [],
        "notes": "PAF-adjusted price-weight in the Nikkei 225 price-weighted index.",
    },
    "log_nky225_weight": {
        "canonical_name": "log_nikkei225_weight",
        "category": "index", "sub_category": "price_weighted_share",
        "direction": 0, "ff_factor": None, "msci_factor": None, "aqr_factor": None,
        "sources": [],
    },
    "excess_weight": {
        "canonical_name": "excess_weight_vs_equal",
        "category": "index", "sub_category": "active_weight",
        "direction": 0, "ff_factor": None, "msci_factor": None, "aqr_factor": None,
        "sources": [],
        "notes": "NKY225 weight minus 1/N equal-weight benchmark.",
    },
    "nky225_weight_chg_1m": {
        "canonical_name": "nky225_weight_monthly_change",
        "category": "index", "sub_category": "weight_momentum",
        "direction": 0, "ff_factor": None, "msci_factor": None, "aqr_factor": None,
        "sources": [],
    },
    "nky225_weight_chg_3m": {
        "canonical_name": "nky225_weight_quarterly_change",
        "category": "index", "sub_category": "weight_momentum",
        "direction": 0, "ff_factor": None, "msci_factor": None, "aqr_factor": None,
        "sources": [],
    },
    "bench_weight": {
        "canonical_name": "benchmark_weight",
        "category": "index", "sub_category": "passive_weight",
        "direction": 0, "ff_factor": None, "msci_factor": None, "aqr_factor": None,
        "sources": [],
    },
    "in_index": {
        "canonical_name": "index_membership_flag",
        "category": "index", "sub_category": "inclusion_flag",
        "direction": +1, "ff_factor": None, "msci_factor": None, "aqr_factor": None,
        "sources": ["Chen-Noronha-Singal (2004) — index addition effect"],
    },

    # ── COMPOSITE SCORES ──────────────────────────────────────────────────────

    "score_momentum": {
        "canonical_name": "composite_momentum_score",
        "category": "composite", "sub_category": "momentum",
        "direction": +1, "ff_factor": "Mom", "msci_factor": "Momentum", "aqr_factor": "Momentum",
        "sources": [],
    },
    "score_reversal": {
        "canonical_name": "composite_reversal_score",
        "category": "composite", "sub_category": "reversal",
        "direction": -1, "ff_factor": "ST_Rev", "msci_factor": "Momentum", "aqr_factor": None,
        "sources": [],
    },
    "score_technical": {
        "canonical_name": "composite_technical_score",
        "category": "composite", "sub_category": "technical",
        "direction": +1, "ff_factor": None, "msci_factor": None, "aqr_factor": None,
        "sources": [],
    },
    "score_liquidity": {
        "canonical_name": "composite_liquidity_score",
        "category": "composite", "sub_category": "liquidity",
        "direction": +1, "ff_factor": None, "msci_factor": None, "aqr_factor": None,
        "sources": [],
    },
    "score_low_vol": {
        "canonical_name": "composite_low_volatility_score",
        "category": "composite", "sub_category": "low_vol",
        "direction": +1, "ff_factor": None, "msci_factor": "Low Volatility", "aqr_factor": "Defensive",
        "sources": [],
    },

    # ── CROSS-SECTIONAL RANKS (cs_rank_ / cs_decile_ / cs_z_) ────────────────
    # These mirror the underlying feature with a cs_ prefix.
    # Only the most important ones are listed individually.

    "cs_rank_ret_1d": {
        "canonical_name": "cs_rank_daily_return",
        "category": "cs_rank", "sub_category": "return",
        "direction": +1, "ff_factor": None, "msci_factor": None, "aqr_factor": None,
        "sources": [],
    },
    "cs_rank_ret_1m": {
        "canonical_name": "cs_rank_monthly_return",
        "category": "cs_rank", "sub_category": "return",
        "direction": +1, "ff_factor": None, "msci_factor": None, "aqr_factor": None,
        "sources": [],
    },
    "cs_rank_vol_20d": {
        "canonical_name": "cs_rank_realized_vol_20d",
        "category": "cs_rank", "sub_category": "volatility",
        "direction": -1, "ff_factor": None, "msci_factor": "Low Volatility", "aqr_factor": None,
        "sources": [],
    },
    "cs_rank_momentum_12_1": {
        "canonical_name": "cs_rank_price_momentum_12_1",
        "category": "cs_rank", "sub_category": "momentum",
        "direction": +1, "ff_factor": "Mom", "msci_factor": "Momentum", "aqr_factor": "Momentum",
        "sources": [],
    },
    "cs_rank_amihud_60d": {
        "canonical_name": "cs_rank_amihud_illiquidity_60d",
        "category": "cs_rank", "sub_category": "liquidity",
        "direction": -1, "ff_factor": None, "msci_factor": None, "aqr_factor": None,
        "sources": [],
    },
    "cs_rank_nky225_weight": {
        "canonical_name": "cs_rank_nikkei225_weight",
        "category": "cs_rank", "sub_category": "index",
        "direction": 0, "ff_factor": None, "msci_factor": None, "aqr_factor": None,
        "sources": [],
    },
    "cs_rank_pe_ttm": {
        "canonical_name": "cs_rank_price_earnings_ttm",
        "category": "cs_rank", "sub_category": "valuation",
        "direction": -1, "ff_factor": "HML", "msci_factor": "Value", "aqr_factor": "Value",
        "sources": [],
    },
    "cs_rank_pb_ratio": {
        "canonical_name": "cs_rank_price_book_ratio",
        "category": "cs_rank", "sub_category": "valuation",
        "direction": -1, "ff_factor": "HML", "msci_factor": "Value", "aqr_factor": "Value",
        "sources": [],
    },
    "cs_rank_roe_ttm": {
        "canonical_name": "cs_rank_return_on_equity_ttm",
        "category": "cs_rank", "sub_category": "profitability",
        "direction": +1, "ff_factor": "RMW", "msci_factor": "Quality", "aqr_factor": "Quality",
        "sources": [],
    },
    "cs_rank_roa_ttm": {
        "canonical_name": "cs_rank_return_on_assets_ttm",
        "category": "cs_rank", "sub_category": "profitability",
        "direction": +1, "ff_factor": "RMW", "msci_factor": "Quality", "aqr_factor": "Quality",
        "sources": [],
    },
    "cs_rank_net_margin_ttm": {
        "canonical_name": "cs_rank_net_profit_margin_ttm",
        "category": "cs_rank", "sub_category": "profitability",
        "direction": +1, "ff_factor": "RMW", "msci_factor": "Quality", "aqr_factor": "Quality",
        "sources": [],
    },
    "cs_rank_op_margin_ttm": {
        "canonical_name": "cs_rank_operating_margin_ttm",
        "category": "cs_rank", "sub_category": "profitability",
        "direction": +1, "ff_factor": "RMW", "msci_factor": "Quality", "aqr_factor": "Quality",
        "sources": [],
    },
    "cs_rank_rev_growth_yoy": {
        "canonical_name": "cs_rank_revenue_growth_yoy",
        "category": "cs_rank", "sub_category": "growth",
        "direction": +1, "ff_factor": None, "msci_factor": None, "aqr_factor": "Quality",
        "sources": [],
    },
    "cs_rank_netinc_growth_yoy": {
        "canonical_name": "cs_rank_net_income_growth_yoy",
        "category": "cs_rank", "sub_category": "growth",
        "direction": +1, "ff_factor": None, "msci_factor": None, "aqr_factor": "Quality",
        "sources": [],
    },
    "cs_rank_accruals_ratio": {
        "canonical_name": "cs_rank_accruals_to_assets",
        "category": "cs_rank", "sub_category": "quality",
        "direction": -1, "ff_factor": "RMW", "msci_factor": "Quality", "aqr_factor": "Quality",
        "sources": [],
    },
    "cs_rank_fcf_yield": {
        "canonical_name": "cs_rank_free_cash_flow_yield",
        "category": "cs_rank", "sub_category": "valuation",
        "direction": +1, "ff_factor": "HML", "msci_factor": "Value", "aqr_factor": "Value",
        "sources": [],
    },
    "cs_rank_ev_ebitda": {
        "canonical_name": "cs_rank_ev_ebitda_ttm",
        "category": "cs_rank", "sub_category": "valuation",
        "direction": -1, "ff_factor": "HML", "msci_factor": "Value", "aqr_factor": "Value",
        "sources": [],
    },
    "cs_rank_ps_ratio": {
        "canonical_name": "cs_rank_price_sales_ttm",
        "category": "cs_rank", "sub_category": "valuation",
        "direction": -1, "ff_factor": "HML", "msci_factor": "Value", "aqr_factor": "Value",
        "sources": [],
    },
    "cs_rank_debt_to_equity": {
        "canonical_name": "cs_rank_book_leverage_ratio",
        "category": "cs_rank", "sub_category": "quality",
        "direction": -1, "ff_factor": None, "msci_factor": "Quality", "aqr_factor": "Quality",
        "sources": [],
    },
    "cs_rank_current_ratio": {
        "canonical_name": "cs_rank_current_ratio",
        "category": "cs_rank", "sub_category": "quality",
        "direction": +1, "ff_factor": None, "msci_factor": "Quality", "aqr_factor": "Quality",
        "sources": [],
    },
    "cs_rank_rsi_14": {
        "canonical_name": "cs_rank_rsi_14",
        "category": "cs_rank", "sub_category": "technical",
        "direction": 0, "ff_factor": None, "msci_factor": None, "aqr_factor": None,
        "sources": [],
    },
    "cs_rank_adx_14": {
        "canonical_name": "cs_rank_adx_14",
        "category": "cs_rank", "sub_category": "technical",
        "direction": +1, "ff_factor": None, "msci_factor": None, "aqr_factor": None,
        "sources": [],
    },
    "cs_rank_bb_pos": {
        "canonical_name": "cs_rank_bollinger_band_position",
        "category": "cs_rank", "sub_category": "technical",
        "direction": 0, "ff_factor": None, "msci_factor": None, "aqr_factor": None,
        "sources": [],
    },
    "cs_rank_stoch_k": {
        "canonical_name": "cs_rank_stochastic_k",
        "category": "cs_rank", "sub_category": "technical",
        "direction": 0, "ff_factor": None, "msci_factor": None, "aqr_factor": None,
        "sources": [],
    },
    "cs_rank_skew_20d": {
        "canonical_name": "cs_rank_return_skewness_20d",
        "category": "cs_rank", "sub_category": "risk",
        "direction": +1, "ff_factor": None, "msci_factor": None, "aqr_factor": None,
        "sources": [],
    },
    "cs_rank_log_yen_volume": {
        "canonical_name": "cs_rank_log_trading_volume_jpy",
        "category": "cs_rank", "sub_category": "liquidity",
        "direction": +1, "ff_factor": None, "msci_factor": None, "aqr_factor": None,
        "sources": [],
    },
    "cs_rank_high_to_price": {
        "canonical_name": "cs_rank_nearness_to_52wk_high",
        "category": "cs_rank", "sub_category": "momentum",
        "direction": +1, "ff_factor": None, "msci_factor": "Momentum", "aqr_factor": None,
        "sources": [],
    },
    "cs_rank_obv_ret_1m": {
        "canonical_name": "cs_rank_obv_monthly_return",
        "category": "cs_rank", "sub_category": "technical",
        "direction": +1, "ff_factor": None, "msci_factor": None, "aqr_factor": None,
        "sources": [],
    },
    "cs_rank_max_ret_1m": {
        "canonical_name": "cs_rank_max_daily_return_1m",
        "category": "cs_rank", "sub_category": "risk",
        "direction": -1, "ff_factor": None, "msci_factor": None, "aqr_factor": None,
        "sources": [],
    },
    "cs_rank_reversal_1m": {
        "canonical_name": "cs_rank_short_term_reversal_1m",
        "category": "cs_rank", "sub_category": "reversal",
        "direction": -1, "ff_factor": "ST_Rev", "msci_factor": "Momentum", "aqr_factor": None,
        "sources": [],
    },
    "cs_rank_ret_1w": {
        "canonical_name": "cs_rank_weekly_return",
        "category": "cs_rank", "sub_category": "return",
        "direction": 0, "ff_factor": None, "msci_factor": None, "aqr_factor": None,
        "sources": [],
    },
    "cs_rank_ret_3m": {
        "canonical_name": "cs_rank_quarterly_return",
        "category": "cs_rank", "sub_category": "return",
        "direction": 0, "ff_factor": None, "msci_factor": None, "aqr_factor": None,
        "sources": [],
    },
    "cs_rank_ret_12m": {
        "canonical_name": "cs_rank_annual_return",
        "category": "cs_rank", "sub_category": "momentum",
        "direction": +1, "ff_factor": "Mom", "msci_factor": "Momentum", "aqr_factor": "Momentum",
        "sources": [],
    },
    "cs_rank_vol_60d": {
        "canonical_name": "cs_rank_realized_vol_60d",
        "category": "cs_rank", "sub_category": "volatility",
        "direction": -1, "ff_factor": None, "msci_factor": "Low Volatility", "aqr_factor": None,
        "sources": [],
    },
    "cs_rank_di_diff_14": {
        "canonical_name": "cs_rank_directional_indicator_diff",
        "category": "cs_rank", "sub_category": "technical",
        "direction": +1, "ff_factor": None, "msci_factor": None, "aqr_factor": None,
        "sources": [],
    },
    "cs_z_amihud_60d": {
        "canonical_name": "cs_zscore_amihud_illiquidity_60d",
        "category": "cs_rank", "sub_category": "liquidity",
        "direction": -1, "ff_factor": None, "msci_factor": None, "aqr_factor": None,
        "sources": [],
    },
    "cs_z_ret_1m": {
        "canonical_name": "cs_zscore_monthly_return",
        "category": "cs_rank", "sub_category": "return",
        "direction": +1, "ff_factor": None, "msci_factor": None, "aqr_factor": None,
        "sources": [],
    },
    "cs_z_ret_12m": {
        "canonical_name": "cs_zscore_annual_return",
        "category": "cs_rank", "sub_category": "momentum",
        "direction": +1, "ff_factor": "Mom", "msci_factor": "Momentum", "aqr_factor": "Momentum",
        "sources": [],
    },
    "cs_z_vol_20d": {
        "canonical_name": "cs_zscore_realized_vol_20d",
        "category": "cs_rank", "sub_category": "volatility",
        "direction": -1, "ff_factor": None, "msci_factor": "Low Volatility", "aqr_factor": None,
        "sources": [],
    },
    "cs_decile_ret_1m": {
        "canonical_name": "cs_decile_monthly_return",
        "category": "cs_rank", "sub_category": "return",
        "direction": +1, "ff_factor": None, "msci_factor": None, "aqr_factor": None,
        "sources": [],
    },
    "cs_decile_vol_20d": {
        "canonical_name": "cs_decile_realized_vol_20d",
        "category": "cs_rank", "sub_category": "volatility",
        "direction": -1, "ff_factor": None, "msci_factor": "Low Volatility", "aqr_factor": None,
        "sources": [],
    },
    "cs_decile_momentum_12_1": {
        "canonical_name": "cs_decile_price_momentum_12_1",
        "category": "cs_rank", "sub_category": "momentum",
        "direction": +1, "ff_factor": "Mom", "msci_factor": "Momentum", "aqr_factor": "Momentum",
        "sources": [],
    },
}

# ─── Flat name map (col_name → canonical_name) ───────────────────────────────
# This is the primary lookup used by normalize_features.py.

NORMALIZED_NAMES: dict[str, str] = {
    k: v["canonical_name"] for k, v in TAXONOMY.items()
}

# ─── Category groupings ───────────────────────────────────────────────────────
# Useful for selecting a feature block in ML pipelines.

CATEGORY_GROUPS: dict[str, list[str]] = {}
for _col, _meta in TAXONOMY.items():
    _cat = _meta["category"]
    CATEGORY_GROUPS.setdefault(_cat, []).append(_col)


# ─── Factor-family groupings ─────────────────────────────────────────────────
# Columns that map to the same FF5/MSCI/AQR factor.

FF_GROUPS: dict[str, list[str]] = {}
MSCI_GROUPS: dict[str, list[str]] = {}
AQR_GROUPS: dict[str, list[str]] = {}
for _col, _meta in TAXONOMY.items():
    if _meta["ff_factor"]:
        FF_GROUPS.setdefault(_meta["ff_factor"], []).append(_col)
    if _meta["msci_factor"]:
        MSCI_GROUPS.setdefault(_meta["msci_factor"], []).append(_col)
    if _meta["aqr_factor"]:
        AQR_GROUPS.setdefault(_meta["aqr_factor"], []).append(_col)


if __name__ == "__main__":
    print(f"FULL TAXONOMY AUDIT  —  {len(TAXONOMY)} features catalogued\n")
    for cat, cols in sorted(CATEGORY_GROUPS.items()):
        print(f"  {cat:<18} {len(cols):>3} features")
    print()
    print("FF5 factor coverage:")
    for ff, cols in sorted(FF_GROUPS.items()):
        print(f"  {ff:<12} {len(cols):>3} cols  eg. {cols[0]}")
    print()
    print("MSCI factor coverage:")
    for m, cols in sorted(MSCI_GROUPS.items()):
        print(f"  {m:<20} {len(cols):>3} cols")
    print()
    print("AQR factor coverage:")
    for a, cols in sorted(AQR_GROUPS.items()):
        print(f"  {a:<20} {len(cols):>3} cols")
