# NKY 225 — ML & Quantitative Techniques Research
## Deep Literature Review: Signal Generation · Feature Selection · Portfolio Optimisation · Risk

> Sources: Gu/Kelly/Xiu (RFS 2020), Lopez de Prado (2018), NBER WP 31502, arXiv, OR Spectrum, Journal of Portfolio Management, Pacific-Basin Finance Journal, SSRN

---

## Quick Reference — Recommended Pipeline

| Stage | Technique | Library | Key Paper |
|-------|-----------|---------|-----------|
| Feature selection | IC/ICIR ranking + SHAP | `alphalens`, `shap` | Grinold-Kahn (1999) |
| Validation | Purged K-Fold + CPCV | `mlfinlab` | Lopez de Prado (2018) |
| Primary signal | LightGBM LambdaRank | `lightgbm` | arXiv:2012.07149 |
| Secondary signal | Deep MLP (3 layers) | `pytorch` | arXiv:1801.01777 |
| Temporal signal | LSTM (12M lookback) | `pytorch` | arXiv:1901.11493 |
| Cross-stock signal | GNN (keiretsu graph) | `torch_geometric` | arXiv:1909.10660 |
| Ensemble blend | ICIR-weighted average | custom | Gu et al. (2020) |
| Alpha scaling | Grinold-Kahn | custom | Clarke et al. (2002) |
| Portfolio opt | QP active weights | `cvxpy` + `clarabel` | arXiv:2405.12762 |
| Covariance | BARRA factor + Ledoit-Wolf | `sklearn`, `riskfolio-lib` | Ledoit-Wolf (2022) |
| Evaluation | ICIR + DSR + PSR | `pyfolio`, `mlfinlab` | Bailey-LdP (2014) |

---

## Section A — Signal Generation / Prediction Models

### A1. Linear Models — Ridge, Lasso, ElasticNet

**What it is.** Penalised linear regressions that regress forward excess returns on a matrix of normalised factor exposures. Ridge shrinks all coefficients uniformly (L2); Lasso induces sparsity by zeroing weak predictors (L1); ElasticNet blends both. Standard baseline against which all ML methods are benchmarked.

**Implementation for cross-sectional equity.**
1. At each month-end, form an N×K matrix of cross-sectionally z-scored characteristics.
2. Stack monthly observations across an expanding training window.
3. Target: 1-month-ahead excess return (or its cross-sectional rank).
4. Cross-validate λ with Purged K-Fold (never standard K-Fold on time series).
5. Rank stocks by predicted score; use ranking for portfolio construction.

**Key hyperparameters.**

| Parameter | Range | Notes |
|-----------|-------|-------|
| λ (regularisation) | `[1e-5, 1e2]` log scale | Cross-validate with purged CV |
| ρ (L1/L2 mix, ElasticNet) | `[0, 1]` | 0 = Ridge, 1 = Lasso |
| Training window | 60–120 months | Expanding preferred |

**Empirical evidence.**
- Gu, Kelly & Xiu (2020, RFS): Lasso/Ridge are competitive on IC but dominated by GBM and neural nets by ~1–3% annualised return on long-short quintiles.
- OR Spectrum (2022) European 12,000-stock study: penalised linear models outperform OLS by ~0.3%/month but trail GBMs by ~0.3%/month.
- Nakagawa et al. (arXiv:2002.06975): linear OLS on 25 factors for MSCI Japan is the comparison baseline that deep learning beats.

**Japan notes.** Useful as an ensemble component or regime-specific signal where tree models overfit. Good interpretability for regulatory / stakeholder review.

**Libraries.** `sklearn.linear_model.{Ridge, Lasso, ElasticNet, LassoCV}`

**Papers.**
- Gu, S., Kelly, B., Xiu, D. (2020). "Empirical Asset Pricing via Machine Learning." *RFS* 33(5). SSRN 3159577.
- Hübler, O. (2022). "ML techniques for cross-sectional equity returns' prediction." *OR Spectrum.*
- Giglio, S., Kelly, B., Xiu, D. (2022). "Factor Models, ML, and Asset Pricing." *Annual Review of Financial Economics*, 14. SSRN 3943284.

---

### A2. Gradient Boosted Trees — LightGBM · XGBoost · CatBoost

**What it is.** GBMs build ensembles of shallow decision trees sequentially, each correcting residuals of the last. They handle non-linear feature interactions, are robust to irrelevant features, and work well on noisy-label tabular equity data. The two key loss variants for cross-sectional prediction are **regression** (predict return magnitude) and **ranking** (predict relative order).

**Regression vs. Ranking objectives.**

| Objective | Loss | Strength |
|-----------|------|----------|
| `MSE/MAE` (regression) | Mean squared/absolute error on returns | Directly predicts return magnitude |
| `LambdaRank` (ranking) | Pairwise ranking loss (NDCG optimised) | Learns ordering; up to 3× Sharpe vs. regression (arXiv:2012.07149) |
| `Huber` (robust regression) | L2 for small errors, L1 for large | Outlier-robust; preferred for Japan with extreme event days |

```python
import lightgbm as lgb

# LGBMRanker — learns cross-sectional ordering
model_rank = lgb.LGBMRanker(
    objective='lambdarank', n_estimators=500,
    learning_rate=0.03, num_leaves=63,
    subsample=0.8, colsample_bytree=0.7,
    reg_lambda=2.0, min_child_samples=30,
)

# LGBMRegressor with Huber loss — predicts return size
model_reg = lgb.LGBMRegressor(
    objective='huber', alpha=0.9,
    n_estimators=500, learning_rate=0.03,
    num_leaves=63, subsample=0.8,
    reg_lambda=2.0, min_child_samples=30,
)
```

**Era Splitting (arXiv:2309.14496 — DeLise 2023).** A critical adaptation for financial GBMs: modifies the tree-split criterion to maximise agreement of split decisions *across time eras* rather than pooled MSE. Equivalent to Invariant Risk Minimisation for trees. Prevents overfitting to specific historical market regimes. Published implementation integrates into scikit-learn's decision tree framework.

**Key hyperparameters.**

| Parameter | LightGBM | XGBoost | Notes |
|-----------|----------|---------|-------|
| `n_estimators` | 300–1000 | 300–1000 | Use early stopping |
| `learning_rate` | 0.01–0.05 | 0.01–0.05 | Lower = more robust |
| `max_depth` | 4–8 | 4–6 | Shallow trees preferred for equity |
| `num_leaves` | 15–127 | — | LightGBM specific |
| `subsample` | 0.6–0.9 | 0.6–0.9 | Row sampling per tree |
| `colsample_bytree` | 0.5–0.9 | 0.5–0.9 | Column sampling |
| `min_child_samples` | 20–100 | `min_child_weight` 5–50 | Prevents leaf overfitting |
| `reg_lambda` (L2) | 0.1–10 | 1–10 | Key regulariser for noisy data |

**Empirical evidence.**
- Gu, Kelly & Xiu (2020): GBMs are top performers with OOS R² ~0.4% for monthly US returns — meaningful given high equity noise.
- LightGBM multi-factor CSI300 study (ScienceDirect 2022): IC = 0.153, annualised return = 31.4%, Sharpe = 2.08.
- OR Spectrum (2022): GBMs dominate linear and RF models by ~0.1–0.3% per month.
- Kelly & Xiu (2023, NBER WP 31502): GBMs and NNs confirmed as state-of-the-art for cross-sectional equity.

**Japan notes.** High applicability — cross-sectional return prediction is structurally identical across markets. Include Japan-specific features: TSE governance disclosure scores, yen sensitivity beta, JPX investor-type flow signals, BOJ policy regime flags.

**Libraries.** `lightgbm`, `xgboost`, `catboost`, `optuna` (HPO)

**Papers.**
- DeLise, T. (2023). "Era Splitting: Invariant Learning for Decision Trees." arXiv:2309.14496.
- "Building Cross-Sectional Strategies by Learning to Rank." arXiv:2012.07149.
- Kelly, B., Xiu, D. (2023). "Financial Machine Learning." NBER WP 31502.

---

### A3. Random Forests

**What it is.** Ensemble of deep decision trees grown on bootstrapped subsamples and random feature subsets (bagging). Trees grown independently — higher variance reduction but lower bias reduction vs. GBMs. Naturally resistant to overfitting in high-noise settings.

**Key hyperparameters.**

| Parameter | Range |
|-----------|-------|
| `n_estimators` | 100–500 |
| `max_depth` | 6–15 |
| `max_features` | `"sqrt"` or 0.3–0.5 |
| `min_samples_leaf` | 10–50 |

**Feature importance.** Use Mean Decrease Accuracy (MDA / permutation importance) over Mean Decrease Impurity (MDI). Lopez de Prado (2018) shows MDI is biased toward high-cardinality features. For correlated factors (common in equity), use **clustered feature importance** — cluster correlated features first, then assess importance at the cluster level.

**Empirical evidence.** Gu et al. (2020) and OR Spectrum (2022) consistently find RF underperforms GBMs for equity return prediction. Primary use: providing **uncorrelated predictions** for ensemble stacking, and robust feature importance ranking for factor selection.

**Japan notes.** Handles missing data gracefully — relevant for Japanese small-caps with intermittent reporting. Good standalone baseline and ensemble diversifier.

**Libraries.** `sklearn.ensemble.RandomForestRegressor`, `sklearn.inspection.permutation_importance`

**Papers.**
- Gu, Kelly, Xiu (2020), RFS 33(5).
- Lopez de Prado, M. (2018). *Advances in Financial Machine Learning.* Wiley. (Chapter 8.)

---

### A4. Deep Neural Networks — MLP and ResNet-Style

**What it is.** Multi-layer perceptrons and residual-connection networks applied to the factor exposure matrix per stock. Learn smooth non-linear functions and latent factor structures better than tree models for large feature sets (>100 factors).

**Architecture from Gu, Kelly & Xiu (2020).**

| Model | Architecture | Notes |
|-------|-------------|-------|
| NN1 | 32 → output | 1 hidden layer |
| NN3 | 32 → 16 → 8 → output | 3 layers (recommended starting point) |
| NN5 | 32 → 16 → 8 → 4 → 2 → output | 5 layers |

- Activation: ReLU at all hidden layers
- Regularisation: L1 weight penalty + early stopping + batch normalisation
- Optimiser: Adam; SGD with momentum also works

**ResNet extension.** Skip connections stabilise training and allow deeper architectures (Feng, He & Polson 2018, arXiv:1804.09314).

**Target transformation (critical).** Rank-normalise cross-sectional returns before training:

```python
from scipy.stats import rankdata, norm

def rank_normalise(returns_per_date):
    """Map each stock's return to its cross-sectional normal quantile."""
    ranks = rankdata(returns_per_date)
    quantiles = ranks / (len(ranks) + 1)
    return norm.ppf(quantiles)  # maps to N(0,1) range
```

This eliminates market-regime effects from the target, stabilising gradients.

**Key hyperparameters.**

| Parameter | Range | Notes |
|-----------|-------|-------|
| Hidden layers | 2–5 | Deeper rarely helps for tabular equity |
| Hidden units | 32–512 | Pyramid (32→16→8) works well |
| Dropout | 0.1–0.5 | After each hidden layer |
| L2 weight decay | `1e-5` to `1e-3` | Key regulariser |
| Batch size | 512–4096 | Larger = more stable gradients |
| Learning rate | `1e-4` to `1e-2` | Adam; cosine annealing schedule |
| Early stopping patience | 10–20 epochs | Monitor validation IC |

**Empirical evidence.** Gu et al. (2020): NN3–NN5 competitive with GBMs; OR Spectrum (2022): NNs yield ~0.6%/month improvement vs. OLS.

**Japan notes.** Abe & Nakayama (arXiv:1801.01777): deep MLP on MSCI Japan (100+ features, monthly) confirms deep > shallow > linear. Abe & Nakagawa (arXiv:2002.06975): profitable in Japan cross-section with next-day-open execution.

**Libraries.** `pytorch`, `pytorch-lightning`, `skorch`, `tensorflow/keras`

**Papers.**
- Gu, Kelly, Xiu (2020). RFS 33(5).
- Abe, M., Nakayama, H. (2018). arXiv:1801.01777.
- Abe, M., Nakagawa, K. (2020). arXiv:2002.06975.
- Feng, G., He, J., Polson, N. (2018). arXiv:1804.09314.

---

### A5. LSTM / Recurrent Models

**What it is.** Long Short-Term Memory networks extend feedforward NNs to handle sequential input. In cross-sectional alpha: input = stock's K-dimensional factor vector at each month; final hidden state predicts next-period return. Captures time-varying factor dynamics that a static cross-sectional model misses.

**Deep Recurrent Factor Model (arXiv:1901.11493 — Nakagawa et al. 2019).** Replaces constant factor loadings in a linear multi-factor model with an LSTM-learned time-varying function. Layer-wise Relevance Propagation (LRP) linearises LSTM predictions post-hoc into an interpretable "factor model with time-varying loadings." Directly applied to the Japanese stock market — the most relevant architecture for this project.

```python
import torch.nn as nn

class StockLSTM(nn.Module):
    def __init__(self, input_size=50, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=num_layers, dropout=dropout, batch_first=True
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):           # x: (batch, seq_len, features)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])   # last timestep prediction
```

**Key hyperparameters.**

| Parameter | Recommended |
|-----------|-------------|
| Sequence length (lookback) | 12–24 months (capture Japan annual seasonality) |
| Hidden size | 32–256 |
| LSTM layers | 1–3 |
| Dropout | 0.1–0.4 |
| Learning rate | `1e-3` to `1e-4` (Adam) |

**Empirical evidence.**
- Nakagawa et al. (arXiv:1901.11493): LSTM with LRP outperforms linear and MLP baselines on Japanese equities.
- Chinese A-share cross-sectional study (2024): LSTM/Transformer deliver superior accuracy and stronger tail-risk control.
- GBM + LSTM hybrid (arXiv:2505.23084): 10–15% improvement in predictive accuracy vs. single models.

**Japan notes.** LSTM is particularly suited to Japan because BOJ policy regime changes (Abenomics 2013, YCC introduction 2016, YCC abandonment 2024) create time-varying factor dynamics that LSTMs capture. Use 12–24 month lookback to cover one fiscal year cycle without excessive stale data.

**Libraries.** `pytorch`, `tensorflow/keras`

**Papers.**
- Nakagawa, K. et al. (2019). "Deep Recurrent Factor Model." arXiv:1901.11493.
- Nakagawa, K., Abe, M. (2020). arXiv:2002.06975.

---

### A6. Transformers / Attention Models

**What it is.** Transformer architectures apply self-attention to learn which time steps and features to weight. For cross-sectional equity: (a) **temporal self-attention** selects informative historical periods; (b) **cross-stock attention** (MASTER) captures inter-stock correlations.

**MASTER (arXiv:2312.15235 — AAAI 2024).** Market-guided Stock TransformER — five-step architecture:
1. Market-guided gating: weights feature importance using market-level macro information.
2. Intra-stock aggregation: temporal self-attention over a stock's history.
3. Inter-stock aggregation: cross-sectional attention across stocks.
4. Temporal aggregation: summarises to a prediction vector.
5. Prediction head: linear → return score.

Explicitly models "momentary and cross-time stock correlations" and allows market conditions to dynamically weight factors — addressing the time-varying factor efficacy problem in Japan.

**MCI-GRU (arXiv:2410.20679).** Multi-head cross-attention + improved GRU. Trains on 60 prior trading days to predict return rankings over next 21 days.

**Key hyperparameters.**

| Parameter | Range |
|-----------|-------|
| Attention heads | 4–16 |
| `d_model` (embed dim) | 64–512 |
| Encoder layers | 2–6 |
| Sequence length | 20–60 time steps |
| Dropout | 0.1–0.3 |
| Optimizer | AdamW with cosine annealing |

**Empirical evidence.** Autoformer-based transformer outperforms simple NNs at 1-, 3-, and 12-month horizons (ScienceDirect 2025). MASTER demonstrates superior stock price forecasting. Kelly & Xiu (2023) note transformers do not uniformly dominate well-tuned GBMs on tabular equity data — excel most when temporal and cross-stock correlations matter.

**Japan notes.** Cross-stock attention captures Nikkei 225's tight sector concentrations (auto, tech hardware, industrials). Computationally expensive — GPU required for training.

**Libraries.** `pytorch`, `huggingface/transformers`, SJTU-DMTai/MASTER (GitHub)

**Papers.**
- MASTER: arXiv:2312.15235.
- MCI-GRU: arXiv:2410.20679.
- Kelly, B., Xiu, D. (2023). NBER WP 31502.

---

### A7. Graph Neural Networks (GNNs)

**What it is.** GNNs model stocks as nodes in a graph where edges represent relationships (sector membership, supply chain, return correlation). Each node aggregates information from its neighbours before generating a prediction — directly capturing cross-stock information propagation.

**Matsunaga et al. (arXiv:1909.10660 — 2019).** The seminal GNN-for-stocks paper; directly applied to Nikkei 225 companies using inter-company knowledge graphs including supplier relationships. 20-year rolling backtest. **Result: 29.5% improvement in return ratio; 2.2× Sharpe ratio vs. market benchmark.**

**Graph construction options for Nikkei 225.**

| Graph Type | Edges | Motivation |
|------------|-------|-----------|
| Sector graph | Stocks in same TOPIX sector (33 sectors) | Sector co-movement |
| Correlation graph | K-nearest by historical return correlation | Data-driven proximity |
| Keiretsu graph | Cross-shareholding relationships | Japan-specific; dense signal |
| Supply-chain graph | Supplier–customer links from filings | Earnings propagation |

```python
import torch
from torch_geometric.nn import GCNConv

class StockGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden, out=1):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.fc    = torch.nn.Linear(hidden, out)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        return self.fc(x)
```

**Japan notes.** Japan's keiretsu structures (Toyota → Aisin, Denso, Toyota Industries; Mitsubishi → cross-holding web) make GNNs especially well-motivated. Construct keiretsu adjacency matrix from TSE large-shareholder filings (大量保有報告書), available free from EDINET.

**Libraries.** `torch_geometric`, `dgl` (Deep Graph Library), `stellargraph`

**Papers.**
- Matsunaga, D. et al. (2019). arXiv:1909.10660.
- Hybrid Transformer-GNN (arXiv:2601.04602).

---

### A8. Hybrid Ensembles

**What it is.** Combining predictions from heterogeneous model classes to produce a blended alpha score. Different families capture different aspects: GBMs = tabular cross-sectional patterns; LSTMs = time-series dynamics; GNNs = cross-stock propagation.

**Blending methods.**

| Method | Description | Use when |
|--------|-------------|----------|
| Simple average | Mean of normalised model z-scores | Default; robust; no look-ahead |
| ICIR-weighted | Weight each model by recent ICIR | Models have meaningfully different ICs |
| Stacking (Ridge meta-learner) | Ridge on model outputs | Enough OOS predictions for meta-training |
| Bayesian model averaging | Posterior weights over models | Formal uncertainty quantification |

**Implementation note.** Cross-sectionally z-score each model's output before blending. The blended score then has consistent scale for portfolio construction.

**Empirical evidence.** GBM + LSTM hybrid: 10–15% improvement over individual models (arXiv:2505.23084). RF + GBM + DNN ensemble significantly improves equity return prediction across international markets.

**Libraries.** `mlflow` (tracking), `sklearn.ensemble.VotingRegressor`, custom stacking

**Papers.**
- arXiv:2505.23084 (GBDT + LSTM).
- arXiv:2507.07107 (ML enhanced multi-factor quantitative trading).

---

## Section B — Feature Selection & Dimensionality Reduction

### B1. IC-Based Univariate Ranking (IC / ICIR)

**Formulae.**

```
IC_t    = Spearman( α_{i,t} , r_{i,t+1} )   across all stocks i at date t
ICIR    = mean(IC_t) / std(IC_t)             across all dates t
```

**Thresholds.**

| ICIR | Interpretation |
|------|---------------|
| < 0.1 | Noise level — exclude |
| 0.2–0.4 | Weak — use only in ensemble |
| 0.5–0.8 | Acceptable standalone signal |
| > 1.0 | Strong (rare in equity) |

**IC decay analysis.** Plot IC(h) vs. holding horizon h to determine factor half-life and rebalancing frequency. Momentum decays fast (6–9 months); value decays slowly (12–36 months). Use IC decay to set training target horizon and rebalance cadence.

**Libraries.** `alphalens-reloaded`, `scipy.stats.spearmanr`

**Papers.** Grinold, R., Kahn, R. (1999). *Active Portfolio Management.* McGraw-Hill.

---

### B2. Purged Walk-Forward Cross-Validation (Lopez de Prado)

**The problem.** Standard K-Fold on time series causes leakage: training and test observations separated by less than the prediction horizon overlap in their forward return windows.

**Solution.** Purge training observations that temporally overlap with the test fold, then add an embargo period:

```
For each test fold:
  1. PURGE:   Remove training obs within prediction_horizon days of any test obs
  2. EMBARGO: Additionally remove next embargo_size days after the test fold
  3. Train on purged set → evaluate on test fold
```

**Recommended settings.**

| Parameter | Value |
|-----------|-------|
| K (folds) | 5–10 |
| Purge | 1 prediction horizon (e.g., 21 days for monthly returns) |
| Embargo | 5 trading days |

```python
from mlfinlab.cross_validation import PurgedKFold

cv = PurgedKFold(n_splits=5, n_embargo=5)
for train_idx, test_idx in cv.split(X, pred_times=times, eval_times=times):
    ...
```

**Libraries.** `mlfinlab` (Hudson & Thames), `fold`

**Papers.**
- Lopez de Prado, M. (2018). *Advances in Financial Machine Learning.* Wiley. Ch. 7.
- Bailey, D., Lopez de Prado, M. (2014). SSRN 2326253.

---

### B3. SHAP Values for Interpretable Feature Importance

**What it is.** Decomposes each model prediction into per-feature Shapley-value contributions. For equity ML: (a) global feature ranking by mean |SHAP|; (b) feature interaction detection; (c) time-period-specific explanations for individual predictions.

**Advantages over MDI/MDA.** SHAP is model-agnostic, consistent (a feature always gets a higher value than one that contributes less), and unbiased toward high-cardinality features (a key MDI failure in tree models).

```python
import shap

explainer   = shap.TreeExplainer(lgbm_model)
shap_values = explainer.shap_values(X_test)

# Global feature ranking
mean_abs_shap = pd.Series(
    np.abs(shap_values).mean(axis=0),
    index=feature_names
).sort_values(ascending=False)

# Production monitoring: track SHAP contributions monthly
# Alert if a high-weight factor's mean SHAP collapses
```

**Production use.** Monitor monthly SHAP contributions as a regime-change detector — if momentum SHAP values collapse, the model is rotating away from momentum and may be detecting a regime shift.

**Libraries.** `shap`

**Papers.**
- Lundberg, S., Lee, S.-I. (2017). "A Unified Approach to Interpreting Model Predictions." NeurIPS 2017. arXiv:1705.07874.
- Lopez de Prado (2018). Ch. 8.

---

### B4. PCA / IPCA (Instrumented PCA)

**Standard PCA.** Extracts orthogonal principal components from the return covariance matrix. Statistical factors with no economic labelling. Useful for covariance estimation (see F3) rather than direct alpha generation.

**IPCA — Kelly, Pruitt & Su (2019, JFE).** *Characteristics Are Covariances.* Allows factor loadings to be time-varying and instrumented by stock characteristics:

```
r_{i,t+1} = β_{i,t}' f_{t+1} + ε_{i,t+1}
β_{i,t}   = Γ' z_{i,t}       (loadings = linear function of characteristics)
```

5 IPCA factors explain the US equity cross-section significantly better than Fama-French 5F or HXZ. Most anomaly returns become insignificant after IPCA adjustment — they are factor risk compensation, not alpha.

```python
from ipca import InstrumentedPCA

ipca = InstrumentedPCA(n_factors=5, intercept=False)
Gamma, Factors = ipca.fit_transform(X=characteristics, y=returns)
```

**Japan notes.** IPCA applied to Japanese equities would identify latent factors driven by yen sensitivity, TSE governance score, BOJ policy exposure — an underexplored research direction.

**Libraries.** `ipca` (Kelly's implementation), `sklearn.decomposition.PCA`

**Papers.** Kelly, B., Pruitt, S., Su, Y. (2019). JFE 134(3). SSRN 2983919.

---

### B5. Mutual Information and mRMR

**What it is.** Mutual Information (MI) measures reduction in target uncertainty given a feature, capturing non-linear dependencies that linear IC misses. Features with high MI but low linear IC indicate non-linear predictive power.

**mRMR (minimum Redundancy Maximum Relevance).** Selects features that maximise relevance to the target while minimising redundancy with each other — addresses the correlated factor problem.

```python
from sklearn.feature_selection import mutual_info_regression

mi_scores = mutual_info_regression(X_train, y_train, random_state=42)
mi_series = pd.Series(mi_scores, index=feature_names).sort_values(ascending=False)
```

**Libraries.** `sklearn.feature_selection.mutual_info_regression`, `pymrmr`, `minepy`

---

## Section C — Model Training & Validation Methodology

### C1. Walk-Forward Expanding Window

**Standard approach.** Train on all data from t₀ to T; test on T → T+h; roll forward. Expanding window preferred (more training data, long-run regime information valuable). Rolling window used when structural breaks make old data counterproductive.

**Typical configuration for monthly alpha.**
- Training: months 1 → T (expanding)
- Test: month T+1 only (roll monthly), or T+1 → T+12 (annual refit)
- For Nikkei 225: initial training period ≥ 36 months; OOS test from 2017 onward

---

### C2. Combinatorial Purged Cross-Validation (CPCV)

**What it is.** Generates thousands of train/test splits using combinatorial construction so every part of the series is tested multiple times. Provides a distribution of backtest outcomes rather than a single historical path — enabling computation of Probability of Backtest Overfitting (PBO) and Deflated Sharpe Ratio.

**Algorithm.**
1. Divide time series into T subsets.
2. For each combination of k test subsets (from T), train on T−k subsets with purging/embargo.
3. Aggregate test predictions across all combinations → backtest path distribution.
4. PBO = fraction of paths with negative OOS Sharpe.

**Interpretation.** If PBO < 5%, the strategy is likely genuine. If PBO > 50%, the backtest Sharpe is mostly luck.

**Libraries.** `mlfinlab.cross_validation.CombinatorialPurgedKFold`

**Papers.** Lopez de Prado (2018). Bailey, Lopez de Prado (2014). SSRN 4778909.

---

### C3. Bayesian Hyperparameter Optimisation (Optuna)

**What it is.** Uses Tree-structured Parzen Estimators (TPE) to model the objective function (validation ICIR) and propose the next hyperparameter setting that maximises expected improvement — far more efficient than grid or random search.

**Key guidance.** Optimise on ICIR (mean IC / std IC), not raw IC — penalises high-variance parameter settings that produce volatile signals.

```python
import optuna

def objective(trial):
    params = {
        'n_estimators':    trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate':   trial.suggest_float('lr', 0.01, 0.1, log=True),
        'max_depth':       trial.suggest_int('max_depth', 3, 8),
        'num_leaves':      trial.suggest_int('num_leaves', 15, 127),
        'subsample':       trial.suggest_float('subsample', 0.5, 1.0),
        'reg_lambda':      trial.suggest_float('lambda', 0.1, 10.0, log=True),
    }
    model = lgb.LGBMRegressor(**params)
    icir = purged_cv_icir(model, X_train, y_train)  # use purged CV
    return icir

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=200)
```

**Libraries.** `optuna`, `hyperopt`, `ray[tune]`

**Papers.** Akiba, T. et al. (2019). "Optuna." KDD 2019.

---

### C4. Regularisation Strategies

| Strategy | Application | Finance Notes |
|----------|-------------|--------------|
| L2 (Ridge / weight decay) | NN, linear models | Most common; prevents large weights |
| L1 (Lasso) | Linear, NN | Sparsity; feature selection |
| Dropout | NN hidden layers | Rate 0.2–0.4; variational for uncertainty |
| Batch normalisation | NN training | Between linear and activation layers |
| Early stopping | All NN models | Patience 10–20 epochs; monitor val IC |
| Gradient clipping | LSTM, Transformer | Clip norm ≤ 1.0–5.0 |

---

### C5. Target Transformation

**Cross-sectional rank normalisation (recommended).**

```python
from scipy.stats import rankdata, norm

def cs_rank_normalise(ret_series):
    """Map returns → cross-sectional N(0,1) quantiles per date."""
    ranks     = rankdata(ret_series)
    quantiles = ranks / (len(ranks) + 1)
    return norm.ppf(quantiles)      # clips to approx ±3 range
```

Eliminates market-wide return level from training target. Alternative: excess return over equal-weighted universe mean (retains distribution shape information).

---

## Section D — Alpha / Signal Combination

### D1. Grinold-Kahn Fundamental Law of Active Management

**Core formula.**

```
IR = IC × √BR                       (basic form)
IR = TC × IC × √BR                  (with portfolio constraints)

α_i = IC × σ_i × score_i           (Grinold-Kahn alpha scaling)
```

Where:
- **IC** = Information Coefficient (signal skill per period)
- **BR** = Breadth (independent bets per year; NKY 225 monthly = 225 × 12 = 2,700)
- **TC** = Transfer Coefficient (0–1; fraction of IC surviving constraints)
- `score_i` = cross-sectionally z-scored ML model output
- `σ_i` = annualised volatility of stock i

**Nikkei 225 example.** IC = 0.05, TC = 0.6, BR = 2,700:
`IR = 0.6 × 0.05 × √2700 ≈ 1.56` — competitive institutional-grade IR.

**Use in pipeline.** The Grinold-Kahn formula converts dimensionless ML scores into expected annual return estimates in the same units as the QP optimiser's objective, properly scaled by each stock's risk (σ_i).

**Papers.**
- Grinold, R., Kahn, R. (1999). *Active Portfolio Management* (2nd ed.). McGraw-Hill.
- Clarke, R., de Silva, H., Thorley, S. (2002). "Portfolio Constraints and the Fundamental Law." *FAJ* 58(5).

---

### D2. Signal Blending Methods

| Method | Formula | Strength |
|--------|---------|---------|
| Equal-weight | mean(z-score_model_k) | Robust; no look-ahead; default |
| ICIR-weighted | Σ ICIR_k × z_k / Σ ICIR_k | Adapts to relative signal strength |
| Ridge meta-learner | Ridge(y = signals, target = return) | Learns combination weights from data |
| Isotonic regression | Monotone step function on combined score | Preserves ordinal structure |

**Bayesian shrinkage of ICIR estimates.** Observed ICIR over a finite window is noisy. Shrink toward cross-sectional mean ICIR with prior variance ∝ 1/T_obs. Prevents over-weighting signals with short high-IC run.

---

### D3. Regime-Switching Signal Weighting

**What it is.** Different signals perform well in different market regimes. Detect regime and adjust signal weights dynamically.

**Implementation.**
1. Classify market regimes with HMM or GMM on macro variables (VIX, JGB yield change, earnings revision breadth, USD/JPY trend).
2. Train a GBM classifier to predict regimes from macro features.
3. For each regime, use pre-calibrated IC-weighted signal weights.
4. Final alpha = Σ P(regime_k) × alpha_k.

**Japan regimes.**

| Regime | Characteristics | Favoured signals |
|--------|----------------|-----------------|
| BOJ easing / QQE | Falling JGB yields; yen weakening | Momentum, growth, exporter beta |
| BOJ tightening / YCC exit | Rising JGB yields | Value, domestic, bank stocks |
| Global risk-off (VIX > 25) | Yen strengthening; Nikkei falling | Low-vol, quality, domestic |
| Reflation | Rising CPI + wages | Domestic consumption, financials |

**Papers.**
- "Dynamic Factor Allocation Leveraging Regime-Switching Signals." arXiv:2410.14841.
- "Constructing Equity Investment Strategies Using Analyst Reports and Regime Switching." *Frontiers in AI* (2022).

---

## Section E — Portfolio Optimisation

### E1. Mean-Variance Optimisation (MVO)

**Standard QP formulation (active weight version).**

```
min   h' Σ h  −  λ · α' h
s.t.  1' h = 0                          (active-neutral: long = short vs bench)
      w_bench + h ≥ 0                   (long-only: no naked shorts)
      (w_bench + h)' Σ (w_bench + h) ≤ TE²   (tracking error budget)
      |h_i| ≤ max_active_weight         (position limits)
      |Δh|_1 ≤ turnover_budget          (turnover constraint)
```

Where h = w − w_bench (active weight), Σ = covariance matrix, α = Grinold-Kahn expected returns, TE = tracking error target.

**Practical problems with raw MVO.** Maximises estimation error in expected returns → extreme weights. Solution: (1) Black-Litterman blending; (2) covariance shrinkage; (3) robust optimisation; (4) explicit turnover constraints.

**Libraries.** `cvxpy`, `riskfolio-lib`, `pypfopt`

**Papers.** Markowitz, H. (1952). *Journal of Finance* 7(1).

---

### E2. Black-Litterman Model

**What it is.** Bayesian blend of CAPM equilibrium returns (from market cap weights) and alpha model views. Prevents extreme weight concentration from naive MVO.

```
μ_BL = [(τΣ)⁻¹ + P'Ω⁻¹P]⁻¹ × [(τΣ)⁻¹ μ_prior + P'Ω⁻¹q]
```

Where:
- `μ_prior` = CAPM reverse-optimised from benchmark weights
- `P, q` = pick matrix and view returns (ML model predictions)
- `Ω` = view uncertainty (calibrate: `Ω_ii ∝ 1/IC_signal²` — high-IC signal gets low uncertainty)
- `τ` ≈ 1/T

**Japan implementation.** Reverse-engineer equilibrium returns from JPY121 ETF (1321.T) price-weighted index constituents. View uncertainty Ω calibrated to historical ICIR of each signal type (Ranker vs. Huber model).

**LLM extension.** arXiv:2504.14345 — use LLM-extracted views from earnings calls and news as additional P-q pairs.

**Libraries.** `pypfopt.BlackLittermanModel`, `riskfolio-lib`

**Papers.** Black, F., Litterman, R. (1992). *FAJ* 48(5). arXiv:2504.14345.

---

### E3. Robust Portfolio Optimisation

**What it is.** Explicitly accounts for uncertainty in μ and Σ by optimising worst-case performance within an uncertainty ellipsoid:

```
max   min_{μ ∈ U} μ'w  −  (1/2) w'Σw
```

Equivalent to standard MVO with μ penalised by √(w' Ω_μ w) — positions in uncertain stocks are penalised. Transforms to a tractable SOCP.

**Goldfarb-Iyengar factor model extension.** Uncertainty set on factor loadings B. Retains QP tractability and integrates naturally with BARRA-style risk models.

**Libraries.** `cvxpy` (SOCP constraints), `riskfolio-lib` (robust methods built-in)

**Papers.** Fabozzi, F. et al. (2007). *Robust Portfolio Optimization and Management.* Wiley. arXiv:2103.13806.

---

### E4. QP Solvers Comparison

| Solver | Type | Speed | Python Interface | Best for |
|--------|------|-------|-----------------|----------|
| **CLARABEL** | Interior point | Very fast | `cvxpy`, `qpsolvers` | Monthly rebalance; new CVXPY default |
| **OSQP** | ADMM | Fast; warm-starts | `osqp`, `cvxpy` | Daily monitoring; warm-starting 10× speedup |
| **Gurobi** | Interior point | Fastest | `gurobipy` | Mixed-integer QP; commercial |
| **MOSEK** | Interior point | Very fast | `mosek`, `cvxpy` | SDP/SOCP; commercial |
| **SCS** | ADMM | Moderate | `scs`, `cvxpy` | Large-scale SDPs |
| **CVXOPT** | Interior point | Moderate | `cvxopt` | Pure Python; N < 200 |

**Recommendation for Nikkei 225 (N=225).** OSQP for daily monitoring (warm-starting from prior solution); CLARABEL for monthly rebalance. Both are open-source and integrate with `cvxpy`.

**Papers.** Goulart, P., Chen, Y. (2024). "Clarabel." arXiv:2405.12762.

---

### E5. Hierarchical Risk Parity (HRP)

**What it is.** Lopez de Prado (2016) — graph theory + hierarchical clustering, no matrix inversion required. Steps:
1. Compute correlation matrix → distance matrix: `d_ij = √(2(1 − ρ_ij))`
2. Hierarchical clustering → dendrogram.
3. Recursive bisection: inverse-variance weights within clusters; proportional across clusters.

**Advantages.**
- Works even when Σ is singular or ill-conditioned.
- More stable OOS than MVO (original paper shows HRP beats CLA and 1/N).
- Naturally diversified across the correlation structure.

**Japan application.** HRP is a viable standalone allocator for the Nikkei 225 (N=225 is small enough that MVO works, but HRP useful as a robustness check and for sectors with high intra-group correlation — e.g., auto-parts keiretsu).

**Libraries.** `riskfolio-lib`, `pypfopt.HRPOpt`, `mlfinlab`

**Papers.** Lopez de Prado, M. (2016). *JPM* 42(4). SSRN 2708678.

---

### E6. Transaction Cost Model

**Linear TC (recommended for Nikkei 225 large-caps).**

```
TC_total = Σ_i  c_i × |Δw_i|
```

Where `c_i` ≈ 5–15bps round-trip for Nikkei 225 large-caps (OQ Funds: "exceptionally low bid-ask spreads ~1bp brokerage"). Add as linear penalty to QP objective:

```
min   h'Σh  −  α'h  +  λ_tc × Σ_i c_i |Δh_i|
```

**Turnover constraint (preferred approach).** Hard-constrain one-way monthly turnover ≤ 20% rather than soft TC penalty. More transparent and easier to monitor in production.

---

## Section F — Risk / Covariance Estimation

### F1. Why Sample Covariance Fails

For N=225 stocks and T monthly observations, you need T > N for the sample covariance to be invertible. With 5 years of data (T=60): the noise eigenvalue range per Marchenko-Pastur law is `[(1−√(225/60))², (1+√(225/60))²] ≈ [0.04, 7.3]` — nearly all eigenvalues are in the noise band. **Shrinkage or factor models are mandatory for N≥50.**

---

### F2. Ledoit-Wolf Shrinkage

**Formula.**

```
Σ̂ = (1 − α) S  +  α T
```

Where S = sample covariance, T = shrinkage target (scaled identity or constant-correlation), and α is the analytically optimal shrinkage intensity.

**Versions.**

| Version | Target | Notes |
|---------|--------|-------|
| Ledoit-Wolf (2004) | Scaled identity | Closed-form α; linear shrinkage |
| OAS (Oracle Approx.) | Scaled identity | Improved finite-sample α estimate |
| Nonlinear / Goldilocks (2022) | None (spectrum-specific) | Best for N>100; nonlinear eigenvalue cleaning |

**Empirical performance.** Ledoit-Wolf consistently outperforms sample covariance for portfolio optimisation; nonlinear (2022 Goldilocks) dominates linear version for N>100.

**Libraries.** `sklearn.covariance.{LedoitWolf, OAS}`

**Papers.** Ledoit, O., Wolf, M. (2004). JMVA 88(2). Ledoit, O., Wolf, M. (2022). *RFS.*

---

### F3. Factor-Based Risk Models (BARRA-Style)

**Structure.** Decompose returns into factor component + idiosyncratic residual:

```
r_i = B_i' f + ε_i        →      Σ = B F B' + D
```

Where B = N×K factor exposure matrix, F = K×K factor covariance, D = diagonal specific risk matrix. Reduces estimation from N(N+1)/2 = 25,650 parameters to K(K+1)/2 + N ≈ 675 (with K=30 factors for N=225).

**Construct an open-source BARRA-style model for Nikkei 225.**

| Factor Group | Variables |
|-------------|-----------|
| Market | Nikkei 225 market return |
| Industry (33) | TOPIX 33 sector dummies |
| Size | Log market cap |
| Value | B/P, EBITDA/EV |
| Momentum | 12-1M return, analyst revision |
| Quality | GPA, ROE, accruals |
| Volatility | 12M realised vol |
| Yen sensitivity | Rolling USD/JPY beta |

**Estimate monthly via cross-sectional GLS.** Specific risk D estimated from trailing 24-month residual volatility with exponential weighting.

**Libraries.** `statsmodels`, `riskfolio-lib`, `pandas`

---

### F4. DCC-GARCH for Dynamic Covariance

**What it is.** Dynamic Conditional Correlation GARCH (Engle 2002) — models time-varying correlations:

```
H_t = D_t R_t D_t
Q_t = (1−a−b) Q̄  +  a z_{t−1} z'_{t−1}  +  b Q_{t−1}
R_t = diag(Q_t)^{−1/2} Q_t diag(Q_t)^{−1/2}
```

Where D_t contains univariate GARCH volatilities and R_t is the dynamic correlation matrix.

**For N=225:** Use DCC-NL (DCC + nonlinear shrinkage on Q̄) or DECO (Dynamic Equicorrelation — imposes a single common correlation at each date) for tractability.

**Deep Learning extension (arXiv:2506.02796).** LSTM to estimate DCC correlation dynamics — improves flexibility and forecast accuracy.

**Libraries.** `arch` (Kevin Sheppard), `rpy2` + `rmgarch` (R interface)

**Papers.** Engle, R. (2002). *JBES* 20(3). arXiv:2506.02796.

---

### F5. Industry Equi-Correlation (IEC)

**What it is (Giller 2024, arXiv:2411.08864).** Isotropic correlation model: all pairwise correlations within a sector = ρ_within, across sectors = ρ_between. Block-equicorrelation structure:

```
Σ_IEC = σ² [ (1−ρ_within)I + ρ_within (11' within each sector) + ρ_between (cross-sector terms) ]
```

Has a closed-form inverse and eigendecomposition — computationally very efficient for N=225. Morimoto, Akama & Kawasaki (2026) apply this structure to Japanese equities and show it substantially improves model performance over Fama-French approaches.

**Libraries.** Custom (simple matrix construction with `numpy`).

**Papers.** Giller, G. (2024). arXiv:2411.08864. Morimoto, T. et al. (2026). *Explaining the Cross-Section of Japanese Equity Returns Using IEC.*

---

### F6. Non-Negative Matrix Factorisation (NMF)

**What it is.** Factors return history X (T×N) into W (T×K factors) and H (K×N loadings) with W≥0, H≥0 constraints. Non-negativity ensures interpretable additive factors — loadings represent "how much" a stock belongs to each factor, not positive/negative bets.

**Risk budget with NMF (arXiv:2204.02757).**
1. Extract K NMF factors from returns → near-orthogonal, interpretable factors.
2. Inverse-variance allocate at factor level.
3. Back-transform to stock weights.

Outperforms HRP and classical allocations on diversification metrics.

**Libraries.** `sklearn.decomposition.NMF`

**Papers.** arXiv:2204.02757.

---

## Section G — Backtesting & Evaluation

### G1. Core Metrics at a Glance

| Metric | Formula | Target (NKY 225 L/S) |
|--------|---------|---------------------|
| IC | Spearman(score, return) | > 0.03 |
| ICIR | mean(IC) / std(IC) | > 0.5 |
| CAGR | (Final/Initial)^(1/T) − 1 | > 10% active |
| Sharpe Ratio | (CAGR − Rf) / σ | > 1.5 |
| Calmar Ratio | CAGR / Max Drawdown | > 2.0 |
| Max Drawdown | Max peak-to-trough | < 15% |
| Turnover (1-way) | Monthly weight change | < 20%/month |
| Transfer Coefficient | Corr(h_model, h_actual) | > 0.5 |

---

### G2. Deflated Sharpe Ratio (DSR)

**The problem.** An observed backtest SR over T months may be due to chance, especially if many strategies were tested. DSR corrects for: (1) non-normality of returns (skewness, kurtosis); (2) selection bias from multiple testing.

**Formula.**

```
PSR(SR*) = Φ[ (SR√T × (1 − γ₃SR + (γ₄−1)/4 × SR²)) / √(1 − γ₃SR + γ₃²SR²/4) ]
```

Where γ₃ = skewness, γ₄ = excess kurtosis. DSR then applies a Bonferroni-style correction for N_trials independent strategies tested.

**Minimum SR for significance.** For 50 independent strategy variants, with realistic return distribution: **SR > 1.2–1.5** required at 95% confidence (vs. naive threshold of SR > 0.65 × √(T/12)).

**Libraries.** `pyfolio`, `mlfinlab.backtest_statistics.deflated_sharpe_ratio`

**Papers.** Bailey, D., Lopez de Prado, M. (2014). *JPM* 40(5). SSRN 2460551.

---

### G3. Transfer Coefficient (TC)

**What it is.** Measures how much IC survives the portfolio construction step:

```
TC = Corr(h_model , h_portfolio)
```

Where h_model = unconstrained optimal active weights from the alpha signal; h_portfolio = actual constrained weights.

**Typical values.**

| Portfolio Type | Typical TC |
|---------------|-----------|
| Unconstrained | 1.0 |
| Long-short, moderate constraints | 0.5–0.8 |
| Long-only index-relative | 0.3–0.5 |

Active value added = TC × IC × σ_P × √BR. TC quantifies the cost of every constraint — use it to decide which constraints are worth keeping.

---

### G4. Multiple Testing Corrections

**Harvey, Liu & Zhu (2016, RFS).** Minimum t-ratio thresholds for factor significance given multiple testing:

| Era | Minimum |t-stat| |
|-----|---------|
| Pre-2000 (single tests) | 2.0 |
| 2000–2012 (Bonferroni era) | 3.0 |
| Post-2012 (BHY FDR 5%) | 3.18 |

**Bonferroni (FWER control).** Divide α by M tests: α_adj = 0.05/M. For M=100 factors: minimum |t| > 4.4. Very conservative.

**BHY (FDR control).** Controls False Discovery Rate; less conservative. Appropriate for factor discovery where some false positives are acceptable.

**Papers.** Harvey, C., Liu, Y., Zhu, H. (2016). RFS 29(1). NBER WP 20592.

---

### G5. IC Decay and Alpha Decay Analysis

Plot IC(h) vs. holding horizon h for each factor to determine:
- **Optimal rebalancing frequency**: rebalance when IC drops to ~50% of peak.
- **Training target horizon**: use h where IC is most stable as the prediction horizon.

| Factor Type | Typical IC Half-Life | Recommended Rebalance |
|-------------|---------------------|-----------------------|
| Short-term reversal | 2–4 weeks | Weekly or biweekly |
| Analyst revision momentum | 4–8 weeks | Monthly |
| Value (B/P, EBITDA/EV) | 6–18 months | Quarterly |
| Quality (GPA, ROE) | 3–12 months | Quarterly |

**Tools.** `alphalens-reloaded` provides IC decay and turnover plots natively.

---

### G6. Turnover-Adjusted Performance

**Formula (Zhang, Wang & Cao 2021, arXiv:2105.10306).**

```
IR_adj = (TC × IC × √BR − κ × Turnover) / Active_Risk
```

Where κ = round-trip TC per unit of turnover. For Japan (κ ≈ 10bps), a monthly turnover of 20% costs 10bps × 20% × 12 = 24bps/year — significant relative to a 50bps/year active return target.

**Key insight.** Turnover-adjusted IR can *improve* when reducing rebalance frequency for slow factors (value), even though raw IC decreases — TC savings more than compensate.

**Papers.** Zhang, F. et al. (2021). arXiv:2105.10306.

---

## Section H — Full Recommended Pipeline for NKY 225

### Stage 1: Data Preparation
- Universe: all Nikkei 225 ever-members from 2014-01-01 (from JPY121 ETF holdings)
- Factors: 50-feature locked set (Section B1 IC ranking)
- Preprocessing: cross-sectional rank-normalise each factor per date
- Target: rank-normalised 1-month excess return over equal-weighted universe

### Stage 2: Model Stack

| Model | Role | Expected ICIR | Library |
|-------|------|--------------|---------|
| LightGBM LambdaRank | Primary alpha; learns relative ordering | 0.5–1.0 | `lightgbm` |
| LightGBM Huber | Return size; outlier-robust | 0.4–0.8 | `lightgbm` |
| Deep MLP (NN3: 32→16→8) | Smooth non-linear complement | 0.3–0.6 | `pytorch` |
| LSTM (24M lookback) | Time-varying Japan macro regimes | 0.3–0.5 | `pytorch` |
| Ridge regression | Baseline anchor; interpretable | 0.2–0.4 | `sklearn` |
| GNN (keiretsu graph) | Cross-stock supply chain signal | 0.1–0.3 | `torch_geometric` |

**Ensemble blend.** ICIR-weighted average of cross-sectional z-scores, updated monthly.

### Stage 3: Validation
- Primary: Purged K-Fold CV (K=5, purge=21d, embargo=5d)
- Confirmation: CPCV — verify PBO < 10% before deployment
- HPO: Optuna (200 trials) on ICIR objective; use Era Splitting for GBMs

### Stage 4: Alpha Construction
- Grinold-Kahn: α_i = IC × σ_i × blended_score_i
- Signal routing: covered NKY 225 members → Ranker + Huber blend; uncovered → Huber only

### Stage 5: Portfolio Optimisation
- Method: QP active weight optimisation (MVO with active constraints)
- Covariance: BARRA-style factor model (33 TOPIX industries + 7 style factors) with Ledoit-Wolf Goldilocks nonlinear shrinkage on residuals
- Constraints: TE ≤ 5%, active weight ±3%, monthly turnover ≤ 20%, long-only (no naked shorts)
- Solver: OSQP (warm-started, daily TE monitoring) / CLARABEL (monthly rebalance)
- TC: linear 10bps; or hard turnover constraint ≤ 20%/month

### Stage 6: Risk Monitoring
- Real-time TE monitoring via OSQP warm-started daily
- SHAP dashboard: monthly factor contribution tracking; alert if dominant factor SHAP collapses
- Stress tests: 2011 Tōhoku, 2013 Abenomics, 2016 Brexit/BOJ shock, 2020 COVID, 2022 YCC shock, 2024 yen carry unwind
- IC decay monitor: rolling 3-month IC; alert if drops below 0.02

### Stage 7: Evaluation
- Primary production: ICIR, Calmar Ratio, TC, Turnover-Adjusted IR
- Statistical: DSR (SSRN 2460551) and PSR before live deployment
- Benchmark: TOPIX Net Return (long-only); equal-weight Nikkei 225 (market-neutral)

---

## Full Citation List

1. Gu, S., Kelly, B., Xiu, D. (2020). "Empirical Asset Pricing via Machine Learning." *RFS* 33(5), 2223–2273. SSRN 3159577.
2. Kelly, B., Xiu, D. (2023). "Financial Machine Learning." NBER WP 31502.
3. Giglio, S., Kelly, B., Xiu, D. (2022). "Factor Models, ML, and Asset Pricing." *Annual Review of Financial Economics* 14. SSRN 3943284.
4. Kelly, B., Pruitt, S., Su, Y. (2019). "Characteristics Are Covariances." *JFE* 134(3). SSRN 2983919.
5. Abe, M., Nakayama, H. (2018). "Deep Learning for Forecasting Stock Returns in the Cross-Section." arXiv:1801.01777. PAKDD 2018.
6. Abe, M., Nakagawa, K. (2020). "Cross-sectional Stock Price Prediction using Deep Learning." arXiv:2002.06975.
7. Nakagawa, K., Ito, T., Abe, M., Izumi, K. (2019). "Deep Recurrent Factor Model." arXiv:1901.11493.
8. Matsunaga, D., Suzumura, T., Takahashi, T. (2019). "Exploring Graph Neural Networks for Stock Market Predictions." arXiv:1909.10660.
9. MASTER (2024). "Market-Guided Stock Transformer." arXiv:2312.15235. AAAI 2024.
10. MCI-GRU (2024). arXiv:2410.20679.
11. DeLise, T. (2023). "Era Splitting: Invariant Learning for Decision Trees." arXiv:2309.14496.
12. "Building Cross-Sectional Strategies by Learning to Rank." arXiv:2012.07149.
13. GBDT + LSTM hybrid. arXiv:2505.23084.
14. Hybrid Transformer-GNN. arXiv:2601.04602.
15. Dynamic Factor Allocation with Regime-Switching. arXiv:2410.14841.
16. Lopez de Prado, M. (2016). "Building Diversified Portfolios that Outperform Out-of-Sample." *JPM* 42(4). SSRN 2708678.
17. Lopez de Prado, M. (2018). *Advances in Financial Machine Learning.* Wiley.
18. Bailey, D., Lopez de Prado, M. (2014). "The Deflated Sharpe Ratio." *JPM* 40(5). SSRN 2460551.
19. Bailey, D., Borwein, J., Lopez de Prado, M., Zhu, Q. (2016). "Probability of Backtest Overfitting." *Journal of Computational Finance* 20(4). SSRN 2326253.
20. Bailey, D., Lopez de Prado, M. (2012). "The Sharpe Ratio Efficient Frontier." *Journal of Risk* 15(2). SSRN 1821643.
21. Hübler, O. (2022). "ML techniques for cross-sectional equity returns' prediction." *OR Spectrum.*
22. Ledoit, O., Wolf, M. (2004). "A well-conditioned estimator for large-dimensional covariance matrices." JMVA 88(2).
23. Ledoit, O., Wolf, M. (2022). "Nonlinear Shrinkage: Markowitz Meets Goldilocks." *RFS.*
24. Goulart, P., Chen, Y. (2024). "Clarabel: Interior-point solver for conic programs." arXiv:2405.12762.
25. Giller, G. (2024). "Isotropic Correlation Models for the Cross-Section of Equity Returns." arXiv:2411.08864.
26. Zhang, F., Wang, X., Cao, H. (2021). "Turnover-Adjusted Information Ratio." arXiv:2105.10306.
27. "Risk budget portfolios with convex Non-negative Matrix Factorisation." arXiv:2204.02757.
28. Harvey, C., Liu, Y., Zhu, H. (2016). "…and the Cross-Section of Expected Returns." *RFS* 29(1). NBER WP 20592.
29. Clarke, R., de Silva, H., Thorley, S. (2002). "Portfolio Constraints and the Fundamental Law." *FAJ* 58(5).
30. Black, F., Litterman, R. (1992). "Global Portfolio Optimization." *FAJ* 48(5).
31. Fabozzi, F. et al. (2007). *Robust Portfolio Optimization and Management.* Wiley.
32. Engle, R. (2002). "Dynamic Conditional Correlations." *JBES* 20(3).
33. Deep Learning Enhanced MGARCH. arXiv:2506.02796.
34. Robust portfolio optimisation review. arXiv:2103.13806.
35. LLM-Enhanced Black-Litterman. arXiv:2504.14345.
36. Feng, G., He, J., Polson, N. (2018). "Deep Learning for Predicting Asset Returns." arXiv:1804.09314.
37. Lundberg, S., Lee, S.-I. (2017). "A Unified Approach to Interpreting Model Predictions." NeurIPS. arXiv:1705.07874.
38. Akiba, T. et al. (2019). "Optuna: A Next-generation Hyperparameter Optimization Framework." KDD 2019.
39. Markowitz, H. (1952). "Portfolio Selection." *Journal of Finance* 7(1).
40. Morimoto, T., Akama, Y., Kawasaki, Y. (2026). "Explaining the Cross-Section of Japanese Equity Returns Using IEC Structures."
41. "Machine learning goes global: Cross-sectional return predictability in international stock markets." *Journal of Economic Dynamics and Control* 155 (2023).
