# 🏎️💡 Team XData -- Insurance Charges ML-based Predictions

### End-to-end repo for predicting **health insurance charges** with:
- **XGBoost** (gradient boosting) + **Bayesian hyperparameter tuning** (`BayesSearchCV`)
- **Random Forest** (bagging) + Bayesian tuning
- **Log-target modeling** with **Duan smearing** (global & subgroup)
- **Presentation-ready visuals** (parity plots, MAE by segment)
- **Sensitivity analysis** (what moves predictions & by how much)
- **Fairness slicing** (MAE by Sex / Region / Children)

---

## 🔗 Table of Contents
### • [Why this project?](#-why-this-project)  
### • [Modeling approach](#-modeling-approach)  
### • [Hyperparameter tuning](#-hyperparameter-tuning)  
### • [Log target & smearing](#-log-target--smearing)  
### • [Evaluation metrics (business-friendly)](#-evaluation-metrics-business-friendly)  
### • [Visualizations](#-visualizations)  
### • [Sensitivity analysis](#-sensitivity-analysis)  
### • [Fairness slices & recommendations](#-fairness-slices--recommendations)  
### • [Reproducibility & seeds](#-reproducibility--seeds)  
### • [Common errors & fixes](#-common-errors--fixes)  

---

## ✨ Why this project?
### Insurance pricing is **noisy** and **nonlinear** (thresholds at **BMI≈30**, smoker interactions).
### We need models that:
- **Learn nonlinearities & interactions** (XGBoost / RandomForest)
- **Tune efficiently** (Bayesian search over focused spaces)
- **Predict on $ scale with minimal bias** (log-model + **Duan smearing**)
- **Remain interpretable** (MAE, parity plots, sensitivity “what-if”)

---

## 🧠 Modeling approach 

### XGBoost (boosting)

1. Stage-wise additive modeling: each tree fits the negative gradient (residuals) of the current ensemble.
2. Handles nonlinearities, interactions, and outliers robustly.
3. Regularized with gamma, reg_alpha, reg_lambda, plus subsampling to reduce overfitting.

### Random Forest (bagging)
1. Many decorrelated trees via bootstrap sampling and feature subsampling.
2. Strong variance reduction and stable baseline; often great MAPE on lower charges.
3. Linear & Elastic Net (optional)
3. With engineered interactions (e.g., Smoker×BMI, Smoker×Age) to reduce bias.

### Elastic Net adds L1/L2 for feature selection + stability under collinearity.

## 🎯 Hyperparameter tuning

### Bayesian optimization (scikit-optimize BayesSearchCV)
- Smarter than exhaustive grid: balances exploration/exploitation via surrogate modeling (Gaussian Process or Tree-Parzen under the hood).
- Great for expensive model evaluations and mixed continuous/discrete spaces.

Recommended amount:
1. XGBoost: n_iter ≈ 40–80
2. Random Forest: n_iter ≈ 30–60 (RF is less sensitive than boosting)

### 📈 Log target & smearing
 **We train on log(charges) to stabilize variance. To go back to dollars without bias, apply Duan smearing:

- Train residuals (log scale): e_i = y_log_true - y_log_pred
- Smearing factor: S = mean(exp(e_i))
- Back-transform: ŷ_$ = exp(ŷ_log) * S
- We also support per-group smearing (e.g., by sex) to reduce subgroup bias if residual distributions differ.

### 📊 Evaluation metrics (business-friendly)
 **MAE — average absolute error in dollars; easy to interpret (“typical error per policy”).
 **RMSE — penalizes big misses; good for risk control.
 **R² — share of variance explained; high-level goodness of fit.
 **MAPE — relative error (%); highlights performance on low-charge cases.

For pricing ops, we typically headline MAE (and optionally WAPE), with RMSE and MAPE as supporting evidence.

### 🖼️ Visualizations
- Actual vs Predicted (parity plot) with optional smearing lines (before/after), LOWESS trend, and metric box.
- MAE by segment (Sex / Region / Children) → flags parity gaps.
- Correlation heatmap (incl. log_charges) for quick EDA.
- All plotting snippets are ready to paste. Colors use accessible palettes (Okabe-Ito, Tableau10, Navy).

### 🔍 Sensitivity analysis
“What moves predictions?” Compute % change in predicted dollars for small perturbations:
Numerics: +k·σ (e.g., +0.1σ, +1σ) per feature.
Categoricals: baseline (mode) → each alternative level.
Outputs:
- Row-wise effects and summary tables (median, 5th–95th percentile).
- Aggregated mean |impact| by base feature → ranked drivers.
- Drop-in cell provided (uses your fitted pipeline + smearing). No refits required.

### ⚖️ Fairness slices & practical recommendations
Slices used: Sex / Region / Children
Compare MAE per group for both RF and XGB.
If Region shows the largest gap, collect:
 - Facility/provider ID, network tier, clinic type/DRG, negotiated unit prices, urban/rural flag.

Utilization proxies: visit counts, chronic flags, medication classes → narrows variance for high-charge tails.

Policy granularity: deductible, co-pay, plan tier → aligns premiums with benefit design.

### 🔁 Reproducibility & seeds
- Set random_state across split, models, and tuning.
- Boosting can be seed-sensitive (train/test splits change residuals, and BayesSearchCV samples different configs).
- To assess robustness: evaluate across a small seed sweep (e.g., 42) and report variability.
- If performance swings a lot with seeds, consider: larger CV folds, more data, tighter search space, or stronger regularization.

### 🛠️ Common errors & fixes

1. ImportError: cannot import name BayesSearchCV from sklearn.model_selection Use from skopt import BayesSearchCV (it’s in scikit-optimize, not sklearn).
2. Import 'skopt.space' could not be resolved pip install -U scikit-optimize
3. XGBoost categorical error If using a pandas DataFrame with object dtypes directly in xgboost.DMatrix, you’ll get Invalid columns: ... object. Fix: OHE in a ColumnTransformer (as in this repo), or enable_categorical=True with pandas Categorical.
4. Duan smearing makes MAE worse Smearing corrects mean bias on the dollar scale; it can reduce RMSE but sometimes increase MAE. Keep both reported; choose the KPI that matches business priorities.
5. Pillow _imaging arch mismatch on Apple Silicon Reinstall for arm64 wheel: pip uninstall pillow && pip install --no-binary=:all: pillow or use pip install --upgrade --force-reinstall pillow in a arm64 Python.
6. Git pull overwrote local work Use feature branches; when in doubt: git stash → git pull → resolve → git stash pop. For PRs: push your branch, open compare main....


