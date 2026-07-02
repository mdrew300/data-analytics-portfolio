# LendingClub Interest Rate Predictor

Predicting loan interest rates at underwriting time using LendingClub data (2007–2020).  
**Final model:** XGBoost (random search) — Holdout RMSE **4.029**, R² **0.405**

Georgetown University · OPAN 6604 Applied AI · Project 1  
*Saxa 4: Matt Drew · Alec Safreno · Stephanie Ong · Clay Cyrus*

---

## Results at a Glance

| Model | Holdout RMSE | Holdout R² |
|---|---|---|
| OLS LinearRegression (baseline) | 4.294 | 0.325 |
| XGBoost — Grid Search | 4.032 | 0.405 |
| **XGBoost — Random Search (selected)** | **4.029** | **0.405** |

XGBoost beat the linear baseline by **0.26 RMSE** and **+8 R² points**, driven by feature interactions that OLS cannot capture (e.g., high DTI matters more on 60-month loans). Random search edged grid search by 0.003 RMSE by exploring deeper trees, lower learning rates, and two additional regularizers (`min_child_weight`, `reg_alpha`) across a wider continuous space.

---

## Top Features (SHAP Global Importance)

| Feature | Mean \|SHAP\| | Interpretation |
|---|---|---|
| FICO score | +2.07 pp | Dominant predictor — compresses full credit history into one score |
| Loan term | +1.18 pp | 60-month loans price materially higher than 36-month |
| DTI | +0.93 pp | Higher debt-to-income ratio signals reduced repayment capacity |
| Purpose (credit card) | +0.69 pp | Unsecured consolidation draws a rate premium |
| Finance inquiries | +0.45 pp | Recent credit-seeking behavior flags elevated risk |

SHAP and permutation importance rankings agreed on the top features, validating the ranking. XGBoost's built-in gain importance overstated high-cardinality categorical dummies (`addr_state`) — SHAP is the more reliable measure here.

---

## Notebook Structure

The analysis runs end-to-end in a single notebook: [interest-rate-predictor.ipynb](interest-rate-predictor.ipynb)

**Step 1 — EDA & Preprocessing**
- Explored target distribution, missingness, and bivariate relationships
- Dropped 17 of 39 features in three documented groups to prevent leakage:
  - **POST (3):** only observable after funding (e.g., `loan_status`, `chargeoff_within_12_mths`)
  - **AMBIG (10):** current-snapshot fields that drift post-origination (e.g., `revol_util`, `tot_cur_bal`)
  - **LOW-VALUE (4):** redundant or too sparse (e.g., `emp_title`, `zip_code`)
- Cleaned string-encoded numerics (`term`, `emp_length`), engineered `fico_score` midpoint
- Removed 637 outlier training rows (`dti > 100` or `annual_inc > $1M`)
- Built a `ColumnTransformer` pipeline (median impute + scale for numerics; mode impute + OHE for categoricals)
- Result: **21 raw features → 83 after encoding**; 80/20 train/holdout split

**Step 2 — Baseline & Hyperparameter Tuning**
- OLS `LinearRegression` baseline (5-fold CV RMSE: 4.340)
- `GridSearchCV`: 48 combos × 5 folds = 240 fits over 5 hyperparameters
- `RandomizedSearchCV`: 80 samples × 5 folds = 400 fits over 7 hyperparameters (added `min_child_weight`, `reg_alpha`)
- Validation curve over `reg_lambda` confirmed the random-search winner sits near the bias-variance optimum

**Step 3 — Explainability & Final Submission**
- **Global:** Permutation Feature Importance, SHAP summary bar + beeswarm, PDP for top continuous features
- **Local:** ICE plots (fan analysis), SHAP waterfalls for 3 selected loans, LIME cross-check
- Aggregated SHAP to parent-feature level — corrects fragmentation across OHE dummies
- Refit final model on combined train + holdout (99,363 rows); generated `LC_test` submission

---

## Key Findings

- **FICO dominates**, but the spread at any given score is wide — term, DTI, and purpose carry substantial additional signal.
- **ICE fanning** on FICO confirms the feature's effect varies by borrower, justifying the non-linear model over OLS.
- **Feature ceiling, not a tuning ceiling:** the 0.43-point train/CV gap persists even with stronger L2 regularization. The next improvement would come from richer origination-time features (employment stability, local economic conditions), not more hyperparameter search.
- **Gain importance is misleading** for high-cardinality categoricals: `addr_state` (50 dummies) appears inflated by gain but ranks much lower by SHAP and permutation importance.

---

## Business Implications

**For borrowers:** The clearest levers for a lower rate are improving your FICO score, choosing a shorter loan term, and keeping DTI below ~36%.

**For lenders:** R² = 0.40 using origination-time data only leaves ~60% of rate variance unexplained — likely reflecting factors unavailable at application (employment trajectory, local conditions). This points to a data collection opportunity, not a modeling failure.

---

## Setup

```bash
pip install numpy pandas scikit-learn xgboost shap lime matplotlib seaborn joblib
```

The notebook expects `LC_train.csv` and `LC_test.csv` in the same directory. Update `DATA_DIR` in Step 1 if your data lives elsewhere.

Intermediate outputs (fitted preprocessor, model weights, SHAP cache) are written to an `artifacts/` subdirectory created automatically on first run.

---

## AI Tool Citation

Claude (Anthropic) was used as a coding assistant for notebook structure, leakage-column triage, and hyperparameter search range suggestions. All decisions, interpretations, and final code were reviewed and approved by the student authors, per course policy.
