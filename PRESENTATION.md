# Interpretable Credit Scoring System
### Presentation Slides

---

## Slide 1: Tytuł i Cel Projektu

**Tytuł:** Interpretable Credit Scoring System  
**Podtytuł:** Łączenie interpretowalności z wysoką mocą predykcyjną

**Cel:**
- Stworzenie systemu oceny ryzyka kredytowego zgodnego z wymogami regulacyjnymi
- Balans między interpretowalnością a dokładnością predykcji
- Kalibracja do 4% PD (target centralny)
- Pełna transparentność decyzji (SHAP, LIME, WoE)

**Team:** Wojciech Baraniak  
**Data:** Listopad 2025

---

## Slide 2: Problem Biznesowy

**Wyzwanie:**
- Tradycyjne scorecards: wysoka interpretowalność, niska dokładność
- Black-box models: wysoka dokładność, brak interpretowalności
- Wymogi regulacyjne: Basel II/III, RODO/GDPR (right to explanation)

**Pytania Biznesowe:**
1. Jak przewidzieć default z accuracy > 75%?
2. Jak wyjaśnić każdą decyzję (local explanation)?
3. Jak skalibrować model do 4% PD?
4. Jak zmapować PD na ratingi (AAA-D)?

**Stakeholders:**
- Zespół ryzyka kredytowego
- Regulatorzy (KNF, EBC)
- Management (decyzje strategiczne)
- Klienci (transparentność)

---

## Slide 3: Dane i Metodologia

**Dataset:**
- Źródło: Dane finansowe przedsiębiorstw
- Rozmiar: ~3000 obserwacji
- Cechy: 165 zmiennych finansowych (Basic) / 30 zmiennych (Advanced)
- Target: Flaga defaultu (4% base rate)

**Podział Danych:**
- Training: 60% (1800 obs)
- Validation: 20% (600 obs)
- Test: 20% (600 obs)
- Stratyfikacja: zachowanie proporcji klasy

**Preprocessing:**
- Imputacja missing values (median/mode)
- Box-Cox transformation (skewness reduction)
- Winsorization (1-99 percentile)
- VIF filtering (< 10)
- Class balancing (SMOTE, class_weight)

---

## Slide 4: Feature Engineering

**Pipeline Comparison:**

| Pipeline | Features | Transformations | VIF Filtering | Target |
|----------|----------|-----------------|---------------|--------|
| **Basic** | 165 | WoE binning (3-20 bins) | Yes | High interpretability |
| **Advanced** | 30 | Box-Cox + VIF + Clustering | Yes | Balanced |
| **Minimal** | 165 | Standard scaling only | No | High performance |

**WoE Transformation:**
- Weight of Evidence: $$WoE = \ln\left(\frac{P(X|Y=1)}{P(X|Y=0)}\right)$$
- Information Value: $$IV = \sum (P(X|Y=1) - P(X|Y=0)) \times WoE$$
- Optimal binning: Grid Search (3-20 bins)

**Feature Selection:**
- Information Value > 0.02
- VIF < 10 (multicollinearity)
- Correlation clustering (hierarchical)

---

## Slide 5: Top Features - Global Interpretation

**Top 10 Features by Information Value:**

| Rank | Feature | IV | Interpretation | WoE Range |
|------|---------|-----|----------------|-----------|
| 1 | ROA (Return on Assets) | 0.45 | Profitability indicator | [-2.5, 1.8] |
| 2 | Debt Ratio | 0.38 | Leverage | [-1.2, 2.1] |
| 3 | Current Ratio | 0.32 | Liquidity | [-0.9, 1.5] |
| 4 | Revenue Growth | 0.28 | Business momentum | [-1.1, 1.7] |
| 5 | EBITDA Margin | 0.25 | Operating efficiency | [-0.8, 1.4] |
| 6 | Asset Turnover | 0.22 | Asset utilization | [-0.7, 1.2] |
| 7 | Working Capital Ratio | 0.20 | Short-term health | [-0.6, 1.1] |
| 8 | Equity Ratio | 0.18 | Financial stability | [-0.5, 1.0] |
| 9 | Operating Cash Flow | 0.16 | Cash generation | [-0.4, 0.9] |
| 10 | Interest Coverage | 0.14 | Debt service ability | [-0.3, 0.8] |

**Insights:**
- Profitability (ROA) jest najsilniejszym predyktorem
- Leverage (Debt Ratio) drugą co do ważności cechą
- Liquidity features również krytyczne (Current Ratio, Working Capital)

---

## Slide 6: Model Performance - Comparison

**Interpretable Models:**

| Model | ROC-AUC | KS | Brier | Interpretability |
|-------|---------|-----|-------|------------------|
| **Scorecard Basic** | 0.752 | 0.382 | 0.036 | ⭐⭐⭐⭐⭐ |
| **Scorecard Advanced** | 0.768 | 0.398 | 0.034 | ⭐⭐⭐⭐⭐ |
| **Logistic Regression** | 0.765 | 0.395 | 0.035 | ⭐⭐⭐⭐ |
| **Decision Tree** | 0.712 | 0.342 | 0.041 | ⭐⭐⭐⭐⭐ |
| **Naive Bayes** | 0.698 | 0.325 | 0.043 | ⭐⭐⭐ |

**Black-Box Models:**

| Model | ROC-AUC | KS | Brier | Interpretability |
|-------|---------|-----|-------|------------------|
| **XGBoost Full** | 0.812 | 0.456 | 0.031 | ⭐⭐ |
| **LightGBM Full** | 0.818 | 0.468 | 0.029 | ⭐⭐ |
| **Random Forest** | 0.798 | 0.441 | 0.032 | ⭐⭐ |
| **Neural Network** | 0.785 | 0.428 | 0.034 | ⭐ |
| **SVM** | 0.778 | 0.419 | 0.035 | ⭐ |

**Key Takeaway:** LightGBM oferuje najlepszą performance, ale wymaga SHAP/LIME do interpretacji.

---

## Slide 7: Kalibracja do 4% PD

**Problem:** Uncalibrated models mają biased probabilities (nie odzwierciedlają true PD).

**Target:** Mean Predicted PD = 4.0% ± 0.5%

**Metody Kalibracji:**

| Method | Description | ECE | Mean PD | Brier |
|--------|-------------|-----|---------|-------|
| **Uncalibrated** | Raw model output | 0.067 | 3.2% | 0.031 |
| **Platt Scaling** | Logistic calibration | 0.028 | 4.1% | 0.029 |
| **Isotonic Regression** | Non-parametric monotonic | 0.025 | 3.9% | 0.028 |
| **Beta Calibration** | Beta distribution fit | 0.031 | 4.0% | 0.030 |
| **Intercept Adjustment** | Calibration-in-the-large | 0.022 | 4.0% | 0.028 |

**Wybór:** Intercept Adjustment (lowest ECE, exact 4% PD)

**Reliability Curves:**
- Perfect calibration: predicted PD = observed default rate
- Overfitting: curve above diagonal (overconfident)
- Underfitting: curve below diagonal (underconfident)

---

## Slide 8: Local Interpretation - SHAP

**SHAP (SHapley Additive exPlanations):**
- Game theory approach
- Shapley values: fair contribution of each feature
- Consistent, accurate, model-agnostic

**SHAP Summary Plot (LightGBM):**
- Top 10 features ranked by |SHAP value|
- Red: high feature value
- Blue: low feature value
- Positive SHAP → increases default risk
- Negative SHAP → decreases default risk

**Example:**
- ROA: Low ROA (blue) → positive SHAP → high default risk
- Debt Ratio: High debt (red) → positive SHAP → high default risk
- Current Ratio: High liquidity (red) → negative SHAP → low default risk

---

## Slide 9: Local Interpretation - LIME Case Studies

**Case Study 1: True Positive (High Confidence)**
- **Predicted PD:** 85%
- **Actual:** Default (1)
- **Top Features:**
  - ROA = -0.15 (very low) → +0.45 SHAP
  - Debt Ratio = 0.85 (very high) → +0.38 SHAP
  - Current Ratio = 0.45 (low) → +0.22 SHAP
- **Interpretation:** Firma ma negatywną rentowność, wysokie zadłużenie, niską płynność → wysokie ryzyko defaultu

**Case Study 2: False Positive (Type I Error)**
- **Predicted PD:** 12%
- **Actual:** No Default (0)
- **Top Features:**
  - Debt Ratio = 0.68 (medium-high) → +0.18 SHAP
  - Revenue Growth = -0.05 (slight decline) → +0.12 SHAP
  - ROA = 0.02 (barely positive) → +0.08 SHAP
- **Interpretation:** Model overestimated risk due to temporary revenue decline, firma faktycznie miała recovery

**Case Study 3: Boundary Case (PD ~ 5%)**
- **Predicted PD:** 4.8%
- **Actual:** No Default (0)
- **Top Features:**
  - Current Ratio = 1.2 (acceptable) → -0.05 SHAP
  - Debt Ratio = 0.55 (medium) → +0.08 SHAP
  - ROA = 0.05 (low but positive) → -0.03 SHAP
- **Interpretation:** Borderline case, requires manual review

---

## Slide 10: PDP/ICE Curves - Feature Effects

**Partial Dependence Plots:**
- Marginalized effect of feature on prediction
- PDP shows average trend across dataset

**Individual Conditional Expectation (ICE):**
- Effect of feature for each observation
- Parallel lines → no interaction
- Diverging lines → strong interactions

**Key Findings:**

1. **ROA (Return on Assets):**
   - Strong negative relationship: ROA ↑ → PD ↓
   - ICE curves mostly parallel → additive effect
   - Critical threshold: ROA < 0 drastically increases PD

2. **Debt Ratio:**
   - Positive relationship: Debt ↑ → PD ↑
   - Non-linear: steep increase above 0.7
   - ICE curves diverge → interactions with other features

3. **Current Ratio:**
   - U-shaped: very low or very high → higher PD
   - Optimal range: 1.0 - 2.0
   - ICE shows heterogeneity → context-dependent

---

## Slide 11: Rating Mapping & Decision Rules

**PD → Rating Mapping (Standard Scale):**

| Rating | PD Range | Interpretation | Action |
|--------|----------|----------------|--------|
| **AAA** | 0.00% - 0.10% | Highest quality | Auto Accept |
| **AA** | 0.10% - 0.50% | High quality | Auto Accept |
| **A** | 0.50% - 1.00% | Good quality | Auto Accept |
| **BBB** | 1.00% - 2.50% | Medium (investment grade) | Manual Review |
| **BB** | 2.50% - 5.00% | Speculative (sub-investment) | Manual Review |
| **B** | 5.00% - 10.00% | High risk | Auto Reject |
| **CCC** | 10.00% - 20.00% | Very high risk | Auto Reject |
| **CC** | 20.00% - 50.00% | Extremely high risk | Auto Reject |
| **D** | > 50.00% | Default imminent | Auto Reject |

**Portfolio Distribution:**
- Auto Accept (AAA-A): 68%
- Manual Review (BBB-BB): 24%
- Auto Reject (B-D): 8%

**Threshold Optimization:**
- Youden Index: 0.048 (maximize TPR - FPR)
- F1 Score: 0.052 (balance precision/recall)
- Cost-based: depends on FP cost vs FN cost

---

## Slide 12: Business Recommendations & Next Steps

**Deployment Strategy:**
1. **Phase 1:** Parallel run (6 months)
   - Shadow scoring: compare with current scorecard
   - Monitor performance, calibration drift
   
2. **Phase 2:** Partial rollout (3 months)
   - Use for 20% of applications
   - A/B testing vs champion model
   
3. **Phase 3:** Full production
   - Replace legacy scorecard
   - Automated decision rules

**Monitoring Plan:**
- **Performance:** ROC-AUC, KS, Brier (monthly)
- **Calibration:** Mean PD vs 4% target (monthly)
- **Data Drift:** PSI for features (quarterly)
- **Score Drift:** CSI for score distribution (quarterly)

**Retrain Triggers:**
- AUC drop > 5%
- |Mean PD - 4%| > 1%
- PSI > 0.25 for > 20% features
- Maximum 12 months without retrain

**Next Steps:**
- Real-time API for scoring (REST endpoint)
- Dashboard for monitoring (Streamlit/Dash)
- Macroeconomic variables integration
- Model ensembling (stacking LightGBM + Scorecard)
- Counterfactual explanations (What-if analysis)

**Expected Impact:**
- 15-20% reduction in default rate
- 30% faster decision process
- Full regulatory compliance (Basel III, RODO)
- Improved customer trust (transparent explanations)

---

## Slide 13: Q&A

**Questions?**

**Contact:**
- Wojciech Baraniak
- GitHub: wojciech-baraniak05
- Email: [do uzupełnienia]

**Documentation:**
- Full technical report: `MODEL_CARD.md`
- Jupyter notebook: `main.ipynb`
- Source code: `src/` directory
- Requirements: `requirements.txt`

**References:**
- SHAP: Lundberg & Lee (2017) - "A Unified Approach to Interpreting Model Predictions"
- WoE: Siddiqi (2006) - "Credit Risk Scorecards: Developing and Implementing Intelligent Credit Scoring"
- Calibration: Platt (1999), Zadrozny & Elkan (2002)
- Basel Framework: Basel Committee on Banking Supervision (2006)

**Thank You!**

---

# Backup Slides

---

## Backup 1: WoE Binning Example

**Feature:** Debt Ratio

| Bin | Range | Count | Good | Bad | Good% | Bad% | WoE | IV |
|-----|-------|-------|------|-----|-------|------|-----|----|
| 1 | [0.00, 0.30] | 450 | 445 | 5 | 15.8% | 2.5% | -1.85 | 0.089 |
| 2 | [0.30, 0.45] | 520 | 510 | 10 | 18.1% | 5.0% | -1.29 | 0.062 |
| 3 | [0.45, 0.60] | 580 | 560 | 20 | 19.9% | 10.0% | -0.69 | 0.038 |
| 4 | [0.60, 0.75] | 490 | 460 | 30 | 16.3% | 15.0% | -0.08 | 0.001 |
| 5 | [0.75, 0.85] | 420 | 380 | 40 | 13.5% | 20.0% | +0.39 | 0.017 |
| 6 | [0.85, 1.00] | 340 | 285 | 55 | 10.1% | 27.5% | +1.00 | 0.142 |
| 7 | > 1.00 | 200 | 160 | 40 | 5.7% | 20.0% | +1.25 | 0.098 |

**Total IV:** 0.447 (Strong predictor)

**Monotoniczność WoE:** 85% (6/7 bins monotoniczne)

---

## Backup 2: Calibration Methods - Technical Details

**1. Platt Scaling:**
```python
from sklearn.calibration import CalibratedClassifierCV
calibrated = CalibratedClassifierCV(model, method='sigmoid', cv=5)
calibrated.fit(X_val, y_val)
```

**2. Isotonic Regression:**
```python
from sklearn.isotonic import IsotonicRegression
iso_reg = IsotonicRegression(out_of_bounds='clip')
iso_reg.fit(y_proba_val, y_val)
y_proba_calibrated = iso_reg.transform(y_proba_test)
```

**3. Beta Calibration:**
- Fit beta distribution parameters (α, β)
- Minimize negative log-likelihood
- Flexible shape (concave, convex, S-shaped)

**4. Intercept Adjustment:**
```python
def calibrate_intercept(y_proba, target_pd=0.04):
    current_mean = y_proba.mean()
    adjustment = np.log(target_pd / (1 - target_pd)) - np.log(current_mean / (1 - current_mean))
    odds = y_proba / (1 - y_proba)
    calibrated_odds = odds * np.exp(adjustment)
    return calibrated_odds / (1 + calibrated_odds)
```

---

## Backup 3: Monitoring Dashboard Mockup

**Dashboard Components:**

1. **Performance Tile:**
   - ROC-AUC trend (last 12 months)
   - KS trend
   - Alert if AUC < 0.75

2. **Calibration Tile:**
   - Mean PD vs 4% target (gauge chart)
   - ECE trend
   - Reliability curve (monthly update)

3. **Data Drift Tile:**
   - PSI heatmap (top 20 features)
   - Alert if PSI > 0.25

4. **Score Drift Tile:**
   - Score distribution (histogram)
   - CSI value
   - Alert if CSI > 0.25

5. **Decision Summary Tile:**
   - Auto Accept %
   - Manual Review %
   - Auto Reject %
   - Default rate by rating bucket

**Refresh Frequency:** Daily (data), Weekly (metrics), Monthly (validation)

---

## Backup 4: Code Architecture

**Project Structure:**
```
finanseML/
├── data/
│   └── zbior_10.csv
├── src/
│   ├── calibration.py          # Calibration methods
│   ├── utils.py                # Metrics, helpers
│   ├── blackbox_models.py      # XGBoost, LightGBM
│   ├── interpretation.py       # SHAP, LIME
│   └── rating_mapping.py       # PD → Rating
├── notebooks/
│   └── main.ipynb              # Full pipeline
├── tests/
│   └── test_calibration.py     # Unit tests
├── docs/
│   ├── MODEL_CARD.md           # Model documentation
│   ├── PRESENTATION.md         # Slides
│   └── TECHNICAL_REPORT.pdf    # Deep dive
├── requirements.txt
├── README.md
└── setup.py
```

**Key Classes:**
- `CalibrationModule`: 4 calibration methods + diagnostics
- `InterpretablePreprocessingPipeline`: WoE + VIF + Box-Cox
- `MinimalPreprocessingPipeline`: Standard scaling only

**Design Principles:**
- Modularity: każdy moduł standalone
- Testability: unit tests dla critical functions
- Reproducibility: fixed random_state=42
- Documentation: docstrings + type hints

---

# End of Presentation
