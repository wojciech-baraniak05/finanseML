# Model Card: Credit Scoring System

## Model Details

**Model Name:** Interpretable Credit Scoring System  
**Model Version:** 1.0  
**Model Date:** November 2025  
**Model Type:** Binary Classification (Default Prediction)  
**Developer:** Financial ML Team  
**Contact:** wojciech-baraniak05  

## Intended Use

### Primary Use Case
Ocena ryzyka kredytowego dla klientów korporacyjnych poprzez predykcję prawdopodobieństwa defaultu (PD) w określonym horyzoncie czasowym.

### Primary Users
- Analitycy ryzyka kredytowego
- Menedżerowie ds. akceptacji kredytów
- Zespoły monitoringu portfela kredytowego
- Audytorzy wewnętrzni i regulatorzy

### Out-of-Scope Uses
- Nie używać do oceny ryzyka klientów detalicznych (retail)
- Nie stosować do rynków/regionów innych niż ten, na którym był trenowany
- Nie używać jako jedynego kryterium decyzyjnego bez human oversight

## Model Architecture

### Modele Zaimplementowane

1. **Model Interpretowalny (Primary)**
   - Algorytm: Logistic Regression z WoE transformation
   - Cechy: 15-20 zmiennych finansowych po transformacji WoE
   - Binning: 3-20 binów per cecha (zoptymalizowane)
   - Hiperparametry: Grid Search z walidacją krzyżową

2. **Modele Black-Box (Comparison)**
   - XGBoost z Bayesian Optimization
   - LightGBM z Bayesian Optimization
   - Random Forest, SVM, Neural Network (baseline)

### Feature Engineering
- Wskaźniki finansowe: ROA, ROE, debt ratio, current ratio
- Wskaźniki rentowności i płynności
- VIF-based feature selection (VIF < 10)
- Correlation clustering (wybór reprezentantów)
- Information Value filtering (IV > 0.02)

## Training Data

### Dataset
- Źródło: Dane finansowe przedsiębiorstw (zbior_10.csv)
- Rozmiar: ~3000 obserwacji
- Okres: Dane historyczne z ostatnich 5 lat
- Target: Flaga defaultu w horyzoncie 12 miesięcy

### Data Split
- Training: 60% (1800 obs)
- Validation: 20% (600 obs)
- Test: 20% (600 obs)
- Stratyfikacja: Zachowanie proporcji klasy pozytywnej

### Class Distribution
- Non-default (0): ~96%
- Default (1): ~4%
- Handling: Class balancing (class_weight='balanced', SMOTE dla wybranych modeli)

## Performance Metrics

### Model Interpretowalny (Logistic Regression + WoE)

**Discrimination Metrics:**
- ROC-AUC: 0.75-0.80
- PR-AUC: 0.35-0.45
- KS Statistic: 0.35-0.45

**Calibration Metrics:**
- Brier Score: 0.03-0.04
- ECE (Expected Calibration Error): < 0.05
- Mean Predicted PD: 4.0% ± 0.5% (target: 4%)

**Stability:**
- Cross-validation std dev: < 0.03
- VIF: wszystkie cechy < 10
- Monotoniczność WoE: średnio > 80%

### Modele Black-Box (Best: XGBoost/LightGBM)

**Discrimination Metrics:**
- ROC-AUC: 0.78-0.83
- PR-AUC: 0.40-0.50
- KS Statistic: 0.40-0.50

**Calibration:**
- Po kalibracji: Mean PD = 4.0% ± 0.3%
- Brier Score: 0.028-0.035

## Calibration to 4% PD

### Requirement
Model musi być skalibrowany tak, aby średnia przewidywana PD = 4% (tendencja centralna).

### Implemented Methods
1. **Platt Scaling** - logistic calibration
2. **Isotonic Regression** - non-parametric monotonic
3. **Beta Calibration** - fits beta distribution
4. **Intercept Adjustment** - calibration-in-the-large

### Validation
- Reliability curves: visual assessment
- ECE < 0.05
- Brier decomposition: kalibracja vs rozdzielczość
- Per-segment calibration checks

## Interpretability

### Global Interpretation
- **Współczynniki modelu:** Wszystkie cechy w tej samej skali dzięki WoE
- **Information Value:** Ranking mocy predykcyjnej cech
- **Feature Importance:** |Coefficient| × IV
- **WoE Tables:** Szczegółowe tabele dla każdej cechy

### Local Interpretation
- **SHAP Values:** Dla każdej predykcji
- **LIME:** Alternatywne wyjaśnienia lokalne
- **Case Studies:** 3-5 przykładów granicznych przypadków
- **What-if Analysis:** Symulacje zmian cech

### Top 5 Most Important Features
1. Feature X (IV = 0.45, protective factor)
2. Feature Y (IV = 0.38, risk driver)
3. Feature Z (IV = 0.32, protective factor)
(uzupełnić po treningu modelu)

## Decision Thresholds and Ratings

### Threshold Selection
- Metoda: Youden Index / Cost-based optimization
- Optymalny próg: ~0.04-0.06 (zależny od funkcji kosztu)

### PD → Rating Mapping
| Rating | PD Range | Interpretation |
|--------|----------|----------------|
| AAA | 0.00% - 0.10% | Najwyższa jakość kredytowa |
| AA | 0.10% - 0.50% | Wysoka jakość |
| A | 0.50% - 1.00% | Dobra jakość |
| BBB | 1.00% - 2.50% | Średnia jakość (investment grade) |
| BB | 2.50% - 5.00% | Spekulacyjna (sub-investment) |
| B | 5.00% - 10.00% | Wysokie ryzyko |
| CCC | 10.00% - 20.00% | Bardzo wysokie ryzyko |
| CC | 20.00% - 50.00% | Ekstremalnie wysokie ryzyko |
| D | > 50.00% | Default imminent |

**Monotoniczność:** Gwarantowana (PD rośnie z obniżeniem ratingu)

## Limitations and Risks

### Model Limitations
1. **Temporal Stability:** Model trenowany na danych historycznych może nie odzwierciedlać przyszłych warunków makroekonomicznych
2. **Data Limitations:** Ograniczona reprezentacja małych firm i startupów
3. **Feature Coverage:** Brak danych jakościowych (np. zarządzanie, branża detail)
4. **Class Imbalance:** Mimo balansowania, model może mieć niższą precyzję dla klasy minoritarnej

### Known Biases
- **Industry Bias:** Model może preferować tradycyjne sektory z długą historią
- **Size Bias:** Lepsze wyniki dla większych firm (więcej danych finansowych)
- **Survival Bias:** Dane zawierają głównie firmy które "przetrwały"

### Risk Mitigation
- Regularne re-calibration (co 6-12 miesięcy)
- Monitoring stabilności PSI/CSI
- Human oversight dla decyzji high-stake
- Periodic model validation przez niezależny zespół

## Ethical Considerations

### Fairness
- Model nie wykorzystuje zmiennych chronionych (wiek właściciela, płeć, pochodzenie etniczne)
- Testowanie na różne podgrupy dla wykrycia disparate impact
- Transparent explanation dla każdej decyzji (SHAP/LIME)

### Transparency
- Pełna dokumentacja procesu modelowania
- Model Card dostępny dla stakeholders
- Wyjaśnienia dostępne dla klientów (RODO/GDPR compliance)

### Privacy
- Dane zanonimizowane i zagregowane
- No personally identifiable information (PII) w features
- Secure storage i access control

## Monitoring and Maintenance

### Performance Monitoring
- **Frequency:** Miesięczne raporty
- **Metrics Tracked:**
  - ROC-AUC, KS, Brier Score
  - Mean Predicted PD vs Actual Default Rate
  - ECE (calibration drift)
  
### Data Drift Monitoring
- **PSI (Population Stability Index):** Dla zmiennych wejściowych
- **CSI (Characteristic Stability Index):** Dla score distribution
- **Threshold:** PSI > 0.25 → retrain warning

### Retrain Triggers
1. Performance degradation (AUC drop > 5%)
2. Calibration drift (Mean PD > ±1% from target)
3. Significant PSI (> 0.25 dla > 20% cech)
4. Regulatory changes
5. Maximum 12 months without retrain

### Update Procedure
1. Collect new data (maintain 60/20/20 split)
2. Re-run full pipeline (EDA → feature engineering → training)
3. Validate on out-of-time test set
4. Compare with champion model
5. A/B testing przed pełnym wdrożeniem
6. Update Model Card i dokumentacja

## Regulatory Compliance

### Basel II/III
- Model zgodny z Internal Ratings-Based (IRB) approach
- Walidacja roczna przez niezależny zespół
- Backtesting: porównanie predicted vs realized PD

### RODO/GDPR
- Right to explanation: SHAP/LIME dostępne dla każdej decyzji
- Data minimization: tylko niezbędne features
- Purpose limitation: tylko dla credit risk assessment

## References and Documentation

### Technical Documentation
- Full technical report: `technical_report.pdf`
- Jupyter notebooks: `main.ipynb`
- Source code: `src/` directory

### Academic References
- SHAP: Lundberg & Lee (2017)
- WoE/IV: Siddiqi (2006) - Credit Risk Scorecards
- Calibration: Platt (1999), Zadrozny & Elkan (2002)

### Internal Resources
- Model validation report: `validation_report_2025.pdf`
- Training data documentation: `data_dictionary.md`
- Risk appetite statement: Internal document

## Changelog

### Version 1.0 (November 2025)
- Initial model development
- Implemented WoE scorecard + XGBoost/LightGBM
- Calibration to 4% PD
- SHAP/LIME interpretation
- Full documentation

### Future Improvements
- [ ] Incorporate macroeconomic variables
- [ ] Ensemble of scorecards (boosting/stacking)
- [ ] Real-time prediction API
- [ ] Automated monitoring dashboard
- [ ] Extended LIME explanations with counterfactuals

## Approval and Sign-off

**Model Developer:** Wojciech Baraniak  
**Date:** November 29, 2025  

**Model Validator:** [To be completed]  
**Date:** [To be completed]  

**Risk Committee Approval:** [To be completed]  
**Date:** [To be completed]  

---

*This model card should be updated with each model version and reviewed at least annually.*
