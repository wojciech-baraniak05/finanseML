# finanseML - Interpretable Credit Scoring System

System oceny ryzyka kredytowego łączący modele interpretowalne (WoE scorecard) z zaawansowanymi modelami black-box (XGBoost, LightGBM). Projekt zawiera kompletną pipeline'ę z kalibracją do 4% PD, globalną i lokalną interpretacją (SHAP, LIME) oraz mapowaniem PD na ratingi.

## Szybki Start (One-Click)

```bash
git clone <repository-url>
cd finanseML
pip install -r requirements.txt
jupyter notebook "calkiemG — kopiaNmgJuz.ipynb"
```

Następnie uruchom wszystkie komórki (`Cell → Run All`).

## Struktura Projektu

```
finanseML/
├── README.md                                 # Ten plik
├── MODEL_CARD.md                             # Pełna dokumentacja modelu
├── requirements.txt                          # Zależności Python
├── zbior_10.csv                              # Dane wejściowe
├── calkiemG — kopiaNmgJuz.ipynb             # Główny notebook
└── src/                                      # Moduły Python
    ├── calibration.py                        # Kalibracja do 4% PD
    ├── utils.py                              # Funkcje pomocnicze
    ├── blackbox_models.py                    # XGBoost, LightGBM
    ├── interpretation.py                     # SHAP, LIME
    └── rating_mapping.py                     # PD → Rating mapping
```

## Wymagania

### Python Packages
```
python>=3.8
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
xgboost>=1.5.0
lightgbm>=3.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scipy>=1.7.0
scikit-optimize>=0.9.0
shap>=0.41.0
lime>=0.2.0.1
jupyter>=1.0.0
```

### Opcjonalne (dla zaawansowanych funkcji)
- `imblearn` - SMOTE dla balancowania klas
- `category_encoders` - dodatkowe enkodery
- `optuna` - alternatywna optymalizacja hiperparametrów

## Kluczowe Komponenty

### 1. Data Preprocessing
- **InterpretablePreprocessingPipeline:** Box-Cox, winsorization, VIF selection
- **MinimalPreprocessingPipeline:** Podstawowe transformacje
- Podział: 60% train / 20% validation / 20% test

### 2. Feature Engineering
- 165 features (Basic pipeline) vs 30 features (Advanced pipeline)
- WoE transformation z optimal binning (3-20 binów)
- Information Value filtering (IV > 0.02)
- VIF-based multicollinearity removal (VIF < 10)

### 3. Modele Interpretowalne
- **Logistic Regression + WoE:** Model główny
- **Decision Tree:** Baseline
- **Naive Bayes:** Baseline
- Grid Search dla optymalizacji hiperparametrów

### 4. Modele Black-Box
- **XGBoost:** Bayesian Optimization (50 iteracji)
- **LightGBM:** Bayesian Optimization (50 iteracji)
- **Random Forest, SVM, Neural Network:** Baseline comparison

### 5. Kalibracja (4% PD Target)
- Platt Scaling
- Isotonic Regression
- Beta Calibration
- Intercept Adjustment
- Metryki: ECE, Brier Score, Reliability Curves

### 6. Interpretacja
- **Global:** WoE tables, Information Value, Feature Importance
- **Local:** SHAP (TreeExplainer, LinearExplainer, KernelExplainer)
- **Local:** LIME dla case studies
- **Partial Dependence Plots (PDP)** i **ICE curves**

### 7. Decision Support
- Threshold optimization (Youden, Cost-based, F1)
- PD → Rating mapping (AAA do D)
- Decision tables dla business users
- Rating distribution analysis

## Kluczowe Wyniki

### Performance (Test Set)
- **ROC-AUC:** 0.75-0.83 (zależnie od modelu)
- **KS Statistic:** 0.35-0.50
- **Brier Score:** 0.028-0.040

### Calibration
- **Mean Predicted PD:** 4.0% ± 0.5% (target: 4%)
- **ECE:** < 0.05
- **Reliability:** High correlation między predicted i observed

### Top 5 Features (Przykład)
1. ROA - Return on Assets
2. Debt Ratio
3. Current Ratio
4. Revenue Growth
5. EBITDA Margin

(Dokładne wartości w notebooks)

## Workflow

1. **Data Loading & EDA**
   - Load zbior_10.csv
   - Eksploracyjna analiza danych
   - Identyfikacja outliers i missing values

2. **Feature Engineering**
   - Transformacje Box-Cox
   - Winsorization (1-99 percentile)
   - VIF-based selection
   - WoE binning z optymalizacją

3. **Model Training**
   - Interpretable models na WoE features
   - Black-box models na raw/transformed features
   - Grid Search / Bayesian Optimization

4. **Calibration**
   - Apply 4 metody kalibracji
   - Porównanie reliability curves
   - Wybór najlepszej metody (lowest ECE)

5. **Interpretation**
   - SHAP summary plots
   - LIME dla 3-5 case studies
   - PDP/ICE dla top features

6. **Business Application**
   - Threshold optimization
   - Rating assignment (AAA-D)
   - Decision tables

## Monitoring & Maintenance

### Co monitorować?
- **Performance:** ROC-AUC, KS, Brier Score (miesięcznie)
- **Calibration Drift:** Mean PD vs 4% target
- **Data Drift:** PSI dla features
- **Score Drift:** CSI dla score distribution

### Kiedy retrenować?
- AUC drop > 5%
- |Mean PD - 4%| > 1%
- PSI > 0.25 dla > 20% features
- Maximum 12 miesięcy od ostatniego treningu

## Compliance

- **Basel II/III:** IRB-compliant
- **RODO/GDPR:** Right to explanation (SHAP/LIME)
- **Validation:** Roczna przez niezależny zespół
- **Documentation:** MODEL_CARD.md

## Kontakt

- **Developer:** Wojciech Baraniak
- **GitHub:** wojciech-baraniak05
- **Email:** [do uzupełnienia]

## Licencja

[Do uzupełnienia]

## Changelog

### Version 1.0 (November 2025)
- Implementacja WoE scorecard
- Dodanie XGBoost/LightGBM z Bayesian Optimization
- Kalibracja do 4% PD (4 metody)
- SHAP i LIME interpretation
- PD → Rating mapping
- Kompletna dokumentacja (MODEL_CARD.md)

## Następne Kroki (Future Work)

- [ ] API REST dla real-time scoring
- [ ] Dashboard monitoringu (Streamlit/Dash)
- [ ] Macroeconomic variables integration
- [ ] Model ensembling
- [ ] Counterfactual explanations
- [ ] Automated retraining pipeline