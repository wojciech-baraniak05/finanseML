# Analiza Ryzyka Kredytowego - Modele ML

Ten projekt zawiera analizę ryzyka kredytowego oraz budowę modeli predykcyjnych, w tym interpretowalnych scorecardów oraz modeli typu "black box".

## Struktura Projektu

Projekt składa się z dwóch głównych notatników Jupyter:

1.  **`finalInterpretowalny.ipynb`**:
    * Ten notatnik koncentruje się na budowie interpretowalnych modeli ryzyka kredytowego, w szczególności scorecardów (kart punktowych).
    * Obejmuje on:
        * **Setup i instalację bibliotek:** Sprawdzenie i instalacja wymaganych pakietów.
        * **Wczytanie i podział danych:** Podział na zbiory treningowe, walidacyjne i testowe.
        * **Preprocessing:** Czyszczenie danych, imputacja braków, winsoryzacja, usuwanie zmiennych skorelowanych (VIF, klastrowanie).
        * **Feature Engineering:** Tworzenie nowych zmiennych, transformacja WoE (Weight of Evidence).
        * **Budowa Scorecardów:**
            * **Basic Scorecard:** Prosty model oparty na podstawowych cechach.
            * **Advanced Scorecard:** Bardziej zaawansowany model z nowymi cechami i interakcjami.
        * **Kalibracja:** Dostrajanie prawdopodobieństw (Calibration-in-the-Large, Isotonic Regression itp.).
        * **Ewaluacja i Interpretacja:** Analiza wyników (ROC-AUC, KS, Gini), analiza stabilności, interpretacja współczynników, wykresy PDP i ICE.
        * **Generowanie kodu SQL:** Eksport modelu do SQL.

2.  **`finalBlackBox.ipynb`**:
    * Ten notatnik skupia się na bardziej złożonych modelach uczenia maszynowego (tzw. "black box"), które często oferują wyższą moc predykcyjną kosztem interpretowalności.
    * Zawiera on:
        * **Trenowanie modeli:** LightGBM, XGBoost, SVM.
        * **Tuning hiperparametrów:** Wykorzystanie Optuna do optymalizacji modeli.
        * **Ensembling:** Łączenie modeli (Seed Ensemble, Blending, Stacking) w celu poprawy wyników.
        * **Ewaluacja:** Porównanie wyników różnych podejść.
        * **Analiza SHAP:** Wyjaśnianie predykcji modeli "black box" za pomocą wartości SHAP (globalna i lokalna interpretowalność).
        * **Analiza biznesowa:** Symulacja zysków i strat, wyznaczanie optymalnego progu odcięcia (threshold), macierz zysków (profit matrix).

## Wymagania

Aby uruchomić projekt, potrzebujesz zainstalowanych następujących bibliotek Pythona. Możesz je zainstalować używając pliku `requirements.txt`.

### Instalacja

1.  Upewnij się, że masz zainstalowanego Pythona (zalecana wersja 3.8+).
2.  Zalecane jest utworzenie wirtualnego środowiska:
    ```bash
    python -m venv venv
    source venv/bin/activate  # Na Linux/macOS
    venv\Scripts\activate     # Na Windows
    ```
3.  Zainstaluj wymagane pakiety:
    ```bash
    pip install -r requirements.txt
    ```

## Uruchomienie

Projekt jest w formie notatników Jupyter (`.ipynb`). Aby je uruchomić:

1.  Uruchom Jupyter Notebook lub JupyterLab:
    ```bash
    jupyter notebook
    ```
2.  W przeglądarce otwórz wybrany plik (`finalInterpretowalny.ipynb` lub `finalBlackBox.ipynb`).
3.  Wykonuj komórki z kodem po kolei (Shift + Enter).

**Uwaga:** Notatniki mogą wymagać dostępu do pliku z danymi (`zbior_10.csv`). Upewnij się, że plik ten znajduje się w tym samym katalogu co notatniki lub zaktualizuj ścieżkę w kodzie.

## Autorzy

[Aleksander Brandt/Wojciech Baraniak]