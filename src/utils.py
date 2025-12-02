"""
Funkcje pomocnicze dla projektu credit scoring.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve, 
    average_precision_score, log_loss, brier_score_loss
)


def calculate_ks_statistic(y_true, y_pred_proba):
    """
    Oblicza statystykę Kołmogorowa-Smirnova.
    
    KS = max|CDF_good - CDF_bad|
    """
    df = pd.DataFrame({'true': y_true, 'pred': y_pred_proba})
    df = df.sort_values('pred', ascending=False).reset_index(drop=True)
    
    df['cumulative_bad'] = df['true'].cumsum() / df['true'].sum()
    df['cumulative_good'] = (1 - df['true']).cumsum() / (1 - df['true']).sum()
    
    ks = (df['cumulative_bad'] - df['cumulative_good']).abs().max()
    return ks


def calculate_all_metrics(y_true, y_pred_proba):
    """
    Oblicza komplet metryk dla modelu klasyfikacji binarnej.
    
    Zwraca dict z:
    - roc_auc, pr_auc, ks, log_loss, brier
    """
    return {
        'roc_auc': roc_auc_score(y_true, y_pred_proba),
        'pr_auc': average_precision_score(y_true, y_pred_proba),
        'ks': calculate_ks_statistic(y_true, y_pred_proba),
        'log_loss': log_loss(y_true, y_pred_proba),
        'brier': brier_score_loss(y_true, y_pred_proba)
    }


def print_metrics(metrics, model_name="Model"):
    """Czytelny wydruk metryk."""
    print(f"\n{model_name}")
    print("="*60)
    print(f"  ROC-AUC:    {metrics['roc_auc']:.4f}")
    print(f"  PR-AUC:     {metrics['pr_auc']:.4f}")
    print(f"  KS:         {metrics['ks']:.4f}")
    print(f"  Log Loss:   {metrics['log_loss']:.4f}")
    print(f"  Brier:      {metrics['brier']:.4f}")
    print("="*60)
