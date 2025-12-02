"""Evaluation metrics for credit scoring models."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, log_loss, brier_score_loss,
    confusion_matrix
)


def calculate_ks_statistic(y_true, y_proba):
    """Calculate Kolmogorov-Smirnov statistic."""
    y_true = np.array(y_true)
    y_proba = np.array(y_proba)
    
    idx_sorted = np.argsort(y_proba)
    y_true_sorted = y_true[idx_sorted]
    
    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos
    
    if n_pos == 0 or n_neg == 0:
        return 0.0
    
    cum_pos = np.cumsum(y_true_sorted) / n_pos
    cum_neg = np.cumsum(1 - y_true_sorted) / n_neg
    
    ks = np.max(np.abs(cum_pos - cum_neg))
    return ks


def calculate_gini(y_true, y_proba):
    """Calculate Gini coefficient (2 * AUC - 1)."""
    auc = roc_auc_score(y_true, y_proba)
    return 2 * auc - 1


def calculate_psi(expected, actual, bins=10):
    """Calculate Population Stability Index."""
    expected = np.array(expected)
    actual = np.array(actual)
    
    breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf
    breakpoints = np.unique(breakpoints)
    
    expected_counts = np.histogram(expected, bins=breakpoints)[0]
    actual_counts = np.histogram(actual, bins=breakpoints)[0]
    
    expected_pct = expected_counts / len(expected)
    actual_pct = actual_counts / len(actual)
    
    expected_pct = np.clip(expected_pct, 0.0001, None)
    actual_pct = np.clip(actual_pct, 0.0001, None)
    
    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return psi


def calculate_all_metrics(y_true, y_proba, threshold=0.5):
    """Calculate all evaluation metrics."""
    y_pred = (np.array(y_proba) >= threshold).astype(int)
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_proba),
        'pr_auc': average_precision_score(y_true, y_proba),
        'ks_statistic': calculate_ks_statistic(y_true, y_proba),
        'gini': calculate_gini(y_true, y_proba),
        'log_loss': log_loss(y_true, y_proba),
        'brier_score': brier_score_loss(y_true, y_proba)
    }


def print_model_metrics(metrics, model_name):
    """Print metrics in formatted table."""
    print(f"\n{model_name}")
    print("-" * 40)
    for name, value in metrics.items():
        print(f"  {name:15}: {value:.4f}")


def compare_models(y_true, proba_dict, threshold=0.5):
    """Compare multiple models metrics in DataFrame."""
    results = {}
    for name, y_proba in proba_dict.items():
        results[name] = calculate_all_metrics(y_true, y_proba, threshold)
    df = pd.DataFrame(results).T
    df.index.name = 'model'
    return df.round(4)


def get_confusion_matrix_stats(y_true, y_pred):
    """Get confusion matrix with derived statistics."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    return {
        'true_negatives': tn,
        'false_positives': fp,
        'false_negatives': fn,
        'true_positives': tp,
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'fpr': fp / (fp + tn) if (fp + tn) > 0 else 0,
        'fnr': fn / (fn + tp) if (fn + tp) > 0 else 0
    }
