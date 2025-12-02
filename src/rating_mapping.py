import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, precision_score, recall_score, f1_score

from .config import RATING_BOUNDARIES


RATING_ORDER = ['AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC', 'CC', 'D']


def _compute_threshold_metrics(y_true, y_proba, threshold, fpr_val, tpr_val):
    y_pred = (y_proba >= threshold).astype(int)
    return {
        'tpr': tpr_val,
        'fpr': fpr_val,
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0)
    }


def find_optimal_threshold_youden(y_true, y_proba):
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    metrics = _compute_threshold_metrics(
        y_true, y_proba, thresholds[optimal_idx], fpr[optimal_idx], tpr[optimal_idx]
    )
    metrics['youden_index'] = j_scores[optimal_idx]
    return thresholds[optimal_idx], metrics


def find_optimal_threshold_cost(y_true, y_proba, cost_fp=1, cost_fn=5):
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    costs = []
    for thresh in thresholds:
        y_pred = (y_proba >= thresh).astype(int)
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()
        costs.append(fp * cost_fp + fn * cost_fn)
    
    optimal_idx = np.argmin(costs)
    metrics = _compute_threshold_metrics(
        y_true, y_proba, thresholds[optimal_idx], fpr[optimal_idx], tpr[optimal_idx]
    )
    metrics['total_cost'] = costs[optimal_idx]
    return thresholds[optimal_idx], metrics


def find_optimal_threshold_f1(y_true, y_proba):
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    f1_scores = [
        f1_score(y_true, (y_proba >= t).astype(int), zero_division=0)
        for t in thresholds
    ]
    optimal_idx = np.argmax(f1_scores)
    metrics = _compute_threshold_metrics(
        y_true, y_proba, thresholds[optimal_idx], fpr[optimal_idx], tpr[optimal_idx]
    )
    metrics['f1'] = f1_scores[optimal_idx]
    return thresholds[optimal_idx], metrics


def find_optimal_threshold(y_true, y_proba, method='youden', cost_fp=1, cost_fn=5):
    if method == 'youden':
        return find_optimal_threshold_youden(y_true, y_proba)
    elif method == 'cost':
        return find_optimal_threshold_cost(y_true, y_proba, cost_fp, cost_fn)
    elif method == 'f1':
        return find_optimal_threshold_f1(y_true, y_proba)
    raise ValueError(f"Unknown method: {method}")


def get_rating_boundaries(scheme='standard'):
    if scheme == 'standard':
        return list(RATING_BOUNDARIES.items())
    return scheme


def map_pd_to_rating(pd_values, rating_scheme='standard'):
    boundaries = get_rating_boundaries(rating_scheme)
    
    ratings = []
    for pd_val in pd_values:
        assigned = 'D'
        for rating, (lower, upper) in boundaries:
            if lower <= pd_val < upper:
                assigned = rating
                break
        ratings.append(assigned)
    return np.array(ratings)


def analyze_rating_distribution(ratings):
    counts = pd.Series(ratings).value_counts()
    total = len(ratings)
    
    result = [
        {'Rating': r, 'Count': counts.get(r, 0), 'Percentage': 100 * counts.get(r, 0) / total}
        for r in RATING_ORDER
    ]
    return pd.DataFrame(result)


def create_decision_table(ratings, y_proba, y_true, threshold):
    y_pred = (y_proba >= threshold).astype(int)
    
    df = pd.DataFrame({
        'Rating': ratings,
        'PD': y_proba,
        'Predicted': y_pred,
        'Actual': y_true
    })
    
    summary = df.groupby('Rating').agg({
        'PD': ['mean', 'min', 'max', 'count'],
        'Predicted': 'sum',
        'Actual': 'sum'
    }).round(4)
    summary.columns = ['Avg_PD', 'Min_PD', 'Max_PD', 'Count', 'Predicted_Defaults', 'Actual_Defaults']
    return summary.reset_index()


def compare_all_thresholds(y_true, y_proba, cost_fp=1, cost_fn=5):
    results = {}
    for method in ['youden', 'cost', 'f1']:
        thresh, metrics = find_optimal_threshold(y_true, y_proba, method, cost_fp, cost_fn)
        results[method] = {'threshold': thresh, **metrics}
    return pd.DataFrame(results).T


def get_threshold_metrics_curve(y_true, y_proba, threshold_range=(0.01, 0.20), n_points=100):
    thresholds = np.linspace(threshold_range[0], threshold_range[1], n_points)
    
    data = []
    for thresh in thresholds:
        y_pred = (y_proba >= thresh).astype(int)
        data.append({
            'threshold': thresh,
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0)
        })
    return pd.DataFrame(data)
