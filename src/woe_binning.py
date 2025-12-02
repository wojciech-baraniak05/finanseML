import numpy as np
import pandas as pd
from .config import MIN_IV_THRESHOLD

N_BINS_DEFAULT = 10


def calculate_woe_iv(df, feature, target, bins=N_BINS_DEFAULT):
    """Calculate WoE and IV for a single feature."""
    data = df[[feature, target]].copy()
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    
    if data[feature].nunique() <= bins:
        data['bin'] = data[feature]
    else:
        data['bin'] = pd.qcut(data[feature], q=bins, duplicates='drop')
    
    grouped = data.groupby('bin', observed=True)[target].agg(['count', 'sum'])
    grouped.columns = ['total', 'events']
    grouped['non_events'] = grouped['total'] - grouped['events']
    
    total_events = grouped['events'].sum()
    total_non_events = grouped['non_events'].sum()
    
    if total_events == 0 or total_non_events == 0:
        grouped['woe'] = 0
        grouped['iv'] = 0
        return grouped.reset_index(), 0.0
    
    grouped['event_rate'] = grouped['events'] / total_events
    grouped['non_event_rate'] = grouped['non_events'] / total_non_events
    
    grouped['event_rate'] = grouped['event_rate'].clip(lower=0.0001)
    grouped['non_event_rate'] = grouped['non_event_rate'].clip(lower=0.0001)
    
    grouped['woe'] = np.log(grouped['non_event_rate'] / grouped['event_rate'])
    grouped['iv'] = (grouped['non_event_rate'] - grouped['event_rate']) * grouped['woe']
    
    iv_total = grouped['iv'].sum()
    
    return grouped.reset_index(), iv_total


def woe_transform(X, y, feature, bins=N_BINS_DEFAULT):
    """Transform feature values to WoE."""
    df = pd.DataFrame({feature: X[feature], 'target': y})
    woe_table, _ = calculate_woe_iv(df, feature, 'target', bins)
    
    if df[feature].nunique() <= bins:
        woe_map = dict(zip(woe_table['bin'], woe_table['woe']))
        return X[feature].map(woe_map)
    
    bin_edges = pd.qcut(df[feature].dropna(), q=bins, duplicates='drop', retbins=True)[1]
    bin_labels = pd.cut(X[feature], bins=bin_edges, labels=False, include_lowest=True)
    
    woe_values = woe_table['woe'].values
    result = bin_labels.map(lambda x: woe_values[int(x)] if pd.notna(x) and int(x) < len(woe_values) else 0)
    
    return result


def calculate_iv_for_selection(X, y, bins=N_BINS_DEFAULT):
    results = []
    for col in X.columns:
        try:
            df = pd.DataFrame({col: X[col], 'target': y})
            _, iv = calculate_woe_iv(df, col, 'target', bins)
            results.append({'Feature': col, 'IV': iv})
        except Exception:
            continue
    return pd.DataFrame(results).sort_values('IV', ascending=False).reset_index(drop=True)


def monotonicity_score(woe_table):
    """Calculate monotonicity score of WoE values."""
    woe_values = woe_table['woe'].values
    
    if len(woe_values) < 2:
        return 1.0
    
    diffs = np.diff(woe_values)
    
    if np.all(diffs >= 0):
        return 1.0
    if np.all(diffs <= 0):
        return 1.0
    
    increasing = np.sum(diffs > 0)
    decreasing = np.sum(diffs < 0)
    total = len(diffs)
    
    return max(increasing, decreasing) / total


def select_features_by_iv(iv_df, min_iv=MIN_IV_THRESHOLD, top_n=None):
    filtered = iv_df[iv_df['IV'] >= min_iv].copy()
    if top_n is not None:
        filtered = filtered.head(top_n)
    return filtered['Feature'].tolist()


def get_woe_mappings(X, y, features, bins=N_BINS_DEFAULT):
    """Get WoE mappings for multiple features."""
    mappings = {}
    
    for feature in features:
        df = pd.DataFrame({feature: X[feature], 'target': y})
        woe_table, iv = calculate_woe_iv(df, feature, 'target', bins)
        mappings[feature] = {
            'woe_table': woe_table,
            'iv': iv
        }
    
    return mappings


def transform_to_woe(X, y, features, bins=N_BINS_DEFAULT):
    """Transform multiple features to WoE values."""
    X_woe = pd.DataFrame(index=X.index)
    
    for feature in features:
        X_woe[feature] = woe_transform(X, y, feature, bins)
    
    return X_woe


def interpret_iv(iv_value):
    """Interpret IV value strength."""
    if iv_value < 0.02:
        return 'Not useful'
    elif iv_value < 0.1:
        return 'Weak'
    elif iv_value < 0.3:
        return 'Medium'
    elif iv_value < 0.5:
        return 'Strong'
    else:
        return 'Suspicious'
