"""Feature engineering for credit scoring models."""

import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from .config import VIF_THRESHOLD, CORRELATION_THRESHOLD


def create_financial_ratios(X):
    """Create financial ratios from raw features."""
    X_new = X.copy()
    
    cols = X_new.columns.str.lower()
    col_map = {c.lower(): c for c in X_new.columns}
    
    if 'attr27' in cols and 'attr1' in cols:
        denom = X_new[col_map['attr1']].replace(0, np.nan)
        X_new['profit_margin'] = X_new[col_map['attr27']] / denom
    
    if 'attr1' in cols and 'attr10' in cols:
        denom = X_new[col_map['attr10']].replace(0, np.nan)
        X_new['asset_turnover'] = X_new[col_map['attr1']] / denom
    
    if 'attr4' in cols and 'attr8' in cols:
        denom = X_new[col_map['attr8']].replace(0, np.nan)
        X_new['current_ratio'] = X_new[col_map['attr4']] / denom
    
    if 'attr8' in cols and 'attr10' in cols:
        denom = X_new[col_map['attr10']].replace(0, np.nan)
        X_new['debt_ratio'] = X_new[col_map['attr8']] / denom
    
    if 'attr27' in cols and 'attr14' in cols:
        denom = X_new[col_map['attr14']].replace(0, np.nan)
        X_new['roa'] = X_new[col_map['attr27']] / denom
    
    if 'attr27' in cols and 'attr10' in cols and 'attr8' in cols:
        equity = X_new[col_map['attr10']] - X_new[col_map['attr8']]
        equity = equity.replace(0, np.nan)
        X_new['roe'] = X_new[col_map['attr27']] / equity
    
    if 'attr10' in cols and 'attr8' in cols:
        equity = X_new[col_map['attr10']] - X_new[col_map['attr8']]
        equity = equity.replace(0, np.nan)
        X_new['leverage'] = X_new[col_map['attr10']] / equity
    
    if 'attr4' in cols and 'attr8' in cols:
        X_new['working_capital'] = X_new[col_map['attr4']] - X_new[col_map['attr8']]
    
    X_new = X_new.replace([np.inf, -np.inf], np.nan)
    
    return X_new


def apply_vif_cleaning(X, threshold=VIF_THRESHOLD):
    """Remove features with high VIF iteratively."""
    X_clean = X.copy()
    X_clean = X_clean.replace([np.inf, -np.inf], np.nan)
    X_clean = X_clean.dropna(axis=1, how='any')
    
    if X_clean.shape[1] < 2:
        return X_clean
    
    removed_features = []
    
    while True:
        if X_clean.shape[1] < 2:
            break
            
        vif_data = []
        for i in range(X_clean.shape[1]):
            try:
                vif = variance_inflation_factor(X_clean.values, i)
                vif_data.append((X_clean.columns[i], vif))
            except Exception:
                vif_data.append((X_clean.columns[i], np.inf))
        
        vif_df = pd.DataFrame(vif_data, columns=['feature', 'vif'])
        vif_df = vif_df.replace([np.inf, -np.inf], 999)
        
        max_vif = vif_df['vif'].max()
        
        if max_vif <= threshold:
            break
        
        feature_to_remove = vif_df.loc[vif_df['vif'].idxmax(), 'feature']
        X_clean = X_clean.drop(columns=[feature_to_remove])
        removed_features.append(feature_to_remove)
    
    return X_clean


def apply_correlation_clustering(X, y, threshold=CORRELATION_THRESHOLD):
    """Select representative features from correlation clusters."""
    corr_matrix = X.corr().abs()
    
    np.fill_diagonal(corr_matrix.values, 0)
    
    distance_matrix = 1 - corr_matrix
    distance_matrix = distance_matrix.clip(lower=0)
    
    try:
        condensed_dist = squareform(distance_matrix.values)
        linkage_matrix = linkage(condensed_dist, method='complete')
        clusters = fcluster(linkage_matrix, t=1-threshold, criterion='distance')
    except Exception:
        return X
    
    cluster_df = pd.DataFrame({
        'feature': X.columns,
        'cluster': clusters
    })
    
    correlations_with_target = X.corrwith(pd.Series(y, index=X.index)).abs()
    
    selected_features = []
    for cluster_id in cluster_df['cluster'].unique():
        cluster_features = cluster_df[cluster_df['cluster'] == cluster_id]['feature'].tolist()
        
        if len(cluster_features) == 1:
            selected_features.append(cluster_features[0])
        else:
            best_feature = max(cluster_features, 
                             key=lambda f: correlations_with_target.get(f, 0))
            selected_features.append(best_feature)
    
    return X[selected_features]


def remove_constant_features(X, threshold=0.01):
    """Remove features with very low variance."""
    variances = X.var()
    low_variance = variances[variances < threshold].index.tolist()
    return X.drop(columns=low_variance, errors='ignore')


def remove_highly_correlated(X, threshold=CORRELATION_THRESHOLD):
    """Remove one of each pair of highly correlated features."""
    corr_matrix = X.corr().abs()
    upper_tri = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    to_drop = [col for col in upper_tri.columns 
               if any(upper_tri[col] > threshold)]
    
    return X.drop(columns=to_drop, errors='ignore')


def get_feature_importance_by_correlation(X, y):
    """Rank features by absolute correlation with target."""
    correlations = X.corrwith(pd.Series(y, index=X.index)).abs()
    return correlations.sort_values(ascending=False)
