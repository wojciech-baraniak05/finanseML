"""Credit Scorecard implementation with WoE transformation."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from .config import (
    BASE_SCORE, PDO, TARGET_ODDS, RANDOM_STATE, 
    CORRELATION_THRESHOLD, N_BINS_DEFAULT
)
from .woe_binning import calculate_woe_iv, transform_to_woe, select_features_by_iv, calculate_iv_for_selection
from .metrics import calculate_all_metrics, calculate_ks_statistic


class ScorecardPreprocessingPipeline:
    """Preprocessing pipeline for scorecard: remove NaN, categorical, constant; winsorize; remove correlated."""
    
    def __init__(self, correlation_threshold=CORRELATION_THRESHOLD):
        self.correlation_threshold = correlation_threshold
        self.columns_to_drop = []
        self.correlated_columns = []
        self.winsorization_limits = {}
        self.final_columns = None
    
    def fit(self, X, y=None):
        X_work = X.copy()
        
        self.columns_to_drop = []
        self.columns_to_drop.extend(X_work.columns[X_work.isna().any()].tolist())
        self.columns_to_drop.extend(X_work.select_dtypes(include=['object']).columns.tolist())
        
        for col in X_work.select_dtypes(include=[float, int]).columns:
            if X_work[col].nunique() == 1:
                self.columns_to_drop.append(col)
        
        self.columns_to_drop = list(set(self.columns_to_drop))
        X_work = X_work.drop(columns=self.columns_to_drop, errors='ignore')
        
        self.winsorization_limits = {}
        for col in X_work.columns:
            lower = X_work[col].quantile(0.01)
            upper = X_work[col].quantile(0.99)
            self.winsorization_limits[col] = {'lower': lower, 'upper': upper}
            X_work[col] = X_work[col].clip(lower=lower, upper=upper)
        
        corr_matrix = X_work.corr().abs()
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        self.correlated_columns = [
            col for col in upper_tri.columns 
            if any(upper_tri[col] > self.correlation_threshold)
        ]
        X_work = X_work.drop(columns=self.correlated_columns, errors='ignore')
        
        self.final_columns = X_work.columns.tolist()
        return self
    
    def transform(self, X):
        X_work = X.copy()
        X_work = X_work.drop(columns=self.columns_to_drop, errors='ignore')
        X_work = X_work.drop(columns=self.correlated_columns, errors='ignore')
        
        for col in self.final_columns:
            if col in X_work.columns:
                limits = self.winsorization_limits[col]
                X_work[col] = X_work[col].clip(lower=limits['lower'], upper=limits['upper'])
        
        return X_work[self.final_columns]
    
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


class CreditScorecard:
    """Credit scorecard using WoE transformation and logistic regression."""
    
    def __init__(self, base_score=BASE_SCORE, pdo=PDO, target_odds=TARGET_ODDS):
        self.base_score = base_score
        self.pdo = pdo
        self.target_odds = target_odds
        
        self.factor = pdo / np.log(2)
        self.offset = base_score - self.factor * np.log(target_odds)
        
        self.model = None
        self.features = None
        self.woe_mappings = {}
        self.feature_bins = {}
        self.scorecard_table = None
    
    def fit(self, X_train, y_train, features=None, bins=N_BINS_DEFAULT, C=1.0):
        """Fit scorecard model."""
        if features is None:
            iv_df = calculate_iv_for_selection(X_train, y_train, bins)
            features = select_features_by_iv(iv_df, top_n=30)
        
        self.features = features
        self.feature_bins = {f: bins for f in features}
        
        for feature in features:
            df = pd.DataFrame({feature: X_train[feature], 'target': y_train})
            woe_table, iv = calculate_woe_iv(df, feature, 'target', bins)
            self.woe_mappings[feature] = {
                'woe_table': woe_table,
                'iv': iv,
                'bins': bins
            }
        
        X_woe = transform_to_woe(X_train, y_train, features, bins)
        X_woe = X_woe.fillna(0)
        
        self.model = LogisticRegression(
            C=C, 
            solver='lbfgs', 
            max_iter=1000, 
            random_state=RANDOM_STATE
        )
        self.model.fit(X_woe, y_train)
        
        self._build_scorecard_table()
        
        return self
    
    def _build_scorecard_table(self):
        """Build scorecard points table."""
        records = []
        
        for i, feature in enumerate(self.features):
            coef = self.model.coef_[0][i]
            woe_table = self.woe_mappings[feature]['woe_table']
            
            for _, row in woe_table.iterrows():
                points = -(coef * row['woe'] + self.model.intercept_[0] / len(self.features)) * self.factor
                
                records.append({
                    'feature': feature,
                    'bin': row['bin'],
                    'woe': row['woe'],
                    'coefficient': coef,
                    'points': points,
                    'count': row['total'],
                    'event_rate': row['events'] / row['total'] if row['total'] > 0 else 0
                })
        
        self.scorecard_table = pd.DataFrame(records)
    
    def transform_to_woe(self, X, y=None):
        """Transform features to WoE values."""
        if y is None:
            X_woe = pd.DataFrame(index=X.index)
            for feature in self.features:
                woe_table = self.woe_mappings[feature]['woe_table']
                bins = self.feature_bins[feature]
                
                if X[feature].nunique() <= bins:
                    woe_map = dict(zip(woe_table['bin'], woe_table['woe']))
                    X_woe[feature] = X[feature].map(woe_map).fillna(0)
                else:
                    bin_edges = pd.qcut(
                        X[feature].dropna(), q=bins, 
                        duplicates='drop', retbins=True
                    )[1]
                    bin_labels = pd.cut(
                        X[feature], bins=bin_edges, 
                        labels=False, include_lowest=True
                    )
                    woe_values = woe_table['woe'].values
                    X_woe[feature] = bin_labels.map(
                        lambda x: woe_values[int(x)] if pd.notna(x) and int(x) < len(woe_values) else 0
                    )
            return X_woe
        else:
            return transform_to_woe(X, y, self.features, N_BINS_DEFAULT)
    
    def predict_proba(self, X):
        """Predict probability of default."""
        X_woe = self.transform_to_woe(X)
        X_woe = X_woe.fillna(0)
        return self.model.predict_proba(X_woe)[:, 1]
    
    def predict(self, X, threshold=0.5):
        """Predict default class."""
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)
    
    def predict_score(self, X):
        """Predict credit score."""
        proba = self.predict_proba(X)
        proba = np.clip(proba, 0.0001, 0.9999)
        odds = (1 - proba) / proba
        scores = self.offset + self.factor * np.log(odds)
        return scores
    
    def get_scorecard_table(self):
        """Get scorecard points table."""
        return self.scorecard_table
    
    def get_feature_importance(self):
        """Get feature importance based on IV and coefficients."""
        importance = []
        
        for i, feature in enumerate(self.features):
            importance.append({
                'feature': feature,
                'iv': self.woe_mappings[feature]['iv'],
                'coefficient': abs(self.model.coef_[0][i]),
                'importance': self.woe_mappings[feature]['iv'] * abs(self.model.coef_[0][i])
            })
        
        return pd.DataFrame(importance).sort_values('importance', ascending=False)
    
    def evaluate(self, X, y):
        """Evaluate scorecard performance."""
        y_proba = self.predict_proba(X)
        y_pred = self.predict(X)
        return calculate_all_metrics(y, y_pred, y_proba)
    
    def grid_search(self, X_train, y_train, X_val, y_val, 
                   C_values=[0.01, 0.1, 1.0, 10.0],
                   bin_options=[5, 10, 15, 20],
                   top_n_features=[10, 20, 30]):
        """Grid search for optimal hyperparameters."""
        best_score = 0
        best_params = {}
        results = []
        
        for n_features in top_n_features:
            for bins in bin_options:
                for C in C_values:
                    try:
                        self.fit(X_train, y_train, features=None, bins=bins, C=C)
                        
                        if len(self.features) > n_features:
                            top_features = self.features[:n_features]
                            self.fit(X_train, y_train, features=top_features, bins=bins, C=C)
                        
                        metrics = self.evaluate(X_val, y_val)
                        
                        results.append({
                            'n_features': len(self.features),
                            'bins': bins,
                            'C': C,
                            'roc_auc': metrics['roc_auc'],
                            'ks': metrics['ks_statistic'],
                            'gini': metrics['gini']
                        })
                        
                        if metrics['roc_auc'] > best_score:
                            best_score = metrics['roc_auc']
                            best_params = {
                                'features': self.features.copy(),
                                'bins': bins,
                                'C': C,
                                'metrics': metrics
                            }
                    
                    except Exception:
                        continue
        
        if best_params:
            self.fit(X_train, y_train, features=best_params['features'], 
                    bins=best_params['bins'], C=best_params['C'])
        
        return best_params, pd.DataFrame(results)
