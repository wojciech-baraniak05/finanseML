"""
Preprocessing pipelines for credit scoring.

PRODUCTION: InterpretablePreprocessingPipeline
EXPERIMENTAL: MinimalPreprocessingPipeline
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, PowerTransformer
from scipy.stats import boxcox
from .config import CORRELATION_THRESHOLD, WINSORIZATION_LOWER, WINSORIZATION_UPPER


def identify_columns_to_drop(X, sparsity_threshold=0.95):
    """Identify columns to drop based on missing data, categorical type, and sparsity."""
    to_drop = []
    sparse_to_binary = []
    
    missing_counts = X.isna().sum()
    to_drop.extend(missing_counts[missing_counts > 0].index.tolist())
    
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    to_drop.extend(categorical_cols)
    
    for col in X.select_dtypes(include=[np.number]).columns:
        if X[col].nunique() == 1:
            to_drop.append(col)
            continue
        
        zero_ratio = (X[col] == 0).sum() / len(X[col])
        if zero_ratio > sparsity_threshold:
            sparse_to_binary.append(col)
            to_drop.append(col)
    
    return {
        'to_drop': list(set(to_drop)),
        'sparse_to_binary': list(set(sparse_to_binary))
    }


def calculate_winsorization_limits(X, lower=WINSORIZATION_LOWER, upper=WINSORIZATION_UPPER):
    """Calculate winsorization limits for each column."""
    limits = {}
    for col in X.columns:
        limits[col] = {
            'lower': X[col].quantile(lower),
            'upper': X[col].quantile(upper)
        }
    return limits


class InterpretableColumnTransformer:
    """Transform single column with Box-Cox, Yeo-Johnson or log1p."""
    
    def __init__(self, column_name, method='box-cox'):
        self.column_name = column_name
        self.method = method
        self.transformer = None
        self.scaler = StandardScaler()
        self.shift = 0
        self.lambda_ = None
        
    def fit(self, X, y=None):
        data = np.array(X).flatten()
        data = data[~np.isnan(data)]
        
        if len(data) == 0:
            return self
        
        if self.method == 'box-cox':
            if (data <= 0).any():
                self.shift = abs(data.min()) + 1
                data = data + self.shift
            
            try:
                _, self.lambda_ = boxcox(data)
                self.transformer = PowerTransformer(method='box-cox', standardize=False)
                self.transformer.fit(data.reshape(-1, 1))
            except Exception:
                self.method = 'yeo-johnson'
                self.transformer = PowerTransformer(method='yeo-johnson', standardize=False)
                self.transformer.fit((data - self.shift).reshape(-1, 1))
        
        elif self.method == 'yeo-johnson':
            self.transformer = PowerTransformer(method='yeo-johnson', standardize=False)
            self.transformer.fit(data.reshape(-1, 1))
        
        elif self.method == 'log1p':
            if (data < 0).any():
                self.shift = abs(data.min())
                data = data + self.shift
        
        if self.method in ['box-cox', 'yeo-johnson']:
            data_transformed = self.transformer.transform(data.reshape(-1, 1)).flatten()
        elif self.method == 'log1p':
            data_transformed = np.log1p(data)
        else:
            data_transformed = data
        
        self.scaler.fit(data_transformed.reshape(-1, 1))
        return self
    
    def transform(self, X):
        data = np.array(X).flatten()
        
        if self.method == 'box-cox':
            data = data + self.shift
        
        if self.method in ['box-cox', 'yeo-johnson']:
            data = self.transformer.transform(data.reshape(-1, 1)).flatten()
        elif self.method == 'log1p':
            data = np.log1p(data + self.shift)
        
        std = self.scaler.scale_[0]
        if std > 1e-10:
            data_transformed = self.scaler.transform(np.array(data).reshape(-1, 1)).flatten()
        else:
            data_transformed = np.zeros_like(data)
        
        return pd.Series(data_transformed, index=X.index)
    
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


class InterpretablePreprocessingPipeline:
    """
    PRODUCTION pipeline with Box-Cox transformations, winsorization and correlation removal.
    
    Steps:
    1. Remove columns with missing values, categorical, constant
    2. Winsorize outliers (1%-99%)
    3. Remove highly correlated features
    4. Apply Box-Cox/Yeo-Johnson transformation
    5. Standardize (mean=0, std=1)
    """
    
    def __init__(self, correlation_threshold=CORRELATION_THRESHOLD):
        self.correlation_threshold = correlation_threshold
        self.columns_info = None
        self.winsorization_limits = None
        self.transformers = {}
        self.final_columns = None
        self.correlated_columns = None
        
    def fit(self, X, y=None):
        X_work = X.copy()
        
        self.columns_info = identify_columns_to_drop(X_work)
        X_work = X_work.drop(columns=self.columns_info['to_drop'], errors='ignore')
        
        numeric_cols = X_work.columns.tolist()
        self.winsorization_limits = calculate_winsorization_limits(X_work)
        
        for col in numeric_cols:
            X_work[col] = X_work[col].clip(
                lower=self.winsorization_limits[col]['lower'],
                upper=self.winsorization_limits[col]['upper']
            )
        
        corr_matrix = X_work.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        self.correlated_columns = [col for col in upper_tri.columns 
                                   if any(upper_tri[col] > self.correlation_threshold)]
        X_work = X_work.drop(columns=self.correlated_columns, errors='ignore')
        numeric_cols = [c for c in numeric_cols if c not in self.correlated_columns]
        
        for col in numeric_cols:
            self.transformers[col] = InterpretableColumnTransformer(col)
            self.transformers[col].fit(X_work[col])
        
        self.final_columns = numeric_cols
        return self
    
    def transform(self, X):
        X_work = X.copy()
        
        X_work = X_work.drop(columns=self.columns_info['to_drop'], errors='ignore')
        X_work = X_work.drop(columns=self.correlated_columns, errors='ignore')
        
        for col in self.transformers.keys():
            if col in X_work.columns:
                X_work[col] = X_work[col].clip(
                    lower=self.winsorization_limits[col]['lower'],
                    upper=self.winsorization_limits[col]['upper']
                )
        
        result = pd.DataFrame(index=X_work.index)
        for col in self.transformers.keys():
            if col in X_work.columns:
                result[col] = self.transformers[col].transform(X_work[col])
        
        return result
    
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


class MinimalPreprocessingPipeline:
    """
    EXPERIMENTAL pipeline with basic cleaning and standardization.
    
    Steps:
    1. Remove columns with missing values, categorical, constant
    2. Winsorize outliers (1%-99%)
    3. Remove highly correlated features
    4. Standardize (mean=0, std=1)
    """
    
    def __init__(self, correlation_threshold=CORRELATION_THRESHOLD, standardize=True):
        self.correlation_threshold = correlation_threshold
        self.standardize = standardize
        self.columns_to_drop = []
        self.correlated_columns = []
        self.winsorization_limits = None
        self.scaler = StandardScaler() if standardize else None
        self.final_columns = None
        
    def fit(self, X, y=None):
        X_work = X.copy()
        
        self.columns_to_drop = []
        self.columns_to_drop.extend(X_work.columns[X_work.isna().any()].tolist())
        self.columns_to_drop.extend(X_work.select_dtypes(include=['object']).columns.tolist())
        
        for col in X_work.select_dtypes(include=[np.number]).columns:
            if X_work[col].nunique() == 1:
                self.columns_to_drop.append(col)
        
        self.columns_to_drop = list(set(self.columns_to_drop))
        X_work = X_work.drop(columns=self.columns_to_drop, errors='ignore')
        
        self.winsorization_limits = calculate_winsorization_limits(X_work)
        for col in X_work.columns:
            X_work[col] = X_work[col].clip(
                lower=self.winsorization_limits[col]['lower'],
                upper=self.winsorization_limits[col]['upper']
            )
        
        corr_matrix = X_work.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        self.correlated_columns = [col for col in upper_tri.columns 
                                   if any(upper_tri[col] > self.correlation_threshold)]
        X_work = X_work.drop(columns=self.correlated_columns, errors='ignore')
        
        if self.standardize:
            self.scaler.fit(X_work)
        
        self.final_columns = X_work.columns.tolist()
        return self
    
    def transform(self, X):
        X_work = X.copy()
        
        X_work = X_work.drop(columns=self.columns_to_drop, errors='ignore')
        X_work = X_work.drop(columns=self.correlated_columns, errors='ignore')
        
        for col in self.final_columns:
            if col in X_work.columns and col in self.winsorization_limits:
                X_work[col] = X_work[col].clip(
                    lower=self.winsorization_limits[col]['lower'],
                    upper=self.winsorization_limits[col]['upper']
                )
        
        X_work = X_work[self.final_columns]
        
        if self.standardize:
            X_work = pd.DataFrame(
                self.scaler.transform(X_work),
                columns=X_work.columns,
                index=X_work.index
            )
        
        return X_work
    
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
