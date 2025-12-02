"""Data loading and preprocessing utilities."""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from .config import RANDOM_STATE, TEST_SIZE, VAL_SIZE


def load_data(filepath, target_col='class'):
    """Load dataset from CSV file."""
    df = pd.read_csv(filepath)
    
    if target_col in df.columns:
        X = df.drop(columns=[target_col])
        y = df[target_col]
        return X, y
    
    return df, None


def split_data(X, y, test_size=TEST_SIZE, val_size=VAL_SIZE, random_state=RANDOM_STATE):
    """Split data into train, validation and test sets."""
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    val_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_adjusted, random_state=random_state, stratify=y_temp
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def clean_data(X, strategy='median'):
    """Clean data: handle NaN and inf values."""
    X_clean = X.copy()
    
    X_clean = X_clean.replace([np.inf, -np.inf], np.nan)
    
    numeric_cols = X_clean.select_dtypes(include=[np.number]).columns
    
    if X_clean[numeric_cols].isna().any().any():
        imputer = SimpleImputer(strategy=strategy)
        X_clean[numeric_cols] = imputer.fit_transform(X_clean[numeric_cols])
    
    return X_clean


def get_data_summary(X, y=None):
    """Get summary statistics of the dataset."""
    summary = {
        'n_samples': len(X),
        'n_features': X.shape[1],
        'numeric_features': X.select_dtypes(include=[np.number]).shape[1],
        'categorical_features': X.select_dtypes(include=['object']).shape[1],
        'missing_values': X.isna().sum().sum(),
        'missing_pct': (X.isna().sum().sum() / X.size) * 100,
        'duplicate_rows': X.duplicated().sum()
    }
    
    if y is not None:
        summary['target_distribution'] = y.value_counts().to_dict()
        summary['target_rate'] = y.mean() if y.dtype in [int, float] else None
        summary['class_balance'] = y.value_counts(normalize=True).to_dict()
    
    return summary


def combine_train_val(X_train, X_val, y_train, y_val):
    """Combine train and validation sets."""
    X_combined = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
    y_combined = pd.concat([y_train, y_val], axis=0).reset_index(drop=True)
    return X_combined, y_combined


def load_and_prepare_data(filepath, target_col='class'):
    """Load data and prepare train/val/test splits."""
    X, y = load_data(filepath, target_col)
    X = clean_data(X)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    
    return {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'feature_names': X.columns.tolist()
    }
