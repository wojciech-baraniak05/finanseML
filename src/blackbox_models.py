"""Black-box models: XGBoost, LightGBM, RandomForest with Bayesian optimization."""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, log_loss
from .config import RANDOM_STATE, BAYESIAN_N_ITER, CV_FOLDS

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from skopt import BayesSearchCV
    from skopt.space import Real, Integer
    BAYESIAN_AVAILABLE = True
except ImportError:
    BAYESIAN_AVAILABLE = False


def train_xgboost_bayesian(X_train, y_train, X_val=None, y_val=None, 
                           n_iter=BAYESIAN_N_ITER, cv=CV_FOLDS, random_state=RANDOM_STATE):
    """Train XGBoost with Bayesian optimization."""
    if not XGBOOST_AVAILABLE:
        raise ImportError("XGBoost not installed")
    
    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    
    if BAYESIAN_AVAILABLE:
        search_spaces = {
            'n_estimators': Integer(50, 300),
            'max_depth': Integer(3, 10),
            'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
            'subsample': Real(0.6, 1.0),
            'colsample_bytree': Real(0.6, 1.0),
            'min_child_weight': Integer(1, 10),
            'gamma': Real(0, 5),
            'reg_alpha': Real(0, 10),
            'reg_lambda': Real(0, 10)
        }
        
        base_model = XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            random_state=random_state,
            eval_metric='logloss',
            use_label_encoder=False,
            verbosity=0
        )
        
        opt = BayesSearchCV(
            base_model, search_spaces, n_iter=n_iter, cv=cv,
            scoring='roc_auc', n_jobs=-1, random_state=random_state, verbose=0
        )
        opt.fit(X_train, y_train)
        
        return opt.best_estimator_, opt.best_params_
    else:
        model = XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=random_state, eval_metric='logloss',
            use_label_encoder=False, verbosity=0
        )
        model.fit(X_train, y_train)
        return model, {}


def train_lightgbm_bayesian(X_train, y_train, X_val=None, y_val=None,
                            n_iter=BAYESIAN_N_ITER, cv=CV_FOLDS, random_state=RANDOM_STATE):
    """Train LightGBM with Bayesian optimization."""
    if not LIGHTGBM_AVAILABLE:
        raise ImportError("LightGBM not installed")
    
    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    
    if BAYESIAN_AVAILABLE:
        search_spaces = {
            'n_estimators': Integer(50, 300),
            'max_depth': Integer(3, 10),
            'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
            'subsample': Real(0.6, 1.0),
            'colsample_bytree': Real(0.6, 1.0),
            'min_child_samples': Integer(10, 100),
            'reg_alpha': Real(0, 10),
            'reg_lambda': Real(0, 10),
            'num_leaves': Integer(20, 150)
        }
        
        base_model = lgb.LGBMClassifier(
            scale_pos_weight=scale_pos_weight,
            random_state=random_state, verbose=-1
        )
        
        opt = BayesSearchCV(
            base_model, search_spaces, n_iter=n_iter, cv=cv,
            scoring='roc_auc', n_jobs=-1, random_state=random_state, verbose=0
        )
        opt.fit(X_train, y_train)
        
        return opt.best_estimator_, opt.best_params_
    else:
        model = lgb.LGBMClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=random_state, verbose=-1
        )
        model.fit(X_train, y_train)
        return model, {}


def train_random_forest(X_train, y_train, n_iter=BAYESIAN_N_ITER, 
                       cv=CV_FOLDS, random_state=RANDOM_STATE):
    """Train Random Forest with Bayesian optimization."""
    class_weight = 'balanced'
    
    if BAYESIAN_AVAILABLE:
        search_spaces = {
            'n_estimators': Integer(50, 300),
            'max_depth': Integer(3, 20),
            'min_samples_split': Integer(2, 20),
            'min_samples_leaf': Integer(1, 10),
            'max_features': Real(0.3, 1.0)
        }
        
        base_model = RandomForestClassifier(
            class_weight=class_weight, random_state=random_state, n_jobs=-1
        )
        
        opt = BayesSearchCV(
            base_model, search_spaces, n_iter=n_iter, cv=cv,
            scoring='roc_auc', n_jobs=-1, random_state=random_state, verbose=0
        )
        opt.fit(X_train, y_train)
        
        return opt.best_estimator_, opt.best_params_
    else:
        model = RandomForestClassifier(
            n_estimators=100, max_depth=10, min_samples_split=5,
            class_weight=class_weight, random_state=random_state, n_jobs=-1
        )
        model.fit(X_train, y_train)
        return model, {}


def check_overfitting(model, X_train, y_train, X_val, y_val):
    """Check overfitting by comparing train vs validation metrics."""
    y_train_proba = model.predict_proba(X_train)[:, 1]
    y_val_proba = model.predict_proba(X_val)[:, 1]
    
    train_auc = roc_auc_score(y_train, y_train_proba)
    val_auc = roc_auc_score(y_val, y_val_proba)
    train_logloss = log_loss(y_train, y_train_proba)
    val_logloss = log_loss(y_val, y_val_proba)
    
    return {
        'train_auc': train_auc,
        'val_auc': val_auc,
        'auc_gap': train_auc - val_auc,
        'train_logloss': train_logloss,
        'val_logloss': val_logloss,
        'logloss_gap': val_logloss - train_logloss,
        'is_overfitting': (train_auc - val_auc) > 0.05
    }
