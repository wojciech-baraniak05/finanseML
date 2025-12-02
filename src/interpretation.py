"""Model interpretation: SHAP, LIME, log-odds decomposition, case studies."""

import numpy as np
import pandas as pd

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    from lime import lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False


def get_shap_values(model, X_background, X_explain, model_type='tree'):
    """Calculate SHAP values for model."""
    if not SHAP_AVAILABLE:
        raise ImportError("SHAP not installed")
    
    if model_type == 'tree':
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_explain)
    elif model_type == 'linear':
        explainer = shap.LinearExplainer(model, X_background)
        shap_values = explainer.shap_values(X_explain)
    else:
        explainer = shap.KernelExplainer(model.predict_proba, X_background)
        shap_values = explainer.shap_values(X_explain)
    
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    
    return shap_values, explainer


def get_shap_explanation(model, X_background, X_explain, model_type='tree'):
    """Get SHAP Explanation object for plotting."""
    if not SHAP_AVAILABLE:
        raise ImportError("SHAP not installed")
    
    shap_values, explainer = get_shap_values(model, X_background, X_explain, model_type)
    
    return shap.Explanation(
        values=shap_values,
        base_values=explainer.expected_value if hasattr(explainer, 'expected_value') else None,
        data=X_explain,
        feature_names=X_explain.columns.tolist() if hasattr(X_explain, 'columns') else None
    )


def get_lime_explanation(model, X_train, X_instance, feature_names=None, num_features=10):
    """Get LIME explanation for single instance."""
    if not LIME_AVAILABLE:
        raise ImportError("LIME not installed")
    
    if feature_names is None:
        feature_names = X_train.columns.tolist() if hasattr(X_train, 'columns') else None
    
    explainer = lime_tabular.LimeTabularExplainer(
        X_train.values if hasattr(X_train, 'values') else X_train,
        feature_names=feature_names,
        class_names=['No Default', 'Default'],
        mode='classification'
    )
    
    instance = X_instance.values if hasattr(X_instance, 'values') else X_instance
    
    return explainer.explain_instance(
        instance.flatten(), model.predict_proba, num_features=num_features
    )


def get_lime_contributions(exp):
    """Extract LIME contributions as dictionary."""
    return dict(exp.as_list())


def decompose_log_odds(model, X_woe, observation_idx, feature_names=None):
    """Decompose log-odds for logistic regression scorecard."""
    if feature_names is None:
        feature_names = X_woe.columns.tolist() if hasattr(X_woe, 'columns') else None
    
    if hasattr(observation_idx, '__iter__') and not isinstance(observation_idx, (str, int)):
        obs = X_woe.iloc[observation_idx[0]] if hasattr(X_woe, 'iloc') else X_woe[observation_idx[0]]
    else:
        obs = X_woe.iloc[observation_idx] if hasattr(X_woe, 'iloc') else X_woe[observation_idx]
    
    intercept = model.intercept_[0]
    coefficients = model.coef_[0]
    
    contributions = {}
    for i, (coef, value) in enumerate(zip(coefficients, obs)):
        feat_name = feature_names[i] if feature_names else f'feature_{i}'
        contributions[feat_name] = coef * value
    
    log_odds = intercept + sum(contributions.values())
    probability = 1 / (1 + np.exp(-log_odds))
    
    return {
        'intercept': intercept,
        'contributions': contributions,
        'log_odds': log_odds,
        'probability': probability
    }


def select_representative_cases(y_true, y_pred, y_proba, n_per_type=2):
    """Select representative cases for each prediction type (TP, TN, FP, FN)."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_proba = np.array(y_proba)
    
    tp_mask = (y_true == 1) & (y_pred == 1)
    tn_mask = (y_true == 0) & (y_pred == 0)
    fp_mask = (y_true == 0) & (y_pred == 1)
    fn_mask = (y_true == 1) & (y_pred == 0)
    
    def select_from_mask(mask, proba, n, high_confidence=True):
        indices = np.where(mask)[0]
        if len(indices) == 0:
            return []
        
        proba_subset = proba[indices]
        if high_confidence:
            sorted_idx = np.argsort(proba_subset)[::-1]
        else:
            sorted_idx = np.argsort(np.abs(proba_subset - 0.5))
        
        return indices[sorted_idx[:n]].tolist()
    
    return {
        'true_positives': select_from_mask(tp_mask, y_proba, n_per_type, True),
        'true_negatives': select_from_mask(tn_mask, 1-y_proba, n_per_type, True),
        'false_positives': select_from_mask(fp_mask, y_proba, n_per_type, True),
        'false_negatives': select_from_mask(fn_mask, 1-y_proba, n_per_type, True)
    }


def analyze_case_study(idx, X_woe, X_raw, y_true, model, woe_mappings=None, feature_names=None):
    """Analyze single case study with detailed breakdown."""
    if feature_names is None:
        feature_names = X_woe.columns.tolist() if hasattr(X_woe, 'columns') else None
    
    decomposition = decompose_log_odds(model, X_woe, idx, feature_names)
    
    raw_values = X_raw.iloc[idx] if hasattr(X_raw, 'iloc') else X_raw[idx]
    woe_values = X_woe.iloc[idx] if hasattr(X_woe, 'iloc') else X_woe[idx]
    
    sorted_contributions = sorted(
        decomposition['contributions'].items(), 
        key=lambda x: abs(x[1]), 
        reverse=True
    )
    
    return {
        'index': idx,
        'true_label': y_true.iloc[idx] if hasattr(y_true, 'iloc') else y_true[idx],
        'predicted_probability': decomposition['probability'],
        'log_odds': decomposition['log_odds'],
        'intercept': decomposition['intercept'],
        'contributions': sorted_contributions,
        'raw_values': raw_values.to_dict() if hasattr(raw_values, 'to_dict') else dict(raw_values),
        'woe_values': woe_values.to_dict() if hasattr(woe_values, 'to_dict') else dict(woe_values)
    }


def get_top_features_by_importance(shap_values, feature_names, n_top=10):
    """Get top N features by mean |SHAP value|."""
    importance = np.abs(shap_values).mean(axis=0)
    
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(len(importance))]
    
    indices = np.argsort(importance)[::-1][:n_top]
    
    return [(feature_names[i], importance[i]) for i in indices]


def get_feature_importance_df(shap_values, feature_names):
    """Get feature importance as DataFrame."""
    importance = np.abs(shap_values).mean(axis=0)
    
    df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance,
        'mean_shap': shap_values.mean(axis=0),
        'std_shap': shap_values.std(axis=0)
    })
    
    return df.sort_values('importance', ascending=False).reset_index(drop=True)
