import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix, brier_score_loss
from sklearn.calibration import calibration_curve


def plot_data_overview(X, y, figsize=(16, 8)):
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    ax = axes[0]
    y.value_counts().plot(kind='bar', ax=ax, color=['#2ecc71', '#e74c3c'])
    ax.set_title('Target Distribution')
    ax.set_xticklabels(['No Default', 'Default'], rotation=0)
    for c in ax.containers:
        ax.bar_label(c, fmt='%d')
    
    ax = axes[1]
    missing = X.isna().sum().sort_values(ascending=False).head(20)
    if missing.sum() > 0:
        missing.plot(kind='barh', ax=ax, color='#e67e22')
        ax.set_title('TOP 20 Missing Values')
    else:
        ax.text(0.5, 0.5, 'No missing values', ha='center', va='center')
        ax.axis('off')
    
    ax = axes[2]
    num_cnt = len(X.select_dtypes(include=[np.number]).columns)
    cat_cnt = len(X.select_dtypes(include=['object']).columns)
    counts = pd.Series({'Numeric': num_cnt, 'Categorical': cat_cnt})
    counts[counts > 0].plot(kind='bar', ax=ax, color=['#3498db', '#9b59b6'])
    ax.set_title('Column Types')
    for c in ax.containers:
        ax.bar_label(c, fmt='%d')
    
    plt.tight_layout()
    return fig


def plot_correlation_matrix(X, y=None, top_n=30, method='pearson', figsize=(14, 12), annot=False):
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    X_num = X[numeric_cols]
    corr = X_num.corr(method=method)
    
    if y is not None and len(numeric_cols) > top_n:
        target_corr = X_num.corrwith(pd.Series(y.values)).abs().sort_values(ascending=False)
        top_features = target_corr.head(top_n).index.tolist()
        corr = corr.loc[top_features, top_features]
    
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(corr, annot=annot, cmap='coolwarm', center=0, vmin=-1, vmax=1,
                square=True, linewidths=0.5, ax=ax)
    ax.set_title(f'Correlation Matrix (TOP {top_n})' if y is not None else 'Correlation Matrix')
    plt.tight_layout()
    return fig, corr


def plot_target_correlation(X, y, top_n=15, figsize=(12, 8)):
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    correlations = X[numeric_cols].corrwith(pd.Series(y.values))
    corr_df = pd.DataFrame({
        'Feature': correlations.index,
        'Correlation': correlations.values,
        'Abs_Correlation': correlations.abs().values
    }).sort_values('Abs_Correlation', ascending=False).head(top_n)
    
    fig, ax = plt.subplots(figsize=figsize)
    colors = ['#e74c3c' if x > 0 else '#3498db' for x in corr_df['Correlation']]
    ax.barh(range(len(corr_df)), corr_df['Correlation'], color=colors, alpha=0.7)
    ax.set_yticks(range(len(corr_df)))
    ax.set_yticklabels(corr_df['Feature'])
    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.set_title('Feature Correlation with Target')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    return fig, corr_df


def plot_distribution_comparison(data_before, data_after, feature_names=None, max_features=10, figsize=None):
    if feature_names is None:
        common = list(set(data_before.columns) & set(data_after.columns))[:max_features]
    else:
        common = [f for f in feature_names if f in data_before.columns and f in data_after.columns][:max_features]
    
    if not common:
        return None
    
    n_rows = (len(common) + 1) // 2
    figsize = figsize or (14, n_rows * 4)
    fig, axes = plt.subplots(n_rows, 2, figsize=figsize)
    axes = np.atleast_2d(axes).flatten()
    
    for idx, feat in enumerate(common):
        ax = axes[idx]
        ax.hist(data_before[feat].dropna(), bins=50, alpha=0.5, label='Before', color='#e74c3c', density=True)
        ax.hist(data_after[feat].dropna(), bins=50, alpha=0.5, label='After', color='#2ecc71', density=True)
        ax.set_title(feat)
        ax.legend()
    
    for idx in range(len(common), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    return fig


def plot_roc_curves(y_true, y_proba_dict, figsize=(10, 8)):
    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.tab10(np.linspace(0, 1, len(y_proba_dict)))
    
    for (name, y_proba), color in zip(y_proba_dict.items(), colors):
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc = np.trapz(tpr, fpr)
        ax.plot(fpr, tpr, label=f'{name} (AUC={auc:.4f})', color=color, linewidth=2)
    
    ax.plot([0, 1], [0, 1], 'k--', label='Random', alpha=0.5)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig


def plot_precision_recall_curves(y_true, y_proba_dict, figsize=(10, 8)):
    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.tab10(np.linspace(0, 1, len(y_proba_dict)))
    baseline = y_true.sum() / len(y_true)
    
    for (name, y_proba), color in zip(y_proba_dict.items(), colors):
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        pr_auc = np.trapz(precision, recall)
        ax.plot(recall, precision, label=f'{name} (PR-AUC={pr_auc:.4f})', color=color, linewidth=2)
    
    ax.axhline(y=baseline, color='k', linestyle='--', label=f'Baseline ({baseline:.4f})', alpha=0.5)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curves')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig


def plot_confusion_matrices(y_true, y_pred_dict, figsize=None, normalize=False):
    n_models = len(y_pred_dict)
    n_cols = min(3, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols
    figsize = figsize or (n_cols * 5, n_rows * 4)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.atleast_2d(axes).flatten()
    
    for idx, (name, y_pred) in enumerate(y_pred_dict.items()):
        cm = confusion_matrix(y_true, y_pred)
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        sns.heatmap(cm, annot=True, fmt='.2%' if normalize else 'd', cmap='Blues', ax=axes[idx])
        axes[idx].set_title(name)
        axes[idx].set_ylabel('Actual')
        axes[idx].set_xlabel('Predicted')
    
    for idx in range(n_models, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    return fig


def plot_model_comparison(results_dict, metrics=None, figsize=(14, 10)):
    df = pd.DataFrame([{'Model': k, **v} for k, v in results_dict.items()])
    metrics = metrics or [c for c in df.columns if c != 'Model']
    metrics = [m for m in metrics if m in df.columns]
    
    n_cols = 3
    n_rows = (len(metrics) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.atleast_2d(axes).flatten()
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        colors = plt.cm.viridis(np.linspace(0, 1, len(df)))
        ax.barh(range(len(df)), df[metric], color=colors, alpha=0.7)
        ax.set_yticks(range(len(df)))
        ax.set_yticklabels(df['Model'])
        ax.set_xlabel(metric.replace('_', ' ').title())
        ax.grid(axis='x', alpha=0.3)
        for i, v in enumerate(df[metric]):
            ax.text(v, i, f'{v:.4f}', va='center')
    
    for idx in range(len(metrics), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    return fig, df


def plot_iv_ranking(iv_df, top_n=20, figsize=(12, 8)):
    df = iv_df.sort_values('IV', ascending=True).tail(top_n)
    
    def get_color(iv):
        if iv >= 0.5:
            return '#27ae60'
        elif iv >= 0.3:
            return '#f39c12'
        elif iv >= 0.1:
            return '#e67e22'
        return '#e74c3c'
    
    colors = [get_color(iv) for iv in df['IV']]
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(range(len(df)), df['IV'], color=colors, alpha=0.8)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df['Feature'] if 'Feature' in df.columns else df.index)
    ax.set_xlabel('Information Value')
    ax.set_title('Feature Ranking by IV')
    ax.axvline(x=0.1, color='gray', linestyle='--', alpha=0.5, label='Weak (0.1)')
    ax.axvline(x=0.3, color='gray', linestyle='-.', alpha=0.5, label='Medium (0.3)')
    ax.legend()
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    return fig


def plot_woe_bars(woe_table, feature_name, figsize=(10, 6)):
    fig, ax = plt.subplots(figsize=figsize)
    colors = ['#e74c3c' if w > 0 else '#3498db' for w in woe_table['WoE']]
    ax.bar(range(len(woe_table)), woe_table['WoE'], color=colors, alpha=0.8)
    ax.set_xticks(range(len(woe_table)))
    ax.set_xticklabels(woe_table['Bin'] if 'Bin' in woe_table.columns else range(len(woe_table)), rotation=45, ha='right')
    ax.axhline(y=0, color='black', linewidth=0.8)
    ax.set_xlabel('Bin')
    ax.set_ylabel('WoE')
    ax.set_title(f'WoE Analysis: {feature_name}')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    return fig


def plot_calibration_curves(y_true, proba_dict, n_bins=10, figsize=(10, 8)):
    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.tab10(np.linspace(0, 1, len(proba_dict)))
    
    for (name, y_proba), color in zip(proba_dict.items(), colors):
        prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=n_bins)
        ax.plot(prob_pred, prob_true, marker='o', label=name, color=color, linewidth=2)
    
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Fraction of Positives')
    ax.set_title('Calibration Curves')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig


def plot_calibration_histogram(y_true, y_proba, figsize=(10, 6)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(y_proba[y_true == 0], bins=50, alpha=0.5, label='Class 0', color='#3498db', density=True)
    ax.hist(y_proba[y_true == 1], bins=50, alpha=0.5, label='Class 1', color='#e74c3c', density=True)
    ax.set_xlabel('Predicted Probability')
    ax.set_ylabel('Density')
    ax.set_title('Probability Distribution by Class')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig


def plot_score_distribution(scores, y_true=None, figsize=(12, 6)):
    fig, ax = plt.subplots(figsize=figsize)
    
    if y_true is not None:
        ax.hist(scores[y_true == 0], bins=50, alpha=0.5, label='No Default', color='#2ecc71', density=True)
        ax.hist(scores[y_true == 1], bins=50, alpha=0.5, label='Default', color='#e74c3c', density=True)
        ax.legend()
    else:
        ax.hist(scores, bins=50, color='#3498db', alpha=0.7, density=True)
    
    ax.set_xlabel('Credit Score')
    ax.set_ylabel('Density')
    ax.set_title('Score Distribution')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig


def plot_waterfall_log_odds(contributions, feature_names, intercept, final_prob, figsize=(12, 8)):
    sorted_idx = np.argsort(np.abs(contributions))[::-1]
    top_n = min(15, len(contributions))
    idx = sorted_idx[:top_n]
    
    feats = [feature_names[i] for i in idx]
    vals = [contributions[i] for i in idx]
    
    fig, ax = plt.subplots(figsize=figsize)
    colors = ['#e74c3c' if v > 0 else '#2ecc71' for v in vals]
    y_pos = range(len(feats))
    
    ax.barh(y_pos, vals, color=colors, alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feats)
    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.set_xlabel('Log-Odds Contribution')
    ax.set_title(f'Log-Odds Decomposition (Intercept: {intercept:.3f}, P(Default): {final_prob:.3f})')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    return fig


def plot_pdp(model, X, features, figsize=(14, 10)):
    try:
        from sklearn.inspection import PartialDependenceDisplay
    except ImportError:
        return None
    
    fig, ax = plt.subplots(figsize=figsize)
    PartialDependenceDisplay.from_estimator(model, X, features, ax=ax)
    plt.tight_layout()
    return fig


def plot_ice(model, X, features, n_samples=100, figsize=(14, 10)):
    try:
        from sklearn.inspection import PartialDependenceDisplay
    except ImportError:
        return None
    
    sample_idx = np.random.choice(len(X), min(n_samples, len(X)), replace=False)
    X_sample = X.iloc[sample_idx] if hasattr(X, 'iloc') else X[sample_idx]
    
    fig, ax = plt.subplots(figsize=figsize)
    PartialDependenceDisplay.from_estimator(model, X_sample, features, kind='both', ax=ax)
    plt.tight_layout()
    return fig


def plot_shap_summary(shap_values, X, max_display=20, figsize=(10, 8)):
    try:
        import shap
        plt.figure(figsize=figsize)
        shap.summary_plot(shap_values, X, max_display=max_display, show=False)
        fig = plt.gcf()
        plt.tight_layout()
        return fig
    except ImportError:
        return None


def plot_shap_waterfall(shap_explanation, idx=0, max_display=10, figsize=(10, 8)):
    try:
        import shap
        plt.figure(figsize=figsize)
        shap.waterfall_plot(shap_explanation[idx], max_display=max_display, show=False)
        fig = plt.gcf()
        return fig
    except ImportError:
        return None


def plot_rating_distribution(rating_dist_df, figsize=(10, 6)):
    fig, ax = plt.subplots(figsize=figsize)
    
    rating_colors = {
        'AAA': '#27ae60', 'AA': '#2ecc71', 'A': '#82e0aa',
        'BBB': '#f7dc6f', 'BB': '#f39c12', 'B': '#e67e22',
        'CCC': '#e74c3c', 'CC': '#c0392b', 'D': '#922b21'
    }
    
    colors = [rating_colors.get(r, '#95a5a6') for r in rating_dist_df['Rating']]
    ax.bar(rating_dist_df['Rating'], rating_dist_df['Percentage'], color=colors, edgecolor='black')
    ax.set_xlabel('Rating')
    ax.set_ylabel('Percentage (%)')
    ax.set_title('Rating Distribution')
    ax.grid(axis='y', alpha=0.3)
    
    for i, row in rating_dist_df.iterrows():
        ax.text(i, row['Percentage'] + 0.5, f"{row['Percentage']:.1f}%", ha='center', fontsize=9)
    
    plt.tight_layout()
    return fig


def plot_threshold_analysis(y_true, y_proba, optimal_thresholds=None, figsize=(14, 5)):
    from sklearn.metrics import precision_score, recall_score, f1_score
    
    thresholds = np.linspace(0.01, 0.20, 100)
    metrics_data = []
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        metrics_data.append({
            'threshold': t,
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0)
        })
    df = pd.DataFrame(metrics_data)
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    axes[0].plot(df['threshold'], df['precision'], label='Precision', linewidth=2)
    axes[0].plot(df['threshold'], df['recall'], label='Recall', linewidth=2)
    axes[0].plot(df['threshold'], df['f1'], label='F1', linewidth=2)
    axes[0].set_xlabel('Threshold')
    axes[0].set_ylabel('Score')
    axes[0].set_title('Metrics vs Threshold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    axes[1].plot(fpr, tpr, linewidth=2, color='darkblue')
    axes[1].plot([0, 1], [0, 1], 'k--')
    
    if optimal_thresholds:
        for name, thresh in optimal_thresholds.items():
            y_pred_opt = (y_proba >= thresh).astype(int)
            fpr_opt = ((y_pred_opt == 1) & (y_true == 0)).sum() / (y_true == 0).sum()
            tpr_opt = ((y_pred_opt == 1) & (y_true == 1)).sum() / (y_true == 1).sum()
            axes[1].scatter([fpr_opt], [tpr_opt], s=100, label=f'{name} ({thresh:.3f})', zorder=5)
        axes[1].legend()
    
    axes[1].set_xlabel('FPR')
    axes[1].set_ylabel('TPR')
    axes[1].set_title('ROC with Optimal Thresholds')
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_gains_chart(y_true, y_proba, figsize=(10, 6)):
    df = pd.DataFrame({'y_true': y_true, 'y_proba': y_proba})
    df = df.sort_values('y_proba', ascending=False).reset_index(drop=True)
    df['cumulative_defaults'] = df['y_true'].cumsum()
    df['cumulative_pct'] = (df.index + 1) / len(df)
    df['gains'] = df['cumulative_defaults'] / df['y_true'].sum()
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(df['cumulative_pct'], df['gains'], linewidth=2, label='Model')
    ax.plot([0, 1], [0, 1], 'k--', label='Random')
    ax.fill_between(df['cumulative_pct'], df['gains'], df['cumulative_pct'], alpha=0.3)
    ax.set_xlabel('% Population')
    ax.set_ylabel('% Defaults Captured')
    ax.set_title('Gains Chart')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig


def plot_lift_chart(y_true, y_proba, n_bins=10, figsize=(10, 6)):
    df = pd.DataFrame({'y_true': y_true, 'y_proba': y_proba})
    df['decile'] = pd.qcut(df['y_proba'], n_bins, labels=False, duplicates='drop')
    
    lift_df = df.groupby('decile').agg({
        'y_true': ['sum', 'count'],
        'y_proba': 'mean'
    })
    lift_df.columns = ['defaults', 'count', 'avg_prob']
    lift_df['rate'] = lift_df['defaults'] / lift_df['count']
    baseline = y_true.sum() / len(y_true)
    lift_df['lift'] = lift_df['rate'] / baseline
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(range(len(lift_df)), lift_df['lift'], color='steelblue', alpha=0.8)
    ax.axhline(y=1, color='red', linestyle='--', label='Baseline')
    ax.set_xlabel('Decile (highest risk first)')
    ax.set_ylabel('Lift')
    ax.set_title('Lift Chart')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    return fig


def plot_coefficient_importance(coef_df, top_n=20, figsize=(12, 8)):
    df = coef_df.copy()
    df['abs_coef'] = df['Coefficient'].abs()
    df = df.sort_values('abs_coef', ascending=True).tail(top_n)
    
    colors = ['#e74c3c' if c > 0 else '#3498db' for c in df['Coefficient']]
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(range(len(df)), df['Coefficient'], color=colors, alpha=0.8)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df['Feature'] if 'Feature' in df.columns else df.index)
    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.set_xlabel('Coefficient')
    ax.set_title('Logistic Regression Coefficients')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    return fig


def plot_calibration_diagnostic(y_true, y_proba, n_bins=10, figsize=(16, 12)):
    """Full calibration diagnostic: reliability curve, histogram, ECE/ACE, Brier decomposition."""
    from sklearn.metrics import brier_score_loss
    
    y_true = np.array(y_true)
    y_proba = np.array(y_proba)
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    ax = axes[0, 0]
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=n_bins, strategy='uniform')
    ax.plot(prob_pred, prob_true, 'o-', color='#3498db', linewidth=2, markersize=8, label='Model')
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect')
    ax.fill_between(prob_pred, prob_true, prob_pred, alpha=0.2, color='#e74c3c')
    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Fraction of Positives')
    ax.set_title('Reliability Curve')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    ax = axes[0, 1]
    ax.hist(y_proba[y_true == 0], bins=30, alpha=0.6, label='Class 0 (No Default)', 
            color='#2ecc71', density=True, edgecolor='white')
    ax.hist(y_proba[y_true == 1], bins=30, alpha=0.6, label='Class 1 (Default)', 
            color='#e74c3c', density=True, edgecolor='white')
    ax.set_xlabel('Predicted Probability')
    ax.set_ylabel('Density')
    ax.set_title('Prediction Distribution by Class')
    ax.legend()
    ax.grid(alpha=0.3)
    
    ax = axes[1, 0]
    ece, ace, bin_data = _calculate_calibration_errors(y_true, y_proba, n_bins)
    
    x = np.arange(n_bins)
    width = 0.35
    ax.bar(x - width/2, bin_data['accuracy'], width, label='Actual (Fraction Positive)', 
           color='#3498db', alpha=0.8)
    ax.bar(x + width/2, bin_data['confidence'], width, label='Predicted (Mean Prob)', 
           color='#e74c3c', alpha=0.8)
    ax.set_xlabel('Probability Bin')
    ax.set_ylabel('Rate')
    ax.set_title(f'ECE = {ece:.4f}, ACE = {ace:.4f}')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{i/n_bins:.1f}-{(i+1)/n_bins:.1f}' for i in range(n_bins)], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    ax = axes[1, 1]
    brier = brier_score_loss(y_true, y_proba)
    calibration_comp, refinement_comp, uncertainty = _brier_decomposition_full(y_true, y_proba, n_bins)
    
    components = ['Brier Score', 'Calibration', 'Refinement', 'Uncertainty']
    values = [brier, calibration_comp, refinement_comp, uncertainty]
    colors = ['#9b59b6', '#e74c3c', '#3498db', '#2ecc71']
    
    bars = ax.bar(components, values, color=colors, alpha=0.8, edgecolor='white', linewidth=2)
    ax.set_ylabel('Score')
    ax.set_title('Brier Score Decomposition')
    ax.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars, values):
        ax.annotate(f'{val:.4f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    
    metrics = {
        'brier_score': brier,
        'calibration': calibration_comp,
        'refinement': refinement_comp,
        'uncertainty': uncertainty,
        'ece': ece,
        'ace': ace,
        'mean_predicted': y_proba.mean(),
        'mean_actual': y_true.mean()
    }
    
    return fig, metrics


def _calculate_calibration_errors(y_true, y_proba, n_bins=10):
    """Calculate ECE, ACE and bin statistics."""
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_proba, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    ece = 0.0
    ace = 0.0
    n_total = len(y_true)
    
    accuracy_list = []
    confidence_list = []
    counts_list = []
    
    for i in range(n_bins):
        mask = bin_indices == i
        count = mask.sum()
        if count > 0:
            acc = y_true[mask].mean()
            conf = y_proba[mask].mean()
            ece += abs(acc - conf) * (count / n_total)
            ace += abs(acc - conf)
            accuracy_list.append(acc)
            confidence_list.append(conf)
        else:
            accuracy_list.append(0)
            confidence_list.append(0)
        counts_list.append(count)
    
    ace /= n_bins
    
    return ece, ace, {
        'accuracy': accuracy_list,
        'confidence': confidence_list,
        'counts': counts_list
    }


def _brier_decomposition_full(y_true, y_proba, n_bins=10):
    """Full Brier score decomposition: calibration, refinement, uncertainty."""
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_proba, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    calibration = 0.0
    refinement = 0.0
    n_total = len(y_true)
    base_rate = y_true.mean()
    uncertainty = base_rate * (1 - base_rate)
    
    for i in range(n_bins):
        mask = bin_indices == i
        n_k = mask.sum()
        if n_k > 0:
            o_k = y_true[mask].mean()
            p_k = y_proba[mask].mean()
            calibration += n_k * (o_k - p_k) ** 2
            refinement += n_k * (o_k - base_rate) ** 2
    
    calibration /= n_total
    refinement /= n_total
    
    return calibration, refinement, uncertainty


def plot_calibration_comparison(y_true, proba_dict, n_bins=10, figsize=(18, 10)):
    """Compare calibration of multiple methods with reliability curves and metrics."""
    n_methods = len(proba_dict)
    fig, axes = plt.subplots(2, n_methods, figsize=figsize)
    
    if n_methods == 1:
        axes = axes.reshape(2, 1)
    
    colors = plt.cm.tab10(np.linspace(0, 1, n_methods))
    metrics_list = []
    
    for idx, ((name, y_proba), color) in enumerate(zip(proba_dict.items(), colors)):
        y_proba = np.array(y_proba)
        
        ax = axes[0, idx]
        prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=n_bins, strategy='uniform')
        ax.plot(prob_pred, prob_true, 'o-', color=color, linewidth=2, markersize=6)
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax.fill_between(prob_pred, prob_true, prob_pred, alpha=0.2, color=color)
        ax.set_xlabel('Mean Predicted')
        ax.set_ylabel('Fraction Positive')
        ax.set_title(f'{name}')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.grid(alpha=0.3)
        
        ax = axes[1, idx]
        ax.hist(y_proba, bins=30, alpha=0.7, color=color, edgecolor='white', density=True)
        ax.axvline(y_proba.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {y_proba.mean():.3f}')
        ax.axvline(y_true.mean(), color='green', linestyle='--', linewidth=2, label=f'Actual: {y_true.mean():.3f}')
        ax.set_xlabel('Predicted Probability')
        ax.set_ylabel('Density')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        
        ece, ace, _ = _calculate_calibration_errors(y_true, y_proba, n_bins)
        brier = brier_score_loss(y_true, y_proba)
        metrics_list.append({
            'Method': name,
            'ECE': ece,
            'ACE': ace,
            'Brier': brier,
            'Mean Pred': y_proba.mean(),
            'Mean True': y_true.mean()
        })
    
    plt.tight_layout()
    metrics_df = pd.DataFrame(metrics_list).set_index('Method')
    return fig, metrics_df
