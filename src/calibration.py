"""Probability calibration module for credit scoring models."""

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from scipy import optimize
from .config import TARGET_PD, CALIBRATION_N_BINS


class CalibrationModule:
    """Probability calibration with target PD adjustment."""
    
    def __init__(self, target_pd=TARGET_PD, n_bins=CALIBRATION_N_BINS):
        self.target_pd = target_pd
        self.n_bins = n_bins
        
        self.platt_calibrator = None
        self.isotonic_calibrator = None
        self.beta_params = None
        self.intercept_adjustment = 0.0
        
        self.metrics_history = {}
    
    def fit_platt(self, y_true, y_proba):
        """Fit Platt Scaling (logistic calibration)."""
        logits = self._to_logits(y_proba)
        self.platt_calibrator = LogisticRegression(max_iter=1000)
        self.platt_calibrator.fit(logits.reshape(-1, 1), y_true)
        return self
    
    def fit_isotonic(self, y_true, y_proba):
        """Fit Isotonic Regression calibration."""
        self.isotonic_calibrator = IsotonicRegression(out_of_bounds='clip')
        self.isotonic_calibrator.fit(y_proba, y_true)
        return self
    
    def fit_beta(self, y_true, y_proba):
        """Fit Beta Calibration."""
        def beta_loss(params, y_t, y_p):
            a, b = params
            eps = 1e-10
            y_p = np.clip(y_p, eps, 1-eps)
            calibrated = (y_p ** a) / ((y_p ** a) + ((1 - y_p) ** b))
            calibrated = np.clip(calibrated, eps, 1-eps)
            return -np.mean(y_t * np.log(calibrated) + (1 - y_t) * np.log(1 - calibrated))
        
        result = optimize.minimize(
            beta_loss, x0=[1.0, 1.0], args=(y_true, y_proba),
            method='L-BFGS-B', bounds=[(0.1, 10.0), (0.1, 10.0)]
        )
        self.beta_params = result.x
        return self
    
    def fit_intercept(self, y_proba):
        """Fit intercept adjustment to match target PD."""
        def objective(offset):
            logits = self._to_logits(y_proba)
            adjusted = self._from_logits(logits + offset)
            return (np.mean(adjusted) - self.target_pd) ** 2
        
        result = optimize.minimize_scalar(objective, bounds=(-10, 10), method='bounded')
        self.intercept_adjustment = result.x
        return self
    
    def transform(self, y_proba, method='platt'):
        """Transform probabilities using fitted calibrator."""
        if method == 'platt':
            if self.platt_calibrator is None:
                raise ValueError("Platt calibrator not fitted")
            logits = self._to_logits(y_proba)
            return self.platt_calibrator.predict_proba(logits.reshape(-1, 1))[:, 1]
        
        elif method == 'isotonic':
            if self.isotonic_calibrator is None:
                raise ValueError("Isotonic calibrator not fitted")
            return self.isotonic_calibrator.transform(y_proba)
        
        elif method == 'beta':
            if self.beta_params is None:
                raise ValueError("Beta calibrator not fitted")
            a, b = self.beta_params
            eps = 1e-10
            y_proba = np.clip(y_proba, eps, 1-eps)
            return np.clip((y_proba ** a) / ((y_proba ** a) + ((1 - y_proba) ** b)), eps, 1-eps)
        
        elif method == 'intercept':
            logits = self._to_logits(y_proba)
            return self._from_logits(logits + self.intercept_adjustment)
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def get_calibration_metrics(self, y_true, y_proba, name="Model"):
        """Calculate calibration metrics."""
        metrics = {
            'mean_predicted_pd': np.mean(y_proba),
            'mean_true_pd': np.mean(y_true),
            'pd_gap': abs(np.mean(y_proba) - self.target_pd),
            'ece': self._calculate_ece(y_true, y_proba),
            'brier': brier_score_loss(y_true, y_proba),
            'log_loss': log_loss(y_true, y_proba)
        }
        
        brier_decomp = self._brier_decomposition(y_true, y_proba)
        metrics['brier_calibration'] = brier_decomp['calibration']
        metrics['brier_refinement'] = brier_decomp['refinement']
        
        self.metrics_history[name] = metrics
        return metrics
    
    def compare_methods(self, y_true, y_proba):
        """Compare all calibration methods."""
        results = {'uncalibrated': y_proba}
        
        self.fit_platt(y_true, y_proba)
        results['platt'] = self.transform(y_proba, 'platt')
        
        self.fit_isotonic(y_true, y_proba)
        results['isotonic'] = self.transform(y_proba, 'isotonic')
        
        self.fit_beta(y_true, y_proba)
        results['beta'] = self.transform(y_proba, 'beta')
        
        self.fit_intercept(y_proba)
        results['intercept'] = self.transform(y_proba, 'intercept')
        
        comparison = {}
        for name, proba in results.items():
            comparison[name] = self.get_calibration_metrics(y_true, proba, name)
        
        return pd.DataFrame(comparison).T
    
    def get_reliability_data(self, y_true, y_proba):
        """Get data for reliability curve plotting."""
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_proba, n_bins=self.n_bins, strategy='uniform'
        )
        return {
            'fraction_of_positives': fraction_of_positives,
            'mean_predicted_value': mean_predicted_value
        }
    
    def _to_logits(self, proba):
        eps = 1e-10
        proba = np.clip(proba, eps, 1-eps)
        return np.log(proba / (1 - proba))
    
    def _from_logits(self, logits):
        return 1 / (1 + np.exp(-logits))
    
    def _calculate_ece(self, y_true, y_proba):
        bins = np.linspace(0, 1, self.n_bins + 1)
        bin_indices = np.digitize(y_proba, bins) - 1
        bin_indices = np.clip(bin_indices, 0, self.n_bins - 1)
        
        ece = 0.0
        n_total = len(y_true)
        
        for i in range(self.n_bins):
            mask = bin_indices == i
            if mask.sum() > 0:
                accuracy = y_true[mask].mean()
                confidence = y_proba[mask].mean()
                ece += abs(accuracy - confidence) * (mask.sum() / n_total)
        
        return ece
    
    def _brier_decomposition(self, y_true, y_proba):
        bins = np.linspace(0, 1, self.n_bins + 1)
        bin_indices = np.digitize(y_proba, bins) - 1
        bin_indices = np.clip(bin_indices, 0, self.n_bins - 1)
        
        calibration = 0.0
        refinement = 0.0
        n_total = len(y_true)
        
        for i in range(self.n_bins):
            mask = bin_indices == i
            if mask.sum() > 0:
                n_k = mask.sum()
                o_k = y_true[mask].mean()
                p_k = y_proba[mask].mean()
                calibration += n_k * (o_k - p_k) ** 2
                refinement += n_k * o_k * (1 - o_k)
        
        return {
            'calibration': calibration / n_total,
            'refinement': refinement / n_total,
            'total': brier_score_loss(y_true, y_proba)
        }


def calibrate_to_target_pd(y_true, y_proba, target_pd=TARGET_PD, method='intercept'):
    """Convenience function to calibrate probabilities to target PD."""
    calibrator = CalibrationModule(target_pd=target_pd)
    
    if method == 'platt':
        calibrator.fit_platt(y_true, y_proba)
    elif method == 'isotonic':
        calibrator.fit_isotonic(y_true, y_proba)
    elif method == 'beta':
        calibrator.fit_beta(y_true, y_proba)
    elif method == 'intercept':
        calibrator.fit_intercept(y_proba)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return calibrator.transform(y_proba, method)
