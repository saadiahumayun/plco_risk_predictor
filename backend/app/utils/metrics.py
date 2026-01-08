# app/utils/metrics.py
"""
Utility functions for metrics calculation.
"""
import numpy as np
from typing import Tuple, List, Dict, Any
from scipy import stats
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import logging


logger = logging.getLogger(__name__)


def calculate_confidence_interval(risk_score: float, 
                                confidence_level: float = 0.95) -> Tuple[float, float]:
    """
    Calculate confidence interval for risk score.
    
    Args:
        risk_score: Predicted risk score
        confidence_level: Confidence level (default 0.95)
        
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    try:
        # Use Wilson score interval for binomial proportion
        # This is more appropriate for probabilities near 0 or 1
        z = stats.norm.ppf((1 + confidence_level) / 2)
        n = 100  # Assumed sample size for estimation
        
        center = (risk_score + z*z/(2*n)) / (1 + z*z/n)
        margin = z / (1 + z*z/n) * np.sqrt(risk_score*(1-risk_score)/n + z*z/(4*n*n))
        
        lower = max(0, center - margin)
        upper = min(1, center + margin)
        
        return float(lower), float(upper)
        
    except Exception as e:
        logger.error(f"Failed to calculate confidence interval: {e}")
        # Return simple interval as fallback
        margin = 0.05
        return max(0, risk_score - margin), min(1, risk_score + margin)


def calculate_calibration_error(y_true: List[int], 
                              y_pred_proba: List[float],
                              n_bins: int = 10) -> float:
    """
    Calculate expected calibration error.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        n_bins: Number of bins for calibration
        
    Returns:
        Expected calibration error
    """
    try:
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find predictions in this bin
            in_bin = (y_pred_proba > bin_lower) & (y_pred_proba <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                # Average predicted probability in bin
                avg_pred_prob = y_pred_proba[in_bin].mean()
                # Fraction of positives in bin
                frac_positives = y_true[in_bin].mean()
                # Weighted absolute difference
                ece += np.abs(avg_pred_prob - frac_positives) * prop_in_bin
        
        return float(ece)
        
    except Exception as e:
        logger.error(f"Failed to calculate calibration error: {e}")
        return 0.0


def calculate_model_metrics(y_true: np.ndarray, 
                          y_pred: np.ndarray,
                          y_pred_proba: np.ndarray) -> Dict[str, float]:
    """
    Calculate comprehensive model metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities
        
    Returns:
        Dictionary of metrics
    """
    try:
        metrics = {
            "auc": roc_auc_score(y_true, y_pred_proba),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "calibration_error": calculate_calibration_error(y_true, y_pred_proba)
        }
        
        # Calculate sensitivity at fixed specificity points
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        
        # Sensitivity at 90% specificity
        spec_90_idx = np.argmin(np.abs(fpr - 0.1))
        metrics["sensitivity_at_90_specificity"] = tpr[spec_90_idx]
        
        # Sensitivity at 95% specificity  
        spec_95_idx = np.argmin(np.abs(fpr - 0.05))
        metrics["sensitivity_at_95_specificity"] = tpr[spec_95_idx]
        
        return metrics
        
    except Exception as e:
        logger.error(f"Failed to calculate model metrics: {e}")
        return {}


def calculate_feature_importance_impact(feature_value: float,
                                      feature_mean: float,
                                      feature_std: float,
                                      feature_importance: float) -> float:
    """
    Calculate the impact of a feature on the prediction.
    
    Args:
        feature_value: Current value of the feature
        feature_mean: Population mean of the feature
        feature_std: Population standard deviation
        feature_importance: Model-derived importance
        
    Returns:
        Impact score (can be positive or negative)
    """
    try:
        if feature_std > 0:
            # Standardize the feature value
            z_score = (feature_value - feature_mean) / feature_std
            # Calculate impact
            impact = z_score * feature_importance
        else:
            impact = 0.0
        
        return float(impact)
        
    except Exception as e:
        logger.error(f"Failed to calculate feature impact: {e}")
        return 0.0


def calculate_population_percentile(risk_score: float,
                                  population_distribution: List[float] = None) -> int:
    """
    Calculate percentile rank in population.
    
    Args:
        risk_score: Individual risk score
        population_distribution: Population risk scores (if available)
        
    Returns:
        Percentile (0-100)
    """
    try:
        if population_distribution:
            percentile = stats.percentileofscore(population_distribution, risk_score)
        else:
            # Use pre-computed distribution based on PLCO data
            # These are approximate values for breast cancer risk
            if risk_score < 0.05:
                percentile = risk_score * 10 * 100  # Linear up to 50th percentile
            elif risk_score < 0.15:
                percentile = 50 + (risk_score - 0.05) * 4 * 100  # 50-90th percentile
            elif risk_score < 0.25:
                percentile = 90 + (risk_score - 0.15) * 0.9 * 100  # 90-99th percentile
            else:
                percentile = 99
        
        return int(min(99, max(0, percentile)))
        
    except Exception as e:
        logger.error(f"Failed to calculate percentile: {e}")
        return 50


def roc_curve(y_true: np.ndarray, 
              y_scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate ROC curve.
    
    Args:
        y_true: True binary labels
        y_scores: Target scores
        
    Returns:
        fpr: False positive rates
        tpr: True positive rates  
        thresholds: Thresholds
    """
    # Sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_scores, kind="mergesort")[::-1]
    y_scores = y_scores[desc_score_indices]
    y_true = y_true[desc_score_indices]
    
    # Get unique scores and threshold indices
    distinct_value_indices = np.where(np.diff(y_scores))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]
    
    # Calculate TPR and FPR
    tps = np.cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps
    
    # Add starting point
    tps = np.r_[0, tps]
    fps = np.r_[0, fps]
    
    # Calculate rates
    if fps[-1] <= 0:
        fpr = np.zeros_like(fps)
    else:
        fpr = fps / fps[-1]
        
    if tps[-1] <= 0:
        tpr = np.zeros_like(tps)
    else:
        tpr = tps / tps[-1]
        
    return fpr, tpr, y_scores[threshold_idxs]