"""
Statistical Tests for Model Comparison
=======================================

HEC Lausanne - Data Science and Advanced Programming
Master in Finance

This module implements the Diebold-Mariano test to compare model accuracy.
It tells us if one model is significantly better than another.

Reference:
    Diebold, F.X. and Mariano, R.S. (1995)
    "Comparing Predictive Accuracy"
    Journal of Business & Economic Statistics, 13(3), 253-263.
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


def diebold_mariano_test(
    errors1: np.ndarray,
    errors2: np.ndarray,
    loss_function: str = "mse",
    h: int = 1
) -> Tuple[float, float]:
    """
    Diebold-Mariano test for comparing two forecasts.
    
    The null hypothesis is that both models have equal accuracy.
    If the p-value is small (< 0.05), we reject H0 and conclude one is better.
    
    Args:
        errors1: Errors from model 1 (actual - predicted)
        errors2: Errors from model 2 (actual - predicted)
        loss_function: "mse" or "mae"
        h: Forecast horizon (for adjusting serial correlation)
        
    Returns:
        (DM_statistic, p_value)
        
    Interpretation:
        - DM < 0 → Model 1 is better (lower loss)
        - DM stat > 0 → Model 2 has lower loss (better)
        - p-value < 0.05 → Difference is statistically significant
        
    Example:
        >>> dm_stat, p_val = diebold_mariano_test(errors_rf, errors_ridge)
        >>> if p_val < 0.05:
        ...     winner = "Model 1" if dm_stat < 0 else "Model 2"
        ...     print(f"{winner} is significantly better")
    """
    # Remove NaN values (pairwise)
    valid = ~(np.isnan(errors1) | np.isnan(errors2))
    e1 = errors1[valid]
    e2 = errors2[valid]
    
    n = len(e1)
    
    if n < 30:
        logger.warning(
            f"Only {n} valid samples for DM test. "
            "Results may be unreliable (recommend n >= 30)."
        )
    
    if n < 2:
        return np.nan, np.nan
    
    # Compute loss differential
    if loss_function.lower() == "mse":
        d = e1**2 - e2**2
    elif loss_function.lower() == "mae":
        d = np.abs(e1) - np.abs(e2)
    else:
        raise ValueError(f"Unknown loss_function: {loss_function}. Use 'mse' or 'mae'.")
    
    # Mean of loss differential
    d_bar = np.mean(d)
    
    # Variance estimation with Newey-West HAC correction for autocorrelation
    # (relevant for multi-step forecasts, h > 1)
    gamma_0 = np.var(d, ddof=1)
    
    # For h=1, use simple variance. For h>1, need HAC adjustment.
    if h == 1:
        var_d = gamma_0 / n
    else:
        # Newey-West HAC estimator
        autocovariances = 0.0
        for lag in range(1, h):
            gamma_lag = np.cov(d[:-lag], d[lag:])[0, 1] if lag < n else 0
            autocovariances += 2 * (1 - lag / h) * gamma_lag
        var_d = (gamma_0 + autocovariances) / n
    
    # Ensure positive variance
    if var_d <= 0:
        logger.warning("Non-positive variance estimate. Setting to small value.")
        var_d = 1e-10
    
    # DM statistic
    dm_stat = d_bar / np.sqrt(var_d)
    
    # Harvey-Leybourne-Newbold small sample correction
    # Adjusts for finite sample bias
    hln_correction = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm_stat_corrected = dm_stat * hln_correction
    
    # Two-sided p-value using t-distribution
    # (more appropriate for small samples than normal)
    df = n - 1
    p_value = 2 * (1 - stats.t.cdf(np.abs(dm_stat_corrected), df=df))
    
    return float(dm_stat_corrected), float(p_value)


def compare_all_models_pairwise(
    results_dict: Dict[str, dict],
    target: np.ndarray,
    alpha: float = 0.05,
    loss_function: str = "mse"
) -> pd.DataFrame:
    """
    Perform pairwise Diebold-Mariano tests for all model pairs.
    
    Args:
        results_dict: Dict from WalkForwardEvaluator
                     {model_name: {'predictions': array, ...}}
        target: True target values (aligned with predictions)
        alpha: Significance level (default: 0.05 for 95% confidence)
        loss_function: Loss function for comparison ('mse' or 'mae')
        
    Returns:
        DataFrame with columns:
        - Model_1, Model_2: Model pair
        - DM_Statistic: Test statistic (negative = Model_1 better)
        - p_value: Two-sided p-value
        - Significant: Boolean (p < alpha)
        - Better_Model: Name of better performing model
        - N_Samples: Number of valid comparison samples
        
    Example:
        >>> dm_results = compare_all_models_pairwise(results, target)
        >>> significant = dm_results[dm_results['Significant']]
        >>> print(significant[['Model_1', 'Model_2', 'Better_Model', 'p_value']])
    """
    model_names = list(results_dict.keys())
    n_models = len(model_names)
    
    if n_models < 2:
        logger.warning("Need at least 2 models for pairwise comparison.")
        return pd.DataFrame()
    
    comparisons = []
    
    # Pairwise comparisons (upper triangle only to avoid duplicates)
    for i in range(n_models):
        for j in range(i + 1, n_models):
            model1 = model_names[i]
            model2 = model_names[j]
            
            # Get predictions
            pred1 = results_dict[model1].get('predictions')
            pred2 = results_dict[model2].get('predictions')
            
            if pred1 is None or pred2 is None:
                logger.warning(f"Missing predictions for {model1} or {model2}")
                continue
            
            # Compute forecast errors
            errors1 = target - pred1
            errors2 = target - pred2
            
            # Count valid samples
            valid = ~(np.isnan(errors1) | np.isnan(errors2) | np.isnan(target))
            n_valid = np.sum(valid)
            
            if n_valid < 30:
                logger.warning(
                    f"Only {n_valid} valid samples for {model1} vs {model2}. "
                    "Results may be unreliable."
                )
            
            if n_valid < 2:
                logger.warning(f"Skipping {model1} vs {model2}: insufficient data.")
                continue
            
            # Run DM test
            try:
                dm_stat, p_val = diebold_mariano_test(
                    errors1, 
                    errors2, 
                    loss_function=loss_function
                )
                
                # Determine better model
                if np.isnan(dm_stat):
                    better = "Inconclusive"
                elif dm_stat < 0:
                    better = model1
                else:
                    better = model2
                
                comparisons.append({
                    'Model_1': model1,
                    'Model_2': model2,
                    'DM_Statistic': round(dm_stat, 4),
                    'p_value': round(p_val, 4),
                    'Significant': p_val < alpha if not np.isnan(p_val) else False,
                    'Better_Model': better,
                    'N_Samples': n_valid,
                    'Loss_Function': loss_function.upper()
                })
                
            except Exception as e:
                logger.error(f"DM test failed for {model1} vs {model2}: {e}")
    
    return pd.DataFrame(comparisons)


def generate_significance_summary(
    dm_results: pd.DataFrame,
    alpha: float = 0.05
) -> str:
    """
    Generate a human-readable summary of DM test results.
    
    Args:
        dm_results: DataFrame from compare_all_models_pairwise()
        alpha: Significance level used
        
    Returns:
        Formatted string summary
    """
    if dm_results.empty:
        return "No comparison results available."
    
    lines = [
        "=" * 60,
        f"DIEBOLD-MARIANO TEST SUMMARY (α = {alpha})",
        "=" * 60,
        ""
    ]
    
    # Significant differences
    significant = dm_results[dm_results['Significant'] == True]
    
    if len(significant) > 0:
        lines.append("STATISTICALLY SIGNIFICANT DIFFERENCES:")
        lines.append("-" * 40)
        
        for _, row in significant.iterrows():
            winner = row['Better_Model']
            loser = row['Model_1'] if winner == row['Model_2'] else row['Model_2']
            lines.append(
                f"  • {winner} significantly outperforms {loser}"
            )
            lines.append(
                f"    (DM = {row['DM_Statistic']:.3f}, p = {row['p_value']:.4f})"
            )
        lines.append("")
    else:
        lines.append("⚠ No statistically significant differences detected.")
        lines.append("  All models perform similarly within statistical noise.")
        lines.append("")
    
    # Non-significant (similar performance)
    non_significant = dm_results[dm_results['Significant'] == False]
    if len(non_significant) > 0:
        lines.append("SIMILAR PERFORMANCE (not significantly different):")
        lines.append("-" * 40)
        for _, row in non_significant.iterrows():
            lines.append(
                f"  • {row['Model_1']} ≈ {row['Model_2']} (p = {row['p_value']:.3f})"
            )
    
    lines.append("")
    lines.append("=" * 60)
    
    return "\n".join(lines)


def rank_models_by_wins(dm_results: pd.DataFrame) -> pd.DataFrame:
    """
    Rank models by number of significant wins.
    
    Args:
        dm_results: DataFrame from compare_all_models_pairwise()
        
    Returns:
        DataFrame with model rankings
    """
    if dm_results.empty:
        return pd.DataFrame()
    
    # Get all unique models
    all_models = set(dm_results['Model_1'].tolist() + dm_results['Model_2'].tolist())
    
    # Count wins
    rankings = []
    for model in all_models:
        # Significant wins
        wins = dm_results[
            (dm_results['Significant'] == True) & 
            (dm_results['Better_Model'] == model)
        ].shape[0]
        
        # Significant losses
        is_model1 = dm_results['Model_1'] == model
        is_model2 = dm_results['Model_2'] == model
        significant = dm_results['Significant'] == True
        better_not_this = dm_results['Better_Model'] != model
        
        losses = dm_results[
            significant & better_not_this & (is_model1 | is_model2)
        ].shape[0]
        
        # Ties (non-significant)
        ties = dm_results[
            (dm_results['Significant'] == False) & 
            ((dm_results['Model_1'] == model) | (dm_results['Model_2'] == model))
        ].shape[0]
        
        rankings.append({
            'Model': model,
            'Significant_Wins': wins,
            'Significant_Losses': losses,
            'Ties': ties,
            'Win_Rate': wins / (wins + losses) if (wins + losses) > 0 else 0.5
        })
    
    rankings_df = pd.DataFrame(rankings)
    rankings_df = rankings_df.sort_values(
        ['Significant_Wins', 'Win_Rate'], 
        ascending=[False, False]
    ).reset_index(drop=True)
    
    rankings_df['Rank'] = range(1, len(rankings_df) + 1)
    
    return rankings_df[['Rank', 'Model', 'Significant_Wins', 'Significant_Losses', 
                        'Ties', 'Win_Rate']]


def main() -> None:
    """Example usage of statistical tests."""
    print("=" * 70)
    print("DIEBOLD-MARIANO TEST DEMONSTRATION")
    print("=" * 70)
    
    np.random.seed(42)
    n = 200
    
    # True values
    target = np.random.randn(n)
    
    # Model predictions with different accuracy
    pred_good = target + np.random.randn(n) * 0.1    # Good model
    pred_medium = target + np.random.randn(n) * 0.3  # Medium model
    pred_bad = target + np.random.randn(n) * 0.5     # Bad model
    
    # Create results dict
    results = {
        'Good_Model': {'predictions': pred_good},
        'Medium_Model': {'predictions': pred_medium},
        'Bad_Model': {'predictions': pred_bad},
    }
    
    # Run pairwise comparisons
    print("\nRunning pairwise Diebold-Mariano tests...")
    dm_results = compare_all_models_pairwise(results, target, alpha=0.05)
    
    print("\n" + dm_results.to_string(index=False))
    
    # Summary
    print("\n" + generate_significance_summary(dm_results))
    
    # Rankings
    print("\nMODEL RANKINGS:")
    rankings = rank_models_by_wins(dm_results)
    print(rankings.to_string(index=False))


if __name__ == "__main__":
    main()
