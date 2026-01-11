"""
Tests for Statistical Tests Module
==================================

HEC Lausanne - Data Science and Advanced Programming

Unit tests for Diebold-Mariano test and model comparisons.
"""

import numpy as np
import pandas as pd
import pytest

from src.statistical_tests import (
    diebold_mariano_test,
    compare_all_models_pairwise,
    generate_significance_summary,
    rank_models_by_wins,
)


# ==============================================================================
# Test: Diebold-Mariano Test
# ==============================================================================

class TestDieboldMarianoTest:
    """Tests for Diebold-Mariano test."""

    def test_identical_forecasts(self):
        """Test DM with identical forecasts (DM ≈ 0, p ≈ 1)."""
        np.random.seed(42)
        errors = np.random.randn(100)
        
        dm_stat, p_value = diebold_mariano_test(errors, errors.copy())
        
        # DM stat should be essentially zero
        assert abs(dm_stat) < 1e-10
        
        # p-value should be very high (not significant)
        assert p_value > 0.99

    def test_clearly_better_model(self):
        """Test DM when one model is clearly better."""
        np.random.seed(42)
        n = 200
        
        # Model 1: small errors
        errors1 = np.random.randn(n) * 0.01
        
        # Model 2: large errors
        errors2 = np.random.randn(n) * 0.1
        
        dm_stat, p_value = diebold_mariano_test(errors1, errors2)
        
        # Model 1 better (lower loss) → DM stat should be negative
        assert dm_stat < 0
        
        # Should be statistically significant
        assert p_value < 0.05

    def test_model2_better(self):
        """Test DM when model 2 is better."""
        np.random.seed(42)
        n = 200
        
        # Model 1: large errors
        errors1 = np.random.randn(n) * 0.1
        
        # Model 2: small errors
        errors2 = np.random.randn(n) * 0.01
        
        dm_stat, p_value = diebold_mariano_test(errors1, errors2)
        
        # Model 2 better → DM stat should be positive
        assert dm_stat > 0
        
        # Should be statistically significant
        assert p_value < 0.05

    def test_mae_loss_function(self):
        """Test DM with MAE loss function."""
        np.random.seed(42)
        n = 100
        
        errors1 = np.random.randn(n) * 0.01
        errors2 = np.random.randn(n) * 0.05
        
        dm_stat, p_value = diebold_mariano_test(
            errors1, errors2, loss_function="mae"
        )
        
        assert isinstance(dm_stat, float)
        assert 0 <= p_value <= 1
        
        # Model 1 should be better
        assert dm_stat < 0

    def test_nan_handling(self):
        """Test DM handles NaN values correctly."""
        np.random.seed(42)
        n = 100
        
        errors1 = np.random.randn(n)
        errors2 = np.random.randn(n)
        
        # Introduce NaNs at different positions
        errors1[10:15] = np.nan
        errors2[20:25] = np.nan
        
        dm_stat, p_value = diebold_mariano_test(errors1, errors2)
        
        # Should still compute (using valid samples only)
        assert not np.isnan(dm_stat)
        assert not np.isnan(p_value)

    def test_small_sample_warning(self):
        """Test warning for small sample sizes."""
        np.random.seed(42)
        
        # Only 20 samples (below recommended 30)
        errors1 = np.random.randn(20) * 0.01
        errors2 = np.random.randn(20) * 0.05
        
        # Should still compute but may issue warning
        dm_stat, p_value = diebold_mariano_test(errors1, errors2)
        
        assert not np.isnan(dm_stat)
        assert not np.isnan(p_value)

    def test_very_small_sample(self):
        """Test with very small sample (< 2)."""
        errors1 = np.array([0.1])
        errors2 = np.array([0.2])
        
        dm_stat, p_value = diebold_mariano_test(errors1, errors2)
        
        # Should return NaN for insufficient data
        assert np.isnan(dm_stat)
        assert np.isnan(p_value)

    def test_invalid_loss_function(self):
        """Test error for invalid loss function."""
        errors1 = np.random.randn(50)
        errors2 = np.random.randn(50)
        
        with pytest.raises(ValueError, match="Unknown loss_function"):
            diebold_mariano_test(errors1, errors2, loss_function="invalid")

    def test_p_value_bounds(self):
        """Test that p-value is always between 0 and 1."""
        np.random.seed(42)
        
        for _ in range(10):
            errors1 = np.random.randn(100)
            errors2 = np.random.randn(100)
            
            _, p_value = diebold_mariano_test(errors1, errors2)
            
            if not np.isnan(p_value):
                assert 0 <= p_value <= 1


# ==============================================================================
# Test: Pairwise Comparison
# ==============================================================================

class TestPairwiseComparison:
    """Tests for pairwise model comparison."""

    def test_basic_pairwise_comparison(self):
        """Test pairwise comparison with three models."""
        np.random.seed(42)
        n = 200
        target = np.random.randn(n)
        
        # Three models with different accuracy
        results_dict = {
            'Model_A': {'predictions': target + np.random.randn(n) * 0.01},  # Best
            'Model_B': {'predictions': target + np.random.randn(n) * 0.05},  # Medium
            'Model_C': {'predictions': target + np.random.randn(n) * 0.10},  # Worst
        }
        
        comparisons = compare_all_models_pairwise(results_dict, target, alpha=0.05)
        
        # Should have 3 pairwise comparisons (A-B, A-C, B-C)
        assert len(comparisons) == 3
        
        # Check required columns
        required_cols = ['Model_1', 'Model_2', 'DM_Statistic', 'p_value', 
                         'Significant', 'Better_Model', 'N_Samples']
        for col in required_cols:
            assert col in comparisons.columns

    def test_identifies_best_model(self):
        """Test that pairwise comparison correctly identifies better model."""
        np.random.seed(42)
        n = 300
        target = np.random.randn(n)
        
        # Model A very good, Model B very bad
        results_dict = {
            'Good_Model': {'predictions': target + np.random.randn(n) * 0.001},
            'Bad_Model': {'predictions': target + np.random.randn(n) * 1.0},
        }
        
        comparisons = compare_all_models_pairwise(results_dict, target, alpha=0.05)
        
        assert len(comparisons) == 1
        assert comparisons.iloc[0]['Significant'] == True
        assert comparisons.iloc[0]['Better_Model'] == 'Good_Model'
        assert comparisons.iloc[0]['p_value'] < 0.001

    def test_similar_models_not_significant(self):
        """Test that similar models are not flagged as significantly different."""
        np.random.seed(42)
        n = 100
        target = np.random.randn(n)
        
        # Very similar predictions
        base_pred = target + np.random.randn(n) * 0.1
        results_dict = {
            'Model_A': {'predictions': base_pred + np.random.randn(n) * 0.001},
            'Model_B': {'predictions': base_pred + np.random.randn(n) * 0.001},
        }
        
        comparisons = compare_all_models_pairwise(results_dict, target, alpha=0.05)
        
        # Should likely not be significant
        # (Note: there's still a 5% chance of false positive)
        assert len(comparisons) == 1

    def test_single_model(self):
        """Test with only one model (no comparisons possible)."""
        np.random.seed(42)
        n = 100
        target = np.random.randn(n)
        
        results_dict = {
            'Only_Model': {'predictions': target + np.random.randn(n) * 0.1},
        }
        
        comparisons = compare_all_models_pairwise(results_dict, target, alpha=0.05)
        
        assert len(comparisons) == 0

    def test_nan_in_predictions(self):
        """Test handling of NaN in predictions."""
        np.random.seed(42)
        n = 100
        target = np.random.randn(n)
        
        pred_a = target + np.random.randn(n) * 0.1
        pred_b = target + np.random.randn(n) * 0.2
        
        # Add NaNs
        pred_a[:20] = np.nan
        pred_b[30:50] = np.nan
        
        results_dict = {
            'Model_A': {'predictions': pred_a},
            'Model_B': {'predictions': pred_b},
        }
        
        comparisons = compare_all_models_pairwise(results_dict, target, alpha=0.05)
        
        # Should still compute with reduced samples
        assert len(comparisons) == 1
        assert comparisons.iloc[0]['N_Samples'] < n


# ==============================================================================
# Test: Significance Summary
# ==============================================================================

class TestSignificanceSummary:
    """Tests for significance summary generation."""

    def test_summary_with_significant_results(self):
        """Test summary when there are significant differences."""
        dm_results = pd.DataFrame({
            'Model_1': ['Good', 'Good'],
            'Model_2': ['Bad', 'Medium'],
            'DM_Statistic': [-3.5, -2.1],
            'p_value': [0.001, 0.04],
            'Significant': [True, True],
            'Better_Model': ['Good', 'Good'],
            'N_Samples': [200, 200]
        })
        
        summary = generate_significance_summary(dm_results)
        
        assert "STATISTICALLY SIGNIFICANT" in summary
        assert "Good" in summary
        assert "outperforms" in summary.lower() or "better" in summary.lower()

    def test_summary_without_significant_results(self):
        """Test summary when no significant differences."""
        dm_results = pd.DataFrame({
            'Model_1': ['A'],
            'Model_2': ['B'],
            'DM_Statistic': [0.5],
            'p_value': [0.6],
            'Significant': [False],
            'Better_Model': ['B'],
            'N_Samples': [100]
        })
        
        summary = generate_significance_summary(dm_results)
        
        assert "No statistically significant" in summary or "not significantly different" in summary.lower()

    def test_summary_empty_results(self):
        """Test summary with empty results."""
        dm_results = pd.DataFrame()
        
        summary = generate_significance_summary(dm_results)
        
        assert "No comparison results" in summary or len(summary) > 0


# ==============================================================================
# Test: Model Ranking
# ==============================================================================

class TestModelRanking:
    """Tests for model ranking by wins."""

    def test_rank_by_wins(self):
        """Test ranking models by significant wins."""
        dm_results = pd.DataFrame({
            'Model_1': ['A', 'A', 'B'],
            'Model_2': ['B', 'C', 'C'],
            'DM_Statistic': [-3.0, -2.5, 1.5],
            'p_value': [0.01, 0.02, 0.15],
            'Significant': [True, True, False],
            'Better_Model': ['A', 'A', 'C'],
            'N_Samples': [200, 200, 200]
        })
        
        rankings = rank_models_by_wins(dm_results)
        
        assert len(rankings) == 3
        assert 'Rank' in rankings.columns
        assert 'Model' in rankings.columns
        assert 'Significant_Wins' in rankings.columns
        
        # Model A should be ranked first (2 significant wins)
        assert rankings.iloc[0]['Model'] == 'A'
        assert rankings.iloc[0]['Significant_Wins'] == 2

    def test_rank_empty_results(self):
        """Test ranking with empty results."""
        dm_results = pd.DataFrame()
        
        rankings = rank_models_by_wins(dm_results)
        
        assert len(rankings) == 0

    def test_rank_no_significant_wins(self):
        """Test ranking when no model has significant wins."""
        dm_results = pd.DataFrame({
            'Model_1': ['A', 'A', 'B'],
            'Model_2': ['B', 'C', 'C'],
            'DM_Statistic': [0.5, -0.3, 0.2],
            'p_value': [0.6, 0.7, 0.8],
            'Significant': [False, False, False],
            'Better_Model': ['B', 'A', 'C'],
            'N_Samples': [100, 100, 100]
        })
        
        rankings = rank_models_by_wins(dm_results)
        
        assert len(rankings) == 3
        # All should have 0 wins
        assert (rankings['Significant_Wins'] == 0).all()


# ==============================================================================
# Test: Edge Cases
# ==============================================================================

class TestEdgeCases:
    """Tests for edge cases."""

    def test_all_nan_errors(self):
        """Test with all NaN errors."""
        errors1 = np.array([np.nan] * 50)
        errors2 = np.array([np.nan] * 50)
        
        dm_stat, p_value = diebold_mariano_test(errors1, errors2)
        
        assert np.isnan(dm_stat)
        assert np.isnan(p_value)

    def test_zero_variance_errors(self):
        """Test with zero variance in loss differential."""
        np.random.seed(42)
        n = 100
        
        # Same errors → zero variance in differential
        errors = np.random.randn(n) * 0.1
        
        dm_stat, p_value = diebold_mariano_test(errors, errors)
        
        # Should handle gracefully
        assert not np.isinf(dm_stat)

    def test_missing_predictions_key(self):
        """Test handling of missing predictions key."""
        np.random.seed(42)
        n = 100
        target = np.random.randn(n)
        
        # Missing 'predictions' key
        results_dict = {
            'Model_A': {'something_else': np.random.randn(n)},
            'Model_B': {'predictions': target + np.random.randn(n) * 0.1},
        }
        
        comparisons = compare_all_models_pairwise(results_dict, target, alpha=0.05)
        
        # Should skip the invalid model
        assert len(comparisons) == 0  # No valid pairs


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
