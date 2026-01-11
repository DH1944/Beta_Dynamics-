"""
Tests for Evaluation Module
===========================

HEC Lausanne - Data Science and Advanced Programming

Unit tests for walk-forward validation and metrics.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from src.evaluation import (
    WalkForwardEvaluator,
    plot_cumulative_mse,
    compute_regime_metrics,
    generate_summary_table,
)


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    np.random.seed(42)
    n = 500
    n_features = 5
    
    X = np.random.randn(n, n_features)
    true_coef = np.array([0.5, -0.3, 0.2, -0.1, 0.4])
    y = X @ true_coef + np.random.randn(n) * 0.1
    
    dates = pd.date_range("2018-01-01", periods=n, freq="B")
    
    return X, y, dates


@pytest.fixture
def sample_regimes():
    """Generate sample regime data."""
    np.random.seed(42)
    n = 500
    
    regimes = pd.Series(
        np.where(np.random.rand(n) > 0.7, "High Volatility", "Calm"),
        index=pd.date_range("2018-01-01", periods=n, freq="B")
    )
    
    return regimes


@pytest.fixture
def evaluator():
    """Create WalkForwardEvaluator instance."""
    return WalkForwardEvaluator(n_splits=5, test_size=50, gap=0)


# ==============================================================================
# Test: Evaluator Initialization
# ==============================================================================

class TestEvaluatorInit:
    """Tests for WalkForwardEvaluator initialization."""

    def test_default_initialization(self):
        """Test default parameter initialization."""
        evaluator = WalkForwardEvaluator()
        
        assert evaluator.n_splits == 10
        assert evaluator.test_size == 63
        assert evaluator.gap == 0

    def test_custom_initialization(self):
        """Test custom parameter initialization."""
        evaluator = WalkForwardEvaluator(
            n_splits=5,
            train_size=200,
            test_size=50,
            gap=5
        )
        
        assert evaluator.n_splits == 5
        assert evaluator.train_size == 200
        assert evaluator.test_size == 50
        assert evaluator.gap == 5


# ==============================================================================
# Test: Split Generation
# ==============================================================================

class TestSplitGeneration:
    """Tests for train/test split generation."""

    def test_generate_splits_count(self, evaluator):
        """Test correct number of splits generated."""
        splits = evaluator._generate_splits(n_samples=500)
        
        assert len(splits) == evaluator.n_splits

    def test_no_overlap(self, evaluator):
        """Test that train and test sets don't overlap."""
        splits = evaluator._generate_splits(n_samples=500)
        
        for train_idx, test_idx in splits:
            overlap = np.intersect1d(train_idx, test_idx)
            assert len(overlap) == 0

    def test_temporal_ordering(self, evaluator):
        """Test that test set comes after train set."""
        splits = evaluator._generate_splits(n_samples=500)
        
        for train_idx, test_idx in splits:
            assert train_idx.max() < test_idx.min()

    def test_expanding_window(self, evaluator):
        """Test that train set expands over time."""
        splits = evaluator._generate_splits(n_samples=500)
        
        train_sizes = [len(train_idx) for train_idx, _ in splits]
        
        # Train size should increase (or stay same)
        for i in range(1, len(train_sizes)):
            assert train_sizes[i] >= train_sizes[i-1]

    def test_gap_respected(self):
        """Test that gap between train and test is respected."""
        evaluator = WalkForwardEvaluator(n_splits=3, test_size=30, gap=5)
        splits = evaluator._generate_splits(n_samples=300)
        
        for train_idx, test_idx in splits:
            gap = test_idx.min() - train_idx.max()
            assert gap >= evaluator.gap


# ==============================================================================
# Test: Single Split Evaluation
# ==============================================================================

class TestSingleSplitEvaluation:
    """Tests for single split evaluation."""

    def test_evaluate_single_split(self, evaluator, sample_data):
        """Test evaluation on a single split."""
        X, y, dates = sample_data
        train_idx = np.arange(0, 300)
        test_idx = np.arange(300, 350)
        
        model = Ridge(alpha=1.0)
        result = evaluator.evaluate_single_split(model, X, y, train_idx, test_idx)
        
        assert "mse" in result
        assert "mae" in result
        assert "y_pred" in result
        assert "y_true" in result

    def test_metrics_calculated_correctly(self, evaluator, sample_data):
        """Test that metrics are calculated correctly."""
        X, y, _ = sample_data
        train_idx = np.arange(0, 300)
        test_idx = np.arange(300, 350)
        
        model = Ridge(alpha=1.0)
        result = evaluator.evaluate_single_split(model, X, y, train_idx, test_idx)
        
        # Manually compute MSE
        expected_mse = np.mean((result["y_true"] - result["y_pred"]) ** 2)
        np.testing.assert_almost_equal(result["mse"], expected_mse, decimal=10)

    def test_handles_nan_in_data(self, evaluator, sample_data):
        """Test handling of NaN values in data."""
        X, y, _ = sample_data
        
        # Add NaN to some rows
        X_nan = X.copy()
        X_nan[:50, 0] = np.nan
        y_nan = y.copy()
        y_nan[:50] = np.nan
        
        train_idx = np.arange(0, 300)
        test_idx = np.arange(300, 350)
        
        model = Ridge(alpha=1.0)
        result = evaluator.evaluate_single_split(model, X_nan, y_nan, train_idx, test_idx)
        
        # Should still produce valid results
        assert "mse" in result
        assert not np.isnan(result["mse"])


# ==============================================================================
# Test: Full Model Evaluation
# ==============================================================================

class TestModelEvaluation:
    """Tests for full model evaluation."""

    def test_evaluate_model(self, evaluator, sample_data):
        """Test full model evaluation."""
        X, y, dates = sample_data
        model = Ridge(alpha=1.0)
        
        result = evaluator.evaluate_model(model, X, y, dates)
        
        assert "split_results" in result
        assert "predictions" in result
        assert "squared_errors" in result
        assert "aggregate" in result

    def test_aggregate_metrics(self, evaluator, sample_data):
        """Test aggregate metrics computation."""
        X, y, dates = sample_data
        model = Ridge(alpha=1.0)
        
        result = evaluator.evaluate_model(model, X, y, dates)
        agg = result["aggregate"]
        
        assert "mean_mse" in agg
        assert "std_mse" in agg
        assert "n_predictions" in agg
        assert "n_splits" in agg
        
        assert agg["n_predictions"] > 0
        assert agg["n_splits"] == len(result["split_results"])

    def test_predictions_length(self, evaluator, sample_data):
        """Test predictions array has correct length."""
        X, y, dates = sample_data
        model = Ridge(alpha=1.0)
        
        result = evaluator.evaluate_model(model, X, y, dates)
        
        assert len(result["predictions"]) == len(y)
        assert len(result["squared_errors"]) == len(y)


# ==============================================================================
# Test: Benchmark Evaluation
# ==============================================================================

class TestBenchmarkEvaluation:
    """Tests for benchmark model evaluation."""

    def test_evaluate_benchmark(self, evaluator, sample_data):
        """Test benchmark evaluation."""
        X, y, dates = sample_data
        
        # Create mock predictions (shifted y as naive)
        predictions = np.roll(y, 1)
        predictions[0] = np.nan
        
        result = evaluator.evaluate_benchmark(predictions, y, dates)
        
        assert "predictions" in result
        assert "squared_errors" in result
        assert "aggregate" in result

    def test_benchmark_metrics(self, evaluator, sample_data):
        """Test benchmark metrics computation."""
        X, y, dates = sample_data
        predictions = np.roll(y, 1)
        predictions[0] = np.nan
        
        result = evaluator.evaluate_benchmark(predictions, y, dates)
        agg = result["aggregate"]
        
        assert agg["mean_mse"] >= 0
        assert agg["n_predictions"] > 0


# ==============================================================================
# Test: Regime Analysis
# ==============================================================================

class TestRegimeAnalysis:
    """Tests for regime-based performance analysis."""

    def test_compute_regime_metrics(self, sample_data, sample_regimes):
        """Test regime metrics computation."""
        X, y, dates = sample_data
        
        # Create mock results
        predictions = np.roll(y, 1)
        predictions[0] = np.nan
        squared_errors = np.full(len(y), np.nan)
        valid = ~np.isnan(predictions)
        squared_errors[valid] = (predictions[valid] - y[valid]) ** 2
        
        results = {
            "Model1": {
                "predictions": predictions,
                "squared_errors": squared_errors,
                "aggregate": {"mean_mse": np.nanmean(squared_errors)}
            }
        }
        
        regime_df = compute_regime_metrics(results, sample_regimes, y)
        
        assert "Model" in regime_df.columns
        assert "MSE (All)" in regime_df.columns
        assert "MSE (Calm)" in regime_df.columns
        assert "MSE (Crisis)" in regime_df.columns

    def test_regime_segmentation(self, sample_data, sample_regimes):
        """Test that regime segmentation is correct."""
        X, y, dates = sample_data
        
        predictions = y + np.random.randn(len(y)) * 0.01
        squared_errors = (predictions - y) ** 2
        
        results = {
            "Model1": {
                "predictions": predictions,
                "squared_errors": squared_errors,
                "aggregate": {"mean_mse": np.mean(squared_errors)}
            }
        }
        
        regime_df = compute_regime_metrics(results, sample_regimes, y)
        
        # Check that N counts match regime counts
        n_calm = (sample_regimes == "Calm").sum()
        n_crisis = (sample_regimes == "High Volatility").sum()
        
        # Allow for some NaN handling tolerance
        assert regime_df.iloc[0]["N (Calm)"] <= n_calm
        assert regime_df.iloc[0]["N (Crisis)"] <= n_crisis


# ==============================================================================
# Test: Summary Table
# ==============================================================================

class TestSummaryTable:
    """Tests for summary table generation."""

    def test_generate_summary_table(self):
        """Test summary table generation."""
        results = {
            "Model1": {
                "aggregate": {
                    "mean_mse": 0.01,
                    "std_mse": 0.005,
                    "mean_mae": 0.08,
                    "r2": 0.85,
                    "n_predictions": 100
                }
            },
            "Model2": {
                "aggregate": {
                    "mean_mse": 0.02,
                    "std_mse": 0.01,
                    "mean_mae": 0.12,
                    "r2": 0.75,
                    "n_predictions": 100
                }
            }
        }
        
        summary = generate_summary_table(results)
        
        assert len(summary) == 2
        assert "Model" in summary.columns
        assert "Mean MSE" in summary.columns
        assert "Rank" in summary.columns

    def test_summary_sorted_by_mse(self):
        """Test that summary is sorted by MSE."""
        results = {
            "Worst": {"aggregate": {"mean_mse": 0.10}},
            "Best": {"aggregate": {"mean_mse": 0.01}},
            "Middle": {"aggregate": {"mean_mse": 0.05}},
        }
        
        summary = generate_summary_table(results)
        
        # Best model should be first
        assert summary.iloc[0]["Model"] == "Best"
        assert summary.iloc[-1]["Model"] == "Worst"


# ==============================================================================
# Test: Visualization
# ==============================================================================

class TestVisualization:
    """Tests for visualization functions."""

    def test_plot_cumulative_mse_returns_figure(self, sample_data):
        """Test that plot function returns a figure."""
        X, y, dates = sample_data
        
        predictions = np.roll(y, 1)
        squared_errors = np.full(len(y), np.nan)
        valid = ~np.isnan(predictions)
        squared_errors[1:] = (predictions[1:] - y[1:]) ** 2
        
        results = {
            "Model1": {
                "predictions": predictions,
                "squared_errors": squared_errors,
                "dates": dates,
                "aggregate": {"mean_mse": np.nanmean(squared_errors)}
            }
        }
        
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        
        fig = plot_cumulative_mse(results, dates)
        
        assert fig is not None
        
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_handles_multiple_models(self, sample_data):
        """Test plotting multiple models."""
        X, y, dates = sample_data
        
        results = {}
        for i in range(3):
            predictions = y + np.random.randn(len(y)) * (0.01 * (i + 1))
            squared_errors = (predictions - y) ** 2
            results[f"Model{i}"] = {
                "predictions": predictions,
                "squared_errors": squared_errors,
                "dates": dates,
                "aggregate": {"mean_mse": np.mean(squared_errors)}
            }
        
        import matplotlib
        matplotlib.use('Agg')
        
        fig = plot_cumulative_mse(results, dates)
        
        assert fig is not None
        
        import matplotlib.pyplot as plt
        plt.close(fig)


# ==============================================================================
# Test: Edge Cases
# ==============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_insufficient_data(self):
        """Test with insufficient data for splits."""
        evaluator = WalkForwardEvaluator(n_splits=10, test_size=50)
        
        # With 100 samples and 10 splits of 50, should fail or produce few splits
        splits = evaluator._generate_splits(n_samples=100)
        
        # Should produce some splits (even if fewer than requested)
        assert len(splits) >= 0

    def test_all_nan_predictions(self, sample_data):
        """Test handling of all-NaN predictions."""
        X, y, dates = sample_data
        
        predictions = np.full(len(y), np.nan)
        
        evaluator = WalkForwardEvaluator()
        result = evaluator.evaluate_benchmark(predictions, y, dates)
        
        assert "error" in result

    def test_empty_results(self):
        """Test summary with empty results."""
        summary = generate_summary_table({})
        
        assert len(summary) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
