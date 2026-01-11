"""
Tests for Feature Importance Analysis Module
============================================

HEC Lausanne - Data Science and Advanced Programming

Unit tests for feature importance extraction.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.feature_analysis import (
    FeatureImportanceAnalyzer,
    analyze_all_models,
)


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def sample_model_and_data():
    """Generate sample model and data for testing."""
    np.random.seed(42)
    n_samples = 200
    n_features = 5
    
    X = np.random.randn(n_samples, n_features)
    # Feature 0 and 1 are most important
    y = 0.8 * X[:, 0] + 0.5 * X[:, 1] + 0.1 * X[:, 2] + np.random.randn(n_samples) * 0.1
    
    model = RandomForestRegressor(n_estimators=20, random_state=42)
    model.fit(X, y)
    
    feature_names = [f"Feature_{i}" for i in range(n_features)]
    
    return model, X, y, feature_names


@pytest.fixture
def sample_pipeline():
    """Generate sample sklearn pipeline."""
    np.random.seed(42)
    X = np.random.randn(100, 4)
    y = X[:, 0] + np.random.randn(100) * 0.1
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', Ridge(alpha=1.0))
    ])
    pipeline.fit(X, y)
    
    return pipeline


# ==============================================================================
# Test: FeatureImportanceAnalyzer Initialization
# ==============================================================================

class TestFeatureImportanceAnalyzerInit:
    """Tests for FeatureImportanceAnalyzer initialization."""

    def test_initialization(self):
        """Test analyzer initialization."""
        feature_names = ["Vol_21d", "Mom_63d", "Beta_Lag1"]
        analyzer = FeatureImportanceAnalyzer(feature_names)
        
        assert analyzer.feature_names == feature_names
        assert len(analyzer.feature_names) == 3

    def test_empty_feature_names(self):
        """Test initialization with empty feature names."""
        analyzer = FeatureImportanceAnalyzer([])
        assert analyzer.feature_names == []


# ==============================================================================
# Test: Importance Extraction
# ==============================================================================

class TestImportanceExtraction:
    """Tests for feature importance extraction."""

    def test_get_importance_tree_model(self, sample_model_and_data):
        """Test importance extraction from tree-based model."""
        model, X, y, feature_names = sample_model_and_data
        
        analyzer = FeatureImportanceAnalyzer(feature_names)
        importance_df = analyzer.get_importance(model, "RandomForest")
        
        assert len(importance_df) == len(feature_names)
        assert "Feature" in importance_df.columns
        assert "Importance" in importance_df.columns
        assert "Model" in importance_df.columns
        assert "Method" in importance_df.columns
        assert "Importance_Pct" in importance_df.columns
        
        # All importances should be non-negative
        assert (importance_df["Importance"] >= 0).all()
        
        # Model name should be set
        assert importance_df["Model"].iloc[0] == "RandomForest"
        
        # Method should be tree_importance
        assert importance_df["Method"].iloc[0] == "tree_importance"

    def test_get_importance_linear_model(self):
        """Test importance extraction from linear model."""
        np.random.seed(42)
        X = np.random.randn(100, 3)
        y = 2 * X[:, 0] - X[:, 1] + 0.5 * X[:, 2] + np.random.randn(100) * 0.1
        
        model = Ridge(alpha=1.0)
        model.fit(X, y)
        
        feature_names = ["F0", "F1", "F2"]
        analyzer = FeatureImportanceAnalyzer(feature_names)
        importance_df = analyzer.get_importance(model, "Ridge")
        
        assert len(importance_df) == 3
        assert importance_df["Method"].iloc[0] == "abs_coefficient"
        
        # Should use absolute coefficients (all non-negative)
        assert (importance_df["Importance"] >= 0).all()
        
        # F0 should be most important (coefficient = 2)
        top_feature = importance_df.iloc[0]["Feature"]
        assert top_feature == "F0"

    def test_get_importance_from_pipeline(self, sample_pipeline):
        """Test importance extraction from sklearn Pipeline."""
        feature_names = ["F0", "F1", "F2", "F3"]
        
        analyzer = FeatureImportanceAnalyzer(feature_names)
        importance_df = analyzer.get_importance(sample_pipeline, "Pipeline_Ridge")
        
        assert len(importance_df) == len(feature_names)
        assert importance_df["Model"].iloc[0] == "Pipeline_Ridge"

    def test_importance_sorted_descending(self, sample_model_and_data):
        """Test that importance is sorted in descending order."""
        model, X, y, feature_names = sample_model_and_data
        
        analyzer = FeatureImportanceAnalyzer(feature_names)
        importance_df = analyzer.get_importance(model, "RF")
        
        importances = importance_df["Importance"].values
        assert all(importances[i] >= importances[i+1] for i in range(len(importances)-1))

    def test_importance_percentages_sum_to_100(self, sample_model_and_data):
        """Test that importance percentages sum to ~100."""
        model, X, y, feature_names = sample_model_and_data
        
        analyzer = FeatureImportanceAnalyzer(feature_names)
        importance_df = analyzer.get_importance(model, "RF")
        
        total_pct = importance_df["Importance_Pct"].sum()
        np.testing.assert_almost_equal(total_pct, 100.0, decimal=1)


# ==============================================================================
# Test: Plotting
# ==============================================================================

class TestPlotting:
    """Tests for importance plotting."""

    def test_plot_importance_creates_figure(self, sample_model_and_data):
        """Test that plot_importance creates a matplotlib figure."""
        model, X, y, feature_names = sample_model_and_data
        
        analyzer = FeatureImportanceAnalyzer(feature_names)
        importance_df = analyzer.get_importance(model, "RF")
        
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        
        fig = analyzer.plot_importance({"RF": importance_df}, top_n=5)
        
        assert fig is not None
        
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_importance_saves_file(self, sample_model_and_data, tmp_path):
        """Test that plot can be saved to file."""
        model, X, y, feature_names = sample_model_and_data
        
        analyzer = FeatureImportanceAnalyzer(feature_names)
        importance_df = analyzer.get_importance(model, "RF")
        
        import matplotlib
        matplotlib.use('Agg')
        
        save_path = tmp_path / "importance_test.png"
        fig = analyzer.plot_importance(
            {"RF": importance_df},
            top_n=5,
            save_path=str(save_path)
        )
        
        assert save_path.exists()
        
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_multiple_models(self, sample_model_and_data):
        """Test plotting with multiple models."""
        model, X, y, feature_names = sample_model_and_data
        
        # Train another model
        ridge = Ridge(alpha=1.0)
        ridge.fit(X, y)
        
        analyzer = FeatureImportanceAnalyzer(feature_names)
        
        importance_dict = {
            "RF": analyzer.get_importance(model, "RF"),
            "Ridge": analyzer.get_importance(ridge, "Ridge"),
        }
        
        import matplotlib
        matplotlib.use('Agg')
        
        fig = analyzer.plot_importance(importance_dict, top_n=5)
        
        assert fig is not None
        
        import matplotlib.pyplot as plt
        plt.close(fig)


# ==============================================================================
# Test: analyze_all_models Integration
# ==============================================================================

class TestAnalyzeAllModels:
    """Tests for analyze_all_models function."""

    def test_analyze_all_models_basic(self, tmp_path):
        """Test analyze_all_models with mock ML results."""
        np.random.seed(42)
        X = np.random.randn(100, 4)
        y = X[:, 0] + np.random.randn(100) * 0.1
        
        # Create mock ML results
        rf = RandomForestRegressor(n_estimators=10, random_state=42)
        rf.fit(X, y)
        
        ridge = Ridge(alpha=1.0)
        ridge.fit(X, y)
        
        ml_results = {
            "random_forest": (rf, {}),
            "ridge": (ridge, {}),
        }
        
        feature_names = ["Vol_21d", "Vol_63d", "Mom_21d", "Beta_Lag1"]
        
        import matplotlib
        matplotlib.use('Agg')
        
        # Analyze
        importance_dict = analyze_all_models(
            ml_results=ml_results,
            feature_names=feature_names,
            output_dir=str(tmp_path)
        )
        
        # Check returns
        assert len(importance_dict) == 2
        assert "random_forest" in importance_dict
        assert "ridge" in importance_dict
        
        # Check files created
        assert (tmp_path / "feature_importance.csv").exists()
        assert (tmp_path / "feature_importance.png").exists()
        
        import matplotlib.pyplot as plt
        plt.close('all')

    def test_analyze_all_models_csv_content(self, tmp_path):
        """Test that CSV output has correct structure."""
        np.random.seed(42)
        X = np.random.randn(100, 3)
        y = X[:, 0] + np.random.randn(100) * 0.1
        
        rf = RandomForestRegressor(n_estimators=10, random_state=42)
        rf.fit(X, y)
        
        ml_results = {"random_forest": (rf, {})}
        feature_names = ["F0", "F1", "F2"]
        
        import matplotlib
        matplotlib.use('Agg')
        
        analyze_all_models(
            ml_results=ml_results,
            feature_names=feature_names,
            output_dir=str(tmp_path)
        )
        
        # Read and verify CSV
        csv_path = tmp_path / "feature_importance.csv"
        df = pd.read_csv(csv_path)
        
        assert "Feature" in df.columns
        assert "Importance" in df.columns
        assert "Model" in df.columns
        assert len(df) == 3  # 3 features
        
        import matplotlib.pyplot as plt
        plt.close('all')


# ==============================================================================
# Test: Edge Cases
# ==============================================================================

class TestEdgeCases:
    """Tests for edge cases."""

    def test_model_without_importance(self):
        """Test error for model without importance attributes."""
        from sklearn.neighbors import KNeighborsRegressor
        
        np.random.seed(42)
        X = np.random.randn(50, 3)
        y = np.random.randn(50)
        
        knn = KNeighborsRegressor(n_neighbors=5)
        knn.fit(X, y)
        
        analyzer = FeatureImportanceAnalyzer(["F0", "F1", "F2"])
        
        with pytest.raises(ValueError, match="does not have"):
            analyzer.get_importance(knn, "KNN")

    def test_mismatched_feature_count(self):
        """Test handling of mismatched feature counts."""
        np.random.seed(42)
        X = np.random.randn(50, 5)
        y = np.random.randn(50)
        
        rf = RandomForestRegressor(n_estimators=10, random_state=42)
        rf.fit(X, y)
        
        # Only 3 feature names for 5 features
        analyzer = FeatureImportanceAnalyzer(["F0", "F1", "F2"])
        
        # Should handle gracefully (with warning)
        importance_df = analyzer.get_importance(rf, "RF")
        
        assert len(importance_df) == 3  # Uses available names


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
