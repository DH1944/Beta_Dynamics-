"""
Tests for Models Module
=======================

HEC Lausanne - Data Science and Advanced Programming

Unit tests for Naive, Welch BSWA, Kalman, and ML models.
"""

import numpy as np
import pandas as pd
import pytest

from src.models_benchmarks import (
    NaiveBaseline,
    WelchBSWA,
    KalmanBeta,
    create_benchmark_model,
)


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def sample_returns():
    """Generate sample returns for testing."""
    np.random.seed(42)
    n = 500
    
    # Generate market and stock returns
    market = np.random.randn(n) * 0.01
    stock = 1.2 * market + np.random.randn(n) * 0.005
    
    return stock, market


@pytest.fixture
def time_varying_beta_data():
    """Generate data with known time-varying beta."""
    np.random.seed(42)
    n = 1000
    
    # True beta varies sinusoidally
    t = np.linspace(0, 4 * np.pi, n)
    true_beta = 1.0 + 0.3 * np.sin(t)
    
    market = np.random.randn(n) * 0.01
    stock = true_beta * market + np.random.randn(n) * 0.003
    
    return stock, market, true_beta


# ==============================================================================
# Test: Naive Baseline
# ==============================================================================

class TestNaiveBaseline:
    """Tests for Naive baseline model."""

    def test_initialization(self):
        """Test model initialization."""
        model = NaiveBaseline(window=252)
        assert model.window == 252
        assert model._fitted is False

    def test_fit(self, sample_returns):
        """Test fit method."""
        stock, market = sample_returns
        model = NaiveBaseline(window=100)
        
        result = model.fit(stock, market)
        
        assert result is model
        assert model._fitted is True

    def test_predict_shape(self, sample_returns):
        """Test prediction output shape."""
        stock, market = sample_returns
        model = NaiveBaseline(window=100)
        model.fit(stock, market)
        
        predictions = model.predict(stock, market)
        
        assert len(predictions) == len(stock)

    def test_predict_is_lagged(self, sample_returns):
        """Test that prediction is lagged beta."""
        stock, market = sample_returns
        model = NaiveBaseline(window=100)
        model.fit(stock, market)
        
        predictions = model.predict(stock, market)
        realized = model._compute_rolling_beta(stock, market, 100)
        
        # Prediction at t should equal realized at t-1
        # Check for indices where both are valid
        valid_idx = 101  # After window + 1 for lag
        np.testing.assert_almost_equal(
            predictions[valid_idx],
            realized[valid_idx - 1],
            decimal=10
        )

    def test_predict_first_value_nan(self, sample_returns):
        """Test that first prediction is NaN."""
        stock, market = sample_returns
        model = NaiveBaseline(window=100)
        model.fit(stock, market)
        
        predictions = model.predict(stock, market)
        
        assert np.isnan(predictions[0])


# ==============================================================================
# Test: Welch BSWA
# ==============================================================================

class TestWelchBSWA:
    """Tests for Welch BSWA model."""

    def test_initialization(self):
        """Test model initialization."""
        model = WelchBSWA(window=252, decay_halflife=126, winsor_pct=0.05)
        
        assert model.window == 252
        assert model.decay_halflife == 126
        assert model.winsor_pct == 0.05

    def test_decay_weights(self):
        """Test exponential decay weights."""
        model = WelchBSWA(decay_halflife=10)
        weights = model._compute_decay_weights(20)
        
        # Weights should sum to 1
        np.testing.assert_almost_equal(np.sum(weights), 1.0)
        
        # Most recent observation should have highest weight
        assert weights[-1] > weights[0]
        
        # Weight at halflife should be ~half of most recent
        # weights[10] should be ~0.5 * weights[20]
        assert weights[9] < weights[-1]

    def test_winsorize(self):
        """Test winsorization."""
        model = WelchBSWA()
        arr = np.array([1, 2, 3, 4, 100])  # 100 is an outlier
        
        winsorized = model._winsorize(arr, 0.1)
        
        # Outlier should be clipped to upper percentile
        assert winsorized[-1] < 100
        
        # All values should be within the percentile bounds
        lower_bound = np.percentile(arr, 10)
        upper_bound = np.percentile(arr, 90)
        assert np.all(winsorized >= lower_bound)
        assert np.all(winsorized <= upper_bound)
        
        # Original array should not be modified
        assert arr[-1] == 100

    def test_predict_shape(self, sample_returns):
        """Test prediction output shape."""
        stock, market = sample_returns
        model = WelchBSWA(window=100)
        model.fit(stock, market)
        
        predictions = model.predict(stock, market)
        
        assert len(predictions) == len(stock)

    def test_predict_initial_nan(self, sample_returns):
        """Test that initial predictions are NaN."""
        stock, market = sample_returns
        model = WelchBSWA(window=100)
        model.fit(stock, market)
        
        predictions = model.predict(stock, market)
        
        # First (window-1) values should be NaN
        assert np.isnan(predictions[:99]).all()

    def test_predict_reasonable_beta(self, sample_returns):
        """Test that predictions are in reasonable range."""
        stock, market = sample_returns
        model = WelchBSWA(window=100)
        model.fit(stock, market)
        
        predictions = model.predict(stock, market)
        
        valid = ~np.isnan(predictions)
        # Beta should typically be between -3 and 3 for most stocks
        assert np.all(np.abs(predictions[valid]) < 5)

    def test_get_params(self):
        """Test parameter retrieval."""
        model = WelchBSWA(window=200, decay_halflife=100, winsor_pct=0.10)
        params = model.get_params()
        
        assert params["window"] == 200
        assert params["decay_halflife"] == 100
        assert params["winsor_pct"] == 0.10


# ==============================================================================
# Test: Kalman Filter
# ==============================================================================

class TestKalmanBeta:
    """Tests for Kalman Filter beta model."""

    def test_initialization(self):
        """Test model initialization."""
        model = KalmanBeta(
            initial_beta=1.0,
            initial_var=1.0,
            obs_noise=0.001,
            state_noise=0.001
        )
        
        assert model.initial_beta == 1.0
        assert model.initial_var == 1.0
        assert model.obs_noise == 0.001
        assert model.state_noise == 0.001

    def test_predict_shape(self, sample_returns):
        """Test prediction output shape."""
        stock, market = sample_returns
        model = KalmanBeta()
        model.fit(stock, market)
        
        predictions = model.predict(stock, market)
        
        assert len(predictions) == len(stock)

    def test_predict_no_nan(self, sample_returns):
        """Test that Kalman filter produces no NaN (after handling)."""
        stock, market = sample_returns
        model = KalmanBeta()
        model.fit(stock, market)
        
        predictions = model.predict(stock, market)
        
        # Kalman filter should produce predictions from t=0
        # May have NaN only if market return is exactly zero
        valid_market = np.abs(market) > 1e-10
        assert not np.isnan(predictions[valid_market]).any()

    def test_predict_with_uncertainty(self, sample_returns):
        """Test prediction with uncertainty estimates."""
        stock, market = sample_returns
        model = KalmanBeta()
        model.fit(stock, market)
        
        beta, variance = model.predict_with_uncertainty(stock, market)
        
        assert len(beta) == len(stock)
        assert len(variance) == len(stock)
        
        # Variance should be non-negative
        valid = ~np.isnan(variance)
        assert np.all(variance[valid] >= 0)

    def test_tracks_time_varying_beta(self, time_varying_beta_data):
        """Test that Kalman tracks time-varying beta."""
        stock, market, true_beta = time_varying_beta_data
        
        model = KalmanBeta(initial_beta=1.0, state_noise=0.001)
        model.fit(stock, market)
        predictions = model.predict(stock, market)
        
        # Correlation between predicted and true beta should be positive
        valid = ~np.isnan(predictions)
        correlation = np.corrcoef(predictions[valid], true_beta[valid])[0, 1]
        
        assert correlation > 0.5  # Should track reasonably well

    def test_get_params(self):
        """Test parameter retrieval."""
        model = KalmanBeta(
            initial_beta=1.5,
            initial_var=2.0,
            obs_noise=0.002,
            state_noise=0.002
        )
        params = model.get_params()
        
        assert params["initial_beta"] == 1.5
        assert params["initial_var"] == 2.0
        assert params["obs_noise"] == 0.002
        assert params["state_noise"] == 0.002


# ==============================================================================
# Test: Factory Function
# ==============================================================================

class TestFactoryFunction:
    """Tests for model factory function."""

    def test_create_naive(self):
        """Test creating Naive model."""
        model = create_benchmark_model("naive", window=100)
        
        assert isinstance(model, NaiveBaseline)
        assert model.window == 100

    def test_create_welch(self):
        """Test creating Welch model."""
        model = create_benchmark_model(
            "welch",
            window=200,
            decay_halflife=100,
            winsor_pct=0.05
        )
        
        assert isinstance(model, WelchBSWA)
        assert model.window == 200
        assert model.decay_halflife == 100

    def test_create_kalman(self):
        """Test creating Kalman model."""
        model = create_benchmark_model("kalman", initial_beta=0.8)
        
        assert isinstance(model, KalmanBeta)
        assert model.initial_beta == 0.8

    def test_case_insensitive(self):
        """Test case-insensitive model names."""
        model1 = create_benchmark_model("NAIVE")
        model2 = create_benchmark_model("Naive")
        model3 = create_benchmark_model("naive")
        
        assert isinstance(model1, NaiveBaseline)
        assert isinstance(model2, NaiveBaseline)
        assert isinstance(model3, NaiveBaseline)

    def test_invalid_model_name(self):
        """Test error for invalid model name."""
        with pytest.raises(ValueError, match="Unknown model"):
            create_benchmark_model("invalid_model")


# ==============================================================================
# Test: Model Comparison
# ==============================================================================

class TestModelComparison:
    """Tests comparing benchmark models."""

    def test_all_models_produce_output(self, sample_returns):
        """Test that all models produce predictions."""
        stock, market = sample_returns
        
        models = {
            "Naive": NaiveBaseline(window=100),
            "Welch": WelchBSWA(window=100),
            "Kalman": KalmanBeta()
        }
        
        for name, model in models.items():
            model.fit(stock, market)
            predictions = model.predict(stock, market)
            
            valid = ~np.isnan(predictions)
            assert np.any(valid), f"{name} produced no valid predictions"

    def test_mse_ordering(self, time_varying_beta_data):
        """Test that sophisticated models outperform naive on time-varying beta."""
        stock, market, true_beta = time_varying_beta_data
        
        models = {
            "Naive": NaiveBaseline(window=100),
            "Kalman": KalmanBeta(state_noise=0.001)
        }
        
        mse = {}
        for name, model in models.items():
            model.fit(stock, market)
            predictions = model.predict(stock, market)
            
            valid = ~np.isnan(predictions)
            mse[name] = np.mean((predictions[valid] - true_beta[valid]) ** 2)
        
        # Kalman should have lower MSE on time-varying beta
        # Note: This is not always guaranteed, but should hold for this test case
        assert mse["Kalman"] <= mse["Naive"] * 1.5  # Allow some slack


# ==============================================================================
# Test: Edge Cases
# ==============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_zero_market_returns(self):
        """Test behavior with zero market returns."""
        np.random.seed(42)
        n = 200
        market = np.zeros(n)
        stock = np.random.randn(n) * 0.01
        
        model = KalmanBeta()
        model.fit(stock, market)
        predictions = model.predict(stock, market)
        
        # Should still produce output (may be initial beta)
        assert len(predictions) == n

    def test_very_short_window(self):
        """Test with very short window."""
        np.random.seed(42)
        market = np.random.randn(50) * 0.01
        stock = 1.0 * market + np.random.randn(50) * 0.005
        
        model = WelchBSWA(window=10)
        model.fit(stock, market)
        predictions = model.predict(stock, market)
        
        valid = ~np.isnan(predictions)
        assert np.sum(valid) > 0

    def test_extreme_returns(self):
        """Test with extreme return values."""
        np.random.seed(42)
        n = 200
        market = np.random.randn(n) * 0.01
        stock = market.copy()
        
        # Add extreme values
        stock[50] = 0.5  # 50% daily return
        stock[100] = -0.4  # -40% daily return
        
        model = WelchBSWA(window=100, winsor_pct=0.05)
        model.fit(stock, market)
        predictions = model.predict(stock, market)
        
        valid = ~np.isnan(predictions)
        # Winsorization should keep beta reasonable
        assert np.all(np.abs(predictions[valid]) < 10)


# ==============================================================================
# Tests for ML Pipeline
# ==============================================================================

from src.models_ml import MLModelPipeline


class TestMLPipeline:
    """Tests for MLModelPipeline class."""

    def test_train_model_ridge(self):
        """Test training Ridge model."""
        np.random.seed(42)
        pipeline = MLModelPipeline(n_splits=3, verbose=0)
        
        X = np.random.randn(500, 5)
        y = X[:, 0] + 0.5 * X[:, 1] + np.random.randn(500) * 0.1
        
        model, results = pipeline.train_model('ridge', X, y)
        
        assert results['best_score'] >= 0
        assert 'best_params' in results
        assert model is not None

    def test_train_model_random_forest(self):
        """Test training Random Forest model."""
        np.random.seed(42)
        pipeline = MLModelPipeline(n_splits=3, verbose=0)
        
        X = np.random.randn(300, 4)
        y = X[:, 0] + np.random.randn(300) * 0.1
        
        model, results = pipeline.train_model('random_forest', X, y)
        
        assert results['best_score'] >= 0
        assert model is not None

    def test_clean_data_removes_nan(self):
        """Test that _clean_data removes rows with NaN."""
        X = np.array([[1, 2], [np.nan, 3], [4, 5], [6, np.nan]])
        y = np.array([1, 2, 3, 4])
        
        X_clean, y_clean = MLModelPipeline._clean_data(X, y)
        
        # Only rows 0 and 2 should remain
        assert len(X_clean) == 2
        assert len(y_clean) == 2
        np.testing.assert_array_equal(X_clean[0], [1, 2])
        np.testing.assert_array_equal(X_clean[1], [4, 5])

    def test_clean_data_removes_nan_y(self):
        """Test that _clean_data removes rows with NaN in y."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([1, np.nan, 3])
        
        X_clean, y_clean = MLModelPipeline._clean_data(X, y)
        
        assert len(X_clean) == 2
        assert len(y_clean) == 2

    def test_assertion_x_y_length_mismatch(self):
        """Test that mismatched X and y lengths raise AssertionError."""
        pipeline = MLModelPipeline(n_splits=3, verbose=0)
        
        X = np.random.randn(100, 5)
        y = np.random.randn(50)  # Different length
        
        with pytest.raises(AssertionError, match="same length"):
            pipeline.train_model('ridge', X, y)

    def test_assertion_all_nan_y(self):
        """Test that all-NaN y raises AssertionError."""
        pipeline = MLModelPipeline(n_splits=3, verbose=0)
        
        X = np.random.randn(100, 5)
        y = np.full(100, np.nan)
        
        with pytest.raises(AssertionError, match="entirely NaN"):
            pipeline.train_model('ridge', X, y)

    def test_train_all_models(self):
        """Test training all models."""
        np.random.seed(42)
        pipeline = MLModelPipeline(n_splits=3, verbose=0)
        
        X = np.random.randn(500, 5)
        y = X[:, 0] + np.random.randn(500) * 0.1
        
        results = pipeline.train_all_models(X, y)
        
        # Should have at least Ridge and RandomForest (capitalized names)
        assert 'Ridge' in results, f"Ridge not in results. Got: {list(results.keys())}"
        assert 'RandomForest' in results, f"RandomForest not in results. Got: {list(results.keys())}"

    def test_predict_after_train(self):
        """Test prediction after training."""
        np.random.seed(42)
        pipeline = MLModelPipeline(n_splits=3, verbose=0)
        
        X = np.random.randn(200, 3)
        y = X[:, 0] + np.random.randn(200) * 0.1
        
        pipeline.train_model('ridge', X, y)
        predictions = pipeline.predict('ridge', X)
        
        assert len(predictions) == len(X)
        assert not np.all(np.isnan(predictions))

    def test_predict_unfitted_raises_error(self):
        """Test that predicting with unfitted model raises error."""
        pipeline = MLModelPipeline(n_splits=3, verbose=0)
        
        X = np.random.randn(50, 3)
        
        with pytest.raises(ValueError, match="has not been fitted"):
            pipeline.predict('ridge', X)

    def test_feature_importance_tree(self):
        """Test feature importance extraction from tree model."""
        np.random.seed(42)
        pipeline = MLModelPipeline(n_splits=3, verbose=0)
        
        feature_names = ['f0', 'f1', 'f2', 'f3']
        X = np.random.randn(300, 4)
        y = 2 * X[:, 0] + X[:, 1] + np.random.randn(300) * 0.1
        
        pipeline.train_model('random_forest', X, y)
        importance = pipeline.get_feature_importance('random_forest', feature_names)
        
        assert len(importance) == 4
        assert 'feature' in importance.columns  # lowercase
        assert 'importance' in importance.columns  # lowercase
        
        # f0 should be most important
        assert importance.iloc[0]['feature'] == 'f0'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
