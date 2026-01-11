"""
Tests for Feature Engineering Module
====================================

HEC Lausanne - Data Science and Advanced Programming

Unit tests for feature computation and Numba functions.
"""

import numpy as np
import pandas as pd
import pytest

from src.features import (
    FeatureEngineer,
    _rolling_mean_numba,
    _rolling_std_numba,
    _rolling_ols_beta_numba,
    _momentum_numba,
    benchmark_numba_vs_pandas,
)


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def sample_returns_data():
    """Generate sample returns data for testing."""
    np.random.seed(42)
    dates = pd.date_range("2018-01-01", periods=1000, freq="B")
    
    # Generate correlated returns
    market_returns = np.random.randn(len(dates)) * 0.01
    stock_returns = 1.2 * market_returns + np.random.randn(len(dates)) * 0.005
    
    data = pd.DataFrame({
        "Market_Return": market_returns,
        "TEST.SW_Return": stock_returns,
    }, index=dates)
    
    return data


@pytest.fixture
def feature_engineer(sample_returns_data):
    """Create FeatureEngineer instance."""
    return FeatureEngineer(
        sample_returns_data,
        beta_window=252,
        vol_window=21,
        momentum_window=63,
    )


# ==============================================================================
# Test: Numba Rolling Mean
# ==============================================================================

class TestRollingMeanNumba:
    """Tests for Numba rolling mean implementation."""

    def test_rolling_mean_basic(self):
        """Test basic rolling mean calculation."""
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        window = 3
        
        result = _rolling_mean_numba(arr, window)
        
        # First two values should be NaN
        assert np.isnan(result[0])
        assert np.isnan(result[1])
        
        # Third value: mean(1, 2, 3) = 2.0
        np.testing.assert_almost_equal(result[2], 2.0)
        
        # Fourth value: mean(2, 3, 4) = 3.0
        np.testing.assert_almost_equal(result[3], 3.0)
        
        # Fifth value: mean(3, 4, 5) = 4.0
        np.testing.assert_almost_equal(result[4], 4.0)

    def test_rolling_mean_vs_pandas(self):
        """Test Numba rolling mean matches Pandas."""
        np.random.seed(42)
        arr = np.random.randn(500)
        window = 20
        
        numba_result = _rolling_mean_numba(arr, window)
        pandas_result = pd.Series(arr).rolling(window).mean().values
        
        # Compare non-NaN values
        valid = ~np.isnan(numba_result)
        np.testing.assert_array_almost_equal(
            numba_result[valid],
            pandas_result[valid],
            decimal=10
        )


# ==============================================================================
# Test: Numba Rolling Std
# ==============================================================================

class TestRollingStdNumba:
    """Tests for Numba rolling standard deviation."""

    def test_rolling_std_basic(self):
        """Test basic rolling std calculation."""
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        window = 3
        
        result = _rolling_std_numba(arr, window)
        
        # First two values should be NaN
        assert np.isnan(result[0])
        assert np.isnan(result[1])
        
        # Third value: std([1, 2, 3]) = 1.0
        np.testing.assert_almost_equal(result[2], 1.0)

    def test_rolling_std_vs_pandas(self):
        """Test Numba rolling std matches Pandas."""
        np.random.seed(42)
        arr = np.random.randn(500)
        window = 20
        
        numba_result = _rolling_std_numba(arr, window)
        pandas_result = pd.Series(arr).rolling(window).std().values
        
        # Compare non-NaN values
        valid = ~np.isnan(numba_result)
        np.testing.assert_array_almost_equal(
            numba_result[valid],
            pandas_result[valid],
            decimal=10
        )


# ==============================================================================
# Test: Numba Rolling OLS Beta
# ==============================================================================

class TestRollingBetaNumba:
    """Tests for Numba rolling OLS beta calculation."""

    def test_rolling_beta_perfect_correlation(self):
        """Test beta = 1 when stock = market."""
        np.random.seed(42)
        n = 300
        market = np.random.randn(n) * 0.01
        stock = market.copy()  # Perfect correlation, beta = 1
        
        result = _rolling_ols_beta_numba(stock, market, 252)
        
        # After window, beta should be ~1
        valid = ~np.isnan(result)
        np.testing.assert_array_almost_equal(
            result[valid],
            np.ones(np.sum(valid)),
            decimal=5
        )

    def test_rolling_beta_known_beta(self):
        """Test beta estimation with known true beta."""
        np.random.seed(42)
        n = 500
        true_beta = 1.5
        
        market = np.random.randn(n) * 0.01
        stock = true_beta * market + np.random.randn(n) * 0.001  # Small noise
        
        result = _rolling_ols_beta_numba(stock, market, 252)
        
        # After window, beta should be close to true_beta
        valid_idx = ~np.isnan(result)
        mean_beta = np.mean(result[valid_idx])
        
        np.testing.assert_almost_equal(mean_beta, true_beta, decimal=1)

    def test_rolling_beta_vs_pandas(self):
        """Test Numba beta matches Pandas covariance method."""
        np.random.seed(42)
        n = 500
        window = 60
        
        market = np.random.randn(n) * 0.01
        stock = 1.2 * market + np.random.randn(n) * 0.005
        
        # Numba result
        numba_result = _rolling_ols_beta_numba(stock, market, window)
        
        # Pandas result
        stock_series = pd.Series(stock)
        market_series = pd.Series(market)
        cov = stock_series.rolling(window).cov(market_series)
        var = market_series.rolling(window).var()
        pandas_result = (cov / var).values
        
        # Compare non-NaN values
        valid = ~(np.isnan(numba_result) | np.isnan(pandas_result))
        np.testing.assert_array_almost_equal(
            numba_result[valid],
            pandas_result[valid],
            decimal=8
        )


# ==============================================================================
# Test: Feature Engineer Initialization
# ==============================================================================

class TestFeatureEngineerInit:
    """Tests for FeatureEngineer initialization."""

    def test_initialization(self, feature_engineer):
        """Test basic initialization."""
        assert feature_engineer.beta_window == 252
        assert feature_engineer.vol_window == 21
        assert feature_engineer.momentum_window == 63

    def test_missing_market_return(self):
        """Test error when Market_Return is missing."""
        data = pd.DataFrame({"Stock_Return": [0.01, 0.02, -0.01]})
        
        with pytest.raises(ValueError, match="Market_Return"):
            FeatureEngineer(data)

    def test_get_stock_columns(self, feature_engineer):
        """Test stock column identification."""
        columns = feature_engineer._get_stock_columns()
        
        assert "TEST.SW_Return" in columns
        assert "Market_Return" not in columns


# ==============================================================================
# Test: Rolling Beta Computation
# ==============================================================================

class TestRollingBeta:
    """Tests for rolling beta computation."""

    def test_compute_rolling_beta(self, feature_engineer):
        """Test rolling beta computation."""
        beta = feature_engineer.compute_rolling_beta("TEST.SW_Return")
        
        assert len(beta) == len(feature_engineer.data)
        assert beta.name == "TEST.SW_Return_Beta"
        
        # First 251 values should be NaN (252 window)
        assert np.isnan(beta.iloc[:251]).all()
        
        # After window, should have valid values
        assert not np.isnan(beta.iloc[251:]).all()

    def test_rolling_beta_numba_vs_pandas(self, feature_engineer):
        """Test Numba and Pandas implementations match."""
        beta_numba = feature_engineer.compute_rolling_beta("TEST.SW_Return", use_numba=True)
        beta_pandas = feature_engineer.compute_rolling_beta("TEST.SW_Return", use_numba=False)
        
        valid = ~(np.isnan(beta_numba) | np.isnan(beta_pandas))
        np.testing.assert_array_almost_equal(
            beta_numba[valid].values,
            beta_pandas[valid].values,
            decimal=8
        )


# ==============================================================================
# Test: Volatility Computation
# ==============================================================================

class TestVolatilityComputation:
    """Tests for volatility calculation."""

    def test_compute_volatility(self, feature_engineer):
        """Test volatility computation."""
        vol = feature_engineer.compute_rolling_volatility("TEST.SW_Return", window=21)
        
        assert len(vol) == len(feature_engineer.data)
        assert vol.name == "TEST.SW_Return_Vol_21d"

    def test_annualized_volatility(self, feature_engineer):
        """Test annualization factor."""
        vol_annual = feature_engineer.compute_rolling_volatility(
            "TEST.SW_Return", window=21, annualize=True
        )
        vol_daily = feature_engineer.compute_rolling_volatility(
            "TEST.SW_Return", window=21, annualize=False
        )
        
        valid = ~(np.isnan(vol_annual) | np.isnan(vol_daily))
        ratio = vol_annual[valid] / vol_daily[valid]
        
        # Ratio should be sqrt(252)
        expected_ratio = np.sqrt(252)
        np.testing.assert_array_almost_equal(
            ratio.values,
            np.full(np.sum(valid), expected_ratio),
            decimal=5
        )


# ==============================================================================
# Test: Feature Creation
# ==============================================================================

class TestFeatureCreation:
    """Tests for full feature set creation."""

    def test_create_features_for_stock(self, feature_engineer):
        """Test feature creation for single stock."""
        features, target = feature_engineer.create_features_for_stock("TEST.SW_Return")
        
        assert isinstance(features, pd.DataFrame)
        assert isinstance(target, pd.Series)
        assert len(features) == len(target)

    def test_feature_columns(self, feature_engineer):
        """Test expected feature columns are created."""
        features, _ = feature_engineer.create_features_for_stock("TEST.SW_Return")
        
        expected_cols = ["Vol_21d", "Vol_63d", "Mom_21d", "Mom_63d", "Beta_Lag1"]
        for col in expected_cols:
            assert col in features.columns, f"Missing column: {col}"

    def test_target_shift(self, feature_engineer):
        """Test target is shifted to prevent look-ahead bias."""
        # With shift_target=True, target at t is beta at t+1
        _, target_shifted = feature_engineer.create_features_for_stock(
            "TEST.SW_Return", shift_target=True
        )
        _, target_unshifted = feature_engineer.create_features_for_stock(
            "TEST.SW_Return", shift_target=False
        )
        
        # Last value of shifted should be NaN
        assert np.isnan(target_shifted.iloc[-1])


# ==============================================================================
# Test: Benchmark Function
# ==============================================================================

class TestBenchmarkFunction:
    """Tests for Numba vs Pandas benchmark."""

    def test_benchmark_runs(self):
        """Test benchmark function runs without error."""
        results = benchmark_numba_vs_pandas(n_samples=500, window=50)
        
        assert "pandas_std_time" in results
        assert "numba_std_time" in results
        assert "pandas_beta_time" in results
        assert "numba_beta_time" in results

    def test_benchmark_positive_times(self):
        """Test that all timing results are positive."""
        results = benchmark_numba_vs_pandas(n_samples=500, window=50)
        
        for key, value in results.items():
            assert value > 0, f"{key} should be positive"


# ==============================================================================
# Test: Momentum Calculation
# ==============================================================================

class TestMomentumNumba:
    """Tests for Numba momentum calculation."""

    def test_momentum_basic(self):
        """Test basic momentum calculation."""
        prices = np.array([100.0, 110.0, 121.0, 133.1, 146.41])
        window = 2
        
        result = _momentum_numba(prices, window)
        
        # First two values should be NaN
        assert np.isnan(result[0])
        assert np.isnan(result[1])
        
        # Third value: 121/100 - 1 = 0.21
        np.testing.assert_almost_equal(result[2], 0.21, decimal=5)


# ==============================================================================
# Test: Edge Cases
# ==============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_short_data_series(self):
        """Test behavior with data shorter than window."""
        data = pd.DataFrame({
            "Market_Return": [0.01, 0.02, -0.01],
            "TEST.SW_Return": [0.02, 0.01, -0.02],
        })
        
        engineer = FeatureEngineer(data, beta_window=252)
        beta = engineer.compute_rolling_beta("TEST.SW_Return")
        
        # All values should be NaN (data shorter than window)
        assert np.isnan(beta).all()

    def test_zero_variance_market(self):
        """Test behavior when market has zero variance."""
        data = pd.DataFrame({
            "Market_Return": np.zeros(300),  # Zero variance
            "TEST.SW_Return": np.random.randn(300) * 0.01,
        })
        
        engineer = FeatureEngineer(data, beta_window=100)
        beta = engineer.compute_rolling_beta("TEST.SW_Return")
        
        # Beta should be NaN when market variance is zero
        # All should be NaN due to zero variance
        assert np.isnan(beta.iloc[99:]).all()


# ==============================================================================
# Test: Market Regime Feature
# ==============================================================================

class TestRegimeFeature:
    """Tests for market regime feature in create_features_for_stock."""
    
    @pytest.fixture
    def sample_data_for_regime(self):
        """Generate sample data for regime testing."""
        np.random.seed(42)
        dates = pd.date_range("2018-01-01", periods=500, freq="B")
        
        market_returns = np.random.randn(len(dates)) * 0.01
        stock_returns = 1.2 * market_returns + np.random.randn(len(dates)) * 0.005
        
        data = pd.DataFrame({
            "Market_Return": market_returns,
            "TEST.SW_Return": stock_returns,
        }, index=dates)
        
        return data
    
    def test_regime_feature_exists_when_enabled(self, sample_data_for_regime):
        """Test that regime feature is created when include_regime=True."""
        engineer = FeatureEngineer(sample_data_for_regime, beta_window=100)
        
        features, _ = engineer.create_features_for_stock(
            "TEST.SW_Return", 
            include_regime=True
        )
        
        assert "Regime_HighVol" in features.columns
        assert "Vol_21d_x_Regime" in features.columns
    
    def test_regime_feature_absent_when_disabled(self, sample_data_for_regime):
        """Test that regime feature is NOT created when include_regime=False."""
        engineer = FeatureEngineer(sample_data_for_regime, beta_window=100)
        
        features, _ = engineer.create_features_for_stock(
            "TEST.SW_Return",
            include_regime=False
        )
        
        assert "Regime_HighVol" not in features.columns
        assert "Vol_21d_x_Regime" not in features.columns
    
    def test_regime_feature_is_binary(self, sample_data_for_regime):
        """Test that regime feature is binary (0 or 1)."""
        engineer = FeatureEngineer(sample_data_for_regime, beta_window=100)
        
        features, _ = engineer.create_features_for_stock(
            "TEST.SW_Return",
            include_regime=True
        )
        
        # Drop NaN and check unique values
        unique_values = features["Regime_HighVol"].dropna().unique()
        
        # Should only contain 0 and/or 1
        assert set(unique_values).issubset({0, 1})
    
    def test_interaction_term_computation(self, sample_data_for_regime):
        """Test that interaction term is computed correctly."""
        engineer = FeatureEngineer(sample_data_for_regime, beta_window=100)
        
        features, _ = engineer.create_features_for_stock(
            "TEST.SW_Return",
            include_regime=True
        )
        
        # Interaction should be Vol_21d * Regime
        vol = features["Vol_21d"].fillna(0)
        regime = features["Regime_HighVol"].fillna(0)
        expected_interaction = vol * regime
        actual_interaction = features["Vol_21d_x_Regime"].fillna(0)
        
        np.testing.assert_array_almost_equal(
            actual_interaction.values,
            expected_interaction.values
        )
    
    def test_regime_when_high_vol(self, sample_data_for_regime):
        """Test regime is 1 when volatility is high."""
        engineer = FeatureEngineer(sample_data_for_regime, beta_window=100)
        
        features, _ = engineer.create_features_for_stock(
            "TEST.SW_Return",
            include_regime=True
        )
        
        # Where regime=1, interaction should equal Vol_21d
        high_vol_mask = features["Regime_HighVol"] == 1
        if high_vol_mask.any():
            vol_vals = features.loc[high_vol_mask, "Vol_21d"]
            interaction_vals = features.loc[high_vol_mask, "Vol_21d_x_Regime"]
            
            np.testing.assert_array_almost_equal(
                vol_vals.dropna().values[:10],  # Check first 10
                interaction_vals.dropna().values[:10]
            )
    
    def test_regime_when_calm(self, sample_data_for_regime):
        """Test regime is 0 when volatility is low (calm)."""
        engineer = FeatureEngineer(sample_data_for_regime, beta_window=100)
        
        features, _ = engineer.create_features_for_stock(
            "TEST.SW_Return",
            include_regime=True
        )
        
        # Where regime=0, interaction should be 0
        calm_mask = features["Regime_HighVol"] == 0
        if calm_mask.any():
            interaction_vals = features.loc[calm_mask, "Vol_21d_x_Regime"]
            
            # All interaction values should be 0 when calm
            assert (interaction_vals.dropna() == 0).all()
    
    def test_backward_compatibility(self, sample_data_for_regime):
        """Test that default behavior still works (include_regime defaults to True)."""
        engineer = FeatureEngineer(sample_data_for_regime, beta_window=100)
        
        # Call without specifying include_regime
        features, target = engineer.create_features_for_stock("TEST.SW_Return")
        
        # Should have regime features by default
        assert "Regime_HighVol" in features.columns
        
        # Target should still be valid (name includes window size)
        assert target.name.startswith("Target_Beta")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
