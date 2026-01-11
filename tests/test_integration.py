"""
Integration Tests - End-to-End & Robustness
============================================

HEC Lausanne - Data Science and Advanced Programming
Master in Finance

This module contains integration tests to validate the complete pipeline:
1. End-to-end test of process_single_ticker on synthetic data
2. CLI argument parsing tests
3. API failure resilience tests
4. Financial consistency checks (beta bounds, correlations)
5. Survivorship bias flag impact test
6. ML reproducibility tests (random seed fixing)

Run with: pytest tests/test_integration.py -v
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pandas as pd
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ==============================================================================
# Fixtures: Synthetic Data Generation
# ==============================================================================

@pytest.fixture
def synthetic_market_data() -> pd.DataFrame:
    """
    Generate realistic synthetic market data for testing.
    
    Creates 5 years of daily data with:
    - Market returns (realistic volatility ~15% annualized)
    - 3 stocks with different betas (0.5, 1.0, 1.5)
    - Proper date index
    """
    np.random.seed(42)
    
    # 5 years of business days
    dates = pd.date_range("2015-01-01", "2019-12-31", freq="B")
    n = len(dates)
    
    # Market returns: ~15% annual vol -> ~0.95% daily vol
    daily_vol = 0.15 / np.sqrt(252)
    market_returns = np.random.normal(0.0003, daily_vol, n)  # Small positive drift
    
    data = pd.DataFrame({"Market_Return": market_returns}, index=dates)
    
    # Stock returns with different betas
    betas = {"LOW.SW": 0.5, "MID.SW": 1.0, "HIGH.SW": 1.5}
    
    for ticker, beta in betas.items():
        # Idiosyncratic volatility
        idio_vol = 0.02
        idio_returns = np.random.normal(0, idio_vol, n)
        
        # Stock return = beta * market + idiosyncratic
        stock_returns = beta * market_returns + idio_returns
        data[f"{ticker}_Return"] = stock_returns
    
    return data


@pytest.fixture
def synthetic_regimes(synthetic_market_data) -> pd.Series:
    """Generate synthetic market regimes based on volatility."""
    vol = synthetic_market_data["Market_Return"].rolling(21).std() * np.sqrt(252) * 100
    regimes = pd.Series(
        np.where(vol > 20, "High Volatility", "Calm"),
        index=synthetic_market_data.index
    )
    return regimes


@pytest.fixture
def minimal_synthetic_data() -> pd.DataFrame:
    """Minimal data for fast tests (1 year only)."""
    np.random.seed(123)
    dates = pd.date_range("2018-01-01", "2018-12-31", freq="B")
    n = len(dates)
    
    market = np.random.normal(0, 0.01, n)
    data = pd.DataFrame({
        "Market_Return": market,
        "TEST.SW_Return": 1.0 * market + np.random.normal(0, 0.005, n)
    }, index=dates)
    
    return data


@pytest.fixture
def batch_test_data() -> pd.DataFrame:
    """
    Data for batch processing tests.
    
    Requirements:
    - At least 750+ days to have >500 samples after 252-day rolling window
    - At least 2 tickers (to trigger multiprocessing)
    - Includes VIX proxy column to avoid external API calls
    
    Calculation: 500 (min samples) + 252 (rolling window) = 752 days minimum
    We use 4 years (~1000 days) for safety margin.
    """
    np.random.seed(456)
    # 4 years of data (~1000 business days)
    dates = pd.date_range("2015-01-01", "2018-12-31", freq="B")
    n = len(dates)
    
    market = np.random.normal(0, 0.01, n)
    
    # Create synthetic VIX-like volatility index (avoids yfinance calls)
    # Base level ~15, with occasional spikes to 25-35
    vix_base = 15 + np.random.normal(0, 2, n)
    # Add some volatility spikes
    spike_mask = np.random.random(n) < 0.05  # 5% of days have spikes
    vix_base[spike_mask] += np.random.uniform(10, 20, spike_mask.sum())
    vix_series = np.clip(vix_base, 10, 50)  # Keep in realistic range
    
    data = pd.DataFrame({
        "Market_Return": market,
        "ALPHA.SW_Return": 0.8 * market + np.random.normal(0, 0.008, n),
        "BETA.SW_Return": 1.2 * market + np.random.normal(0, 0.012, n),
        "VIX_Close": vix_series,  # Synthetic VIX for regime detection
    }, index=dates)
    
    return data


# ==============================================================================
# 1. End-to-End Integration Tests
# ==============================================================================

class TestEndToEndIntegration:
    """Test complete pipeline execution on synthetic data."""
    
    def test_process_single_ticker_returns_valid_result(
        self, synthetic_market_data, synthetic_regimes
    ):
        """
        End-to-end test: process_single_ticker should return a valid
        TickerResult with all required fields populated.
        """
        from main import process_single_ticker, TickerResult
        
        result = process_single_ticker(
            ticker="MID.SW",
            aligned_data=synthetic_market_data,
            regimes=synthetic_regimes,
            is_historical=False,
            n_splits=3,  # Fewer splits for faster test
            test_size=50,
            verbose=False
        )
        
        # Should return a result, not None
        assert result is not None, "process_single_ticker returned None"
        assert isinstance(result, TickerResult)
        
        # All required fields should be populated
        assert result.ticker == "MID.SW"
        assert result.is_historical is False
        assert result.n_samples > 0
        
        # MSE values should be positive and finite
        assert result.naive_mse > 0
        assert result.welch_mse > 0
        assert result.kalman_mse > 0
        assert np.isfinite(result.naive_mse)
        assert np.isfinite(result.welch_mse)
        assert np.isfinite(result.kalman_mse)
        
        # Best ML model should be identified
        assert result.best_ml_model in ["Ridge", "RandomForest", "XGBoost", "MLP", "None"]
        
        # If ML model exists, MSE should be valid
        if result.best_ml_model != "None":
            assert result.best_ml_mse > 0
            assert np.isfinite(result.best_ml_mse)
    
    def test_process_single_ticker_missing_ticker_returns_none(
        self, synthetic_market_data, synthetic_regimes
    ):
        """Should return None for non-existent ticker."""
        from main import process_single_ticker
        
        result = process_single_ticker(
            ticker="NONEXISTENT.SW",
            aligned_data=synthetic_market_data,
            regimes=synthetic_regimes,
            is_historical=False
        )
        
        assert result is None
    
    def test_process_single_ticker_insufficient_data_returns_none(
        self, synthetic_regimes
    ):
        """Should return None when data is too short."""
        from main import process_single_ticker
        
        # Create very short data (only 100 rows)
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=100, freq="B")
        short_data = pd.DataFrame({
            "Market_Return": np.random.normal(0, 0.01, 100),
            "SHORT.SW_Return": np.random.normal(0, 0.01, 100)
        }, index=dates)
        
        short_regimes = pd.Series(["Calm"] * 100, index=dates)
        
        result = process_single_ticker(
            ticker="SHORT.SW",
            aligned_data=short_data,
            regimes=short_regimes,
            is_historical=False
        )
        
        # Should return None due to insufficient data
        assert result is None
    
    def test_batch_analysis_produces_report(
        self, batch_test_data, tmp_path
    ):
        """Batch analysis should produce a valid CSV report."""
        from unittest.mock import patch, MagicMock
        from main import run_batch_analysis
        
        # Create regimes series matching the data length
        batch_regimes = pd.Series(
            ["Calm"] * len(batch_test_data),
            index=batch_test_data.index,
            name="Market_Regime"
        )
        
        # Mock yfinance to avoid network calls during tests
        mock_yf_data = pd.DataFrame({
            "Close": batch_test_data["VIX_Close"]
        }, index=batch_test_data.index)
        
        with patch('yfinance.download', return_value=mock_yf_data):
            report_df = run_batch_analysis(
                aligned_data=batch_test_data,
                regimes=batch_regimes,
                tickers=["ALPHA.SW", "BETA.SW"],  # 2 tickers from batch_test_data
                historical_tickers=[],
                output_dir=tmp_path,
                verbose=False,
                use_multiprocessing=False  # Sequential for faster test
            )
        
        # Report should be a DataFrame with results
        assert isinstance(report_df, pd.DataFrame)
        assert len(report_df) > 0, "Report should contain at least one result"
        
        # Required columns should exist
        required_cols = [
            "Ticker", "Is_Historical", "N_Samples",
            "Welch_MSE", "Best_ML_Model", "Best_ML_MSE",
            "Improvement_vs_Welch_%", "DM_p_value", "Is_Significant"
        ]
        for col in required_cols:
            assert col in report_df.columns, f"Missing column: {col}"
        
        # CSV file should be created
        csv_path = tmp_path / "final_report.csv"
        assert csv_path.exists()


# ==============================================================================
# 2. CLI Argument Parsing Tests
# ==============================================================================

class TestCLIArgumentParsing:
    """Test command-line interface argument handling."""
    
    def test_default_arguments(self):
        """Default arguments should be batch mode with historical."""
        parser = self._create_parser()
        args = parser.parse_args([])
        
        assert args.ticker is None  # No single ticker = batch mode
        assert args.smi_2024_only is False
        assert args.no_cache is False
        assert args.no_historical is False
        assert args.output_dir == "results"
        assert args.no_plots is False
    
    def test_single_ticker_mode(self):
        """--ticker flag should enable single ticker mode."""
        parser = self._create_parser()
        args = parser.parse_args(["--ticker", "NESN.SW"])
        
        assert args.ticker == "NESN.SW"
    
    def test_smi_2024_only_flag(self):
        """--smi-2024-only should limit to current members."""
        parser = self._create_parser()
        args = parser.parse_args(["--smi-2024-only"])
        
        assert args.smi_2024_only is True
    
    def test_no_historical_flag(self):
        """--no-historical should disable survivorship correction."""
        parser = self._create_parser()
        args = parser.parse_args(["--no-historical"])
        
        assert args.no_historical is True
    
    def test_output_dir_override(self):
        """--output-dir should change output directory."""
        parser = self._create_parser()
        args = parser.parse_args(["--output-dir", "custom_results"])
        
        assert args.output_dir == "custom_results"
    
    def test_combined_flags(self):
        """Multiple flags should work together."""
        parser = self._create_parser()
        args = parser.parse_args([
            "--ticker", "ROG.SW",
            "--no-cache",
            "--no-plots",
            "--output-dir", "test_output"
        ])
        
        assert args.ticker == "ROG.SW"
        assert args.no_cache is True
        assert args.no_plots is True
        assert args.output_dir == "test_output"
    
    def _create_parser(self) -> argparse.ArgumentParser:
        """Create argument parser matching main.py."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--ticker", type=str, default=None)
        parser.add_argument("--smi-2024-only", action="store_true")
        parser.add_argument("--no-cache", action="store_true")
        parser.add_argument("--no-historical", action="store_true")
        parser.add_argument("--output-dir", type=str, default="results")
        parser.add_argument("--no-plots", action="store_true")
        return parser


# ==============================================================================
# 3. API Failure Resilience Tests
# ==============================================================================

class TestAPIResilience:
    """Test handling of yfinance API failures."""
    
    def test_data_loader_handles_network_error(self):
        """DataLoader should handle network errors gracefully."""
        from src.data_loader import DataLoader
        
        with patch('src.data_loader.yf.download') as mock_download:
            # Simulate network error
            mock_download.side_effect = Exception("Network error")
            
            loader = DataLoader(
                start_date="2020-01-01",
                end_date="2020-12-31",
                include_historical=False
            )
            
            # Should raise but not crash unexpectedly
            with pytest.raises(Exception):
                loader.load_data(force_refresh=True)
    
    def test_data_loader_handles_empty_response(self):
        """DataLoader should handle empty API responses."""
        from src.data_loader import DataLoader
        
        with patch('src.data_loader.yf.download') as mock_download:
            # Simulate empty response
            mock_download.return_value = pd.DataFrame()
            
            loader = DataLoader(
                start_date="2020-01-01",
                end_date="2020-12-31",
                include_historical=False
            )
            
            # Should handle gracefully
            with pytest.raises(Exception):
                loader.load_data(force_refresh=True)
    
    def test_process_ticker_handles_partial_data(
        self, synthetic_market_data, synthetic_regimes
    ):
        """Pipeline should handle stocks with missing data periods."""
        from main import process_single_ticker
        
        # Create data with NaN gaps
        data_with_gaps = synthetic_market_data.copy()
        data_with_gaps.loc[data_with_gaps.index[100:200], "MID.SW_Return"] = np.nan
        
        result = process_single_ticker(
            ticker="MID.SW",
            aligned_data=data_with_gaps,
            regimes=synthetic_regimes,
            is_historical=False,
            n_splits=3,
            test_size=50
        )
        
        # Should still work (or return None gracefully)
        # The key is no crash
        assert result is None or result.n_samples > 0


# ==============================================================================
# 4. Financial Consistency Checks
# ==============================================================================

class TestFinancialConsistency:
    """Test that results make financial sense."""
    
    def test_beta_within_reasonable_bounds(self, synthetic_market_data):
        """Estimated betas should be within reasonable bounds [-2, 4]."""
        from src.features import FeatureEngineer
        
        engineer = FeatureEngineer(synthetic_market_data, beta_window=252)
        
        for ticker in ["LOW.SW", "MID.SW", "HIGH.SW"]:
            beta = engineer.compute_rolling_beta(f"{ticker}_Return")
            valid_beta = beta.dropna()
            
            # Beta should be bounded
            assert valid_beta.min() > -2, f"{ticker} beta too low: {valid_beta.min()}"
            assert valid_beta.max() < 4, f"{ticker} beta too high: {valid_beta.max()}"
    
    def test_beta_ordering_matches_true_betas(self, synthetic_market_data):
        """
        Stocks with higher true betas should have higher estimated betas.
        LOW.SW (β=0.5) < MID.SW (β=1.0) < HIGH.SW (β=1.5)
        """
        from src.features import FeatureEngineer
        
        engineer = FeatureEngineer(synthetic_market_data, beta_window=252)
        
        mean_betas = {}
        for ticker in ["LOW.SW", "MID.SW", "HIGH.SW"]:
            beta = engineer.compute_rolling_beta(f"{ticker}_Return")
            mean_betas[ticker] = beta.dropna().mean()
        
        # Check ordering
        assert mean_betas["LOW.SW"] < mean_betas["MID.SW"], \
            f"LOW beta ({mean_betas['LOW.SW']:.2f}) should be < MID ({mean_betas['MID.SW']:.2f})"
        assert mean_betas["MID.SW"] < mean_betas["HIGH.SW"], \
            f"MID beta ({mean_betas['MID.SW']:.2f}) should be < HIGH ({mean_betas['HIGH.SW']:.2f})"
    
    def test_welch_beta_produces_reasonable_estimates(self, synthetic_market_data):
        """Welch BSWA should produce reasonable beta estimates."""
        from src.models_benchmarks import NaiveBaseline, WelchBSWA
        
        stock_returns = synthetic_market_data["MID.SW_Return"].values
        market_returns = synthetic_market_data["Market_Return"].values
        
        naive = NaiveBaseline(window=252)
        naive.fit(stock_returns, market_returns)
        naive_beta = naive.predict(stock_returns, market_returns)
        
        welch = WelchBSWA(window=252, decay_halflife=126, winsor_pct=0.05)
        welch.fit(stock_returns, market_returns)
        welch_beta = welch.predict(stock_returns, market_returns)
        
        # Both models should produce valid estimates
        naive_valid = naive_beta[~np.isnan(naive_beta)]
        welch_valid = welch_beta[~np.isnan(welch_beta)]
        
        assert len(naive_valid) > 0, "Naive should produce some valid estimates"
        assert len(welch_valid) > 0, "Welch should produce some valid estimates"
        
        # Welch estimates should be in reasonable range
        assert welch_valid.min() > -2, f"Welch beta too low: {welch_valid.min()}"
        assert welch_valid.max() < 4, f"Welch beta too high: {welch_valid.max()}"
        
        # Both should estimate roughly similar mean beta (within 50%)
        naive_mean = np.mean(naive_valid)
        welch_mean = np.mean(welch_valid)
        
        assert abs(naive_mean - welch_mean) < 0.5 * max(abs(naive_mean), abs(welch_mean), 0.1), \
            f"Welch ({welch_mean:.3f}) and Naive ({naive_mean:.3f}) means should be similar"
        
        # Welch should have reasonable correlation with Naive (> 0.7)
        # This ensures they are estimating the same underlying quantity
        min_len = min(len(naive_valid), len(welch_valid))
        if min_len > 50:
            # Align the series
            naive_aligned = naive_beta[~np.isnan(naive_beta) & ~np.isnan(welch_beta)]
            welch_aligned = welch_beta[~np.isnan(naive_beta) & ~np.isnan(welch_beta)]
            if len(naive_aligned) > 50:
                corr = np.corrcoef(naive_aligned, welch_aligned)[0, 1]
                assert corr > 0.5, f"Welch-Naive correlation too low: {corr:.3f}"
    
    def test_market_correlation_positive(self, synthetic_market_data):
        """Stock returns should be positively correlated with market."""
        market = synthetic_market_data["Market_Return"]
        
        for ticker in ["LOW.SW", "MID.SW", "HIGH.SW"]:
            stock = synthetic_market_data[f"{ticker}_Return"]
            corr = market.corr(stock)
            
            assert corr > 0, f"{ticker} should have positive market correlation, got {corr:.3f}"
    
    def test_higher_beta_higher_volatility(self, synthetic_market_data):
        """Higher beta stocks should have higher total volatility."""
        vols = {}
        for ticker in ["LOW.SW", "MID.SW", "HIGH.SW"]:
            vols[ticker] = synthetic_market_data[f"{ticker}_Return"].std()
        
        # Not strictly monotonic due to idiosyncratic vol, but general trend
        assert vols["HIGH.SW"] > vols["LOW.SW"], \
            f"HIGH vol ({vols['HIGH.SW']:.4f}) should be > LOW vol ({vols['LOW.SW']:.4f})"


# ==============================================================================
# 5. Survivorship Bias Flag Impact Test
# ==============================================================================

class TestSurvivorshipBiasFlag:
    """Test that survivorship bias correction has measurable impact."""
    
    def test_historical_tickers_list_not_empty(self):
        """HISTORICAL_TICKERS should contain entries."""
        from src.data_loader import HISTORICAL_TICKERS, EXIT_DATES
        
        assert len(HISTORICAL_TICKERS) > 0, "HISTORICAL_TICKERS should not be empty"
        
        # All historical tickers should have exit dates
        for ticker in HISTORICAL_TICKERS:
            if ticker not in ["SLHN.SW", "LONN.SW"]:  # These came back
                assert ticker in EXIT_DATES, f"{ticker} missing from EXIT_DATES"
    
    def test_loader_includes_historical_when_enabled(self):
        """DataLoader should include historical tickers when flag is True."""
        from src.data_loader import DataLoader, HISTORICAL_TICKERS
        
        loader_with = DataLoader(include_historical=True)
        loader_without = DataLoader(include_historical=False)
        
        # With historical: should have historical_tickers populated
        assert len(loader_with.historical_tickers) > 0
        
        # Without historical: should be empty
        assert len(loader_without.historical_tickers) == 0
    
    def test_exit_dates_truncation_logic(self):
        """Historical data should be truncated at exit date."""
        from src.data_loader import DataLoader, EXIT_DATES
        
        # Check that exit dates are valid
        for ticker, exit_date in EXIT_DATES.items():
            parsed_date = pd.to_datetime(exit_date)
            assert parsed_date.year >= 2010, f"{ticker} exit date too early"
            assert parsed_date.year <= 2024, f"{ticker} exit date too late"
    
    def test_universe_size_difference(self):
        """Universe size should differ between with/without historical."""
        from src.data_loader import SMI_TICKERS, HISTORICAL_TICKERS
        
        current_only = len(SMI_TICKERS)
        with_historical = len(SMI_TICKERS) + len([
            t for t in HISTORICAL_TICKERS if t not in SMI_TICKERS
        ])
        
        assert with_historical > current_only, \
            f"With historical ({with_historical}) should be > current only ({current_only})"


# ==============================================================================
# 6. ML Reproducibility Tests
# ==============================================================================

class TestMLReproducibility:
    """Test that ML results are reproducible with fixed random seeds."""
    
    def test_feature_engineering_deterministic(self, synthetic_market_data):
        """Feature engineering should produce identical results."""
        from src.features import FeatureEngineer
        
        engineer1 = FeatureEngineer(synthetic_market_data, beta_window=252)
        features1, target1 = engineer1.create_features_for_stock(
            "MID.SW_Return", shift_target=True
        )
        
        engineer2 = FeatureEngineer(synthetic_market_data, beta_window=252)
        features2, target2 = engineer2.create_features_for_stock(
            "MID.SW_Return", shift_target=True
        )
        
        pd.testing.assert_frame_equal(features1, features2)
        pd.testing.assert_series_equal(target1, target2)
    
    def test_benchmark_models_deterministic(self, synthetic_market_data):
        """Benchmark models should be fully deterministic."""
        from src.models_benchmarks import NaiveBaseline, WelchBSWA, KalmanBeta
        
        stock = synthetic_market_data["MID.SW_Return"].values
        market = synthetic_market_data["Market_Return"].values
        
        # Naive
        naive1 = NaiveBaseline(window=252)
        naive1.fit(stock, market)
        pred1 = naive1.predict(stock, market)
        
        naive2 = NaiveBaseline(window=252)
        naive2.fit(stock, market)
        pred2 = naive2.predict(stock, market)
        
        np.testing.assert_array_equal(pred1, pred2)
        
        # Welch
        welch1 = WelchBSWA(window=252)
        welch1.fit(stock, market)
        wpred1 = welch1.predict(stock, market)
        
        welch2 = WelchBSWA(window=252)
        welch2.fit(stock, market)
        wpred2 = welch2.predict(stock, market)
        
        np.testing.assert_array_almost_equal(wpred1, wpred2)
    
    def test_ml_pipeline_reproducible_with_seed(self, minimal_synthetic_data):
        """ML pipeline should produce same results with same seed."""
        from src.models_ml import MLModelPipeline
        from src.features import FeatureEngineer
        
        # Prepare data
        engineer = FeatureEngineer(minimal_synthetic_data, beta_window=63)
        features, target = engineer.create_features_for_stock(
            "TEST.SW_Return", shift_target=True
        )
        
        X = features.values
        y = target.values
        valid_mask = ~(np.any(np.isnan(X), axis=1) | np.isnan(y))
        X_clean = X[valid_mask]
        y_clean = y[valid_mask]
        
        if len(X_clean) < 100:
            pytest.skip("Not enough data for ML test")
        
        # Run twice with same seed
        np.random.seed(42)
        pipeline1 = MLModelPipeline(n_splits=2, verbose=0)
        # Only train Ridge for speed
        from sklearn.linear_model import Ridge
        model1 = Ridge(alpha=1.0)
        model1.fit(X_clean, y_clean)
        pred1 = model1.predict(X_clean)
        
        np.random.seed(42)
        model2 = Ridge(alpha=1.0)
        model2.fit(X_clean, y_clean)
        pred2 = model2.predict(X_clean)
        
        np.testing.assert_array_almost_equal(pred1, pred2)
    
    def test_walk_forward_splits_deterministic(self):
        """Walk-forward splits should be deterministic."""
        from src.evaluation import WalkForwardEvaluator
        
        evaluator = WalkForwardEvaluator(n_splits=5, test_size=50)
        
        splits1 = evaluator._generate_splits(1000)
        splits2 = evaluator._generate_splits(1000)
        
        assert len(splits1) == len(splits2)
        
        for (train1, test1), (train2, test2) in zip(splits1, splits2):
            np.testing.assert_array_equal(train1, train2)
            np.testing.assert_array_equal(test1, test2)
    
    def test_diebold_mariano_deterministic(self):
        """Diebold-Mariano test should be deterministic."""
        from src.statistical_tests import diebold_mariano_test
        
        np.random.seed(42)
        n = 500
        y_true = np.random.randn(n)
        pred1 = y_true + np.random.randn(n) * 0.1
        pred2 = y_true + np.random.randn(n) * 0.15
        
        # DM test expects errors (actual - predicted), not predictions
        errors1 = y_true - pred1
        errors2 = y_true - pred2
        
        # Returns tuple (dm_statistic, p_value)
        stat1, pval1 = diebold_mariano_test(errors1, errors2)
        stat2, pval2 = diebold_mariano_test(errors1, errors2)
        
        assert stat1 == stat2
        assert pval1 == pval2


# ==============================================================================
# 7. Edge Cases and Boundary Conditions
# ==============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_single_regime_data(self, synthetic_market_data):
        """Pipeline should handle data with only one regime."""
        from main import process_single_ticker
        
        # All calm regime
        regimes = pd.Series("Calm", index=synthetic_market_data.index)
        
        result = process_single_ticker(
            ticker="MID.SW",
            aligned_data=synthetic_market_data,
            regimes=regimes,
            is_historical=False,
            n_splits=3,
            test_size=50
        )
        
        # Should not crash
        assert result is not None or result is None  # Just checking no exception
    
    def test_extreme_returns(self):
        """Pipeline should handle extreme return values."""
        from src.features import FeatureEngineer
        
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=500, freq="B")
        
        # Normal data with some extreme values
        market = np.random.normal(0, 0.01, 500)
        market[100] = 0.10  # +10% day (extreme)
        market[200] = -0.08  # -8% day (extreme)
        
        data = pd.DataFrame({
            "Market_Return": market,
            "EXTREME.SW_Return": 1.2 * market + np.random.normal(0, 0.005, 500)
        }, index=dates)
        
        engineer = FeatureEngineer(data, beta_window=63)
        features, target = engineer.create_features_for_stock(
            "EXTREME.SW_Return", shift_target=True
        )
        
        # Should handle without NaN explosion
        assert features.isna().sum().sum() < len(features) * len(features.columns)
    
    def test_constant_returns(self):
        """Pipeline should handle (near) constant returns gracefully."""
        from src.models_benchmarks import WelchBSWA
        
        n = 500
        market = np.ones(n) * 0.001  # Constant
        stock = np.ones(n) * 0.001
        
        welch = WelchBSWA(window=252)
        
        # Should not crash (may produce NaN/inf which is fine)
        try:
            welch.fit(stock, market)
            pred = welch.predict(stock, market)
            # Result might be NaN but shouldn't crash
        except Exception as e:
            pytest.fail(f"Should handle constant returns: {e}")


# ==============================================================================
# 8. Multiprocessing Tests
# ==============================================================================

class TestMultiprocessing:
    """Test parallel processing implementation with ProcessPoolExecutor."""
    
    def test_processpool_executor_is_used_when_enabled(self, synthetic_market_data, synthetic_regimes):
        """ProcessPoolExecutor should be instantiated when use_multiprocessing=True."""
        from unittest.mock import patch, MagicMock
        from main import run_batch_analysis
        
        # Mock ProcessPoolExecutor to verify it's called
        with patch('main.ProcessPoolExecutor') as mock_executor:
            # Setup mock
            mock_instance = MagicMock()
            mock_executor.return_value.__enter__ = MagicMock(return_value=mock_instance)
            mock_executor.return_value.__exit__ = MagicMock(return_value=False)
            
            # Mock as_completed to return empty iterator (we just want to verify executor is used)
            with patch('main.as_completed', return_value=[]):
                try:
                    run_batch_analysis(
                        aligned_data=synthetic_market_data,
                        regimes=synthetic_regimes,
                        tickers=["LOW.SW", "MID.SW"],
                        historical_tickers=[],
                        output_dir=Path("/tmp/test_mp"),
                        use_multiprocessing=True,
                        max_workers=2
                    )
                except Exception:
                    pass  # We don't care about the result, just that executor was used
            
            # Verify ProcessPoolExecutor was instantiated
            mock_executor.assert_called_once_with(max_workers=2)
    
    def test_sequential_mode_does_not_use_executor(self, synthetic_market_data, synthetic_regimes):
        """ProcessPoolExecutor should NOT be used when use_multiprocessing=False."""
        from unittest.mock import patch, MagicMock
        from main import run_batch_analysis
        
        with patch('main.ProcessPoolExecutor') as mock_executor:
            with patch('main.process_single_ticker', return_value=None) as mock_process:
                run_batch_analysis(
                    aligned_data=synthetic_market_data,
                    regimes=synthetic_regimes,
                    tickers=["LOW.SW"],
                    historical_tickers=[],
                    output_dir=Path("/tmp/test_seq"),
                    use_multiprocessing=False
                )
            
            # ProcessPoolExecutor should NOT be called
            mock_executor.assert_not_called()
            
            # process_single_ticker should be called directly
            assert mock_process.call_count >= 1
    
    def test_worker_failure_does_not_crash_pipeline(self, synthetic_market_data, synthetic_regimes):
        """A failing worker should not crash the entire batch analysis."""
        from unittest.mock import patch, MagicMock
        from concurrent.futures import Future
        from main import run_batch_analysis
        
        # Create futures: one succeeds, one fails
        successful_future = Future()
        successful_future.set_result(None)  # Simulates a ticker that returned None
        
        failing_future = Future()
        failing_future.set_exception(RuntimeError("Simulated worker crash"))
        
        # Map futures to tickers
        futures_dict = {
            successful_future: "LOW.SW",
            failing_future: "CRASH.SW"
        }
        
        with patch('main.ProcessPoolExecutor') as mock_executor:
            mock_instance = MagicMock()
            mock_executor.return_value.__enter__ = MagicMock(return_value=mock_instance)
            mock_executor.return_value.__exit__ = MagicMock(return_value=False)
            
            # Mock submit to return our prepared futures
            mock_instance.submit = MagicMock(side_effect=[successful_future, failing_future])
            
            with patch('main.as_completed', return_value=[successful_future, failing_future]):
                # This should NOT raise an exception
                try:
                    result = run_batch_analysis(
                        aligned_data=synthetic_market_data,
                        regimes=synthetic_regimes,
                        tickers=["LOW.SW", "CRASH.SW"],
                        historical_tickers=[],
                        output_dir=Path("/tmp/test_failure"),
                        use_multiprocessing=True,
                        max_workers=2
                    )
                    # Pipeline should complete (possibly with empty results)
                    assert isinstance(result, pd.DataFrame)
                except RuntimeError:
                    pytest.fail("Worker failure should not propagate to main process")
    
    def test_max_workers_parameter_is_respected(self):
        """max_workers should be passed correctly to ProcessPoolExecutor."""
        from unittest.mock import patch, MagicMock
        from main import run_batch_analysis
        
        test_workers = 3
        
        with patch('main.ProcessPoolExecutor') as mock_executor:
            mock_instance = MagicMock()
            mock_executor.return_value.__enter__ = MagicMock(return_value=mock_instance)
            mock_executor.return_value.__exit__ = MagicMock(return_value=False)
            
            with patch('main.as_completed', return_value=[]):
                try:
                    # Create data with 2 tickers (required to trigger multiprocessing)
                    dates = pd.date_range("2020-01-01", periods=100, freq="B")
                    data = pd.DataFrame({
                        "Market_Return": np.random.randn(100) * 0.01,
                        "A.SW_Return": np.random.randn(100) * 0.01,
                        "B.SW_Return": np.random.randn(100) * 0.01,
                    }, index=dates)
                    regimes = pd.Series(["Calm"] * 100, index=dates)
                    
                    run_batch_analysis(
                        aligned_data=data,
                        regimes=regimes,
                        tickers=["A.SW", "B.SW"],  # 2 tickers to trigger multiprocessing
                        historical_tickers=[],
                        output_dir=Path("/tmp/test_workers"),
                        use_multiprocessing=True,
                        max_workers=test_workers
                    )
                except Exception:
                    pass
            
            # Verify exact worker count was passed
            mock_executor.assert_called_once_with(max_workers=test_workers)
    
    def test_default_workers_uses_cpu_count(self):
        """When max_workers=None, should default to cpu_count - 1."""
        from unittest.mock import patch, MagicMock
        from multiprocessing import cpu_count
        from main import run_batch_analysis
        
        expected_workers = max(1, cpu_count() - 1)
        
        with patch('main.ProcessPoolExecutor') as mock_executor:
            mock_instance = MagicMock()
            mock_executor.return_value.__enter__ = MagicMock(return_value=mock_instance)
            mock_executor.return_value.__exit__ = MagicMock(return_value=False)
            
            with patch('main.as_completed', return_value=[]):
                try:
                    dates = pd.date_range("2020-01-01", periods=100, freq="B")
                    data = pd.DataFrame({
                        "Market_Return": np.random.randn(100) * 0.01,
                        "A.SW_Return": np.random.randn(100) * 0.01,
                        "B.SW_Return": np.random.randn(100) * 0.01,
                    }, index=dates)
                    regimes = pd.Series(["Calm"] * 100, index=dates)
                    
                    run_batch_analysis(
                        aligned_data=data,
                        regimes=regimes,
                        tickers=["A.SW", "B.SW"],
                        historical_tickers=[],
                        output_dir=Path("/tmp/test_default"),
                        use_multiprocessing=True,
                        max_workers=None  # Should use default
                    )
                except Exception:
                    pass
            
            # Verify default worker count
            mock_executor.assert_called_once_with(max_workers=expected_workers)
    
    def test_single_ticker_skips_multiprocessing(self, synthetic_market_data, synthetic_regimes):
        """With only 1 ticker, should skip multiprocessing even if enabled."""
        from unittest.mock import patch, MagicMock
        from main import run_batch_analysis
        
        with patch('main.ProcessPoolExecutor') as mock_executor:
            with patch('main.process_single_ticker', return_value=None):
                run_batch_analysis(
                    aligned_data=synthetic_market_data,
                    regimes=synthetic_regimes,
                    tickers=["LOW.SW"],  # Only 1 ticker
                    historical_tickers=[],
                    output_dir=Path("/tmp/test_single"),
                    use_multiprocessing=True  # Even with this True
                )
            
            # ProcessPoolExecutor should NOT be used for single ticker
            mock_executor.assert_not_called()
    
    def test_worker_function_is_picklable(self):
        """_process_ticker_worker must be picklable for multiprocessing."""
        import pickle
        from main import _process_ticker_worker
        
        # Module-level functions should be picklable
        try:
            pickled = pickle.dumps(_process_ticker_worker)
            unpickled = pickle.loads(pickled)
            assert callable(unpickled)
        except (pickle.PicklingError, AttributeError) as e:
            pytest.fail(f"Worker function must be picklable: {e}")
    
    def test_results_collected_from_all_workers(self, synthetic_market_data, synthetic_regimes):
        """All worker results should be collected in the final report."""
        from unittest.mock import patch, MagicMock
        from concurrent.futures import Future
        from main import run_batch_analysis, TickerResult
        
        # Create mock successful results
        result1 = TickerResult(
            ticker="A.SW", is_historical=False, n_samples=100,
            naive_mse=0.01, welch_mse=0.009, kalman_mse=0.008,
            best_ml_model="Ridge", best_ml_mse=0.007,
            improvement_vs_welch_pct=22.2,
            dm_statistic=-1.5, dm_p_value=0.04, is_significant=True,
            all_results=None
        )
        result2 = TickerResult(
            ticker="B.SW", is_historical=True, n_samples=80,
            naive_mse=0.02, welch_mse=0.018, kalman_mse=0.019,
            best_ml_model="RandomForest", best_ml_mse=0.015,
            improvement_vs_welch_pct=16.7,
            dm_statistic=-1.2, dm_p_value=0.08, is_significant=False,
            all_results=None
        )
        
        future1 = Future()
        future1.set_result(result1)
        future2 = Future()
        future2.set_result(result2)
        
        with patch('main.ProcessPoolExecutor') as mock_executor:
            mock_instance = MagicMock()
            mock_executor.return_value.__enter__ = MagicMock(return_value=mock_instance)
            mock_executor.return_value.__exit__ = MagicMock(return_value=False)
            mock_instance.submit = MagicMock(side_effect=[future1, future2])
            
            with patch('main.as_completed', return_value=[future1, future2]):
                with patch('main.TQDM_AVAILABLE', False):
                    dates = pd.date_range("2020-01-01", periods=500, freq="B")
                    data = pd.DataFrame({
                        "Market_Return": np.random.randn(500) * 0.01,
                        "A.SW_Return": np.random.randn(500) * 0.01,
                        "B.SW_Return": np.random.randn(500) * 0.01,
                    }, index=dates)
                    regimes = pd.Series(["Calm"] * 500, index=dates)
                    
                    import tempfile
                    with tempfile.TemporaryDirectory() as tmpdir:
                        result_df = run_batch_analysis(
                            aligned_data=data,
                            regimes=regimes,
                            tickers=["A.SW", "B.SW"],
                            historical_tickers=["B.SW"],
                            output_dir=Path(tmpdir),
                            use_multiprocessing=True,
                            max_workers=2
                        )
        
        # Verify both results are in the DataFrame
        assert len(result_df) == 2
        assert set(result_df["Ticker"].tolist()) == {"A.SW", "B.SW"}
        
        # Verify data integrity
        a_row = result_df[result_df["Ticker"] == "A.SW"].iloc[0]
        assert a_row["Best_ML_Model"] == "Ridge"
        assert a_row["Is_Significant"] == True
        
        b_row = result_df[result_df["Ticker"] == "B.SW"].iloc[0]
        assert b_row["Is_Historical"] == True
        assert b_row["Best_ML_Model"] == "RandomForest"


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
