"""
Tests for Data Loader Module
============================

HEC Lausanne - Data Science and Advanced Programming

Unit tests for data loading, caching, and alignment.
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data_loader import DataLoader, SMI_TICKERS, MARKET_TICKER


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def temp_data_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def loader(temp_data_dir):
    """Create DataLoader with temporary directories."""
    return DataLoader(
        start_date="2020-01-01",
        end_date="2020-12-31",
        tickers=["NESN.SW"],  # Single ticker for faster tests
        raw_data_path=temp_data_dir / "raw",
        processed_data_path=temp_data_dir / "processed",
    )


@pytest.fixture
def sample_stock_data():
    """Generate sample stock data for testing."""
    dates = pd.date_range("2020-01-01", "2020-12-31", freq="B")
    np.random.seed(42)
    
    data = pd.DataFrame({
        "Open": 100 + np.cumsum(np.random.randn(len(dates)) * 0.5),
        "High": 101 + np.cumsum(np.random.randn(len(dates)) * 0.5),
        "Low": 99 + np.cumsum(np.random.randn(len(dates)) * 0.5),
        "Close": 100 + np.cumsum(np.random.randn(len(dates)) * 0.5),
        "Volume": np.random.randint(1000000, 10000000, len(dates)),
    }, index=dates)
    
    return data


@pytest.fixture
def sample_market_data():
    """Generate sample market data for testing."""
    dates = pd.date_range("2020-01-01", "2020-12-31", freq="B")
    np.random.seed(123)
    
    data = pd.DataFrame({
        "Open": 10000 + np.cumsum(np.random.randn(len(dates)) * 50),
        "High": 10050 + np.cumsum(np.random.randn(len(dates)) * 50),
        "Low": 9950 + np.cumsum(np.random.randn(len(dates)) * 50),
        "Close": 10000 + np.cumsum(np.random.randn(len(dates)) * 50),
        "Volume": np.random.randint(10000000, 100000000, len(dates)),
    }, index=dates)
    
    return data


# ==============================================================================
# Test: Initialization
# ==============================================================================

class TestDataLoaderInit:
    """Tests for DataLoader initialization."""

    def test_default_initialization(self):
        """Test default parameter initialization."""
        loader = DataLoader()
        
        assert loader.start_date == "2010-01-01"
        assert loader.end_date == "2024-12-31"
        assert loader.tickers == SMI_TICKERS
        assert loader.market_ticker == MARKET_TICKER

    def test_custom_initialization(self, temp_data_dir):
        """Test custom parameter initialization."""
        custom_tickers = ["NESN.SW", "ROG.SW"]
        
        loader = DataLoader(
            start_date="2015-01-01",
            end_date="2020-12-31",
            tickers=custom_tickers,
            raw_data_path=temp_data_dir / "raw",
        )
        
        assert loader.start_date == "2015-01-01"
        assert loader.end_date == "2020-12-31"
        assert loader.tickers == custom_tickers

    def test_directory_creation(self, temp_data_dir):
        """Test that data directories are created."""
        raw_path = temp_data_dir / "raw"
        processed_path = temp_data_dir / "processed"
        
        loader = DataLoader(
            raw_data_path=raw_path,
            processed_data_path=processed_path,
        )
        
        assert raw_path.exists()
        assert processed_path.exists()


# ==============================================================================
# Test: Caching
# ==============================================================================

class TestCaching:
    """Tests for data caching functionality."""

    def test_cache_filename_generation(self, loader):
        """Test cache filename format."""
        filename = loader._get_cache_filename("NESN.SW")
        
        assert "NESN_SW" in str(filename)
        assert "2020-01-01" in str(filename)
        assert "2020-12-31" in str(filename)
        assert str(filename).endswith(".csv")

    def test_cache_filename_special_characters(self, loader):
        """Test cache filename with special ticker characters."""
        filename = loader._get_cache_filename("^SSMI")
        
        assert "IDX_SSMI" in str(filename)
        assert "^" not in str(filename)

    def test_is_cached_false_for_missing(self, loader):
        """Test cache check returns False for missing files."""
        assert loader._is_cached("NONEXISTENT") is False

    def test_save_and_load_cache(self, loader, sample_stock_data):
        """Test saving and loading from cache."""
        ticker = "TEST.SW"
        
        # Save to cache
        loader._save_to_cache(ticker, sample_stock_data)
        
        # Verify cache exists
        assert loader._is_cached(ticker)
        
        # Load from cache
        loaded_data = loader._load_from_cache(ticker)
        
        # Verify data integrity
        assert len(loaded_data) == len(sample_stock_data)
        assert list(loaded_data.columns) == list(sample_stock_data.columns)


# ==============================================================================
# Test: Return Calculation
# ==============================================================================

class TestReturnsCalculation:
    """Tests for return computation."""

    def test_compute_returns_basic(self, loader, sample_stock_data):
        """Test basic return calculation."""
        returns = loader.compute_returns(sample_stock_data, "Close")
        
        assert len(returns) == len(sample_stock_data)
        assert returns.iloc[0] != returns.iloc[0]  # First value should be NaN
        assert not np.isnan(returns.iloc[1])

    def test_compute_returns_log(self, loader):
        """Test that returns are log returns."""
        dates = pd.date_range("2020-01-01", periods=3, freq="B")
        prices = pd.DataFrame({"Close": [100.0, 110.0, 105.0]}, index=dates)
        
        returns = loader.compute_returns(prices, "Close")
        
        # Log return from 100 to 110
        expected_return = np.log(110.0 / 100.0)
        np.testing.assert_almost_equal(returns.iloc[1], expected_return, decimal=10)


# ==============================================================================
# Test: Data Alignment
# ==============================================================================

class TestDataAlignment:
    """Tests for stock/market data alignment."""

    def test_align_data_basic(self, loader, sample_stock_data, sample_market_data):
        """Test basic data alignment."""
        stock_data = {"NESN.SW": sample_stock_data}
        
        aligned = loader.align_data(stock_data, sample_market_data)
        
        assert "Market_Return" in aligned.columns
        assert "NESN.SW_Return" in aligned.columns
        assert len(aligned) > 0

    def test_align_data_inner_join(self, loader):
        """Test alignment with inner join (explicit)."""
        dates_stock = pd.date_range("2020-01-01", "2020-01-10", freq="B")
        dates_market = pd.date_range("2020-01-03", "2020-01-15", freq="B")
        
        np.random.seed(42)
        stock_data = {
            "TEST.SW": pd.DataFrame({
                "Close": np.random.randn(len(dates_stock)).cumsum() + 100
            }, index=dates_stock)
        }
        market_data = pd.DataFrame({
            "Close": np.random.randn(len(dates_market)).cumsum() + 10000
        }, index=dates_market)
        
        # Use inner join explicitly (default is now outer)
        aligned = loader.align_data(stock_data, market_data, method="inner")
        
        # Should only include overlapping dates
        assert aligned.index.min() >= max(dates_stock.min(), dates_market.min())
        assert aligned.index.max() <= min(dates_stock.max(), dates_market.max())

    def test_align_data_drops_nan(self, loader, sample_stock_data, sample_market_data):
        """Test that alignment drops NaN rows."""
        stock_data = {"NESN.SW": sample_stock_data}
        
        aligned = loader.align_data(stock_data, sample_market_data)
        
        assert not aligned.isnull().any().any()


# ==============================================================================
# Test: Survivorship Bias Disclaimer
# ==============================================================================

class TestSurvivorshipBias:
    """Tests for survivorship bias handling."""

    def test_disclaimer_generation(self):
        """Test that disclaimer is generated."""
        disclaimer = DataLoader.get_survivorship_bias_disclaimer()
        
        assert len(disclaimer) > 100
        assert "survivorship" in disclaimer.lower()
        assert "bias" in disclaimer.lower()

    def test_disclaimer_content(self):
        """Test disclaimer contains required info about survivorship bias."""
        disclaimer = DataLoader.get_survivorship_bias_disclaimer()
        
        # Check key terms are present
        assert "SURVIVORSHIP" in disclaimer.upper()
        assert "bias" in disclaimer.lower()
        assert "historical" in disclaimer.lower()


# ==============================================================================
# Test: Edge Cases
# ==============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_stock_data(self, loader, sample_market_data):
        """Test alignment with no stock data."""
        aligned = loader.align_data({}, sample_market_data)
        
        assert "Market_Return" in aligned.columns
        assert len([c for c in aligned.columns if c != "Market_Return"]) == 0

    def test_single_day_data(self, loader):
        """Test with single day of data."""
        dates = pd.date_range("2020-01-01", periods=2, freq="B")
        
        stock_data = {
            "TEST.SW": pd.DataFrame({"Close": [100.0, 101.0]}, index=dates)
        }
        market_data = pd.DataFrame({"Close": [10000.0, 10010.0]}, index=dates)
        
        aligned = loader.align_data(stock_data, market_data)
        
        # After computing returns and dropping NaN, should have 1 row
        assert len(aligned) == 1


# ==============================================================================
# Test: Historical Data Loading (Survivorship Bias Correction)
# ==============================================================================

from src.data_loader import (
    HISTORICAL_TICKERS, 
    EXIT_DATES, 
    HISTORICAL_CSV_MAPPING
)


class TestHistoricalDataConfiguration:
    """Tests for historical data configuration."""
    
    def test_historical_tickers_not_empty(self):
        """Test historical tickers list is populated."""
        assert len(HISTORICAL_TICKERS) >= 5
    
    def test_exit_dates_match_historical_tickers(self):
        """Test that all historical tickers have exit dates."""
        for ticker in HISTORICAL_TICKERS:
            assert ticker in EXIT_DATES, f"Missing exit date for {ticker}"
    
    def test_exit_dates_format(self):
        """Test exit dates are valid date strings."""
        for ticker, date_str in EXIT_DATES.items():
            # Should be parseable as date
            date = pd.Timestamp(date_str)
            assert date.year >= 2010
            assert date.year <= 2025
    
    def test_csv_mapping_exists(self):
        """Test CSV mapping is defined for all historical tickers."""
        for ticker in HISTORICAL_TICKERS:
            assert ticker in HISTORICAL_CSV_MAPPING, f"Missing CSV mapping for {ticker}"
    
    def test_exit_dates_chronological_order(self):
        """Test exit dates are in expected chronological range."""
        dates = [pd.Timestamp(d) for d in EXIT_DATES.values()]
        min_date = min(dates)
        max_date = max(dates)
        
        assert min_date.year >= 2010
        assert max_date.year <= 2024


class TestHistoricalDataLoading:
    """Tests for historical data loading functionality."""
    
    @pytest.fixture
    def temp_historical_dir(self, temp_data_dir):
        """Create temporary historical data directory."""
        historical_dir = temp_data_dir / "historical"
        historical_dir.mkdir(parents=True, exist_ok=True)
        return historical_dir
    
    @pytest.fixture
    def sample_historical_csv(self, temp_historical_dir):
        """Create sample historical CSV file."""
        dates = pd.date_range("2018-01-01", "2020-12-31", freq="B")
        np.random.seed(42)
        
        data = pd.DataFrame({
            "Open": 50 + np.cumsum(np.random.randn(len(dates)) * 0.3),
            "High": 51 + np.cumsum(np.random.randn(len(dates)) * 0.3),
            "Low": 49 + np.cumsum(np.random.randn(len(dates)) * 0.3),
            "Close": 50 + np.cumsum(np.random.randn(len(dates)) * 0.3),
            "Volume": np.random.randint(500000, 5000000, len(dates)),
        }, index=dates)
        
        csv_path = temp_historical_dir / "TEST_HIST.csv"
        data.to_csv(csv_path)
        return csv_path, data
    
    def test_historical_csv_loading(self, temp_data_dir, sample_historical_csv):
        """Test loading data from historical CSV."""
        csv_path, expected_data = sample_historical_csv
        
        loader = DataLoader(
            start_date="2018-01-01",
            end_date="2020-12-31",
            tickers=[],  # No current tickers
            historical_tickers=["TEST_HIST.SW"],
            include_historical=True,
            raw_data_path=temp_data_dir / "raw",
            historical_data_path=temp_data_dir / "historical",
        )
        
        # Manually add mapping for test ticker
        from src import data_loader
        original_mapping = data_loader.HISTORICAL_CSV_MAPPING.copy()
        data_loader.HISTORICAL_CSV_MAPPING["TEST_HIST.SW"] = "TEST_HIST"
        
        try:
            historical_data = loader.load_historical_data()
            
            assert "TEST_HIST.SW" in historical_data
            assert len(historical_data["TEST_HIST.SW"]) > 0
        finally:
            # Restore original mapping
            data_loader.HISTORICAL_CSV_MAPPING.clear()
            data_loader.HISTORICAL_CSV_MAPPING.update(original_mapping)
    
    def test_exit_date_truncation(self, temp_data_dir, temp_historical_dir):
        """Test that data is truncated at exit date."""
        # Create CSV with data extending past exit date
        dates = pd.date_range("2018-01-01", "2022-12-31", freq="B")
        np.random.seed(42)
        
        data = pd.DataFrame({
            "Close": 50 + np.cumsum(np.random.randn(len(dates)) * 0.3),
        }, index=dates)
        
        csv_path = temp_historical_dir / "TRUNC_TEST.csv"
        data.to_csv(csv_path)
        
        # Set exit date in the middle of the data
        from src import data_loader
        original_exits = data_loader.EXIT_DATES.copy()
        original_mapping = data_loader.HISTORICAL_CSV_MAPPING.copy()
        
        data_loader.EXIT_DATES["TRUNC_TEST.SW"] = "2020-06-15"
        data_loader.HISTORICAL_CSV_MAPPING["TRUNC_TEST.SW"] = "TRUNC_TEST"
        
        try:
            loader = DataLoader(
                start_date="2018-01-01",
                end_date="2022-12-31",
                tickers=[],
                historical_tickers=["TRUNC_TEST.SW"],
                include_historical=True,
                raw_data_path=temp_data_dir / "raw",
                historical_data_path=temp_historical_dir,
            )
            
            historical_data = loader.load_historical_data()
            
            assert "TRUNC_TEST.SW" in historical_data
            loaded_data = historical_data["TRUNC_TEST.SW"]
            
            # Data should be truncated at exit date
            assert loaded_data.index.max() <= pd.Timestamp("2020-06-15")
            
        finally:
            data_loader.EXIT_DATES.clear()
            data_loader.EXIT_DATES.update(original_exits)
            data_loader.HISTORICAL_CSV_MAPPING.clear()
            data_loader.HISTORICAL_CSV_MAPPING.update(original_mapping)
    
    def test_include_historical_false(self, temp_data_dir):
        """Test that historical data is not loaded when disabled."""
        loader = DataLoader(
            start_date="2020-01-01",
            end_date="2020-12-31",
            tickers=[],
            include_historical=False,
            raw_data_path=temp_data_dir / "raw",
        )
        
        historical_data = loader.load_historical_data()
        
        assert len(historical_data) == 0
    
    def test_missing_csv_handled_gracefully(self, temp_data_dir):
        """Test that missing CSV files are handled without error."""
        loader = DataLoader(
            start_date="2020-01-01",
            end_date="2020-12-31",
            tickers=[],
            historical_tickers=["NONEXISTENT.SW"],
            include_historical=True,
            raw_data_path=temp_data_dir / "raw",
            historical_data_path=temp_data_dir / "historical",
        )
        
        # Should not raise error
        historical_data = loader.load_historical_data()
        
        # Should return empty dict for missing file
        assert "NONEXISTENT.SW" not in historical_data
    
    def test_historical_members_info(self, temp_data_dir, temp_historical_dir):
        """Test get_historical_members_info method."""
        # Create one CSV file
        csv_path = temp_historical_dir / "CSGN.csv"
        pd.DataFrame({"Close": [100]}, index=[pd.Timestamp("2020-01-01")]).to_csv(csv_path)
        
        loader = DataLoader(
            include_historical=True,
            historical_data_path=temp_historical_dir,
        )
        
        info = loader.get_historical_members_info()
        
        assert len(info) > 0
        assert "ticker" in info.columns
        assert "exit_date" in info.columns
        assert "csv_available" in info.columns


class TestOuterJoinAlignment:
    """Tests for outer join alignment with discontinuous series."""
    
    def test_outer_join_preserves_market_dates(self, temp_data_dir):
        """Test that outer join keeps all market dates."""
        loader = DataLoader(
            raw_data_path=temp_data_dir / "raw",
        )
        
        # Market has 10 days
        market_dates = pd.date_range("2020-01-01", periods=10, freq="B")
        market_data = pd.DataFrame({
            "Close": [10000 + i * 10 for i in range(10)]
        }, index=market_dates)
        
        # Stock only has 5 days (exits mid-period)
        stock_dates = market_dates[:5]
        stock_data = {
            "EXIT.SW": pd.DataFrame({
                "Close": [100 + i for i in range(5)]
            }, index=stock_dates)
        }
        
        aligned = loader.align_data(stock_data, market_data, method="outer")
        
        # Should have all market dates minus 1 (first day dropped for return calc)
        assert len(aligned) == 9
        
        # Stock has 5 price days -> 4 valid returns (days 2-5)
        # Aligned has 9 rows, so 9 - 4 = 5 NaN values for stock
        assert aligned["EXIT.SW_Return"].isna().sum() == 5
        
        # Market should have no NaN
        assert aligned["Market_Return"].isna().sum() == 0
    
    def test_discontinuous_series_handling(self, temp_data_dir):
        """Test handling of multiple stocks with different date ranges."""
        loader = DataLoader(
            raw_data_path=temp_data_dir / "raw",
        )
        
        # Market: full period
        market_dates = pd.date_range("2020-01-01", periods=20, freq="B")
        market_data = pd.DataFrame({
            "Close": [10000 + i * 10 for i in range(20)]
        }, index=market_dates)
        
        # Stock A: first half
        stock_a_data = pd.DataFrame({
            "Close": [100 + i for i in range(10)]
        }, index=market_dates[:10])
        
        # Stock B: second half
        stock_b_data = pd.DataFrame({
            "Close": [200 + i for i in range(10)]
        }, index=market_dates[10:])
        
        stock_data = {
            "A.SW": stock_a_data,
            "B.SW": stock_b_data,
        }
        
        aligned = loader.align_data(stock_data, market_data, method="outer")
        
        # Market is complete
        assert aligned["Market_Return"].isna().sum() == 0
        
        # Stock A has data for first half
        assert aligned["A.SW_Return"][:9].notna().all()
        
        # Stock B has data for second half
        assert aligned["B.SW_Return"][-9:].notna().all()


# ==============================================================================
# Test: Constants
# ==============================================================================

class TestConstants:
    """Tests for module constants."""

    def test_smi_tickers_not_empty(self):
        """Test SMI tickers list is populated."""
        assert len(SMI_TICKERS) >= 10

    def test_smi_tickers_format(self):
        """Test SMI tickers have correct format."""
        for ticker in SMI_TICKERS:
            assert ticker.endswith(".SW")

    def test_market_ticker(self):
        """Test market ticker is set."""
        assert MARKET_TICKER == "^SSMI"
    
    def test_smi_tickers_count(self):
        """Test SMI has exactly 20 tickers."""
        assert len(SMI_TICKERS) == 20
    
    def test_historical_tickers_count(self):
        """Test historical tickers list has expected count."""
        assert len(HISTORICAL_TICKERS) == 11


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
