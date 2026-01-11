"""
Data Loader Module
==================

HEC Lausanne - Data Science and Advanced Programming
Master in Finance

This module handles data loading for the SMI beta prediction project.
We fetch price data from Yahoo Finance for current SMI members, and we also
load historical CSV files for companies that left the index (to correct
survivorship bias).

Main features:
- Download OHLCV data via yfinance
- Cache data locally to avoid repeated API calls
- Load historical members from CSV with exit date truncation
- Align stock and market returns
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# yfinance might not be installed everywhere
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ==============================================================================
# Constants
# ==============================================================================

# Current SMI members (20 stocks, December 2024)
# These are fetched from Yahoo Finance
SMI_TICKERS: List[str] = [
    # Top 12 blue chips
    "NESN.SW",   # Nestlé
    "ROG.SW",    # Roche Holding AG
    "NOVN.SW",   # Novartis AG
    "UBSG.SW",   # UBS Group AG
    "ZURN.SW",   # Zurich Insurance Group AG
    "CFR.SW",    # Compagnie Financière Richemont SA
    "ABBN.SW",   # ABB Ltd
    "SIKA.SW",   # Sika AG
    "LONN.SW",   # Lonza Group AG
    "GIVN.SW",   # Givaudan SA
    "ALC.SW",    # Alcon AG
    "HOLN.SW",   # Holcim AG
    
    # Other 8 constituents
    "GEBN.SW",   # Geberit AG
    "SLHN.SW",   # Swiss Life Holding AG
    "SCMN.SW",   # Swisscom AG
    "KNIN.SW",   # Kühne + Nagel International AG
    "PGHN.SW",   # Partners Group Holding AG
    "SREN.SW",   # Swiss Re AG
    "LOGN.SW",   # Logitech International SA (added Dec 2024)
    "SOON.SW",   # Sonova Holding AG (added Dec 2024)
]
# Total: 20 tickers


# ==============================================================================
# Historical Members (for survivorship bias correction)
# ==============================================================================

# Former SMI members (delisted or acquired)
# We load these from local CSV files to avoid survivorship bias
HISTORICAL_TICKERS: List[str] = [
    "CSGN.SW",   # Credit Suisse (acquired by UBS 2023)
    "SYST.SW",   # Synthes (acquired by J&J 2012)
    "ATLN.SW",   # Actelion (acquired by J&J 2017)
    "SYNN.SW",   # Syngenta (acquired by ChemChina 2017)
    "RIGN.SW",   # Transocean (removed 2016)
    "BAER.SW",   # Julius Bär (removed 2019)
    "UHR.SW",    # Swatch Group (removed 2021)
    "ADEN.SW",   # Adecco (removed 2020)
    "SGSN.SW",   # SGS (removed 2022)
    "SLHN.SW",   # Swiss Life (left then came back)
    "LONN.SW",   # Lonza (left then came back)
]

# Exit dates: we cut the data at this date to avoid look-ahead bias
# Format: YYYY-MM-DD (last day in the index)
EXIT_DATES: Dict[str, str] = {
    "SLHN.SW": "2010-06-21",   # Swiss Life
    "LONN.SW": "2011-09-19",   # Lonza
    "SYST.SW": "2012-06-18",   # Synthes (acquired by J&J)
    "RIGN.SW": "2016-03-21",   # Transocean
    "ATLN.SW": "2017-05-03",   # Actelion (acquired by J&J)
    "SYNN.SW": "2017-05-15",   # Syngenta (acquired by ChemChina)
    "BAER.SW": "2019-04-10",   # Julius Bär
    "ADEN.SW": "2020-09-21",   # Adecco
    "UHR.SW":  "2021-09-20",   # Swatch
    "SGSN.SW": "2022-09-19",   # SGS
    "CSGN.SW": "2023-06-13",   # Credit Suisse (acquired by UBS)
}

# Map ticker to CSV filename (files are in data/historical/)
HISTORICAL_CSV_MAPPING: Dict[str, str] = {
    "CSGN.SW": "CSGN",
    "SYST.SW": "SYST",
    "ATLN.SW": "ATLN",
    "SYNN.SW": "SYNN",
    "RIGN.SW": "RIGN",
    "BAER.SW": "BAER",
    "UHR.SW":  "UHR",
    "ADEN.SW": "ADEN",
    "SGSN.SW": "SGSN",
    "SLHN.SW": "SLHN",
    "LONN.SW": "LONN",
}

# Market benchmark
MARKET_TICKER: str = "^SSMI"  # Swiss Market Index

# Default paths for data storage
DEFAULT_RAW_PATH: Path = Path("data/raw")
DEFAULT_PROCESSED_PATH: Path = Path("data/processed")
DEFAULT_HISTORICAL_PATH: Path = Path("data/historical")


# ==============================================================================
# DataLoader Class
# ==============================================================================

class DataLoader:
    """
    Loads and manages data for SMI beta analysis.

    This class uses a hybrid approach:
    1. Current members: downloaded from Yahoo Finance
    2. Historical members: loaded from CSV files, truncated at exit date

    We do this to correct survivorship bias (companies that left the index
    are often ignored in backtests, which gives overly optimistic results).

    Attributes:
        start_date: Start of data period (YYYY-MM-DD)
        end_date: End of data period (YYYY-MM-DD)
        tickers: Current SMI members to fetch
        historical_tickers: Former members to load from CSV
        market_ticker: Benchmark index
        include_historical: If True, load historical members too
        raw_data_path: Where to cache downloaded data
        historical_data_path: Where to find historical CSV files

    Example:
        >>> loader = DataLoader(
        ...     start_date='2010-01-01',
        ...     end_date='2024-12-31',
        ...     include_historical=True
        ... )
        >>> stock_data, market_data = loader.load_data()
        >>> aligned = loader.align_data(stock_data, market_data)
    """

    def __init__(
        self,
        start_date: str = "2010-01-01",
        end_date: str = "2024-12-31",
        tickers: Optional[List[str]] = None,
        historical_tickers: Optional[List[str]] = None,
        market_ticker: str = MARKET_TICKER,
        include_historical: bool = True,
        raw_data_path: Optional[Path] = None,
        processed_data_path: Optional[Path] = None,
        historical_data_path: Optional[Path] = None,
    ) -> None:
        """
        Set up the loader with date range and tickers.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            tickers: Current SMI tickers, defaults to SMI_TICKERS
            historical_tickers: Former members, defaults to HISTORICAL_TICKERS
            market_ticker: Index ticker, defaults to ^SSMI
            include_historical: Load historical members from CSV
            raw_data_path: Where to cache data
            processed_data_path: Where to save processed data
            historical_data_path: Where historical CSV files are
        """
        self.start_date = start_date
        self.end_date = end_date
        self.tickers = tickers if tickers is not None else SMI_TICKERS.copy()
        self.market_ticker = market_ticker
        self.include_historical = include_historical
        
        # Only populate historical_tickers if include_historical is True
        if include_historical:
            self.historical_tickers = historical_tickers if historical_tickers is not None else HISTORICAL_TICKERS.copy()
        else:
            self.historical_tickers = []
        
        self.raw_data_path = Path(raw_data_path) if raw_data_path else DEFAULT_RAW_PATH
        self.processed_data_path = Path(processed_data_path) if processed_data_path else DEFAULT_PROCESSED_PATH
        self.historical_data_path = Path(historical_data_path) if historical_data_path else DEFAULT_HISTORICAL_PATH

        # Create directories if needed
        self._create_directories()
        
        # Keep track of which historical members we loaded
        self._loaded_historical: Dict[str, pd.DataFrame] = {}

    def _create_directories(self) -> None:
        """Create data directories if they don't exist."""
        self.raw_data_path.mkdir(parents=True, exist_ok=True)
        self.processed_data_path.mkdir(parents=True, exist_ok=True)
        self.historical_data_path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Ensured directories exist: {self.raw_data_path}, {self.historical_data_path}")

    # ==========================================================================
    # Caching Methods
    # ==========================================================================

    def _get_cache_filename(self, ticker: str) -> Path:
        """
        Build the cache filename for a ticker.
        We replace special chars to make it a valid filename.
        """
        safe_ticker = ticker.replace("^", "IDX_").replace(".", "_")
        filename = f"{safe_ticker}_{self.start_date}_{self.end_date}.csv"
        return self.raw_data_path / filename

    def _is_cached(self, ticker: str) -> bool:
        """Check if we already have cached data for this ticker."""
        cache_file = self._get_cache_filename(ticker)
        if cache_file.exists():
            if cache_file.stat().st_size > 0:
                logger.debug(f"Cache hit for {ticker}")
                return True
        return False

    def _load_from_cache(self, ticker: str) -> pd.DataFrame:
        """Load data from the local cache file."""
        cache_file = self._get_cache_filename(ticker)
        logger.debug(f"Loading {ticker} from cache: {cache_file}")
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        return df

    def _save_to_cache(self, ticker: str, data: pd.DataFrame) -> None:
        """Save data to cache so we don't download it again."""
        cache_file = self._get_cache_filename(ticker)
        data.to_csv(cache_file)
        logger.debug(f"Cached {ticker} to: {cache_file}")

    # ==========================================================================
    # Yahoo Finance Methods (Current Members)
    # ==========================================================================

    def _fetch_from_yfinance(self, ticker: str) -> pd.DataFrame:
        """
        Fetch OHLCV data from Yahoo Finance.

        Args:
            ticker: Stock or index ticker symbol.

        Returns:
            DataFrame with OHLCV data.

        Raises:
            ValueError: If no data found for this ticker
            RuntimeError: If yfinance is not installed
        """
        if not YFINANCE_AVAILABLE:
            raise RuntimeError(
                "yfinance is not installed. Install with: pip install yfinance"
            )
        
        logger.info(f"Fetching {ticker} from Yahoo Finance...")
        try:
            data = yf.download(
                ticker,
                start=self.start_date,
                end=self.end_date,
                progress=False,
                auto_adjust=True,  # Adjust for splits/dividends
            )
            if data.empty:
                raise ValueError(f"No data retrieved for ticker: {ticker}")
            
            # yfinance sometimes returns MultiIndex columns, we flatten them
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.droplevel(1)
            
            return data
        except Exception as e:
            logger.error(f"Failed to fetch {ticker}: {e}")
            raise

    def fetch_single_ticker(self, ticker: str, force_refresh: bool = False) -> pd.DataFrame:
        """
        Fetch data for one ticker, using cache if available.
        If force_refresh is True, we download again even if cached.
        """
        if not force_refresh and self._is_cached(ticker):
            return self._load_from_cache(ticker)
        
        data = self._fetch_from_yfinance(ticker)
        self._save_to_cache(ticker, data)
        return data

    # ==========================================================================
    # Historical CSV Methods (for former SMI members)
    # ==========================================================================

    def _get_historical_csv_path(self, ticker: str) -> Path:
        """Get the path to the CSV file for a historical ticker."""
        csv_name = HISTORICAL_CSV_MAPPING.get(ticker, ticker.replace(".SW", ""))
        return self.historical_data_path / f"{csv_name}.csv"

    def _load_historical_csv(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        Load data from a local CSV file for a former SMI member.
        
        The CSV should have Yahoo Finance format: Date index, OHLCV columns.
        Returns None if file not found.
        """
        csv_path = self._get_historical_csv_path(ticker)
        
        if not csv_path.exists():
            logger.warning(f"Historical CSV not found for {ticker}: {csv_path}")
            return None
        
        logger.info(f"Loading historical data for {ticker} from: {csv_path}")
        
        try:
            # Load CSV (parse_dates=True handles datetime inference automatically)
            df = pd.read_csv(
                csv_path,
                index_col=0,
                parse_dates=True
            )
            
            # Make sure index is datetime
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            
            # Use Adj Close as the Close price (for proper returns calculation)
            # First, drop the original Close column if Adj Close exists
            if 'Adj Close' in df.columns:
                if 'Close' in df.columns:
                    df = df.drop(columns=['Close'])
                df = df.rename(columns={'Adj Close': 'Close'})
            elif 'Adj. Close' in df.columns:
                if 'Close' in df.columns:
                    df = df.drop(columns=['Close'])
                df = df.rename(columns={'Adj. Close': 'Close'})
            elif 'adj_close' in df.columns:
                if 'Close' in df.columns:
                    df = df.drop(columns=['Close'])
                df = df.rename(columns={'adj_close': 'Close'})
            
            # Ensure required columns exist
            if 'Close' not in df.columns:
                logger.error(f"Missing 'Close' column in {csv_path}")
                return None
            
            # Sort by date
            df = df.sort_index()
            
            logger.info(f"Loaded {ticker}: {len(df)} rows ({df.index.min().date()} to {df.index.max().date()})")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load historical CSV for {ticker}: {e}")
            return None

    def _truncate_at_exit_date(self, ticker: str, data: pd.DataFrame) -> pd.DataFrame:
        """
        Truncate historical data at the ticker's exit date from the index.

        This prevents look-ahead bias by not including data after the company
        left the SMI index.

        Args:
            ticker: Ticker symbol.
            data: DataFrame with OHLCV data.

        Returns:
            DataFrame truncated at exit date.
        """
        if ticker not in EXIT_DATES:
            return data
        
        exit_date = pd.Timestamp(EXIT_DATES[ticker])
        original_end = data.index.max()
        
        # Keep only data up to and including exit date
        truncated = data[data.index <= exit_date]
        
        if len(truncated) < len(data):
            logger.info(
                f"Truncated {ticker} at exit date {exit_date.date()} "
                f"(removed {len(data) - len(truncated)} rows after exit)"
            )
        
        return truncated

    def _filter_by_date_range(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Filter data to the configured date range.

        Args:
            data: DataFrame with DatetimeIndex.

        Returns:
            Filtered DataFrame.
        """
        start = pd.Timestamp(self.start_date)
        end = pd.Timestamp(self.end_date)
        return data[(data.index >= start) & (data.index <= end)]

    def load_historical_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load all historical member data from CSV files.

        Applies:
        1. Date range filtering
        2. Exit date truncation (prevents look-ahead bias)

        Returns:
            Dictionary mapping ticker symbols to DataFrames.
        """
        historical_data: Dict[str, pd.DataFrame] = {}
        
        if not self.include_historical:
            logger.info("Historical data loading disabled")
            return historical_data
        
        logger.info(f"Loading {len(self.historical_tickers)} historical tickers...")
        
        for ticker in self.historical_tickers:
            data = self._load_historical_csv(ticker)
            
            if data is None:
                continue
            
            # Apply exit date truncation FIRST (before date range filter)
            data = self._truncate_at_exit_date(ticker, data)
            
            # Then filter to configured date range
            data = self._filter_by_date_range(data)
            
            if data.empty:
                logger.warning(f"No data for {ticker} in date range {self.start_date} to {self.end_date}")
                continue
            
            historical_data[ticker] = data
            self._loaded_historical[ticker] = data
        
        logger.info(f"Loaded {len(historical_data)} historical tickers")
        return historical_data

    # ==========================================================================
    # Main Loading Methods
    # ==========================================================================

    def load_stock_data(self, force_refresh: bool = False) -> Dict[str, pd.DataFrame]:
        """
        Load OHLCV data for all configured stock tickers (current + historical).

        Args:
            force_refresh: If True, bypass cache and re-download current members.

        Returns:
            Dictionary mapping ticker symbols to DataFrames.
        """
        stock_data: Dict[str, pd.DataFrame] = {}
        
        # Load current members via Yahoo Finance
        logger.info(f"Loading {len(self.tickers)} current SMI members...")
        for ticker in self.tickers:
            try:
                stock_data[ticker] = self.fetch_single_ticker(ticker, force_refresh)
                logger.debug(f"Loaded {ticker}: {len(stock_data[ticker])} rows")
            except Exception as e:
                logger.warning(f"Skipping {ticker} due to error: {e}")
        
        # Load historical members from CSV files
        if self.include_historical:
            historical_data = self.load_historical_data()
            
            # Merge historical data (historical takes precedence for overlapping tickers)
            for ticker, data in historical_data.items():
                if ticker in stock_data:
                    # Handle tickers that are both current AND have historical data
                    # (e.g., SLHN.SW, LONN.SW which left and re-entered)
                    # Combine historical + current data
                    current_data = stock_data[ticker]
                    exit_date = pd.Timestamp(EXIT_DATES.get(ticker, self.end_date))
                    
                    # Historical: up to exit date
                    # Current: after exit date (if re-entered)
                    combined = pd.concat([
                        data,
                        current_data[current_data.index > exit_date]
                    ]).sort_index()
                    
                    # Remove duplicates (keep first)
                    combined = combined[~combined.index.duplicated(keep='first')]
                    stock_data[ticker] = combined
                    
                    logger.info(f"Combined historical + current data for {ticker}")
                else:
                    stock_data[ticker] = data
        
        logger.info(f"Total stocks loaded: {len(stock_data)}")
        return stock_data

    def load_market_data(self, force_refresh: bool = False) -> pd.DataFrame:
        """
        Load OHLCV data for the market benchmark.

        Args:
            force_refresh: If True, bypass cache and re-download.

        Returns:
            DataFrame with market index OHLCV data.
        """
        return self.fetch_single_ticker(self.market_ticker, force_refresh)

    def load_data(
        self, force_refresh: bool = False
    ) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
        """
        Load all stock and market data.

        Args:
            force_refresh: If True, bypass cache and re-download all.

        Returns:
            Tuple of (stock_data_dict, market_data_df).
        """
        stock_data = self.load_stock_data(force_refresh)
        market_data = self.load_market_data(force_refresh)
        return stock_data, market_data

    # ==========================================================================
    # Data Processing Methods
    # ==========================================================================

    def compute_returns(self, prices: pd.DataFrame, column: str = "Close") -> pd.Series:
        """
        Compute daily log returns from price series.

        Args:
            prices: DataFrame with OHLCV data.
            column: Column name to use for return calculation.

        Returns:
            Series of log returns.
        """
        return np.log(prices[column] / prices[column].shift(1))

    def align_data(
        self,
        stock_data: Dict[str, pd.DataFrame],
        market_data: pd.DataFrame,
        method: str = "outer",
    ) -> pd.DataFrame:
        """
        Align stock returns with market returns.

        This method handles discontinuous series (stocks that exit mid-period)
        by using an OUTER join to preserve the full market history, then
        filling stock data with NaN where not available.

        Args:
            stock_data: Dictionary of stock DataFrames.
            market_data: Market index DataFrame.
            method: Join method:
                - "outer": Keep all market dates, NaN for missing stocks (recommended)
                - "inner": Keep only dates where ALL stocks have data (loses history)

        Returns:
            DataFrame with aligned returns for all assets and the market.
        """
        # Compute market returns
        market_returns = self.compute_returns(market_data, "Close")
        market_returns.name = "Market_Return"
        
        # Start with market returns as base
        aligned = pd.DataFrame(market_returns)
        
        # Track statistics
        stocks_added = 0
        stocks_skipped = 0
        
        # Add each stock's returns
        for ticker, data in stock_data.items():
            try:
                stock_returns = self.compute_returns(data, "Close")
                stock_returns.name = f"{ticker}_Return"
                
                # Use outer join to preserve market history
                if method == "outer":
                    aligned = aligned.join(stock_returns, how="left")
                else:
                    aligned = aligned.join(stock_returns, how="inner")
                
                stocks_added += 1
                
            except Exception as e:
                logger.warning(f"Could not compute returns for {ticker}: {e}")
                stocks_skipped += 1
        
        # For outer join: drop ONLY the first row (NaN from returns calculation on market)
        # Keep NaN values for stocks that don't have data on certain dates
        if method == "outer":
            # Only drop rows where Market_Return is NaN
            aligned = aligned.dropna(subset=["Market_Return"])
        else:
            # For inner join: drop any row with NaN
            aligned = aligned.dropna()
        
        logger.info(
            f"Aligned data: {len(aligned)} trading days, "
            f"{len(aligned.columns)} columns "
            f"({stocks_added} stocks added, {stocks_skipped} skipped)"
        )
        
        # Log coverage statistics for historical members
        if self.include_historical:
            self._log_coverage_statistics(aligned)
        
        return aligned

    def _log_coverage_statistics(self, aligned: pd.DataFrame) -> None:
        """
        Log data coverage statistics for historical members.

        Args:
            aligned: Aligned returns DataFrame.
        """
        logger.info("\n" + "=" * 60)
        logger.info("HISTORICAL MEMBER COVERAGE")
        logger.info("=" * 60)
        
        for ticker in self.historical_tickers:
            col = f"{ticker}_Return"
            if col not in aligned.columns:
                continue
            
            series = aligned[col]
            valid_count = series.notna().sum()
            total_count = len(series)
            coverage_pct = (valid_count / total_count) * 100
            
            first_date = series.first_valid_index()
            last_date = series.last_valid_index()
            
            exit_date = EXIT_DATES.get(ticker, "N/A")
            
            logger.info(
                f"  {ticker:10s}: {valid_count:4d}/{total_count} days "
                f"({coverage_pct:5.1f}%) | "
                f"{first_date.date() if first_date else 'N/A'} to "
                f"{last_date.date() if last_date else 'N/A'} | "
                f"Exit: {exit_date}"
            )

    def save_processed_data(self, data: pd.DataFrame, filename: str) -> Path:
        """
        Save processed data to the processed data directory.

        Args:
            data: DataFrame to save.
            filename: Output filename (without path).

        Returns:
            Path to saved file.
        """
        filepath = self.processed_data_path / filename
        data.to_csv(filepath)
        logger.info(f"Saved processed data to: {filepath}")
        return filepath

    # ==========================================================================
    # Information Methods
    # ==========================================================================

    def get_historical_members_info(self) -> pd.DataFrame:
        """
        Get information about historical members and their exit dates.

        Returns:
            DataFrame with ticker, exit_date, and csv_available columns.
        """
        info = []
        for ticker in HISTORICAL_TICKERS:
            csv_path = self._get_historical_csv_path(ticker)
            info.append({
                "ticker": ticker,
                "exit_date": EXIT_DATES.get(ticker, "Unknown"),
                "csv_available": csv_path.exists(),
                "csv_path": str(csv_path)
            })
        return pd.DataFrame(info)

    @staticmethod
    def get_survivorship_bias_disclaimer() -> str:
        """
        Generate a disclaimer about survivorship bias correction.

        Returns:
            Disclaimer string for documentation/reporting.
        """
        disclaimer = """
        ============================================================================
        SURVIVORSHIP BIAS CORRECTION - HYBRID DATA LOADING
        ============================================================================
        
        This analysis implements a HYBRID data loading approach to address
        survivorship bias:
        
        1. CURRENT SMI MEMBERS (20 stocks as of Dec 31, 2024):
           - Loaded via Yahoo Finance API
           - Full historical data available
        
        2. HISTORICAL MEMBERS (11 former constituents):
           - Loaded from local CSV files
           - Data TRUNCATED at official exit date from the index
           - Prevents look-ahead bias from knowing future delisting
        
        HISTORICAL MEMBERS INCLUDED:
        - Credit Suisse (CSGN.SW): Exited 2023-06-13 (UBS acquisition)
        - SGS (SGSN.SW): Exited 2022-09-19
        - Swatch (UHR.SW): Exited 2021-09-20
        - Adecco (ADEN.SW): Exited 2020-09-21
        - Julius Bär (BAER.SW): Exited 2019-04-10
        - Syngenta (SYNN.SW): Exited 2017-05-15 (ChemChina acquisition)
        - Actelion (ATLN.SW): Exited 2017-05-03 (J&J acquisition)
        - Transocean (RIGN.SW): Exited 2016-03-21
        - Synthes (SYST.SW): Exited 2012-06-18 (J&J acquisition)
        - Lonza (LONN.SW): Exited 2011-09-19 (later re-entered)
        - Swiss Life (SLHN.SW): Exited 2010-06-21 (later re-entered)
        
        DATA ALIGNMENT:
        - Uses OUTER join to preserve full market history (2010-2024)
        - Stocks have NaN values for dates outside their index membership
        - This allows proper walk-forward validation without artificial truncation
        
        REMAINING LIMITATIONS:
        - IPO bias: Stocks entering the index mid-period start at their IPO/entry date
        - Delisted companies only included if CSV data is provided
        
        Reference: Elton, E., Gruber, M., & Blake, C. (1996). "Survivorship Bias and
        Mutual Fund Performance." Review of Financial Studies.
        ============================================================================
        """
        return disclaimer


# ==============================================================================
# Main Function
# ==============================================================================

def main() -> None:
    """Example usage of the DataLoader module with historical data."""
    # Initialize loader with historical data enabled
    loader = DataLoader(
        start_date="2010-01-01",
        end_date="2024-12-31",
        include_historical=True,
    )
    
    # Print survivorship bias disclaimer
    print(loader.get_survivorship_bias_disclaimer())
    
    # Show historical members info
    print("\nHistorical Members Configuration:")
    print(loader.get_historical_members_info().to_string(index=False))
    
    # Load data (will use cache for current members, CSV for historical)
    try:
        stock_data, market_data = loader.load_data()
        
        # Align data with outer join (preserves full market history)
        aligned_data = loader.align_data(stock_data, market_data, method="outer")
        
        # Display summary
        print(f"\n{'=' * 60}")
        print("DATA SUMMARY")
        print("=" * 60)
        print(f"  - Current SMI members: {len(loader.tickers)}")
        print(f"  - Historical members loaded: {len(loader._loaded_historical)}")
        print(f"  - Total stocks: {len(stock_data)}")
        print(f"  - Date range: {aligned_data.index.min().date()} to {aligned_data.index.max().date()}")
        print(f"  - Trading days: {len(aligned_data)}")
        print(f"\nFirst 5 rows:")
        print(aligned_data.head())
        
    except Exception as e:
        print(f"\nError loading data: {e}")
        print("Note: Historical CSV files may not be present yet.")
        print(f"Expected location: {loader.historical_data_path}/")


if __name__ == "__main__":
    main()
