"""
Feature Engineering Module
==========================

HEC Lausanne - Data Science and Advanced Programming
Master in Finance

This module creates features for predicting beta. We use Numba to speed up
the rolling calculations (otherwise it would be too slow for large datasets).

Main features computed:
- Rolling beta (252-day OLS regression)
- Volatility indicators
- Momentum
- Market regime (based on VIX/VSMI)
"""

import logging
import time
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from numba import jit, prange

logger = logging.getLogger(__name__)

# Hide Numba warnings (they are not important for us)
warnings.filterwarnings("ignore", category=Warning, module="numba")


# ==============================================================================
# Constants
# ==============================================================================

# Small value to avoid division by zero
VARIANCE_EPSILON = 1e-10

# Above this threshold (annualized vol %), we consider the market in "high volatility"
DEFAULT_VOLATILITY_THRESHOLD = 20.0

# Trading days per year (for annualization)
TRADING_DAYS_PER_YEAR = 252


# ==============================================================================
# Numba-Optimized Functions
# ==============================================================================

@jit(nopython=True, cache=True)
def _rolling_mean_numba(arr: np.ndarray, window: int) -> np.ndarray:
    """
    Compute rolling mean with Numba (much faster than pandas for large arrays).
    Returns NaN for the first (window-1) values.
    """
    n = len(arr)
    result = np.full(n, np.nan)
    
    for i in range(window - 1, n):
        total = 0.0
        for j in range(window):
            total += arr[i - j]
        result[i] = total / window
    
    return result


@jit(nopython=True, cache=True)
def _rolling_std_numba(arr: np.ndarray, window: int) -> np.ndarray:
    """
    Compute rolling standard deviation with Numba.
    Returns NaN for the first (window-1) values.
    """
    n = len(arr)
    result = np.full(n, np.nan)
    
    for i in range(window - 1, n):
        # First compute mean
        total = 0.0
        for j in range(window):
            total += arr[i - j]
        mean = total / window
        
        # Then variance
        var_sum = 0.0
        for j in range(window):
            diff = arr[i - j] - mean
            var_sum += diff * diff
        
        # Sample standard deviation (N-1 denominator)
        result[i] = np.sqrt(var_sum / (window - 1))
    
    return result


@jit(nopython=True, cache=True)
def _rolling_ols_beta_numba(
    y: np.ndarray,
    x: np.ndarray,
    window: int
) -> np.ndarray:
    """
    Compute rolling OLS beta (slope coefficient) using Numba JIT.

    Beta = Cov(Y, X) / Var(X)

    Handles NaN values by skipping them in the calculation.
    Requires at least 50% of window to be valid.

    Args:
        y: Dependent variable (stock returns).
        x: Independent variable (market returns).
        window: Rolling window size.

    Returns:
        Array of rolling beta estimates (NaN for insufficient data).
    """
    n = len(y)
    result = np.full(n, np.nan)
    
    # Minimum valid observations: 50% of window, at least 5
    min_required = max(5, window // 2)
    
    for i in range(window - 1, n):
        # Extract window data, handling NaN
        sum_x = 0.0
        sum_y = 0.0
        sum_xy = 0.0
        sum_xx = 0.0
        count = 0
        
        for j in range(window):
            idx = i - j
            xi = x[idx]
            yi = y[idx]
            
            # Skip NaN values
            if np.isnan(xi) or np.isnan(yi):
                continue
            
            sum_x += xi
            sum_y += yi
            sum_xy += xi * yi
            sum_xx += xi * xi
            count += 1
        
        # Need minimum valid observations
        if count < min_required:
            continue
        
        # Compute covariance and variance
        mean_x = sum_x / count
        mean_y = sum_y / count
        
        cov_xy = (sum_xy / count) - (mean_x * mean_y)
        var_x = (sum_xx / count) - (mean_x * mean_x)
        
        # Avoid division by zero (VARIANCE_EPSILON = 1e-10)
        # Note: Cannot use module constant inside Numba nopython mode
        if var_x > 1e-10:
            result[i] = cov_xy / var_x
    
    return result


@jit(nopython=True, cache=True, parallel=True)
def _rolling_ols_beta_parallel(
    y_matrix: np.ndarray,
    x: np.ndarray,
    window: int
) -> np.ndarray:
    """
    Compute rolling OLS beta for multiple stocks in parallel.

    Handles NaN values by skipping them in the calculation.

    Args:
        y_matrix: 2D array (n_samples, n_stocks) of stock returns.
        x: 1D array of market returns.
        window: Rolling window size.

    Returns:
        2D array of rolling betas for each stock.
    """
    n_samples, n_stocks = y_matrix.shape
    result = np.full((n_samples, n_stocks), np.nan)
    
    # Minimum valid observations: 50% of window, at least 5
    min_required = max(5, window // 2)
    
    for stock in prange(n_stocks):
        y = y_matrix[:, stock]
        
        for i in range(window - 1, n_samples):
            sum_x = 0.0
            sum_y = 0.0
            sum_xy = 0.0
            sum_xx = 0.0
            count = 0
            
            for j in range(window):
                idx = i - j
                xi = x[idx]
                yi = y[idx]
                
                # Skip NaN values
                if np.isnan(xi) or np.isnan(yi):
                    continue
                
                sum_x += xi
                sum_y += yi
                sum_xy += xi * yi
                sum_xx += xi * xi
                count += 1
            
            # Need minimum valid observations
            if count < min_required:
                continue
            
            mean_x = sum_x / count
            mean_y = sum_y / count
            
            cov_xy = (sum_xy / count) - (mean_x * mean_y)
            var_x = (sum_xx / count) - (mean_x * mean_x)
            
            if var_x > 1e-10:
                result[i, stock] = cov_xy / var_x
    
    return result


@jit(nopython=True, cache=True)
def _momentum_numba(prices: np.ndarray, window: int) -> np.ndarray:
    """
    Compute momentum (rate of change) using Numba.

    Momentum = (Price_t / Price_{t-window}) - 1

    Args:
        prices: 1D array of prices.
        window: Lookback period.

    Returns:
        Array of momentum values.
    """
    n = len(prices)
    result = np.full(n, np.nan)
    
    for i in range(window, n):
        if prices[i - window] != 0:
            result[i] = (prices[i] / prices[i - window]) - 1.0
    
    return result


# ==============================================================================
# Benchmark Function: Numba vs Pandas
# ==============================================================================

def benchmark_numba_vs_pandas(n_samples: int = 5000, window: int = 252) -> Dict[str, float]:
    """
    Benchmark Numba-optimized functions against Pandas implementations.

    This function measures and compares execution times for:
    1. Rolling standard deviation
    2. Rolling OLS beta

    Args:
        n_samples: Number of data points to generate.
        window: Rolling window size.

    Returns:
        Dictionary with timing results for each implementation.

    Example:
        >>> results = benchmark_numba_vs_pandas(n_samples=5000, window=252)
        >>> print(f"Speedup: {results['pandas_std'] / results['numba_std']:.1f}x")
    """
    # Generate synthetic data
    np.random.seed(42)
    stock_returns = np.random.randn(n_samples) * 0.02
    market_returns = np.random.randn(n_samples) * 0.015
    
    results = {}
    
    # -------------------------------------------------------------------------
    # Rolling Standard Deviation
    # -------------------------------------------------------------------------
    
    # Pandas implementation
    series = pd.Series(stock_returns)
    start = time.perf_counter()
    _ = series.rolling(window).std()
    results["pandas_std_time"] = time.perf_counter() - start
    
    # Numba implementation (warm-up run first)
    _ = _rolling_std_numba(stock_returns, window)  # JIT compile
    start = time.perf_counter()
    _ = _rolling_std_numba(stock_returns, window)
    results["numba_std_time"] = time.perf_counter() - start
    
    # -------------------------------------------------------------------------
    # Rolling OLS Beta
    # -------------------------------------------------------------------------
    
    # Pandas implementation using rolling covariance/variance
    stock_series = pd.Series(stock_returns)
    market_series = pd.Series(market_returns)
    
    start = time.perf_counter()
    cov = stock_series.rolling(window).cov(market_series)
    var = market_series.rolling(window).var()
    _ = cov / var
    results["pandas_beta_time"] = time.perf_counter() - start
    
    # Numba implementation (warm-up run first)
    _ = _rolling_ols_beta_numba(stock_returns, market_returns, window)  # JIT compile
    start = time.perf_counter()
    _ = _rolling_ols_beta_numba(stock_returns, market_returns, window)
    results["numba_beta_time"] = time.perf_counter() - start
    
    # -------------------------------------------------------------------------
    # Print Results
    # -------------------------------------------------------------------------
    
    print("=" * 70)
    print("NUMBA vs PANDAS BENCHMARK RESULTS")
    print("=" * 70)
    print(f"Data size: {n_samples:,} samples, Window: {window} days")
    print("-" * 70)
    print("\nRolling Standard Deviation:")
    print(f"  Pandas:  {results['pandas_std_time']*1000:.3f} ms")
    print(f"  Numba:   {results['numba_std_time']*1000:.3f} ms")
    print(f"  Speedup: {results['pandas_std_time']/results['numba_std_time']:.1f}x")
    print("\nRolling OLS Beta:")
    print(f"  Pandas:  {results['pandas_beta_time']*1000:.3f} ms")
    print(f"  Numba:   {results['numba_beta_time']*1000:.3f} ms")
    print(f"  Speedup: {results['pandas_beta_time']/results['numba_beta_time']:.1f}x")
    print("=" * 70)
    
    return results


# ==============================================================================
# Feature Engineer Class
# ==============================================================================

class FeatureEngineer:
    """
    Feature engineering for beta prediction models.

    This class computes technical indicators, rolling betas, and market regime
    classifications for use in ML models.

    Attributes:
        data (pd.DataFrame): Aligned returns data.
        beta_window (int): Window for rolling beta calculation (default 252).
        vol_window (int): Window for volatility calculation (default 21).
        momentum_window (int): Window for momentum calculation (default 63).

    Example:
        >>> engineer = FeatureEngineer(aligned_returns)
        >>> features, target = engineer.create_features_for_stock('NESN.SW')
        >>> regimes = engineer.detect_market_regime()
    """

    def __init__(
        self,
        data: pd.DataFrame,
        beta_window: int = 252,
        vol_window: int = 21,
        momentum_window: int = 63,
    ) -> None:
        """
        Initialize FeatureEngineer with aligned returns data.

        Args:
            data: DataFrame with aligned returns (must include 'Market_Return' column).
            beta_window: Window for rolling beta calculation (trading days).
            vol_window: Window for volatility calculation (trading days).
            momentum_window: Window for momentum calculation (trading days).
        """
        self.data = data.copy()
        self.beta_window = beta_window
        self.vol_window = vol_window
        self.momentum_window = momentum_window
        
        # Cache for regime detection (avoid repeated API calls)
        self._cached_regime: Optional[pd.Series] = None
        self._cached_vol_source: Optional[str] = None
        
        # Validate required columns
        if "Market_Return" not in self.data.columns:
            raise ValueError("Data must contain 'Market_Return' column")

    def _get_stock_columns(self) -> List[str]:
        """Get list of stock return column names."""
        return [col for col in self.data.columns if col.endswith("_Return") and col != "Market_Return"]

    def compute_rolling_beta(
        self,
        stock_col: str,
        use_numba: bool = True
    ) -> pd.Series:
        """
        Compute rolling OLS beta for a stock.

        Args:
            stock_col: Column name for stock returns.
            use_numba: Whether to use Numba optimization.

        Returns:
            Series of rolling beta values.
            
        Raises:
            AssertionError: If inputs are invalid.
        """
        # Input validation (defensive programming)
        assert stock_col in self.data.columns, \
            f"Column '{stock_col}' not found in data. Available: {list(self.data.columns)}"
        assert self.beta_window > 1, \
            f"beta_window must be > 1. Got: {self.beta_window}"
        
        stock_returns = self.data[stock_col].values
        market_returns = self.data["Market_Return"].values
        
        if use_numba:
            beta_values = _rolling_ols_beta_numba(
                stock_returns, market_returns, self.beta_window
            )
        else:
            # Pandas fallback
            stock_series = pd.Series(stock_returns, index=self.data.index)
            market_series = pd.Series(market_returns, index=self.data.index)
            cov = stock_series.rolling(self.beta_window).cov(market_series)
            var = market_series.rolling(self.beta_window).var()
            beta_values = (cov / var).values
        
        return pd.Series(beta_values, index=self.data.index, name=f"{stock_col}_Beta")

    def compute_rolling_volatility(
        self,
        col: str,
        window: Optional[int] = None,
        annualize: bool = True
    ) -> pd.Series:
        """
        Compute rolling volatility (standard deviation of returns).

        Args:
            col: Column name for returns.
            window: Rolling window. Defaults to self.vol_window.
            annualize: Whether to annualize volatility (multiply by sqrt(252)).

        Returns:
            Series of rolling volatility values.
        """
        window = window or self.vol_window
        returns = self.data[col].values
        
        vol = _rolling_std_numba(returns, window)
        
        if annualize:
            vol = vol * np.sqrt(TRADING_DAYS_PER_YEAR)
        
        return pd.Series(vol, index=self.data.index, name=f"{col}_Vol_{window}d")

    def compute_momentum(
        self,
        col: str,
        window: Optional[int] = None
    ) -> pd.Series:
        """
        Compute momentum (cumulative return over window).

        Args:
            col: Column name for returns.
            window: Lookback period. Defaults to self.momentum_window.

        Returns:
            Series of momentum values.
        """
        window = window or self.momentum_window
        
        # Cumulative return over window
        cum_returns = (1 + self.data[col]).rolling(window).apply(
            lambda x: x.prod() - 1, raw=True
        )
        
        return cum_returns.rename(f"{col}_Mom_{window}d")

    def compute_cumulative_returns(
        self,
        col: str,
        windows: List[int] = [5, 21, 63]
    ) -> pd.DataFrame:
        """
        Compute cumulative returns over multiple windows.

        Args:
            col: Column name for returns.
            windows: List of lookback periods.

        Returns:
            DataFrame with cumulative returns for each window.
        """
        results = {}
        
        for window in windows:
            cum_ret = (1 + self.data[col]).rolling(window).apply(
                lambda x: x.prod() - 1, raw=True
            )
            results[f"{col}_CumRet_{window}d"] = cum_ret
        
        return pd.DataFrame(results, index=self.data.index)

    def create_features_for_stock(
        self,
        stock_col: str,
        shift_target: bool = True,
        include_regime: bool = True,
        precomputed_regimes: Optional[pd.Series] = None,
        target_window: int = 63
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Create full feature set for a single stock.

        CRITICAL: Target is shifted by 1 day to prevent look-ahead bias.
        Features at time t are used to predict realized beta at t+1.

        Args:
            stock_col: Stock return column name.
            shift_target: Whether to shift target (must be True for proper validation).
            include_regime: If True, add market regime as binary feature.
            precomputed_regimes: Pre-computed regime Series to avoid repeated API calls.
                                 If None and include_regime=True, will call detect_market_regime().
            target_window: Rolling window for target beta calculation (default: 63 days).
                          Use shorter windows (21-63 days) for harder, more volatile targets.
                          Use longer windows (252 days) for easier, more persistent targets.
                          RECOMMENDED: 63 days (quarterly beta) balances signal/noise.

        Returns:
            Tuple of (features_df, target_series).
        """
        import logging
        logger = logging.getLogger(__name__)
        
        features = pd.DataFrame(index=self.data.index)
        
        # -------------------------------------------------------------------------
        # Stock-specific features
        # CRITICAL: All features must be shifted by 1 to prevent look-ahead bias
        # At time t, we can only use information available at t-1 to predict t+1
        # -------------------------------------------------------------------------
        
        # Rolling volatility (multiple windows) - SHIFTED
        features["Vol_21d"] = self.compute_rolling_volatility(stock_col, 21, annualize=True).shift(1)
        features["Vol_63d"] = self.compute_rolling_volatility(stock_col, 63, annualize=True).shift(1)
        
        # Momentum - SHIFTED
        features["Mom_21d"] = self.compute_momentum(stock_col, 21).shift(1)
        features["Mom_63d"] = self.compute_momentum(stock_col, 63).shift(1)
        
        # Cumulative returns - SHIFTED
        cum_rets = self.compute_cumulative_returns(stock_col, [5, 21, 63])
        for col in cum_rets.columns:
            # Simplify column names
            simple_name = col.replace(f"{stock_col}_", "")
            features[simple_name] = cum_rets[col].shift(1)
        
        # -------------------------------------------------------------------------
        # Market features - SHIFTED
        # -------------------------------------------------------------------------
        
        features["Market_Vol_21d"] = self.compute_rolling_volatility("Market_Return", 21, annualize=True).shift(1)
        features["Market_Mom_21d"] = self.compute_momentum("Market_Return", 21).shift(1)
        
        # -------------------------------------------------------------------------
        # Lagged beta (yesterday's beta as feature)
        # Already shifted - Beta_Lag1 uses beta[t-1] to predict target at t
        # -------------------------------------------------------------------------
        
        current_beta = self.compute_rolling_beta(stock_col)
        features["Beta_Lag1"] = current_beta.shift(1)   # Beta from t-1
        features["Beta_Lag5"] = current_beta.shift(5)   # Beta from t-5
        # MA5 must be computed on LAGGED values only (shift first, then MA)
        features["Beta_MA5"] = current_beta.shift(1).rolling(5).mean()  # MA of betas from t-1 to t-5
        
        # -------------------------------------------------------------------------
        # Market Regime Feature - SHIFTED
        # -------------------------------------------------------------------------
        
        if include_regime:
            # Use precomputed regimes if provided (avoids repeated API calls)
            if precomputed_regimes is not None:
                regimes = precomputed_regimes
                logger.debug("Using precomputed regimes (no API call)")
            else:
                # Fallback: compute regimes (may trigger API call)
                try:
                    regimes = self.detect_market_regime(threshold=20.0)
                except Exception as e:
                    logger.warning(f"Could not detect regime via VIX/VSMI: {e}")
                    # Use realized volatility as fallback
                    vol_proxy = features["Market_Vol_21d"].ffill() * 100
                    regimes = pd.Series(
                        np.where(vol_proxy > DEFAULT_VOLATILITY_THRESHOLD, "High Volatility", "Calm"),
                        index=self.data.index
                    )
            
            # Convert to binary and SHIFT: 1 = High Volatility, 0 = Calm
            regime_binary = (regimes == "High Volatility").astype(int)
            features["Regime_HighVol"] = regime_binary.shift(1)  # Shifted!
            
            # Interaction term: Vol * Regime (both already shifted)
            features["Vol_21d_x_Regime"] = (
                features["Vol_21d"] * features["Regime_HighVol"]
            )
        
        # -------------------------------------------------------------------------
        # Target: Short-term Beta (MORE VOLATILE = harder prediction task)
        # -------------------------------------------------------------------------
        # Using a shorter window for target creates a more challenging prediction
        # problem that better reveals the value of ML models and historical data.
        # 
        # Window comparison:
        # - 252 days: High persistence, ~95% overlap day-to-day → trivial prediction
        # - 63 days: Moderate persistence, ~98% overlap → challenging but learnable
        # - 21 days: Low persistence, high noise → very hard
        # -------------------------------------------------------------------------
        
        # Compute target beta with shorter window (default 63 days)
        stock_returns = self.data[stock_col].values
        market_returns = self.data["Market_Return"].values
        
        target_beta_values = _rolling_ols_beta_numba(
            stock_returns, market_returns, target_window
        )
        target_beta = pd.Series(target_beta_values, index=self.data.index, name=f"{stock_col}_Target_Beta")
        
        if shift_target:
            # Target at time t is the beta realized at t+1
            # This means features at t predict beta at t+1
            target = target_beta.shift(-1)
        else:
            target = target_beta
        
        target.name = f"Target_Beta_{target_window}d"
        
        logger.debug(f"Target beta window: {target_window} days (feature beta: {self.beta_window} days)")
        
        return features, target

    def create_all_features(
        self,
        shift_target: bool = True
    ) -> Dict[str, Tuple[pd.DataFrame, pd.Series]]:
        """
        Create features for all stocks in the dataset.

        Args:
            shift_target: Whether to shift target for look-ahead prevention.

        Returns:
            Dictionary mapping stock tickers to (features, target) tuples.
        """
        stock_cols = self._get_stock_columns()
        all_features = {}
        
        for col in stock_cols:
            ticker = col.replace("_Return", "")
            features, target = self.create_features_for_stock(col, shift_target)
            all_features[ticker] = (features, target)
        
        return all_features

    def detect_market_regime(
        self,
        vix_data: Optional[pd.DataFrame] = None,
        threshold: float = 20.0,
        force_refresh: bool = False,
        use_realized_vol: bool = True
    ) -> pd.Series:
        """
        Detect market regime based on volatility.

        By default, uses REALIZED volatility of the SMI index for geographic
        consistency with Swiss market data. This avoids issues with:
        - VIX (US market, different timezone, not always synchronized)
        - VSMI (often delisted/unavailable on Yahoo Finance)

        Args:
            vix_data: Optional DataFrame with VIX/VSMI data.
            threshold: Volatility threshold (above = "High Volatility").
            force_refresh: If True, ignore cache and re-fetch data.
            use_realized_vol: If True (default), use realized SMI volatility
                              instead of VIX/VSMI. Recommended for Swiss market.

        Returns:
            Series with regime labels ("Calm" or "High Volatility").
        """
        # Return cached result if available
        if self._cached_regime is not None and not force_refresh:
            return self._cached_regime
        
        vol_index = None
        vol_source = None
        
        if use_realized_vol:
            # PREFERRED: Use realized volatility of SMI for geographic consistency
            # This is calculated from the actual Market_Return data
            vol_index = self.compute_rolling_volatility("Market_Return", 21, annualize=True) * 100
            vol_source = "SMI_Realized_Vol"
            logger.info("Using realized SMI volatility for regime detection (recommended)")
        else:
            # Alternative: Try external volatility indices
            import yfinance as yf
            
            for ticker, name in [("^VSMI", "VSMI"), ("^VIX", "VIX")]:
                try:
                    data = yf.download(
                        ticker,
                        start=self.data.index.min(),
                        end=self.data.index.max(),
                        progress=False
                    )
                    if not data.empty:
                        # Handle MultiIndex columns
                        if isinstance(data.columns, pd.MultiIndex):
                            data.columns = data.columns.droplevel(1)
                        vol_index = data["Close"]
                        vol_source = name
                        logger.info(f"Loaded volatility index: {name}")
                        break
                except Exception as e:
                    logger.debug(f"Could not load {name}: {e}")
                    continue
            
            if vol_index is None:
                # Fallback to realized vol if external indices unavailable
                vol_index = self.compute_rolling_volatility("Market_Return", 21, annualize=True) * 100
                vol_source = "SMI_Realized_Vol_Fallback"
                logger.warning("External vol indices unavailable, using realized SMI volatility")
        
        # Align with data index
        vol_index = vol_index.reindex(self.data.index, method="ffill")
        
        # Classify regime
        regime = pd.Series(
            np.where(vol_index > threshold, "High Volatility", "Calm"),
            index=self.data.index,
            name="Market_Regime"
        )
        
        # Cache the result
        self._cached_regime = regime
        self._cached_vol_source = vol_source
        
        logger.info(f"Market regime detection using: {vol_source}")
        logger.info(f"  - High Volatility periods: {(regime == 'High Volatility').sum()} days")
        logger.info(f"  - Calm periods: {(regime == 'Calm').sum()} days")
        
        return regime

    def get_feature_summary(self) -> pd.DataFrame:
        """
        Generate summary statistics for all computed features.

        Returns:
            DataFrame with feature descriptions and statistics.
        """
        stock_cols = self._get_stock_columns()
        
        if not stock_cols:
            return pd.DataFrame()
        
        # Get features for first stock as example
        features, target = self.create_features_for_stock(stock_cols[0])
        
        summary = features.describe().T
        summary["non_null_pct"] = (1 - features.isnull().mean()) * 100
        
        return summary


def main() -> None:
    """Example usage and benchmark of FeatureEngineer."""
    print("Running Numba vs Pandas Benchmark...")
    print()
    
    # Run benchmark
    benchmark_numba_vs_pandas(n_samples=5000, window=252)
    
    print("\n" + "=" * 70)
    print("FEATURE ENGINEERING EXAMPLE")
    print("=" * 70)
    
    # Generate synthetic data for demonstration
    np.random.seed(42)
    dates = pd.date_range("2010-01-01", periods=3000, freq="B")
    
    data = pd.DataFrame({
        "Market_Return": np.random.randn(len(dates)) * 0.01,
        "NESN.SW_Return": np.random.randn(len(dates)) * 0.015,
        "ROG.SW_Return": np.random.randn(len(dates)) * 0.012,
    }, index=dates)
    
    # Initialize engineer
    engineer = FeatureEngineer(data)
    
    # Create features for one stock
    features, target = engineer.create_features_for_stock("NESN.SW_Return")
    
    print(f"\nFeatures shape: {features.shape}")
    print(f"Target shape: {target.shape}")
    print(f"\nFeature columns: {list(features.columns)}")
    print(f"\nFeature summary (non-null rows):")
    print(features.dropna().describe().round(4))


if __name__ == "__main__":
    main()
