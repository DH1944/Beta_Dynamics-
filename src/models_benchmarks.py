"""
Benchmark Models Module
=======================

HEC Lausanne - Data Science and Advanced Programming
Master in Finance

This module implements academic benchmarks for beta estimation:
- Naive Baseline: just predict yesterday's beta (random walk assumption)
- Welch BSWA: Slope-Winsorized Age-Decayed Beta (from Welch 2021 paper)
- Kalman Filter: dynamic state-space estimation

These benchmarks help us evaluate if our ML models actually add value.

References:
    Welch, I. (2021). "Simply Better Market Betas." SSRN Working Paper.
    Adrian, T., & Franzoni, F. (2009). "Learning about beta."
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from numba import jit


# ==============================================================================
# Abstract Base Class
# ==============================================================================

class BetaEstimator(ABC):
    """
    Base class for all beta estimation models.
    Every model must implement fit() and predict() methods.
    """

    @abstractmethod
    def fit(self, y: np.ndarray, X: np.ndarray) -> "BetaEstimator":
        """
        Fit the model on historical data.
        
        Args:
            y: Stock returns
            X: Market returns
        """
        pass

    @abstractmethod
    def predict(self, y: np.ndarray, X: np.ndarray) -> np.ndarray:
        """
        Predict beta values.
        
        Args:
            y: Stock returns
            X: Market returns
            
        Returns:
            Array of beta estimates
        """
        pass


# ==============================================================================
# Naive Baseline
# ==============================================================================

class NaiveBaseline(BetaEstimator):
    """
    Naive model: Beta(t) = Beta(t-1)
    
    This is the simplest possible model. It assumes beta follows a random walk,
    so the best prediction for tomorrow is just today's value.
    
    Any serious model should beat this baseline, otherwise it's not useful.

    Attributes:
        window (int): Window for initial beta calculation.

    Example:
        >>> model = NaiveBaseline(window=252)
        >>> predictions = model.predict(stock_returns, market_returns)
    """

    def __init__(self, window: int = 252) -> None:
        """
        Initialize NaiveBaseline.

        Args:
            window: Window for calculating initial realized beta.
        """
        self.window = window
        self._fitted = False

    def fit(self, y: np.ndarray, X: np.ndarray) -> "NaiveBaseline":
        """
        Fit is a no-op for the naive model (stateless).

        Args:
            y: Stock returns (unused).
            X: Market returns (unused).

        Returns:
            Self.
        """
        self._fitted = True
        return self

    def predict(self, y: np.ndarray, X: np.ndarray) -> np.ndarray:
        """
        Generate naive predictions: Beta(t) = Beta(t-1).

        Args:
            y: Stock returns array.
            X: Market returns array.

        Returns:
            Array where each beta prediction is the previous day's realized beta.
        """
        # First compute realized betas
        realized_betas = self._compute_rolling_beta(y, X, self.window)
        
        # Naive prediction: shift by 1 (yesterday's beta predicts today)
        predictions = np.roll(realized_betas, 1)
        predictions[0] = np.nan
        
        return predictions

    @staticmethod
    @jit(nopython=True, cache=True)
    def _compute_rolling_beta(y: np.ndarray, X: np.ndarray, window: int) -> np.ndarray:
        """
        Compute rolling OLS beta with NaN handling.

        Args:
            y: Stock returns.
            X: Market returns.
            window: Rolling window size.

        Returns:
            Array of rolling beta estimates.
        """
        n = len(y)
        result = np.full(n, np.nan)
        
        # Minimum valid observations: 50% of window, at least 5
        min_required = max(5, window // 2)
        
        for i in range(window - 1, n):
            sum_x = 0.0
            sum_y = 0.0
            sum_xy = 0.0
            sum_xx = 0.0
            count = 0
            
            for j in range(window):
                idx = i - j
                xi = X[idx]
                yi = y[idx]
                
                # Skip NaN values
                if np.isnan(xi) or np.isnan(yi):
                    continue
                
                sum_x += xi
                sum_y += yi
                sum_xy += xi * yi
                sum_xx += xi * xi
                count += 1
            
            # Need at least min_required valid observations
            if count < min_required:
                continue
            
            mean_x = sum_x / count
            mean_y = sum_y / count
            
            cov_xy = (sum_xy / count) - (mean_x * mean_y)
            var_x = (sum_xx / count) - (mean_x * mean_x)
            
            if var_x > 1e-10:
                result[i] = cov_xy / var_x
        
        return result


# ==============================================================================
# Welch BSWA (Slope-Winsorized Age-Decayed Beta)
# ==============================================================================

class WelchBSWA(BetaEstimator):
    """
    Welch's Slope-Winsorized Age-Decayed Beta estimator.

    Implements the methodology from Welch (2021) "Simply Better Market Betas":
    1. Slope Winsorization: Reduces impact of extreme return days on the regression
       slope by winsorizing the regression residuals/deviations.
    2. Age Decay: Applies exponential decay weighting so that recent observations
       have more influence than older ones.

    The key insight is that standard OLS betas are noisy due to:
    - Outlier return days that distort the regression
    - Equal weighting of old vs. recent observations

    Attributes:
        window (int): Estimation window (default 252 trading days).
        decay_halflife (int): Half-life for exponential decay in days.
        winsor_pct (float): Percentile for winsorization (e.g., 0.05 for 5th/95th).

    References:
        Welch, I. (2021). "Simply Better Market Betas." SSRN Working Paper.
        https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3371240

    Example:
        >>> model = WelchBSWA(window=252, decay_halflife=126, winsor_pct=0.05)
        >>> model.fit(stock_returns, market_returns)
        >>> predictions = model.predict(stock_returns, market_returns)
    """

    def __init__(
        self,
        window: int = 252,
        decay_halflife: int = 126,
        winsor_pct: float = 0.05
    ) -> None:
        """
        Initialize Welch BSWA model.

        Args:
            window: Rolling window size (trading days).
            decay_halflife: Half-life for exponential decay weighting.
            winsor_pct: Winsorization percentile (applied to both tails).
        """
        self.window = window
        self.decay_halflife = decay_halflife
        self.winsor_pct = winsor_pct
        self._fitted = False

    def fit(self, y: np.ndarray, X: np.ndarray) -> "WelchBSWA":
        """
        Fit is a no-op (model is computed on predict).

        Args:
            y: Stock returns (unused for fitting).
            X: Market returns (unused for fitting).

        Returns:
            Self.
        """
        self._fitted = True
        return self

    def _compute_decay_weights(self, n: int) -> np.ndarray:
        """
        Compute exponential decay weights for observations.

        Weight = 2^(-age/halflife) where age is days from most recent.

        Args:
            n: Number of observations in window.

        Returns:
            Array of weights (most recent = highest weight).
        """
        # age goes from n-1 (oldest) to 0 (most recent)
        ages = np.arange(n - 1, -1, -1)
        weights = np.power(2.0, -ages / self.decay_halflife)
        return weights / weights.sum()  # Normalize

    @staticmethod
    def _winsorize(arr: np.ndarray, pct: float) -> np.ndarray:
        """
        Winsorize array at given percentile on both tails.

        Args:
            arr: Input array.
            pct: Percentile (e.g., 0.05 for 5th/95th percentiles).

        Returns:
            Winsorized array.
        """
        lower = np.nanpercentile(arr, pct * 100)
        upper = np.nanpercentile(arr, (1 - pct) * 100)
        return np.clip(arr, lower, upper)

    def _compute_single_beta(
        self,
        y: np.ndarray,
        X: np.ndarray
    ) -> float:
        """
        Compute single BSWA beta estimate for a window of data.

        The algorithm:
        1. Compute initial OLS beta
        2. Calculate residuals
        3. Winsorize the cross-products (regression contributions)
        4. Apply decay weights
        5. Re-estimate weighted beta

        CRITICAL: Winsorization is applied to the slope contributions
        (x_i * resid_i), not raw returns. This reduces the impact of
        extreme return days on the regression slope.

        Args:
            y: Stock returns in window.
            X: Market returns in window.

        Returns:
            BSWA beta estimate, or NaN if insufficient valid data.
        """
        # Handle NaN values: only use valid (non-NaN) observations
        valid_mask = ~(np.isnan(y) | np.isnan(X))
        n_valid = valid_mask.sum()
        n_window = len(y)
        
        # Need at least 50% of window to be valid, with absolute minimum of 5
        # This allows short windows (e.g., 10 days) while requiring sufficient data
        min_required = max(5, n_window // 2)
        if n_valid < min_required:
            return np.nan
        
        y_valid = y[valid_mask]
        X_valid = X[valid_mask]
        n = len(y_valid)
        
        # Step 1: Initial OLS estimate
        mean_x = np.mean(X_valid)
        mean_y = np.mean(y_valid)
        
        x_centered = X_valid - mean_x
        y_centered = y_valid - mean_y
        
        var_x = np.sum(x_centered ** 2)
        if var_x < 1e-10:
            return np.nan
        
        beta_ols = np.sum(x_centered * y_centered) / var_x
        
        # Step 2: Calculate slope contributions
        # Each observation's contribution to the slope: x_i * (y_i - mean_y - beta*(x_i - mean_x))
        residuals = y_centered - beta_ols * x_centered
        slope_contributions = x_centered * residuals
        
        # Step 3: Winsorize slope contributions (key innovation from Welch)
        # This reduces the impact of extreme return days on the slope
        slope_contrib_winsor = self._winsorize(slope_contributions, self.winsor_pct)
        
        # Step 4: Apply decay weights
        weights = self._compute_decay_weights(n)
        
        # Step 5: Re-estimate using weighted, winsorized contributions
        # Weighted covariance and variance
        weighted_cov = np.sum(weights * x_centered * y_centered)
        weighted_cov_adjustment = np.sum(weights * slope_contrib_winsor) - np.sum(weights * slope_contributions)
        
        # Adjusted covariance (original + winsorization adjustment)
        adj_cov = weighted_cov + weighted_cov_adjustment
        
        # Weighted variance of X
        weighted_var_x = np.sum(weights * (x_centered ** 2))
        
        if weighted_var_x < 1e-10:
            return np.nan
        
        beta_bswa = adj_cov / weighted_var_x
        
        return beta_bswa

    def predict(self, y: np.ndarray, X: np.ndarray) -> np.ndarray:
        """
        Generate BSWA beta predictions.

        Args:
            y: Stock returns array.
            X: Market returns array.

        Returns:
            Array of BSWA beta estimates.
        """
        n = len(y)
        predictions = np.full(n, np.nan)
        
        for i in range(self.window - 1, n):
            y_window = y[i - self.window + 1:i + 1]
            X_window = X[i - self.window + 1:i + 1]
            
            predictions[i] = self._compute_single_beta(y_window, X_window)
        
        return predictions

    def get_params(self) -> dict:
        """Return model parameters."""
        return {
            "window": self.window,
            "decay_halflife": self.decay_halflife,
            "winsor_pct": self.winsor_pct
        }


# ==============================================================================
# Kalman Filter Beta
# ==============================================================================

class KalmanBeta(BetaEstimator):
    """
    Kalman Filter for dynamic beta estimation.

    Implements a state-space model where beta is a latent state that evolves
    over time. The observation equation is:

        y_t = alpha_t + beta_t * X_t + epsilon_t

    And the state transition equation is:

        beta_t = beta_{t-1} + eta_t

    This approach allows beta to vary smoothly over time while filtering
    out observation noise.

    Attributes:
        initial_beta (float): Prior mean for initial beta.
        initial_var (float): Prior variance for initial beta.
        obs_noise (float): Observation noise variance.
        state_noise (float): State transition noise variance.

    References:
        Adrian, T., & Franzoni, F. (2009). "Learning about beta: Time-varying
        factor loadings, expected returns, and the conditional CAPM."

    Example:
        >>> model = KalmanBeta(initial_beta=1.0, state_noise=0.001)
        >>> predictions = model.predict(stock_returns, market_returns)
    """

    def __init__(
        self,
        initial_beta: float = 1.0,
        initial_var: float = 1.0,
        obs_noise: float = 0.0001,
        state_noise: float = 0.0001
    ) -> None:
        """
        Initialize Kalman Filter beta model.

        Args:
            initial_beta: Prior mean for beta at t=0.
            initial_var: Prior variance for beta at t=0.
            obs_noise: Variance of observation noise (epsilon).
            state_noise: Variance of state transition noise (eta).
        """
        self.initial_beta = initial_beta
        self.initial_var = initial_var
        self.obs_noise = obs_noise
        self.state_noise = state_noise
        self._fitted = False

    def fit(self, y: np.ndarray, X: np.ndarray) -> "KalmanBeta":
        """
        Fit is a no-op (Kalman filter is computed online).

        Args:
            y: Stock returns.
            X: Market returns.

        Returns:
            Self.
        """
        self._fitted = True
        return self

    def predict(self, y: np.ndarray, X: np.ndarray) -> np.ndarray:
        """
        Generate Kalman-filtered beta predictions.

        Implements the standard Kalman filter recursions:
        1. Predict step: project state forward
        2. Update step: incorporate new observation

        Handles missing data (NaN) gracefully by skipping the update step
        and only performing the prediction step for those observations.

        Args:
            y: Stock returns array.
            X: Market returns array.

        Returns:
            Array of filtered beta estimates.
        """
        n = len(y)
        
        # Initialize state
        beta_filtered = np.full(n, np.nan)
        var_filtered = np.zeros(n)
        
        beta_t = self.initial_beta
        P_t = self.initial_var
        
        for t in range(n):
            # Handle NaN values: skip update step, only predict
            if np.isnan(y[t]) or np.isnan(X[t]):
                # Still do prediction step (state evolves)
                P_t = P_t + self.state_noise
                beta_filtered[t] = beta_t
                var_filtered[t] = P_t
                continue
            
            # Skip if market return is too small (avoid numerical issues)
            if np.abs(X[t]) < 1e-10:
                beta_filtered[t] = beta_t
                var_filtered[t] = P_t
                continue
            
            # =====================================================
            # Predict Step
            # =====================================================
            # State prediction: beta_t|t-1 = beta_{t-1}|{t-1}
            beta_pred = beta_t
            
            # Variance prediction: P_t|t-1 = P_{t-1}|{t-1} + Q
            P_pred = P_t + self.state_noise
            
            # =====================================================
            # Update Step
            # =====================================================
            # Innovation: y_t - X_t * beta_t|t-1
            # Note: We're treating alpha as zero (intercept)
            innovation = y[t] - X[t] * beta_pred
            
            # Innovation variance: X_t^2 * P_t|t-1 + R
            S = X[t] ** 2 * P_pred + self.obs_noise
            
            # Kalman gain: K = P_t|t-1 * X_t / S
            K = P_pred * X[t] / S
            
            # State update: beta_t|t = beta_t|t-1 + K * innovation
            beta_t = beta_pred + K * innovation
            
            # Variance update: P_t|t = (1 - K * X_t) * P_t|t-1
            P_t = (1 - K * X[t]) * P_pred
            
            # Store filtered estimates
            beta_filtered[t] = beta_t
            var_filtered[t] = P_t
        
        return beta_filtered

    def predict_with_uncertainty(
        self,
        y: np.ndarray,
        X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate predictions with uncertainty estimates.

        Args:
            y: Stock returns array.
            X: Market returns array.

        Returns:
            Tuple of (beta_estimates, variance_estimates).
        """
        n = len(y)
        
        beta_filtered = np.zeros(n)
        var_filtered = np.zeros(n)
        
        beta_t = self.initial_beta
        P_t = self.initial_var
        
        for t in range(n):
            if np.abs(X[t]) < 1e-10:
                beta_filtered[t] = beta_t
                var_filtered[t] = P_t
                continue
            
            # Predict
            beta_pred = beta_t
            P_pred = P_t + self.state_noise
            
            # Update
            innovation = y[t] - X[t] * beta_pred
            S = X[t] ** 2 * P_pred + self.obs_noise
            K = P_pred * X[t] / S
            
            beta_t = beta_pred + K * innovation
            P_t = (1 - K * X[t]) * P_pred
            
            beta_filtered[t] = beta_t
            var_filtered[t] = P_t
        
        return beta_filtered, var_filtered

    def get_params(self) -> dict:
        """Return model parameters."""
        return {
            "initial_beta": self.initial_beta,
            "initial_var": self.initial_var,
            "obs_noise": self.obs_noise,
            "state_noise": self.state_noise
        }


# ==============================================================================
# Factory Function
# ==============================================================================

def create_benchmark_model(model_name: str, **kwargs) -> BetaEstimator:
    """
    Factory function to create benchmark models.

    Args:
        model_name: One of 'naive', 'welch', 'kalman'.
        **kwargs: Model-specific parameters.

    Returns:
        Initialized benchmark model.

    Raises:
        ValueError: If model_name is not recognized.
    """
    models = {
        "naive": NaiveBaseline,
        "welch": WelchBSWA,
        "kalman": KalmanBeta
    }
    
    if model_name.lower() not in models:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(models.keys())}")
    
    return models[model_name.lower()](**kwargs)


# ==============================================================================
# Main Example
# ==============================================================================

def main() -> None:
    """Example usage of benchmark models."""
    print("=" * 70)
    print("BENCHMARK MODELS DEMONSTRATION")
    print("=" * 70)
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000
    
    # True time-varying beta
    true_beta = 1.0 + 0.3 * np.sin(np.linspace(0, 4 * np.pi, n_samples))
    
    # Generate returns
    market_returns = np.random.randn(n_samples) * 0.01
    stock_returns = true_beta * market_returns + np.random.randn(n_samples) * 0.005
    
    # Initialize models
    models = {
        "Naive": NaiveBaseline(window=252),
        "Welch BSWA": WelchBSWA(window=252, decay_halflife=126, winsor_pct=0.05),
        "Kalman": KalmanBeta(initial_beta=1.0, state_noise=0.0001)
    }
    
    # Generate predictions
    print("\nGenerating predictions...")
    predictions = {}
    
    for name, model in models.items():
        model.fit(stock_returns, market_returns)
        predictions[name] = model.predict(stock_returns, market_returns)
        
        # Compute MSE (where predictions are valid)
        valid_mask = ~np.isnan(predictions[name])
        mse = np.mean((predictions[name][valid_mask] - true_beta[valid_mask]) ** 2)
        print(f"  {name}: MSE = {mse:.6f}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
