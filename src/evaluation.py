"""
Module of Evaluation
====================

HEC Lausanne - Data Science and Advanced Programming
Master in Finance

This module handles the evaluation of the models utilising the walk-forward validation.
Indeed, the walk-forward validation signifies that the system always trains on the past
data and tests on the future data, which constitutes the correct approach for the time
series (preventing any data leakage).

The principal functionalities of this module are the following:
- Walk-forward validation
- Metrics of evaluation: MSE, MAE, R-squared
- Graphical representations of the cumulative MSE
- Comparison of the performance by regime of the market
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ==============================================================================
# Evaluator of Walk-Forward Validation
# ==============================================================================

class WalkForwardEvaluator:
    """
    Walk-forward validation for the time series.
    
    The principle is the following: the system trains on the data up to time T,
    then tests on T+1. Subsequently, the system advances: it trains on the data
    up to T+1, and tests on T+2. And so on.
    
    This approach permits us to never "cheat" by utilising future data to predict the past.

    Attributes:
        n_splits: The number of splits of train/test to execute.
        train_size: The minimum number of samples of training (None corresponds to the utilisation of all the available data).
        test_size: The number of samples of test per split.
        gap: The gap between train and test (for additional precaution).

    Example:
        >>> evaluator = WalkForwardEvaluator(n_splits=10)
        >>> results = evaluator.evaluate_model(model, features, target)
        >>> evaluator.plot_cumulative_mse(results)
    """

    def __init__(
        self,
        n_splits: int = 10,
        train_size: Optional[int] = None,
        test_size: int = 63,  # Approximately 3 months
        gap: int = 0,
    ) -> None:
        """
        Initialise the evaluator of walk-forward validation.

        Args:
            n_splits: The number of splits for the evaluation.
            train_size: The minimum number of samples of training. If None, an expanding window is utilised.
            test_size: The number of samples of test per split.
            gap: The gap between train and test (in order to prevent any potential leakage).
        """
        self.n_splits = n_splits
        self.train_size = train_size
        self.test_size = test_size
        self.gap = gap

    def _generate_splits(
        self,
        n_samples: int
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate the index splits of train/test for the walk-forward validation.

        Args:
            n_samples: The total number of samples.

        Returns:
            A list of tuples (train_indices, test_indices).
        """
        splits = []
        
        # Calculation of the sizes
        total_test = self.n_splits * self.test_size
        min_train = self.train_size or (n_samples - total_test - self.gap * self.n_splits) // 2
        
        if min_train < 100:
            min_train = 100  # Ensuring a minimum of training data
        
        available = n_samples - min_train - self.gap
        step = available // self.n_splits
        
        for i in range(self.n_splits):
            # Expanding window: the training set grows over time
            train_end = min_train + i * step
            test_start = train_end + self.gap
            test_end = min(test_start + self.test_size, n_samples)
            
            if test_end <= test_start:
                continue
            
            train_idx = np.arange(0, train_end)
            test_idx = np.arange(test_start, test_end)
            
            splits.append((train_idx, test_idx))
        
        return splits

    def evaluate_single_split(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Evaluate a model on a single split of train/test.

        Args:
            model: A fitted model compatible with sklearn.
            X: The matrix of features.
            y: The vector of target.
            train_idx: The indices of training.
            test_idx: The indices of test.

        Returns:
            A dictionary containing the predictions and the metrics.
        """
        # Extraction of the data of train/test
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Suppression of the rows containing NaN
        train_mask = ~(np.any(np.isnan(X_train), axis=1) | np.isnan(y_train))
        test_mask = ~(np.any(np.isnan(X_test), axis=1) | np.isnan(y_test))
        
        X_train_clean = X_train[train_mask]
        y_train_clean = y_train[train_mask]
        X_test_clean = X_test[test_mask]
        y_test_clean = y_test[test_mask]
        
        if len(X_train_clean) < 10 or len(X_test_clean) < 1:
            return {"error": "Insufficient data after cleaning"}
        
        # Fitting of the model
        model.fit(X_train_clean, y_train_clean)
        
        # Prediction
        y_pred = model.predict(X_test_clean)
        
        # Computation of the metrics
        mse = mean_squared_error(y_test_clean, y_pred)
        mae = mean_absolute_error(y_test_clean, y_pred)
        r2 = r2_score(y_test_clean, y_pred) if len(y_test_clean) > 1 else np.nan
        
        return {
            "train_idx": train_idx,
            "test_idx": test_idx[test_mask],
            "y_true": y_test_clean,
            "y_pred": y_pred,
            "mse": mse,
            "mae": mae,
            "r2": r2,
            "train_size": len(X_train_clean),
            "test_size": len(X_test_clean),
        }

    def evaluate_model(
        self,
        model: Any,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        dates: Optional[pd.DatetimeIndex] = None,
    ) -> Dict[str, Any]:
        """
        Perform the complete walk-forward evaluation of a model.

        Args:
            model: A model compatible with sklearn (which will be cloned for each split).
            X: The matrix of features.
            y: The vector of target.
            dates: An optional datetime index for the tracking of the time.

        Returns:
            A dictionary containing all the results of the splits and the aggregate metrics.
        """
        from sklearn.base import clone
        
        X_array = np.asarray(X)
        y_array = np.asarray(y)
        n_samples = len(y_array)
        
        # Generation of the splits
        splits = self._generate_splits(n_samples)
        
        if not splits:
            raise ValueError("No valid splits generated. Check data size and parameters.")
        
        # Evaluation of each split
        split_results = []
        all_predictions = np.full(n_samples, np.nan)
        all_squared_errors = np.full(n_samples, np.nan)
        
        for i, (train_idx, test_idx) in enumerate(splits):
            model_clone = clone(model)
            result = self.evaluate_single_split(model_clone, X_array, y_array, train_idx, test_idx)
            
            if "error" not in result:
                split_results.append(result)
                
                # Storage of the predictions
                test_indices = result["test_idx"]
                all_predictions[test_indices] = result["y_pred"]
                
                # Storage of the squared errors
                squared_errors = (result["y_true"] - result["y_pred"]) ** 2
                all_squared_errors[test_indices] = squared_errors
        
        # Aggregation of the metrics
        valid_mask = ~np.isnan(all_squared_errors)
        
        # Raw errors for the test of Diebold-Mariano
        all_errors = np.full(n_samples, np.nan)
        all_errors[valid_mask] = np.sqrt(all_squared_errors[valid_mask]) * np.sign(
            all_predictions[valid_mask] - y_array[valid_mask]
        )
        # Computation of the raw errors in a proper manner
        all_errors = np.full(n_samples, np.nan)
        all_errors[valid_mask] = y_array[valid_mask] - all_predictions[valid_mask]
        
        return {
            "split_results": split_results,
            "predictions": all_predictions,
            "squared_errors": all_squared_errors,
            "errors": all_errors,
            "mse": np.nanmean(all_squared_errors),
            "dates": dates,
            "aggregate": {
                "mean_mse": np.nanmean(all_squared_errors),
                "std_mse": np.nanstd(all_squared_errors),
                "mean_mae": np.nanmean(np.sqrt(all_squared_errors)),
                "n_predictions": np.sum(valid_mask),
                "n_splits": len(split_results),
            }
        }

    def evaluate_benchmark(
        self,
        predictions: np.ndarray,
        y_true: np.ndarray,
        dates: Optional[pd.DatetimeIndex] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate the pre-computed predictions of the benchmarks.

        For the benchmark models (Naive, Welch, Kalman) which do not utilise
        the standard paradigm of train/predict.

        Args:
            predictions: The array of pre-computed predictions.
            y_true: The true values of the target.
            dates: An optional datetime index.

        Returns:
            A dictionary containing the results of the evaluation.
        """
        # Computation of the metrics where both are valid
        valid_mask = ~(np.isnan(predictions) | np.isnan(y_true))
        
        if not np.any(valid_mask):
            return {"error": "No valid predictions"}
        
        y_valid = y_true[valid_mask]
        pred_valid = predictions[valid_mask]
        
        squared_errors = np.full(len(y_true), np.nan)
        squared_errors[valid_mask] = (y_valid - pred_valid) ** 2
        
        # Raw errors for the test of Diebold-Mariano
        errors = np.full(len(y_true), np.nan)
        errors[valid_mask] = y_valid - pred_valid
        
        return {
            "predictions": predictions,
            "squared_errors": squared_errors,
            "errors": errors,
            "mse": np.mean(squared_errors[valid_mask]),
            "dates": dates,
            "aggregate": {
                "mean_mse": np.mean(squared_errors[valid_mask]),
                "std_mse": np.std(squared_errors[valid_mask]),
                "mean_mae": np.mean(np.abs(y_valid - pred_valid)),
                "r2": r2_score(y_valid, pred_valid),
                "n_predictions": np.sum(valid_mask),
            }
        }


# ==============================================================================
# Functions of Visualisation
# ==============================================================================

def plot_cumulative_mse(
    results: Dict[str, Dict],
    dates: Optional[pd.DatetimeIndex] = None,
    title: str = "Cumulative MSE Over Time",
    figsize: Tuple[int, int] = (14, 7),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot the cumulative MSE over time for multiple models.

    CRITICAL: This visualisation shows WHERE the models fail or succeed,
    which permits the analysis of the behaviour of the models during different
    conditions of the market.

    Args:
        results: A dictionary mapping the names of the models to the results of the evaluation.
        dates: A datetime index for the x-axis.
        title: The title of the plot.
        figsize: The size of the figure.
        save_path: An optional path to save the figure.

    Returns:
        An object of type Matplotlib figure.
    """
    fig, axes = plt.subplots(2, 1, figsize=figsize, height_ratios=[3, 1])
    
    # Colours for the different models
    colors = plt.cm.Set1(np.linspace(0, 1, len(results)))
    
    ax1 = axes[0]
    ax2 = axes[1]
    
    for (model_name, result), color in zip(results.items(), colors):
        if "error" in result:
            continue
        
        squared_errors = result["squared_errors"]
        
        # Obtention of the dates
        if dates is not None:
            x_axis = dates
        elif result.get("dates") is not None:
            x_axis = result["dates"]
        else:
            x_axis = np.arange(len(squared_errors))
        
        # Computation of the cumulative MSE
        valid_mask = ~np.isnan(squared_errors)
        cum_mse = np.nancumsum(squared_errors)
        cum_count = np.cumsum(valid_mask.astype(float))
        cum_count[cum_count == 0] = np.nan
        cum_avg_mse = cum_mse / cum_count
        
        # Plot of the cumulative MSE
        ax1.plot(x_axis, cum_avg_mse, label=model_name, color=color, linewidth=2)
        
        # Plot of the rolling MSE (window of 21 days) on the second axis
        rolling_mse = pd.Series(squared_errors).rolling(21, min_periods=1).mean()
        ax2.plot(x_axis, rolling_mse, label=model_name, color=color, alpha=0.7)
    
    # Configuration of the axes
    ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.set_ylabel("Cumulative Average MSE", fontsize=12)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel("Date" if dates is not None else "Observation", fontsize=12)
    ax2.set_ylabel("Rolling MSE (21d)", fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Formatting of the x-axis dates if applicable
    if dates is not None:
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax1.xaxis.set_major_locator(mdates.YearLocator())
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax2.xaxis.set_major_locator(mdates.YearLocator())
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    return fig


def plot_predictions_vs_actual(
    result: Dict,
    dates: Optional[pd.DatetimeIndex] = None,
    model_name: str = "Model",
    figsize: Tuple[int, int] = (14, 5),
) -> plt.Figure:
    """
    Plot the predicted values versus the actual values of beta.

    Args:
        result: The dictionary of evaluation results for a single model.
        dates: The datetime index.
        model_name: The name for the title of the plot.
        figsize: The size of the figure.

    Returns:
        An object of type Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    predictions = result["predictions"]
    
    # Utilisation of the provided dates or the dates of the result
    if dates is not None:
        x_axis = dates
    elif result.get("dates") is not None:
        x_axis = result["dates"]
    else:
        x_axis = np.arange(len(predictions))
    
    # Plot of the predictions
    valid_mask = ~np.isnan(predictions)
    ax.plot(x_axis, predictions, label=f"{model_name} Prediction", alpha=0.8)
    
    ax.set_title(f"{model_name}: Predictions Over Time", fontsize=14)
    ax.set_xlabel("Date" if dates is not None else "Observation")
    ax.set_ylabel("Beta")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


# ==============================================================================
# Analysis by Regime
# ==============================================================================

def compute_regime_metrics(
    results: Dict[str, Dict],
    regimes: pd.Series,
    target: np.ndarray,
) -> pd.DataFrame:
    """
    Compute the metrics of performance segmented by regime of the market.

    This function generates a table of comparison showing the performance of the models in:
    - All the periods
    - The Calm regime
    - The regime of High Volatility (Crisis)

    Args:
        results: A dictionary mapping the names of the models to the results of the evaluation.
        regimes: A Series containing the labels of the regime ("Calm" or "High Volatility").
        target: The true values of the target.

    Returns:
        A DataFrame containing the MSE by model and by regime.
    """
    regime_metrics = []
    
    regime_array = regimes.values if isinstance(regimes, pd.Series) else regimes
    
    for model_name, result in results.items():
        if "error" in result:
            continue
        
        squared_errors = result["squared_errors"]
        
        # MSE for all the periods
        valid_all = ~np.isnan(squared_errors)
        mse_all = np.mean(squared_errors[valid_all]) if np.any(valid_all) else np.nan
        
        # MSE for the Calm regime
        calm_mask = (regime_array == "Calm") & valid_all
        mse_calm = np.mean(squared_errors[calm_mask]) if np.any(calm_mask) else np.nan
        
        # MSE for the regime of High Volatility
        crisis_mask = (regime_array == "High Volatility") & valid_all
        mse_crisis = np.mean(squared_errors[crisis_mask]) if np.any(crisis_mask) else np.nan
        
        regime_metrics.append({
            "Model": model_name,
            "MSE (All)": mse_all,
            "MSE (Calm)": mse_calm,
            "MSE (Crisis)": mse_crisis,
            "N (All)": np.sum(valid_all),
            "N (Calm)": np.sum(calm_mask),
            "N (Crisis)": np.sum(crisis_mask),
        })
    
    df = pd.DataFrame(regime_metrics)
    df = df.sort_values("MSE (All)").reset_index(drop=True)
    
    return df


def generate_summary_table(
    results: Dict[str, Dict],
    regimes: Optional[pd.Series] = None,
    target: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Generate a comprehensive table of summary comparing all the models.

    Args:
        results: A dictionary of the results of the models.
        regimes: Optional labels of the regime for the segmented analysis.
        target: Optional values of the target.

    Returns:
        A DataFrame containing the comparison of the models.
    """
    summary = []
    
    for model_name, result in results.items():
        if "error" in result or "aggregate" not in result:
            continue
        
        agg = result["aggregate"]
        
        row = {
            "Model": model_name,
            "Mean MSE": agg.get("mean_mse", np.nan),
            "Std MSE": agg.get("std_mse", np.nan),
            "Mean MAE": agg.get("mean_mae", np.nan),
            "R²": agg.get("r2", np.nan),
            "N Predictions": agg.get("n_predictions", 0),
        }
        
        summary.append(row)
    
    df = pd.DataFrame(summary)
    
    if len(df) > 0:
        df = df.sort_values("Mean MSE").reset_index(drop=True)
        
        # Addition of the column of rank
        df.insert(0, "Rank", range(1, len(df) + 1))
    
    return df


# ==============================================================================
# Principal Example of Utilisation
# ==============================================================================

def main() -> None:
    """Example of utilisation of the module of evaluation."""
    print("=" * 70)
    print("WALK-FORWARD EVALUATION DEMONSTRATION")
    print("=" * 70)
    
    from sklearn.linear_model import Ridge
    
    # Generation of synthetic data
    np.random.seed(42)
    n_samples = 1000
    n_features = 5
    
    dates = pd.date_range("2018-01-01", periods=n_samples, freq="B")
    X = np.random.randn(n_samples, n_features)
    true_coef = np.array([0.5, -0.3, 0.2, -0.1, 0.4])
    y = X @ true_coef + np.random.randn(n_samples) * 0.1
    
    # Addition of NaN for the initial period
    X[:100] = np.nan
    y[:100] = np.nan
    
    # Creation of the regimes
    regimes = pd.Series(
        np.where(np.random.rand(n_samples) > 0.7, "High Volatility", "Calm"),
        index=dates
    )
    
    # Initialisation of the evaluator
    evaluator = WalkForwardEvaluator(n_splits=5, test_size=50)
    
    # Evaluation of the model Ridge
    print("\nEvaluating Ridge Regression...")
    ridge = Ridge(alpha=1.0)
    ridge_results = evaluator.evaluate_model(ridge, X, y, dates)
    
    # Creation of the benchmark prediction (naive)
    naive_pred = np.roll(y, 1)
    naive_pred[0] = np.nan
    naive_results = evaluator.evaluate_benchmark(naive_pred, y, dates)
    
    # Compilation of the results
    all_results = {
        "Ridge": ridge_results,
        "Naive": naive_results,
    }
    
    # Generation of the table of summary
    print("\n" + "=" * 70)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 70)
    summary = generate_summary_table(all_results)
    print(summary.to_string(index=False))
    
    # Analysis by regime
    print("\n" + "=" * 70)
    print("REGIME ANALYSIS")
    print("=" * 70)
    regime_df = compute_regime_metrics(all_results, regimes, y)
    print(regime_df.to_string(index=False))
    
    # Plot of the cumulative MSE
    print("\nGenerating cumulative MSE plot...")
    fig = plot_cumulative_mse(all_results, dates, save_path=None)
    plt.close(fig)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
