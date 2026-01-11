"""
Module of the Models of Machine Learning
=========================================

HEC Lausanne - Data Science and Advanced Programming
Master in Finance

This module implements the models of Machine Learning for the prediction of the beta:
- Ridge Regression: a linear model with regularisation of type L2
- Random Forest: an ensemble method based on the aggregation of decision trees
- XGBoost: a method of gradient boosting
- MLP: a simple neural network (Multi-Layer Perceptron)

It is important to note that the system utilises GridSearchCV with TimeSeriesSplit for the
cross-validation, which permits us to avoid the data leakage (the model never trains on
future data). Moreover, all the models execute in parallel with the parameter n_jobs=-1
in order to accelerate the computation.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# XGBoost is optional: it is possible that the library is not installed
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("XGBoost not available. XGBoost models will be skipped.")


# ==============================================================================
# Constants of the Module
# ==============================================================================

# Minimum number of samples required for the training of the models
MIN_SAMPLES_FOR_TRAINING = 50

# Grids of parameters for the optimisation of the hyperparameters
# It is important to note that the grids are kept small in order to avoid excessively long training times
PARAM_GRIDS = {
    "ridge": {
        "regressor__alpha": [0.1, 1.0, 10.0],
    },
    "random_forest": {
        "regressor__n_estimators": [50, 100],
        "regressor__max_depth": [5, 10],
        "regressor__min_samples_leaf": [5, 10],
    },
    "xgboost": {
        "regressor__n_estimators": [50, 100],
        "regressor__max_depth": [3, 5],
        "regressor__learning_rate": [0.05, 0.1],
    },
    "mlp": {
        "regressor__hidden_layer_sizes": [(50,), (100,), (50, 25)],
        "regressor__alpha": [0.001, 0.01],
    },
}


# ==============================================================================
# Class of the Pipeline of Machine Learning
# ==============================================================================

class MLModelPipeline:
    """
    Pipeline of Machine Learning for the prediction of the beta.

    This class provides a unified interface for the training, the optimisation,
    and the evaluation of multiple models of ML with a proper cross-validation
    for the time series.

    The principal functionalities of this class are the following:
    - Automatic scaling of the features via StandardScaler
    - GridSearchCV with TimeSeriesSplit, which permits us to avoid the look-ahead bias
    - Configurable parallelism (n_jobs) for the environments of HPC
    - Support for Ridge, Random Forest, XGBoost, and MLP

    Attributes:
        n_splits (int): The number of splits for the cross-validation via TimeSeriesSplit.
        scoring (str): The metric of scoring for the CV (default: neg_mean_squared_error).
        random_state (int): The seed for the reproducibility of the results.
        verbose (int): The level of verbosity for GridSearchCV.
        n_jobs (int): The number of parallel jobs (-1 for all the cores, 1 for single thread).

    Example:
        >>> pipeline = MLModelPipeline(n_splits=5)
        >>> best_model, cv_results = pipeline.train_model('ridge', X_train, y_train)
        >>> predictions = best_model.predict(X_test)
    """

    def __init__(
        self,
        n_splits: int = 5,
        scoring: str = "neg_mean_squared_error",
        random_state: int = 42,
        verbose: int = 0,
        n_jobs: int = -1,
    ) -> None:
        """
        Initialise the Pipeline of Machine Learning.

        Args:
            n_splits: The number of splits for the cross-validation via TimeSeriesSplit.
            scoring: The metric of scoring for GridSearchCV.
            random_state: The seed for the reproducibility of the results.
            verbose: The level of verbosity (0 for silent, 1 for progression, 2 for detailed).
            n_jobs: The number of parallel jobs for the estimators and the CV.
                    It is recommended to utilise -1 for all the cores, or 1 for
                    single-threaded execution (when ProcessPoolExecutor is utilised
                    for the outer parallelism).
        """
        self.n_splits = n_splits
        self.scoring = scoring
        self.random_state = random_state
        self.verbose = verbose
        self.n_jobs = n_jobs
        
        # Initialisation of the TimeSeriesSplit
        self.cv = TimeSeriesSplit(n_splits=n_splits)
        
        # Storage of the fitted models
        self._fitted_models: Dict[str, Any] = {}

    def _get_base_estimator(self, model_name: str) -> BaseEstimator:
        """
        Obtain the base estimator for a given name of model.

        Args:
            model_name: One of 'ridge', 'random_forest', 'xgboost', 'mlp'.

        Returns:
            An unfitted sklearn estimator.

        Raises:
            ValueError: If the name of the model is not recognised.
        """
        estimators = {
            "ridge": Ridge(random_state=self.random_state),
            "random_forest": RandomForestRegressor(
                random_state=self.random_state,
                n_jobs=self.n_jobs  # Parallelism configurable
            ),
            "mlp": MLPRegressor(
                random_state=self.random_state,
                max_iter=2000,  # Increased for a better convergence
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=20,  # Early stopping if no improvement
            ),
        }
        
        if XGBOOST_AVAILABLE:
            estimators["xgboost"] = xgb.XGBRegressor(
                random_state=self.random_state,
                n_jobs=self.n_jobs,  # Parallelism configurable
                verbosity=0,
            )
        
        if model_name.lower() not in estimators:
            available = list(estimators.keys())
            raise ValueError(f"Unknown model: {model_name}. Available: {available}")
        
        return estimators[model_name.lower()]

    def _create_pipeline(self, model_name: str) -> Pipeline:
        """
        Create a Pipeline of sklearn with a scaler and a regressor.

        Args:
            model_name: The identifier of the model.

        Returns:
            An object of type sklearn Pipeline.
        """
        estimator = self._get_base_estimator(model_name)
        
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", estimator),
        ])
        
        return pipeline

    def train_model(
        self,
        model_name: str,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        param_grid: Optional[Dict] = None,
        use_randomized_search: bool = False,
        n_iter: int = 10,
    ) -> Tuple[Pipeline, Dict]:
        """
        Train a model with the optimisation of the hyperparameters.

        This method utilises GridSearchCV (or RandomizedSearchCV) with TimeSeriesSplit
        in order to find the optimal hyperparameters without temporal leakage.

        CRITICAL: The parameter n_jobs=-1 permits the parallel processing across all the cores of the CPU.

        Args:
            model_name: The identifier of the model ('ridge', 'random_forest', 'xgboost', 'mlp').
            X: The matrix of features.
            y: The vector of target.
            param_grid: A custom grid of parameters. The default grid is utilised if None.
            use_randomized_search: If True, RandomizedSearchCV is utilised instead of GridSearchCV.
            n_iter: The number of iterations for RandomizedSearchCV.

        Returns:
            A tuple containing (best_pipeline, cv_results_dict).
            
        Raises:
            ValueError: If the input data is invalid or insufficient.
            AssertionError: If X and y have mismatched lengths.
        """
        # =====================================================================
        # Validation of the Input (Defensive Programming)
        # =====================================================================
        X_arr = np.asarray(X)
        y_arr = np.asarray(y)
        
        assert len(X_arr) == len(y_arr), \
            f"X and y must have same length. Got X={len(X_arr)}, y={len(y_arr)}"
        assert not np.all(np.isnan(y_arr)), \
            "y cannot be entirely NaN"
        assert X_arr.ndim == 2, \
            f"X must be 2-dimensional. Got {X_arr.ndim} dimensions"
        
        # Creation of the pipeline
        pipeline = self._create_pipeline(model_name)
        
        # Obtention of the grid of parameters
        if param_grid is None:
            param_grid = PARAM_GRIDS.get(model_name.lower(), {})
        
        # Handling of the missing data
        X_clean, y_clean = self._clean_data(X, y)
        
        if len(X_clean) < MIN_SAMPLES_FOR_TRAINING:
            raise ValueError(
                f"Insufficient samples after cleaning: {len(X_clean)}. "
                f"Minimum required: {MIN_SAMPLES_FOR_TRAINING}"
            )
        
        if len(X_clean) < self.n_splits + 1:
            raise ValueError(
                f"Not enough samples ({len(X_clean)}) for {self.n_splits} CV splits"
            )
        
        # Selection of the strategy of search
        if use_randomized_search:
            search = RandomizedSearchCV(
                pipeline,
                param_distributions=param_grid,
                n_iter=n_iter,
                cv=self.cv,
                scoring=self.scoring,
                n_jobs=self.n_jobs,  # Parallelism configurable
                verbose=self.verbose,
                random_state=self.random_state,
            )
        else:
            search = GridSearchCV(
                pipeline,
                param_grid=param_grid,
                cv=self.cv,
                scoring=self.scoring,
                n_jobs=self.n_jobs,  # Parallelism configurable
                verbose=self.verbose,
            )
        
        # Fitting of the model
        logger.info(f"Training {model_name} with {'Randomized' if use_randomized_search else 'Grid'}SearchCV...")
        search.fit(X_clean, y_clean)
        
        # Storage of the fitted model
        self._fitted_models[model_name] = search.best_estimator_
        
        # Preparation of the results
        cv_results = {
            "best_params": search.best_params_,
            "best_score": -search.best_score_,  # Conversion to positive MSE
            "cv_results": pd.DataFrame(search.cv_results_),
        }
        
        logger.info(f"  Best MSE: {cv_results['best_score']:.6f}")
        logger.info(f"  Best params: {cv_results['best_params']}")
        
        return search.best_estimator_, cv_results

    def train_all_models(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
    ) -> Dict[str, Tuple[Pipeline, Dict]]:
        """
        Train all the available models of Machine Learning.

        Args:
            X: The matrix of features.
            y: The vector of target.

        Returns:
            A dictionary mapping the names of the models to the tuples (pipeline, cv_results).
        """
        results = {}
        
        # Mapping from the display names to the internal identifiers of the models
        model_mapping = {
            "Ridge": "ridge",
            "RandomForest": "random_forest",
            "MLP": "mlp",
        }
        if XGBOOST_AVAILABLE:
            model_mapping["XGBoost"] = "xgboost"
        
        for display_name, internal_name in model_mapping.items():
            try:
                pipeline, cv_results = self.train_model(internal_name, X, y)
                results[display_name] = (pipeline, cv_results)
            except ValueError as e:
                logger.warning(f"Invalid data for {display_name}: {e}")
            except np.linalg.LinAlgError as e:
                logger.warning(f"Linear algebra error for {display_name}: {e}")
            except MemoryError as e:
                logger.error(f"Memory error for {display_name}: {e}")
            except Exception as e:
                logger.error(f"Unexpected error training {display_name}: {e}")
                # Re-raise of the unexpected errors in the debug mode
                if self.verbose > 1:
                    raise
        
        return results

    def predict(
        self,
        model_name: str,
        X: Union[np.ndarray, pd.DataFrame],
    ) -> np.ndarray:
        """
        Generate the predictions utilising a fitted model.

        Args:
            model_name: The name of the fitted model.
            X: The matrix of features.

        Returns:
            An array of predictions.

        Raises:
            ValueError: If the model has not been fitted.
        """
        if model_name not in self._fitted_models:
            raise ValueError(f"Model '{model_name}' has not been fitted. Call train_model first.")
        
        # Handling of the NaN in the features
        X_array = np.asarray(X)
        predictions = np.full(len(X_array), np.nan)
        
        valid_mask = ~np.any(np.isnan(X_array), axis=1)
        if np.any(valid_mask):
            predictions[valid_mask] = self._fitted_models[model_name].predict(X_array[valid_mask])
        
        return predictions

    def get_feature_importance(
        self,
        model_name: str,
        feature_names: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Extract the importance of the features from a fitted model.

        Args:
            model_name: The name of the fitted model.
            feature_names: A list of the names of the features (optional).

        Returns:
            A DataFrame containing the importances of the features sorted in descending order.
        """
        if model_name not in self._fitted_models:
            raise ValueError(f"Model '{model_name}' has not been fitted.")
        
        pipeline = self._fitted_models[model_name]
        regressor = pipeline.named_steps["regressor"]
        
        # Extraction of the importances based on the type of model
        if hasattr(regressor, "feature_importances_"):
            # Models based on trees (RF, XGBoost)
            importances = regressor.feature_importances_
        elif hasattr(regressor, "coef_"):
            # Linear models (Ridge)
            importances = np.abs(regressor.coef_)
        else:
            # MLP: utilisation of the weights of the first layer
            if hasattr(regressor, "coefs_"):
                importances = np.abs(regressor.coefs_[0]).sum(axis=1)
            else:
                return pd.DataFrame({"feature": [], "importance": []})
        
        # Creation of the DataFrame
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importances))]
        
        df = pd.DataFrame({
            "feature": feature_names,
            "importance": importances
        })
        
        return df.sort_values("importance", ascending=False).reset_index(drop=True)

    @staticmethod
    def _clean_data(
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Suppress the rows containing NaN values.

        Args:
            X: The matrix of features.
            y: The vector of target.

        Returns:
            A tuple of cleaned arrays (X, y).
        """
        X_array = np.asarray(X)
        y_array = np.asarray(y)
        
        # Identification of the valid rows: no NaN in X or y
        valid_X = ~np.any(np.isnan(X_array), axis=1)
        valid_y = ~np.isnan(y_array)
        valid_mask = valid_X & valid_y
        
        return X_array[valid_mask], y_array[valid_mask]


# ==============================================================================
# Custom Wrapper Compatible with Sklearn for the Benchmark Models
# ==============================================================================

class BenchmarkWrapper(BaseEstimator, RegressorMixin):
    """
    Wrapper compatible with sklearn for the benchmark models.

    This class permits the utilisation of the benchmark models (Naive, Welch, Kalman)
    in the pipelines of sklearn and in the frameworks of evaluation.

    Attributes:
        benchmark_model: An instance of a subclass of BetaEstimator.
        market_returns_col: The index of the column of the market returns in the matrix of features.

    Example:
        >>> from src.models_benchmarks import WelchBSWA
        >>> wrapper = BenchmarkWrapper(WelchBSWA(window=252))
        >>> wrapper.fit(X_train, y_train)
        >>> predictions = wrapper.predict(X_test)
    """

    def __init__(
        self,
        benchmark_model: Any,
        market_returns_idx: int = -1,
    ) -> None:
        """
        Initialise the wrapper.

        Args:
            benchmark_model: An instance of the benchmark model.
            market_returns_idx: The index of the column of the market returns in X.
        """
        self.benchmark_model = benchmark_model
        self.market_returns_idx = market_returns_idx
        self._fitted = False

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        stock_returns: Optional[np.ndarray] = None,
        market_returns: Optional[np.ndarray] = None,
    ) -> "BenchmarkWrapper":
        """
        Fit the benchmark model.

        It is important to note that for the benchmark models, the method 'fit' often
        requires the original returns of the stock and the market, not just the features.

        Args:
            X: The matrix of features (which may contain the market returns as a column).
            y: The target (the actual values of beta or the stock returns).
            stock_returns: An optional array of the stock returns.
            market_returns: An optional array of the market returns.

        Returns:
            Self.
        """
        self._fitted = True
        self._X_fitted = X
        self._y_fitted = y
        self._stock_returns = stock_returns
        self._market_returns = market_returns
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate the predictions.

        Args:
            X: The matrix of features.

        Returns:
            An array of the predictions of beta.
        """
        if not self._fitted:
            raise ValueError("Model must be fitted before predict.")
        
        # For the benchmark models, the actual returns are required
        if self._stock_returns is not None and self._market_returns is not None:
            return self.benchmark_model.predict(
                self._stock_returns[:len(X)],
                self._market_returns[:len(X)]
            )
        
        # Fallback: return of the stored predictions
        n = len(X)
        return np.full(n, np.nan)


# ==============================================================================
# Utilities of Comparison
# ==============================================================================

def compare_models(
    results: Dict[str, Tuple[Pipeline, Dict]],
    metric: str = "mse"
) -> pd.DataFrame:
    """
    Create a table of comparison of all the trained models.

    Args:
        results: A dictionary from the method train_all_models().
        metric: The metric to compare (default: 'mse').

    Returns:
        A DataFrame containing the comparison of the models.
    """
    comparison = []
    
    for model_name, (pipeline, cv_results) in results.items():
        row = {
            "model": model_name,
            "best_cv_mse": cv_results["best_score"],
            "best_params": str(cv_results["best_params"]),
        }
        comparison.append(row)
    
    df = pd.DataFrame(comparison)
    df = df.sort_values("best_cv_mse").reset_index(drop=True)
    
    return df


# ==============================================================================
# Principal Example of Utilisation
# ==============================================================================

def main() -> None:
    """Example of utilisation of the pipeline of the models of Machine Learning."""
    print("=" * 70)
    print("ML MODEL PIPELINE DEMONSTRATION")
    print("=" * 70)
    
    # Generation of synthetic data
    np.random.seed(42)
    n_samples = 1500
    n_features = 10
    
    # Synthetic features and target
    X = np.random.randn(n_samples, n_features)
    true_coef = np.random.randn(n_features) * 0.5
    y = X @ true_coef + np.random.randn(n_samples) * 0.1
    
    # Addition of NaN for realism
    X[:252, :] = np.nan  # The first year has no features (rolling window)
    y[:252] = np.nan
    
    # Initialisation of the pipeline
    print("\nInitializing ML Pipeline...")
    pipeline = MLModelPipeline(n_splits=5, verbose=0)
    
    # Training of all the models
    print("\nTraining all models...")
    results = pipeline.train_all_models(X, y)
    
    # Comparison of the results
    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)
    comparison = compare_models(results)
    print(comparison.to_string(index=False))
    
    # Importance of the features for the best model
    best_model = comparison.iloc[0]["model"]
    print(f"\nFeature Importance for {best_model}:")
    importance = pipeline.get_feature_importance(best_model)
    print(importance.head(5).to_string(index=False))
    
    print("\nDone!")


if __name__ == "__main__":
    main()
