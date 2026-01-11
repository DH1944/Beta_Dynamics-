"""
Beta Dynamics & Portfolio Resilience
=====================================

Project for the course "Data Science and Advanced Programming"
Master in Finance, HEC Lausanne (UNIL)

This package predicts systematic risk (Beta) for SMI stocks using ML models.
We compare them against academic benchmarks like Welch BSWA and Kalman Filter.

Modules:
- data_loader: loads data from Yahoo Finance or local CSV, handles caching
- features: creates features for ML, uses Numba for speed
- models_benchmarks: Welch BSWA, Kalman Filter, Naive baseline
- models_ml: ML models with cross-validation (Ridge, RF, XGBoost, MLP)
- evaluation: walk-forward validation and metrics
- feature_analysis: feature importance
- statistical_tests: Diebold-Mariano tests
"""

__version__ = "1.5.0"
__author__ = "HEC Lausanne Student"

from src.data_loader import (
    DataLoader,
    SMI_TICKERS,
    HISTORICAL_TICKERS,
    EXIT_DATES,
    MARKET_TICKER,
)
from src.features import FeatureEngineer, benchmark_numba_vs_pandas
from src.models_benchmarks import NaiveBaseline, WelchBSWA, KalmanBeta
from src.models_ml import MLModelPipeline
from src.evaluation import WalkForwardEvaluator
from src.feature_analysis import FeatureImportanceAnalyzer, analyze_all_models
from src.statistical_tests import diebold_mariano_test, compare_all_models_pairwise

__all__ = [
    # Data loading
    "DataLoader",
    "SMI_TICKERS",
    "HISTORICAL_TICKERS",
    "EXIT_DATES",
    "MARKET_TICKER",
    # Feature engineering
    "FeatureEngineer",
    "benchmark_numba_vs_pandas",
    # Benchmark models
    "NaiveBaseline",
    "WelchBSWA",
    "KalmanBeta",
    # ML models
    "MLModelPipeline",
    # Evaluation
    "WalkForwardEvaluator",
    # Analysis
    "FeatureImportanceAnalyzer",
    "analyze_all_models",
    "diebold_mariano_test",
    "compare_all_models_pairwise",
]
