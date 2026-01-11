#!/usr/bin/env python3
"""
Beta Dynamics & Portfolio Resilience: Principal Script of Execution
====================================================================

HEC Lausanne - Data Science and Advanced Programming
Master in Finance

This script constitutes the principal entry point of the application, and it permits
the execution of the complete pipeline for the prediction of the systematic risk (Beta).
Indeed, the system supports two distinct modes of operation:
- Batch mode: this mode permits the analysis of all the tickers and the production of a consolidated report
- Single ticker mode: this mode is utilised for the detailed analysis and the debugging of individual stocks

The principal functionalities of this module are the following:
- Multiprocessing for the parallel analysis of the tickers (via ProcessPoolExecutor)
- Walk-forward validation in order to prevent any contamination of the data (data leakage)
- Correction of the survivorship bias via the integration of the historical members of the SMI

Usage:
    python main.py                      # Batch mode (31 tickers, with correction of the survivorship bias)
    python main.py --smi-2024-only      # Batch mode (20 current members of the SMI)
    python main.py --ticker NESN.SW     # Single ticker mode (legacy)
    python main.py --workers 4          # Limitation of the number of parallel workers
"""

# =============================================================================
# CRITICAL: Configuration of the limits of the threads BEFORE the importation of numpy/sklearn
# It is important to note that this configuration permits us to prevent the warnings of
# OpenBLAS/MKL from polluting the bars of progression of tqdm during the utilisation
# of the multiprocessing (race condition on the standard output)
# =============================================================================
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["BLIS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import argparse
import logging
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Addition of the directory src to the path in order to permit the importation of the internal modules
sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import DataLoader, SMI_TICKERS, HISTORICAL_TICKERS, EXIT_DATES
from src.features import FeatureEngineer
from src.models_benchmarks import NaiveBaseline, WelchBSWA, KalmanBeta
from src.models_ml import MLModelPipeline, compare_models
from src.evaluation import (
    WalkForwardEvaluator,
    plot_cumulative_mse,
    compute_regime_metrics,
    generate_summary_table,
)
from src.statistical_tests import diebold_mariano_test

# Tentative of importation of tqdm for the bars of progression
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    # Fallback function in the case where tqdm is not installed
    def tqdm(iterable, **kwargs):
        return iterable

# Suppression of the warnings in order to maintain a clean output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Configuration of the logging system
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ==============================================================================
# Data Classes for the Encapsulation of the Results
# ==============================================================================

@dataclass
class TickerResult:
    """
    Container class for the encapsulation of the results of the analysis of a single ticker.
    
    Indeed, this dataclass permits the structured storage of all the metrics computed
    during the analysis, including the performance of the benchmarks, the performance
    of the models of Machine Learning, and the results of the statistical tests.
    """
    ticker: str
    is_historical: bool
    n_samples: int
    
    # Metrics of the benchmark models
    naive_mse: float
    welch_mse: float
    kalman_mse: float
    
    # Information concerning the best model of Machine Learning
    best_ml_model: str
    best_ml_mse: float
    
    # Comparison with the benchmark of Welch
    improvement_vs_welch_pct: float
    
    # Results of the statistical test of Diebold-Mariano: best ML versus Welch
    dm_statistic: float
    dm_p_value: float
    is_significant: bool
    
    # Ensemble of the results for the detailed analysis (optional)
    all_results: Optional[Dict] = None


# ==============================================================================
# Pure Function: Processing of a Single Ticker
# ==============================================================================

def process_single_ticker(
    ticker: str,
    aligned_data: pd.DataFrame,
    regimes: pd.Series,
    is_historical: bool = False,
    n_splits: int = 10,
    test_size: int = 63,
    verbose: bool = False,
    n_jobs: int = -1,
    target_window: int = 63
) -> Optional[TickerResult]:
    """
    Process a single ticker and return the results of the analysis.
    
    It is important to note that this function is a pure function: it does not produce
    any output to the console (unless the parameter verbose is set to True), and it
    does not generate any side effects. The function returns None in the case where
    the processing of the ticker fails.
    
    Args:
        ticker: The symbol of the stock (for example, "NESN.SW").
        aligned_data: A DataFrame containing the columns Market_Return and {ticker}_Return.
        regimes: A Series containing the labels of the market regimes.
        is_historical: A boolean indicating if the ticker is a former member of the SMI.
        n_splits: The number of splits for the walk-forward validation.
        test_size: The size of each window of test.
        verbose: If True, the function prints the information of progression.
        n_jobs: The number of parallel jobs for the training of the ML models.
                It is recommended to utilise 1 when the function is called from the
                workers of the multiprocessing in order to avoid the oversubscription of the threads.
        target_window: The rolling window for the calculation of the target beta (default: 63 days).
                      Indeed, shorter windows correspond to a more difficult task of prediction.
        
    Returns:
        An object of type TickerResult, or None if the processing has failed.
    """
    ticker_col = f"{ticker}_Return"
    
    # Verification of the existence of the ticker in the data
    if ticker_col not in aligned_data.columns:
        logger.warning(f"Ticker {ticker} not found in data")
        return None
    
    # Verification of the sufficiency of the data
    valid_returns = aligned_data[ticker_col].dropna()
    if len(valid_returns) < 500:
        logger.warning(f"Ticker {ticker}: insufficient data ({len(valid_returns)} < 500)")
        return None
    
    try:
        # Engineering of the features: the precomputed regimes are passed in order to avoid the API calls
        engineer = FeatureEngineer(aligned_data, beta_window=252)
        features, target = engineer.create_features_for_stock(
            ticker_col,
            shift_target=True,
            include_regime=True,
            precomputed_regimes=regimes,  # Transmission of the pre-computed regimes
            target_window=target_window   # A shorter window corresponds to a more difficult task
        )
        
        # Extraction of the returns for the benchmark models
        stock_returns = aligned_data[ticker_col].values
        market_returns = aligned_data["Market_Return"].values
        
        # =======================================================================
        # Benchmark Models: Initialisation and Training
        # =======================================================================
        benchmark_models = {
            "Naive": NaiveBaseline(window=252),
            "Welch_BSWA": WelchBSWA(window=252, decay_halflife=126, winsor_pct=0.05),
            "Kalman": KalmanBeta(initial_beta=1.0, state_noise=0.0001),
        }
        
        benchmark_predictions = {}
        for name, model in benchmark_models.items():
            model.fit(stock_returns, market_returns)
            benchmark_predictions[name] = model.predict(stock_returns, market_returns)
        
        # =======================================================================
        # Models of Machine Learning: Training and Optimisation
        # =======================================================================
        X = features.values
        y = target.values
        
        # Cleaning of the data: suppression of the rows containing NaN values
        valid_mask = ~(np.any(np.isnan(X), axis=1) | np.isnan(y))
        X_clean = X[valid_mask]
        y_clean = y[valid_mask]
        
        n_samples = len(X_clean)
        
        if n_samples < 500:
            logger.warning(f"Ticker {ticker}: insufficient clean samples ({n_samples} < 500)")
            return None
        
        # Training of the ML models: the parameter n_jobs permits the control of the parallelism
        ml_pipeline = MLModelPipeline(n_splits=5, verbose=0, n_jobs=n_jobs)
        ml_results = ml_pipeline.train_all_models(X_clean, y_clean)
        
        # =======================================================================
        # Evaluation via Walk-Forward Validation
        # =======================================================================
        evaluator = WalkForwardEvaluator(n_splits=n_splits, test_size=test_size)
        
        all_results = {}
        
        # Evaluation of the benchmark models
        for name, predictions in benchmark_predictions.items():
            result = evaluator.evaluate_benchmark(
                predictions,
                target.values,
                dates=aligned_data.index
            )
            all_results[name] = result
        
        # Evaluation of the models of Machine Learning
        for name, (pipeline, _) in ml_results.items():
            try:
                result = evaluator.evaluate_model(
                    pipeline,
                    X,
                    y,
                    dates=aligned_data.index
                )
                all_results[f"ML_{name}"] = result
            except Exception as e:
                logger.warning(f"Ticker {ticker}: ML model {name} failed: {e}")
        
        # =======================================================================
        # Computation of the Metrics: FAIR COMPARISON on the common valid indices
        # =======================================================================
        
        # Verification that all the results of the benchmarks exist
        required_benchmarks = ["Naive", "Welch_BSWA", "Kalman"]
        for bench in required_benchmarks:
            if bench not in all_results:
                logger.error(f"Ticker {ticker}: Missing benchmark result '{bench}'")
                return None
        
        # Identification of the models of Machine Learning
        ml_models = {k: v for k, v in all_results.items() if k.startswith("ML_")}
        
        if not ml_models:
            logger.error(f"Ticker {ticker}: No ML models succeeded")
            return None
        
        # CRITICAL: Identification of the indices where ALL the models have valid predictions
        # This approach permits us to ensure a fair comparison of the MSE
        n_points = len(target.values)
        common_valid = np.ones(n_points, dtype=bool)
        
        # Verification of the validity of the target
        common_valid &= ~np.isnan(target.values)
        
        # Verification of all the benchmarks
        for bench in required_benchmarks:
            if "predictions" in all_results[bench]:
                common_valid &= ~np.isnan(all_results[bench]["predictions"])
        
        # Verification of all the models of Machine Learning
        for ml_name in ml_models:
            if "predictions" in ml_models[ml_name]:
                common_valid &= ~np.isnan(ml_models[ml_name]["predictions"])
        
        n_common = common_valid.sum()
        if n_common < 100:
            logger.warning(f"Ticker {ticker}: Only {n_common} common valid points, skipping")
            return None
        
        # Recalculation of the MSE on the common valid indices for a FAIR comparison
        target_common = target.values[common_valid]
        
        # MSE of the benchmarks on the common indices
        naive_pred_common = all_results["Naive"]["predictions"][common_valid]
        welch_pred_common = all_results["Welch_BSWA"]["predictions"][common_valid]
        kalman_pred_common = all_results["Kalman"]["predictions"][common_valid]
        
        naive_mse = np.mean((target_common - naive_pred_common) ** 2)
        welch_mse = np.mean((target_common - welch_pred_common) ** 2)
        kalman_mse = np.mean((target_common - kalman_pred_common) ** 2)
        
        # MSE of the ML models on the common indices
        ml_mse_common = {}
        for ml_name, ml_result in ml_models.items():
            ml_pred_common = ml_result["predictions"][common_valid]
            ml_mse_common[ml_name] = np.mean((target_common - ml_pred_common) ** 2)
        
        # Identification of the best model of Machine Learning
        best_ml_name = min(ml_mse_common.keys(), key=lambda k: ml_mse_common[k])
        best_ml_mse = ml_mse_common[best_ml_name]
        best_ml_model = best_ml_name.replace("ML_", "")
        
        # Computation of the improvement versus the benchmark of Welch (on the same indices)
        if welch_mse > 0:
            improvement_pct = ((welch_mse - best_ml_mse) / welch_mse) * 100
        else:
            improvement_pct = 0.0
        
        # =======================================================================
        # Test of Diebold-Mariano: best ML versus Welch, on the common indices
        # =======================================================================
        dm_statistic = np.nan
        dm_p_value = np.nan
        is_significant = False
        
        if best_ml_model != "None":
            try:
                # Utilisation of the same mask common_valid for the test of DM
                welch_pred_common = all_results["Welch_BSWA"]["predictions"][common_valid]
                ml_pred_common = all_results[f"ML_{best_ml_model}"]["predictions"][common_valid]
                
                # Computation of the errors on the common indices: actual minus predicted
                welch_errors = target_common - welch_pred_common
                ml_errors = target_common - ml_pred_common
                
                # Execution of the test of DM: the function expects the errors and returns a tuple
                dm_statistic, dm_p_value = diebold_mariano_test(
                    welch_errors,
                    ml_errors,
                    loss_function="mse"
                )
                
                is_significant = dm_p_value < 0.05 if not np.isnan(dm_p_value) else False
                
            except Exception as e:
                logger.warning(f"Ticker {ticker}: DM test failed: {e}")
        
        return TickerResult(
            ticker=ticker,
            is_historical=is_historical,
            n_samples=n_common,  # Utilisation of the count of the common valid samples
            naive_mse=naive_mse,
            welch_mse=welch_mse,
            kalman_mse=kalman_mse,
            best_ml_model=best_ml_model,
            best_ml_mse=best_ml_mse,
            improvement_vs_welch_pct=improvement_pct,
            dm_statistic=dm_statistic,
            dm_p_value=dm_p_value,
            is_significant=is_significant,
            all_results=all_results if verbose else None
        )
        
    except Exception as e:
        logger.error(f"Ticker {ticker}: processing failed: {e}")
        return None


# ==============================================================================
# Batch Processing with Multiprocessing
# ==============================================================================

def _process_ticker_worker(
    ticker: str,
    aligned_data: pd.DataFrame,
    regimes: pd.Series,
    is_historical: bool,
    verbose: bool = False,
    target_window: int = 63
) -> Optional[TickerResult]:
    """
    Worker function for the multiprocessing of the tickers.
    
    It is important to note that this function is defined at the level of the module
    (and not as a method of a class), which permits its serialisation via pickle
    for the utilisation with ProcessPoolExecutor.
    
    Moreover, the limits of the threads are configured at the level of the module
    (before the importation of numpy), which permits us to prevent the warnings
    of OpenBLAS/MKL from polluting the output of the console.
    
    Args:
        ticker: The symbol of the ticker to process.
        aligned_data: A DataFrame containing the aligned returns.
        regimes: A Series containing the regimes of the market.
        is_historical: A boolean indicating if the ticker is historical (delisted).
        verbose: If True, the detailed output is printed.
        target_window: The rolling window for the calculation of the target beta.
        
    Returns:
        An object of type TickerResult if the processing is successful, None otherwise.
    """
    # Suppression of all the logging in the worker processes in order to maintain tqdm clean
    logging.getLogger().setLevel(logging.CRITICAL)
    
    # Suppression of the warnings which might escape to stderr
    warnings.filterwarnings("ignore")
    
    return process_single_ticker(
        ticker=ticker,
        aligned_data=aligned_data,
        regimes=regimes,
        is_historical=is_historical,
        verbose=verbose,
        n_jobs=1,  # Single-threaded in order to avoid the oversubscription
        target_window=target_window
    )


def run_batch_analysis(
    aligned_data: pd.DataFrame,
    regimes: pd.Series,
    tickers: List[str],
    historical_tickers: List[str],
    output_dir: Path,
    verbose: bool = False,
    max_workers: Optional[int] = None,
    use_multiprocessing: bool = True,
    target_window: int = 63
) -> pd.DataFrame:
    """
    Execute the batch analysis on all the tickers with parallel processing.
    
    This function utilises ProcessPoolExecutor in order to parallelise the analysis
    of the tickers across the cores of the CPU. Indeed, each ticker is processed in
    a separate process, which permits us to bypass the GIL and to obtain a true
    parallelism for the tasks of training of the ML models which are CPU-bound.
    
    Args:
        aligned_data: A DataFrame containing all the aligned returns.
        regimes: A Series containing the regimes of the market.
        tickers: A list of the current tickers of the SMI.
        historical_tickers: A list of the historical tickers.
        output_dir: The directory where the results will be saved.
        verbose: If True, the detailed progression is printed.
        max_workers: The maximum number of parallel processes (default: CPU count minus 1).
        use_multiprocessing: If False, the sequential processing is utilised.
        target_window: The rolling window for the calculation of the target beta.
        
    Returns:
        A DataFrame containing the consolidated results of the analysis.
    """
    results = []
    failed_tickers = []
    
    # Combination of all the tickers
    all_tickers = list(tickers)
    historical_set = set(historical_tickers)
    
    # Addition of the historical tickers which are present in the data
    for t in historical_tickers:
        if t not in all_tickers and f"{t}_Return" in aligned_data.columns:
            all_tickers.append(t)
    
    n_tickers = len(all_tickers)
    
    # Determination of the number of workers
    if max_workers is None:
        # Utilisation of the number of CPUs minus 1 in order to leave one core free for the system
        max_workers = max(1, cpu_count() - 1)
    
    print(f"\nProcessing {n_tickers} tickers...")
    
    if use_multiprocessing and n_tickers > 1:
        # =======================================================================
        # PARALLEL PROCESSING with ProcessPoolExecutor
        # =======================================================================
        print(f"Using {max_workers} parallel workers (ProcessPoolExecutor)")
        print(f"Target beta window: {target_window} days")
        print("-" * 50)
        
        # Preparation of the arguments of the tasks
        tasks = [
            (ticker, ticker in historical_set)
            for ticker in all_tickers
        ]
        
        # Processing with the bar of progression
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submission of all the tasks
            futures = {
                executor.submit(
                    _process_ticker_worker,
                    ticker,
                    aligned_data,
                    regimes,
                    is_hist,
                    verbose,
                    target_window
                ): ticker
                for ticker, is_hist in tasks
            }
            
            # Collection of the results as they complete
            if TQDM_AVAILABLE:
                pbar = tqdm(total=n_tickers, desc="Analyzing tickers", unit="ticker")
            
            for future in as_completed(futures):
                ticker = futures[future]
                try:
                    result = future.result()
                    if result is not None:
                        results.append(result)
                    else:
                        failed_tickers.append(ticker)
                except Exception as e:
                    logger.error(f"Ticker {ticker}: worker failed with {e}")
                    failed_tickers.append(ticker)
                
                if TQDM_AVAILABLE:
                    pbar.update(1)
            
            if TQDM_AVAILABLE:
                pbar.close()
    else:
        # =======================================================================
        # SEQUENTIAL PROCESSING (fallback mode)
        # =======================================================================
        print("Using sequential processing")
        print(f"Target beta window: {target_window} days")
        print("-" * 50)
        
        iterator = tqdm(all_tickers, desc="Analyzing tickers", unit="ticker") if TQDM_AVAILABLE else all_tickers
        
        for ticker in iterator:
            is_hist = ticker in historical_set
            
            result = process_single_ticker(
                ticker=ticker,
                aligned_data=aligned_data,
                regimes=regimes,
                is_historical=is_hist,
                verbose=verbose,
                target_window=target_window
            )
            
            if result is not None:
                results.append(result)
            else:
                failed_tickers.append(ticker)
    
    # Construction of the DataFrame of the report
    if not results:
        print("\n⚠ No tickers processed successfully!")
        return pd.DataFrame()
    
    report_data = []
    for r in results:
        report_data.append({
            "Ticker": r.ticker,
            "Is_Historical": r.is_historical,
            "N_Samples": r.n_samples,
            "Naive_MSE": r.naive_mse,
            "Welch_MSE": r.welch_mse,
            "Kalman_MSE": r.kalman_mse,
            "Best_ML_Model": r.best_ml_model,
            "Best_ML_MSE": r.best_ml_mse,
            "Improvement_vs_Welch_%": r.improvement_vs_welch_pct,
            "DM_Statistic": r.dm_statistic,
            "DM_p_value": r.dm_p_value,
            "Is_Significant": r.is_significant,
        })
    
    report_df = pd.DataFrame(report_data)
    
    # Sorting by the improvement in descending order
    report_df = report_df.sort_values("Improvement_vs_Welch_%", ascending=False)
    
    # Saving of the report
    report_path = output_dir / "final_report.csv"
    report_df.to_csv(report_path, index=False)
    
    # Printing of the summary
    print("\n" + "=" * 70)
    print("BATCH ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\n✓ Processed: {len(results)} tickers")
    if failed_tickers:
        print(f"✗ Failed: {len(failed_tickers)} tickers: {failed_tickers}")
    
    print(f"\n✓ Report saved: {report_path}")
    
    # Statistics of the summary
    print("\n" + "-" * 50)
    print("SUMMARY STATISTICS")
    print("-" * 50)
    
    print(f"\nMean Welch MSE: {report_df['Welch_MSE'].mean():.6f}")
    print(f"Mean Best ML MSE: {report_df['Best_ML_MSE'].mean():.6f}")
    print(f"Mean Improvement: {report_df['Improvement_vs_Welch_%'].mean():.2f}%")
    
    sig_count = report_df["Is_Significant"].sum()
    print(f"\nStatistically significant improvements: {sig_count}/{len(report_df)} ({100*sig_count/len(report_df):.1f}%)")
    
    # Distribution of the best model of Machine Learning
    print("\nBest ML Model Distribution:")
    model_counts = report_df["Best_ML_Model"].value_counts()
    for model, count in model_counts.items():
        print(f"  {model}: {count} tickers ({100*count/len(report_df):.1f}%)")
    
    return report_df


# ==============================================================================
# Legacy Single Ticker Mode: Detailed Analysis for Debugging
# ==============================================================================

def run_single_ticker_analysis(
    ticker: str,
    aligned_data: pd.DataFrame,
    regimes: pd.Series,
    output_dir: Path,
    save_plots: bool = True,
    is_historical: bool = False
) -> None:
    """
    Execute the detailed analysis for a single ticker (legacy mode).
    
    This function prints the detailed output to the console and saves the graphical
    representations. Indeed, this mode is principally utilised for the debugging
    and the detailed inspection of the results.
    """
    import matplotlib.pyplot as plt
    from src.feature_analysis import analyze_all_models
    from src.statistical_tests import (
        compare_all_models_pairwise,
        generate_significance_summary,
        rank_models_by_wins
    )
    
    print("\n" + "=" * 70)
    print(f"SINGLE TICKER ANALYSIS: {ticker}")
    print("=" * 70)
    
    ticker_col = f"{ticker}_Return"
    
    if ticker_col not in aligned_data.columns:
        print(f"\n✗ ERROR: Ticker {ticker} not found in data")
        available = [c.replace("_Return", "") for c in aligned_data.columns 
                    if c.endswith("_Return") and c != "Market_Return"]
        print(f"Available tickers: {available[:10]}...")
        return
    
    # Engineering of the features
    print("\n[1/6] Feature Engineering...")
    engineer = FeatureEngineer(aligned_data, beta_window=252)
    features, target = engineer.create_features_for_stock(
        ticker_col,
        shift_target=True,
        include_regime=True
    )
    print(f"  ✓ Features: {list(features.columns)}")
    print(f"  ✓ Samples: {len(features)}")
    
    # Training of the benchmark models
    print("\n[2/6] Training Benchmark Models...")
    stock_returns = aligned_data[ticker_col].values
    market_returns = aligned_data["Market_Return"].values
    
    benchmark_models = {
        "Naive": NaiveBaseline(window=252),
        "Welch_BSWA": WelchBSWA(window=252, decay_halflife=126, winsor_pct=0.05),
        "Kalman": KalmanBeta(initial_beta=1.0, state_noise=0.0001),
    }
    
    benchmark_predictions = {}
    for name, model in benchmark_models.items():
        model.fit(stock_returns, market_returns)
        predictions = model.predict(stock_returns, market_returns)
        benchmark_predictions[name] = predictions
        
        valid = ~(np.isnan(predictions) | np.isnan(target.values))
        if np.any(valid):
            mse = np.mean((predictions[valid] - target.values[valid]) ** 2)
            print(f"  ✓ {name}: MSE = {mse:.6f}")
    
    # Training of the models of Machine Learning
    print("\n[3/6] Training ML Models...")
    X = features.values
    y = target.values
    
    valid_mask = ~(np.any(np.isnan(X), axis=1) | np.isnan(y))
    X_clean = X[valid_mask]
    y_clean = y[valid_mask]
    
    print(f"  Valid samples: {len(X_clean)}/{len(X)}")
    
    ml_pipeline = MLModelPipeline(n_splits=5, verbose=0)
    ml_results = ml_pipeline.train_all_models(X_clean, y_clean)
    
    comparison = compare_models(ml_results)
    print("\n  ML Model Comparison (CV MSE):")
    print(comparison.to_string(index=False))
    
    # Evaluation via Walk-Forward Validation
    print("\n[4/6] Walk-Forward Evaluation...")
    evaluator = WalkForwardEvaluator(n_splits=10, test_size=63)
    
    all_results = {}
    
    for name, predictions in benchmark_predictions.items():
        result = evaluator.evaluate_benchmark(
            predictions, target.values, dates=aligned_data.index
        )
        all_results[name] = result
    
    for name, (pipeline, _) in ml_results.items():
        try:
            result = evaluator.evaluate_model(
                pipeline, X, y, dates=aligned_data.index
            )
            all_results[f"ML_{name}"] = result
        except Exception as e:
            print(f"  ⚠ {name} failed: {e}")
    
    # Summary of the results
    print("\n[5/6] Results Summary...")
    summary = generate_summary_table(all_results)
    print("\n" + summary.to_string(index=False))
    
    # Analysis by regime
    regime_df = compute_regime_metrics(all_results, regimes, target.values)
    print("\nRegime Analysis:")
    print(regime_df.to_string(index=False))
    
    # Statistical tests
    print("\n[6/6] Statistical Tests (Diebold-Mariano)...")
    dm_test_data = {name: {"predictions": res["predictions"]} 
                   for name, res in all_results.items()}
    
    try:
        dm_results = compare_all_models_pairwise(
            results_dict=dm_test_data,
            target=target.values,
            alpha=0.05
        )
        
        if len(dm_results) > 0:
            print("\n" + dm_results.to_string(index=False))
            print("\n" + generate_significance_summary(dm_results))
    except Exception as e:
        print(f"  ⚠ DM tests failed: {e}")
    
    # Saving of the results
    summary.to_csv(output_dir / f"{ticker}_summary.csv", index=False)
    regime_df.to_csv(output_dir / f"{ticker}_regime.csv", index=False)
    
    # Generation of the graphical representations
    if save_plots:
        print("\nGenerating plots...")
        
        # Cumulative MSE
        fig = plot_cumulative_mse(
            all_results,
            dates=aligned_data.index,
            title=f"Cumulative MSE - {ticker}",
            save_path=str(output_dir / f"{ticker}_cumulative_mse.png")
        )
        plt.close(fig)
        
        # Time series of the beta
        fig, ax = plt.subplots(figsize=(14, 6))
        
        realized_beta = engineer.compute_rolling_beta(ticker_col)
        ax.plot(realized_beta.index, realized_beta.values,
                label="Realized Beta", color="black", linewidth=1.5)
        
        ax.plot(aligned_data.index, benchmark_predictions["Welch_BSWA"],
                label="Welch BSWA", color="blue", alpha=0.7)
        
        crisis_mask = regimes == "High Volatility"
        ymin, ymax = ax.get_ylim()
        ax.fill_between(
            aligned_data.index, ymin, ymax,
            where=crisis_mask, alpha=0.2, color="red",
            label="High Volatility"
        )
        
        ax.set_title(f"Beta Estimation: {ticker}", fontsize=14)
        ax.set_xlabel("Date")
        ax.set_ylabel("Beta")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / f"{ticker}_beta.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        
        print(f"  ✓ Plots saved to {output_dir}/")
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


# ==============================================================================
# Main Entry Point of the Application
# ==============================================================================

def main(
    mode: str = "batch",
    ticker: Optional[str] = None,
    smi_2024_only: bool = False,
    force_refresh: bool = False,
    save_plots: bool = True,
    output_dir: str = "results",
    include_historical: bool = True,
    max_workers: Optional[int] = None,
    use_multiprocessing: bool = True,
    target_window: int = 63,
) -> None:
    """
    Principal entry point of the pipeline of the application.
    
    This function orchestrates the complete execution of the analysis, from the
    loading of the data to the generation of the final report.
    
    Args:
        mode: The mode of execution, either "batch" or "single".
        ticker: The symbol of the stock for the single mode.
        smi_2024_only: If True, only the current 20 members of the SMI are utilised.
        force_refresh: If True, the data is re-downloaded.
        save_plots: If True, the graphical representations are saved.
        output_dir: The directory of output for the results.
        include_historical: If True, the historical members are included.
        max_workers: The maximum number of parallel workers (None corresponds to CPU count minus 1).
        use_multiprocessing: If True, ProcessPoolExecutor is utilised.
        target_window: The rolling window for the target beta (63 for quarterly, 252 for annual).
    """
    start_time = datetime.now()
    
    print("=" * 70)
    print("BETA DYNAMICS & PORTFOLIO RESILIENCE")
    print("=" * 70)
    print(f"Date: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Mode: {mode.upper()}")
    print(f"Target Beta Window: {target_window} days")
    
    if smi_2024_only:
        print("Universe: SMI 2024 (20 current members)")
        include_historical = False
    else:
        print(f"Universe: {'Full (31 tickers)' if include_historical else 'Current SMI only'}")
    
    # Creation of the directory of output
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Loading of the data
    print("\n" + "-" * 50)
    print("LOADING DATA")
    print("-" * 50)
    
    loader = DataLoader(
        start_date="2010-01-01",
        end_date="2024-12-31",
        include_historical=include_historical,
    )
    
    if include_historical:
        print("\nHistorical members (survivorship bias correction):")
        historical_info = loader.get_historical_members_info()
        for _, row in historical_info.iterrows():
            status = "✓" if row["csv_available"] else "✗"
            print(f"  {status} {row['ticker']:10s} (exit: {row['exit_date']})")
    
    try:
        stock_data, market_data = loader.load_data(force_refresh=force_refresh)
        aligned_data = loader.align_data(stock_data, market_data)
        
        current_count = sum(1 for t in stock_data if t in loader.tickers)
        historical_count = len(loader._loaded_historical)
        
        print(f"\n✓ Loaded {len(stock_data)} stocks")
        print(f"  - Current SMI: {current_count}")
        if include_historical:
            print(f"  - Historical: {historical_count}")
        print(f"✓ Trading days: {len(aligned_data)}")
        print(f"✓ Period: {aligned_data.index.min().date()} to {aligned_data.index.max().date()}")
        
    except Exception as e:
        print(f"\n✗ Data loading failed: {e}")
        print("Cannot proceed without data. Exiting.")
        return
    
    # Detection of the market regimes
    print("\nDetecting market regimes...")
    engineer = FeatureEngineer(aligned_data, beta_window=252)
    
    try:
        regimes = engineer.detect_market_regime(threshold=20.0)
    except Exception:
        vol = aligned_data["Market_Return"].rolling(21).std() * np.sqrt(252) * 100
        vol = vol.ffill()
        regimes = pd.Series(
            np.where(vol > 20, "High Volatility", "Calm"),
            index=aligned_data.index
        )
    
    crisis_days = (regimes == "High Volatility").sum()
    print(f"  - High Volatility: {crisis_days} days ({100*crisis_days/len(regimes):.1f}%)")
    print(f"  - Calm: {len(regimes) - crisis_days} days")
    
    # Execution of the analysis
    if mode == "single" and ticker:
        # Single ticker mode (legacy)
        is_hist = ticker in HISTORICAL_TICKERS
        run_single_ticker_analysis(
            ticker=ticker,
            aligned_data=aligned_data,
            regimes=regimes,
            output_dir=output_path,
            save_plots=save_plots,
            is_historical=is_hist
        )
    else:
        # Batch mode
        tickers_to_process = SMI_TICKERS if smi_2024_only else SMI_TICKERS
        historical_to_process = [] if smi_2024_only else HISTORICAL_TICKERS
        
        report_df = run_batch_analysis(
            aligned_data=aligned_data,
            regimes=regimes,
            tickers=tickers_to_process,
            historical_tickers=historical_to_process,
            output_dir=output_path,
            verbose=False,
            max_workers=max_workers,
            use_multiprocessing=use_multiprocessing,
            target_window=target_window
        )
    
    # Completion of the execution
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print("\n" + "=" * 70)
    print(f"COMPLETE - Duration: {duration:.1f}s")
    print(f"Results: {output_path}/")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Beta Dynamics & Portfolio Resilience Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                      # Batch mode (31 tickers)
  python main.py --smi-2024-only      # Batch mode (20 tickers)
  python main.py --ticker NESN.SW     # Single ticker mode
        """
    )
    
    # Selection of the mode
    parser.add_argument(
        "--ticker",
        type=str,
        default=None,
        help="Single ticker to analyze (enables single-ticker mode)"
    )
    parser.add_argument(
        "--smi-2024-only",
        action="store_true",
        help="Use only current 20 SMI members (no historical)"
    )
    
    # Options concerning the data
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Force re-download of data"
    )
    parser.add_argument(
        "--no-historical",
        action="store_true",
        help="Disable survivorship bias correction"
    )
    
    # Options concerning the output
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory (default: results)"
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plot generation (single-ticker mode)"
    )
    
    # Options concerning the parallelisation
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Max parallel workers for batch mode (default: CPU count - 1)"
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Disable multiprocessing (use sequential processing)"
    )
    
    # Options concerning the models
    parser.add_argument(
        "--target-window",
        type=int,
        default=63,
        help="Rolling window for target beta in days (default: 63). "
             "Shorter = harder prediction task. Options: 21, 63, 126, 252"
    )
    
    args = parser.parse_args()
    
    # Determination of the mode
    if args.ticker:
        mode = "single"
    else:
        mode = "batch"
    
    # Handling of the flag --smi-2024-only
    include_historical = not args.no_historical
    if args.smi_2024_only:
        include_historical = False
    
    main(
        mode=mode,
        ticker=args.ticker,
        smi_2024_only=args.smi_2024_only,
        force_refresh=args.no_cache,
        save_plots=not args.no_plots,
        output_dir=args.output_dir,
        include_historical=include_historical,
        max_workers=args.workers,
        use_multiprocessing=not args.sequential,
        target_window=args.target_window,
    )
