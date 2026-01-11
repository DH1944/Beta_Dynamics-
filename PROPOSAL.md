# Project Proposal: Beta Dynamics & Portfolio Resilience

## Objective

The principal objective of this project is to compare the performance of robust econometric benchmarks against Machine Learning (ML) models for the estimation of the dynamic beta of the stocks. Indeed, the scope is strictly restricted to the prediction of the risk (Beta) on the components of the SMI, excluding all prediction of Alpha or static hedging assets.

## Methodology

The methodology confronts two approaches:

1. **Benchmarks:** The BSWA model of Welch (2021) serves as a strict reference (replacing the OLS), completed by the filter of Kalman.

2. **ML Models:** Ridge, Random Forest, XGBoost and MLP are utilized to capture the non-linearities.

Moreover, the strategy of features integrates market data and a deterministic definition of the regimes (threshold of volatility) to avoid the instability of the clustering. The implementation follows an architecture oriented object with HPC optimization via Numba.

## Data

The data (Yahoo Finance) include the current and historical constituents (ex: CS Group): this permits the neutralization of the bias of survival.

## Evaluation

Finally, the evaluation relies on a validation "walk-forward", measuring the precision (MSE/MAE) and the statistical significativity (Diebold-Mariano) against the BSWA, with a specific analysis by regime of market (calm vs volatile).
