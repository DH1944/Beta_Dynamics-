# Beta Dynamics & Portfolio Resilience

**Version 2.26**

Course: Advanced Programming 2025 (HEC Lausanne)  
Student: Daniel Haas  
Student ID: 21401542

---

## Research Question

**Which predictive model performs best for the estimation of the systematic risk (Beta) of the constituents of the Swiss Market Index: traditional econometric benchmarks or Machine Learning approaches?**

This project aims to determine whether advanced ML models (Ridge, Random Forest, XGBoost, MLP) can outperform standard financial benchmarks (Naive, Welch BSWA, Kalman Filter) in the prediction of beta, while accounting for the survivorship bias through the incorporation of historical data from delisted companies.

---

## Abstract

This project implements a comprehensive system for the estimation and prediction of the systematic risk (Beta) of the constituents of the Swiss Market Index (SMI).

The core academic contribution of this work resides in the rigorous comparison between **standard financial benchmarks** (Naive, Welch BSWA, Kalman Filter) and **Machine Learning models** (Ridge, Random Forest, XGBoost, MLP). Moreover, the project addresses the problem of **survivorship bias** in a significant manner: it incorporates historical data from 11 former members of the SMI (e.g., Credit Suisse, Actelion), which permits us to ensure a robust and realistic evaluation of the performance of the models across different market regimes.

---

## Key Features

- **Survivorship Bias Correction:** The system integrates historical CSV data for delisted companies, which permits us to prevent look-ahead bias in the context of long-term analysis.
- **Advanced Benchmarking:** The project implements sophisticated financial models, notably the Welch BSWA (Slope-Winsorized Age-Decayed) and the Kalman Filter for the estimation of beta.
- **High-Performance Computing:** The utilisation of Numba (`@jit`) allows us to accelerate the critical calculations of the rolling windows (Beta, Volatility).
- **Rigorous Evaluation:** The system features a Walk-Forward Validation framework and Diebold-Mariano statistical tests, which permit the determination of the significance of the differences in performance.
- **Interactive Dashboard:** A Streamlit web application is included in order to visualise the dynamics of beta and to compare the metrics of the models in a dynamic manner.

---

## Installation & Setup

> **Note to Graders:** It is important to note that the project is entirely self-contained. Please follow the steps detailed below in order to replicate the results in a fresh environment.

### Requirements

| Requirement | Version |
|-------------|---------|
| Python | 3.10+ |
| pandas | ≥ 2.0.0 |
| numpy | ≥ 1.24.0 |
| scipy | ≥ 1.11.0 |
| scikit-learn | ≥ 1.3.0 |
| xgboost | ≥ 2.0.0 |
| yfinance | ≥ 0.2.36 |
| pykalman | ≥ 0.9.5 |
| numba | ≥ 0.58.0 |
| streamlit | ≥ 1.29.0 |
| pytest | ≥ 7.4.0 |

The complete list of dependencies with their exact versions is available in `requirements.txt`.

### Step-by-Step Installation

**1. Clone the Repository**

Navigate to the root of the project:

```bash
cd beta_dynamics
```

**2. Create a Virtual Environment (Recommended)**

```bash
# macOS / Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
.\venv\Scripts\activate
```

**3. Install the Dependencies**

All the required libraries (including `yfinance`, `scikit-learn`, `xgboost`, `streamlit`) are listed in the file `requirements.txt`.

```bash
pip install -r requirements.txt
```

---

## Usage Guide

### Running the Full Analysis (CLI)

The primary entry point of the application is `main.py`. By default, this script executes a batch analysis on all the current and historical tickers of the SMI.

```bash
python main.py
```

The results are saved in the directory `results/`, including a summary CSV report which permits the ranking of the models by the improvement of MSE.

### Advanced Options (CLI Arguments)

| Option | Description |
|--------|-------------|
| `--ticker NESN.SW` | Analyse a single ticker (enables single-ticker mode) |
| `--smi-2024-only` | Use only the current 20 members of the SMI (no historical data) |
| `--no-historical` | Deactivate the correction of the survivorship bias |
| `--no-cache` | Force the re-download of data |
| `--output-dir PATH` | Specify the directory of output (default: `results`) |
| `--workers N` | Limit the number of parallel workers for the batch mode |
| `--sequential` | Deactivate the multiprocessing (use sequential processing) |
| `--target-window N` | Rolling window for the target beta in days (default: 63) |

**Examples:**

```bash
python main.py                      # Batch mode (31 tickers)
python main.py --smi-2024-only      # Batch mode (20 tickers, no historical)
python main.py --ticker NESN.SW     # Single ticker mode
python main.py --workers 4          # Limit parallel workers
```

### Launching the Dashboard

In order to explore the results in an interactive manner (visualise the time-series of Beta, compare the "Robust" versus "Biased" datasets):

```bash
streamlit run app/app.py
```

Open your browser at `http://localhost:8501` to visualise the dashboard.

### Running the Tests

In order to execute the unit and integration tests of the project, it is necessary to utilise `pytest`. The following command permits the launching of all the tests:

```bash
pytest tests/
```

Moreover, it is possible to obtain a verbose output with the coverage of the code:

```bash
pytest tests/ -v --cov=src
```

### Expected Output

Upon execution of `main.py`, the system produces the following outputs in the directory `results/`:

- A comprehensive CSV report ranking all models by MSE improvement relative to the benchmarks
- Statistical significance tests (Diebold-Mariano) comparing each ML model against Welch BSWA
- Visualisations of the beta time-series and cumulative MSE comparisons (in single-ticker mode)

The console output displays the progress of the analysis, the detection of market regimes, and a summary of the performance of the models.

---

## Repository Structure

The architecture of the project follows a modular organisation:

```
beta_dynamics_V26/
├── main.py                  # CLI Entry Point: orchestrates the pipeline
├── requirements.txt         # Dependencies of the project
├── README.md                # This documentation
├── TECHNICAL_REPORT.pdf     # Full technical report
├── AI_USAGE.md              # Disclosure of the usage of AI tools
├── PROPOSAL.md              # Original proposal of the project
│
├── data/
│   ├── raw/                 # Cached data from Yahoo Finance
│   └── historical/          # CSV files for former members of SMI (CSGN, etc.)
│
├── src/                     # Core Source Code
│   ├── __init__.py          # Initialisation of the module
│   ├── data_loader.py       # Handles the ingestion of data & survivorship bias
│   ├── features.py          # Feature engineering accelerated by Numba
│   ├── feature_analysis.py  # Analysis and selection of features
│   ├── models_benchmarks.py # Naive, Welch BSWA, Kalman Filter
│   ├── models_ml.py         # ML Pipeline (Ridge, RF, XGB, MLP)
│   ├── evaluation.py        # Walk-forward validation & metrics
│   └── statistical_tests.py # Diebold-Mariano significance tests
│
├── app/
│   └── app.py               # Streamlit Interactive Dashboard
│
└── tests/                   # Unit and Integration tests (pytest)
```

---

## Methodology

### 1. Data Ingestion (`src/data_loader.py`)

The module of data ingestion is responsible for the following operations:

- Fetching of live data via the API of `yfinance`
- Loading of historical CSV files for the delisted firms
- Correction of the duplication of columns and alignment of the time series, which permits us to prevent the "look-ahead" bias

### 2. Feature Engineering (`src/features.py`)

This module computes the rolling volatility, the momentum, and the lagged betas. Indeed, the utilisation of Numba allows us to achieve high-performance calculation of the Rolling OLS Beta.

### 3. Modeling

**Benchmarks:**

| Model | Description |
|-------|-------------|
| Naive | Assumption of the Random Walk (β_t = β_{t-1}) |
| Welch BSWA | Beta with slope winsorisation and age-dependent decay |
| Kalman Filter | Dynamic state-space estimation |

**Machine Learning (`src/models_ml.py`):**

| Model | Type |
|-------|------|
| Ridge | Linear |
| Random Forest | Ensemble |
| XGBoost | Ensemble |
| MLP | Neural Network |

All the ML models are optimised via TimeSeriesSplit Cross-Validation.

### 4. Evaluation (`src/evaluation.py`)

- **Walk-Forward Validation:** This approach simulates the conditions of real-world trading: no future data is utilised in the process.
- **Metric:** Mean Squared Error (MSE)
- **Statistical Significance:** The Diebold-Mariano test (`src/statistical_tests.py`) permits the comparison of the ML predictions against the benchmark of Welch.

---

## Results

### Summary of the Performance of the Models

The evaluation was conducted on 31 tickers (20 current members of the SMI + 11 historical constituents) using Walk-Forward Validation. The following table presents the Mean Squared Errors (MSE) for three archetypes of assets, illustrating the divergence of performance according to the regime of the asset:

| Profile | Ticker | Naive MSE | Welch MSE | Kalman MSE | Best ML MSE | Winner | Improvement vs Welch |
|---------|--------|-----------|-----------|------------|-------------|--------|----------------------|
| Distress / Bankruptcy | CSGN.SW | 0.154 | 0.388 | 0.214 | 0.117 | Ridge | **+69.8%** |
| Stable / Industrial | SGSN.SW | 0.019 | 0.018 | 0.012 | 0.021 | Kalman | -16.6% (ML fails) |
| Refuge / Macro | NESN.SW | 0.014 | 0.013 | 0.010 | 0.011 | Ridge/XGB | +11.8% |

### Impact of the Survivorship Bias

The aggregated analysis reveals the amplitude of the bias:

- The average MSE of the Welch benchmark passes from **0.065** (Robust) to **0.052** (Biased)
- The exclusion of the historical assets masks approximately **20%** of the real risk of the market
- The average improvement of the ML diminishes (from 8.33% to 7.82%) if one ignores the bankruptcies, because it is precisely on these extreme events that the ML generates the most of value (+69.8% on CSGN)

### Conclusion

**The results demonstrate a dichotomy of performance marked according to the regime of the asset:**

1. **In situation of financial distress:** The Ridge regression emerges as the best performer. Indeed, on the case of Credit Suisse (CSGN), the model Ridge outperforms the benchmark of Welch by 69.8%, reducing drastically the MSE from 0.388 to 0.117.

2. **On stable assets (Blue Chips):** The Kalman Filter remains the gold standard. On SGS (SGSN), it achieves the lowest MSE of 0.012, while the ML models introduce more variance than bias (-16.6%).

3. **Key insight:** The ML models do not serve to beat the market on the daily basis, but to offer a critical resilience during the changes of regime (bankruptcies or macroeconomic shocks). Moreover, it is concluded that the exclusion of the disappeared assets biases the evaluation of the risk of market of nearly 20%.

---

## AI Usage Policy

In compliance with the rules of the course, the utilisation of AI assistants (e.g., ChatGPT, Copilot) for debugging, code optimisation, and the drafting of documentation is fully detailed in the following document:

📄 **[AI_USAGE.md](AI_USAGE.md)**

---

*Project submitted for the Final Evaluation of Advanced Programming 2025.*
