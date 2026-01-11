# Usage of Artificial Intelligence

## 1. General Overview

This document details the usage of Artificial Intelligence tools during the development of the project **Beta Dynamics & Portfolio Resilience**. In accordance with the course policy, the assistants Claude, Gemini, and GitHub Copilot were utilized strictly as learning aids to assist with debugging, optimization, and architectural guidance. This report documents the specific contexts where these tools were applied, ensuring transparency and academic integrity.

## 2. Detailed Usage by Technical Domain

### A. Software Architecture & Observability

**From Scripts to Application:** The project required a robust user interface involving complex argument parsing. GitHub Copilot was utilized to rapidly generate boilerplate code for the CLI and to propose reactive patterns for the Streamlit dashboard.

**Data Structures:** To ensure type safety beyond standard dictionaries, Claude was consulted regarding best practices for financial data objects. It recommended implementing `@dataclass` for the `TickerResult` structure, enforcing strict typing throughout the codebase.

**Industrial Observability:** To implement a persistent logging system, Gemini guided the configuration of Python's `logging` library. This enabled separation of INFO and ERROR streams with timestamps, ensuring proper traceability.

### B. High-Performance Computing (HPC)

**Just-In-Time Compilation (Numba):** Confronted with performance bottlenecks in Pandas rolling windows, Claude was consulted for optimization strategies. It introduced Just-In-Time compilation with Numba, demonstrating how to decorate functions to compile explicit loops into machine code.

**Thread Management:** When the application experienced CPU saturation, Gemini helped diagnose a thread oversubscription conflict between Python's multiprocessing and NumPy's internal threads. It proposed modifying environment variables such as `OPENBLAS_NUM_THREADS=1` to resolve the issue.

### C. Data Pipeline & Finance Specificities

**Survivorship Bias:** Claude was utilized to validate the logic for delisted stocks, ensuring that data is properly truncated at the exit date to prevent look-ahead bias.

**Complex Debugging:** Gemini helped diagnose a complex Pandas indexing error during historical data merging. It identified that the merge logic created duplicate columns and proposed suppressing specific columns before renaming.

**Smart Caching:** GitHub Copilot was relied upon to implement decorator logic for the caching system, automating verification of CSV file existence before triggering Yahoo Finance API calls.

### D. Mathematical Modeling

**Dynamic State-Space Models:** For the Kalman Filter implementation, Claude was used to translate the mathematical state-space equations (Prediction and Update steps) into vectorized Python code found in the `KalmanBeta` class.

### E. Rigorous Statistical Validation

**Time-Series Validation:** To avoid data leakage in time-series evaluation, Gemini was asked for a statistically sound alternative to K-Fold. It explained Walk-Forward Validation and helped design a splitter respecting chronological order.

**Hypothesis Testing:** Claude was utilized to understand and implement the Diebold-Mariano test, enabling rigorous statistical comparison of models.

### F. Quality of Code & Optimization of Documentation

Indeed, in strict compliance with the policy of "Good Uses" defined in the rulebook of the course (specifically regarding the categories of Code review suggestions and Documentation writing), the tools of AI were utilized to ameliorate the readability and the professional standard of the project.

**Standardization of the Code:** It is important to note that the central logic, the methodology of finance, and the decisions of architecture reflect my own engineering decisions. However, the assistants permitted us to standardize the style of the writing (compliance with PEP8), to refactor the blocks of code that were repetitive, and to propose names of variables that are more explicit: this ensures a structure of code that is rigorous and easy to maintain.

**Refinement of the Documentation:** Moreover, as a student who is not a native speaker of English, I utilized the artificial intelligence to verify the grammar and the syntax of the docstrings and the comments. This approach permits us to guarantee that the technical explanations are accurate and clear for the correctors: this explains the high level of linguistic consistency that can be observed in the files of the source code.

## 3. Examples of Prompts

To illustrate the nature of the interactions:

> **Claude:** "Explain why standard K-Fold is unsuitable for financial time series and propose a Python implementation for a Walk-Forward splitter."

> **Gemini:** "My CPU usage is at 100% but processing is slow. Could there be a conflict between ProcessPoolExecutor and MKL threads?"

> **Claude:** "Explain the mathematics behind the Diebold-Mariano test and how to implement it in Python."

> **GitHub Copilot:** *(Comment)* `# Decorator to check if the file exists in the cache directory before downloading`

## 4. Declaration of Academic Integrity

I certify the following:

- **Understanding:** I possess complete understanding of the submitted code. AI tools were utilized to explain concepts and optimize specific functions, not to blindly generate core logic.

- **Verification:** All optimizations proposed by AI (e.g., Numba vs. Pandas) were verified and benchmarked.

- **Originality:** Architectural decisions, financial methodology, and the overall project structure reflect my own engineering decisions.
