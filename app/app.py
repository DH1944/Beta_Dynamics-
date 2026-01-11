"""
Beta Dynamics - Streamlit Dashboard
====================================

HEC Lausanne - Data Science and Advanced Programming
Master in Finance

Interactive web app to visualize beta predictions and compare models.
Includes Global Results tab for comparing Robust vs Biased batch runs.

Run with: streamlit run app/app.py
"""

import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# Add parent directory so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import DataLoader, SMI_TICKERS, HISTORICAL_TICKERS
from src.features import FeatureEngineer
from src.models_benchmarks import NaiveBaseline, WelchBSWA, KalmanBeta
from src.models_ml import MLModelPipeline
from src.evaluation import WalkForwardEvaluator, generate_summary_table
from src.statistical_tests import diebold_mariano_test

# Import the pure processing function from main
from main import process_single_ticker, TickerResult


# ==============================================================================
# Page Setup
# ==============================================================================

st.set_page_config(
    page_title="Beta Dynamics Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .comparison-better { color: #28a745; font-weight: bold; }
    .comparison-worse { color: #dc3545; font-weight: bold; }
    .metric-highlight {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# Helper Functions
# ==============================================================================

def get_company_name(ticker: str) -> str:
    """Get company name from ticker."""
    names = {
        "NESN.SW": "Nestlé", "ROG.SW": "Roche", "NOVN.SW": "Novartis",
        "UBSG.SW": "UBS Group", "ZURN.SW": "Zurich Insurance", "CFR.SW": "Richemont",
        "ABBN.SW": "ABB", "SIKA.SW": "Sika", "LONN.SW": "Lonza",
        "GIVN.SW": "Givaudan", "ALC.SW": "Alcon", "HOLN.SW": "Holcim",
        "GEBN.SW": "Geberit", "SLHN.SW": "Swiss Life", "SCMN.SW": "Swisscom",
        "KNIN.SW": "Kühne + Nagel", "PGHN.SW": "Partners Group", "SREN.SW": "Swiss Re",
        "LOGN.SW": "Logitech", "SOON.SW": "Sonova",
        "CSGN.SW": "Credit Suisse", "SYST.SW": "Synthes", "ATLN.SW": "Actelion",
        "SYNN.SW": "Syngenta", "RIGN.SW": "Transocean", "BAER.SW": "Julius Bär",
        "UHR.SW": "Swatch Group", "ADEN.SW": "Adecco", "SGSN.SW": "SGS",
    }
    return names.get(ticker, ticker)


def load_batch_report(results_dir: str = "results") -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Load batch reports if they exist.
    
    Returns:
        Tuple of (robust_report, biased_report) DataFrames, None if not found
    """
    results_path = Path(results_dir)
    
    # Try to load reports with different naming conventions
    robust_report = None
    biased_report = None
    
    # Check for robust report (full universe)
    robust_paths = [
        results_path / "final_report.csv",
        results_path / "final_report_robust.csv",
        results_path / "report_31_tickers.csv",
    ]
    for path in robust_paths:
        if path.exists():
            df = pd.read_csv(path)
            # Verify it's the robust version (has historical tickers)
            if "Is_Historical" in df.columns and df["Is_Historical"].any():
                robust_report = df
                break
            elif len(df) > 25:  # Likely robust if > 25 tickers
                robust_report = df
                break
    
    # Check for biased report (current SMI only)
    biased_paths = [
        results_path / "final_report_biased.csv",
        results_path / "report_20_tickers.csv",
        results_path / "final_report_smi2024.csv",
    ]
    for path in biased_paths:
        if path.exists():
            biased_report = pd.read_csv(path)
            break
    
    # If only one report exists and it's small, it might be biased
    if robust_report is None and biased_report is None:
        default_path = results_path / "final_report.csv"
        if default_path.exists():
            df = pd.read_csv(default_path)
            if "Is_Historical" in df.columns:
                if df["Is_Historical"].any():
                    robust_report = df
                else:
                    biased_report = df
            elif len(df) <= 22:
                biased_report = df
            else:
                robust_report = df
    
    return robust_report, biased_report


# ==============================================================================
# Cache Functions
# ==============================================================================

@st.cache_data(ttl=3600)
def load_data(start_date: str, end_date: str, include_historical: bool = False):
    """Load and cache market data."""
    loader = DataLoader(
        start_date=start_date,
        end_date=end_date,
        include_historical=include_historical
    )
    try:
        stock_data, market_data = loader.load_data()
        aligned_data = loader.align_data(stock_data, market_data)
        return aligned_data, True
    except Exception as e:
        st.warning(f"Could not load real data: {e}. Using synthetic data.")
        return generate_synthetic_data(), False


def generate_synthetic_data():
    """Generate synthetic data for demonstration."""
    np.random.seed(42)
    dates = pd.date_range("2015-01-01", "2024-12-31", freq="B")
    n = len(dates)
    market_returns = np.random.randn(n) * 0.01
    data = pd.DataFrame({"Market_Return": market_returns}, index=dates)
    for ticker in SMI_TICKERS[:8]:
        beta = 0.7 + np.random.rand() * 0.6
        data[f"{ticker}_Return"] = beta * market_returns + np.random.randn(n) * 0.005
    return data


@st.cache_data
def compute_betas_for_display(aligned_data: pd.DataFrame, ticker_col: str):
    """Compute betas using benchmark models for visualization."""
    engineer = FeatureEngineer(aligned_data, beta_window=252)
    stock_returns = aligned_data[ticker_col].values
    market_returns = aligned_data["Market_Return"].values
    
    realized_beta = engineer.compute_rolling_beta(ticker_col)
    
    welch = WelchBSWA(window=252, decay_halflife=126, winsor_pct=0.05)
    welch.fit(stock_returns, market_returns)
    
    kalman = KalmanBeta(initial_beta=1.0, state_noise=0.0001)
    kalman.fit(stock_returns, market_returns)
    
    naive = NaiveBaseline(window=252)
    naive.fit(stock_returns, market_returns)
    
    return {
        "Realized": realized_beta.values,
        "Welch_BSWA": welch.predict(stock_returns, market_returns),
        "Kalman": kalman.predict(stock_returns, market_returns),
        "Naive": naive.predict(stock_returns, market_returns),
    }


@st.cache_data
def detect_regimes(aligned_data: pd.DataFrame, threshold: float = 20.0):
    """Detect market regimes based on volatility."""
    vol = pd.Series(aligned_data["Market_Return"]).rolling(21).std() * np.sqrt(252) * 100
    regimes = np.where(vol > threshold, "High Volatility", "Calm")
    return pd.Series(regimes, index=aligned_data.index)


@st.cache_data
def run_full_analysis(_aligned_data: pd.DataFrame, ticker: str, _regimes: pd.Series):
    """Run full analysis using the same logic as main.py."""
    is_historical = ticker in HISTORICAL_TICKERS
    result = process_single_ticker(
        ticker=ticker,
        aligned_data=_aligned_data,
        regimes=_regimes,
        is_historical=is_historical,
        n_splits=10,
        test_size=63,
        verbose=True
    )
    return result


# ==============================================================================
# Global Results Tab
# ==============================================================================

def render_global_results_tab():
    """Render the Global Results comparison tab."""
    st.header("🌍 Global Results: Survivorship Bias Analysis")
    
    st.markdown("""
    This section compares batch analysis results between:
    - **Robust Mode** (`python main.py`): 31 tickers including historical SMI members
    - **Biased Mode** (`python main.py --no-historical`): 20 current SMI members only
    
    The comparison highlights the **impact of survivorship bias** on model performance metrics.
    """)
    
    # File upload section
    st.subheader("📁 Load Reports")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Robust Report** (31 tickers)")
        robust_file = st.file_uploader(
            "Upload final_report.csv (Robust)", 
            type="csv", 
            key="robust_upload",
            help="Generated with: python main.py"
        )
    
    with col2:
        st.markdown("**Biased Report** (20 tickers)")
        biased_file = st.file_uploader(
            "Upload final_report.csv (Biased)", 
            type="csv", 
            key="biased_upload",
            help="Generated with: python main.py --no-historical"
        )
    
    # Try to load from results directory if no upload
    robust_report, biased_report = None, None
    
    if robust_file is not None:
        robust_report = pd.read_csv(robust_file)
    
    if biased_file is not None:
        biased_report = pd.read_csv(biased_file)
    
    # If no uploads, try loading from disk
    if robust_report is None and biased_report is None:
        disk_robust, disk_biased = load_batch_report("results")
        if disk_robust is not None:
            robust_report = disk_robust
            st.info("📂 Loaded Robust report from `results/final_report.csv`")
        if disk_biased is not None:
            biased_report = disk_biased
            st.info("📂 Loaded Biased report from `results/`")
    
    # Display analysis if we have at least one report
    if robust_report is None and biased_report is None:
        st.warning("""
        ⚠️ No batch reports found. To generate them:
        
        ```bash
        # Generate Robust report (31 tickers)
        python main.py --output-dir results
        mv results/final_report.csv results/final_report_robust.csv
        
        # Generate Biased report (20 tickers)
        python main.py --no-historical --output-dir results
        mv results/final_report.csv results/final_report_biased.csv
        ```
        
        Then upload both files above or place them in the `results/` directory.
        """)
        return
    
    st.markdown("---")
    
    # Display individual reports
    if robust_report is not None:
        render_single_report(robust_report, "Robust (31 tickers)", "🟢")
    
    if biased_report is not None:
        render_single_report(biased_report, "Biased (20 tickers)", "🔴")
    
    # Comparative analysis if both reports exist
    if robust_report is not None and biased_report is not None:
        st.markdown("---")
        render_comparison(robust_report, biased_report)


def render_single_report(df: pd.DataFrame, title: str, emoji: str):
    """Render summary for a single report."""
    with st.expander(f"{emoji} {title} - Summary", expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Tickers Analyzed", len(df))
        
        with col2:
            if "Is_Historical" in df.columns:
                hist_count = df["Is_Historical"].sum()
                st.metric("Historical", f"{hist_count}")
            else:
                st.metric("Historical", "0")
        
        with col3:
            mean_welch = df["Welch_MSE"].mean()
            st.metric("Mean Welch MSE", f"{mean_welch:.6f}")
        
        with col4:
            mean_improvement = df["Improvement_vs_Welch_%"].mean()
            st.metric("Mean ML Improvement", f"{mean_improvement:+.2f}%")
        
        # Show detailed table
        st.dataframe(
            df.style.format({
                "Welch_MSE": "{:.6f}",
                "Best_ML_MSE": "{:.6f}",
                "Improvement_vs_Welch_%": "{:+.2f}%",
                "DM_p_value": "{:.4f}",
            }).background_gradient(subset=["Improvement_vs_Welch_%"], cmap="RdYlGn"),
            use_container_width=True,
            height=300
        )


def render_comparison(robust_df: pd.DataFrame, biased_df: pd.DataFrame):
    """Render detailed comparison between robust and biased reports."""
    st.subheader("📊 Comparative Analysis: Robust vs Biased")
    
    # Find common tickers
    robust_tickers = set(robust_df["Ticker"])
    biased_tickers = set(biased_df["Ticker"])
    common_tickers = robust_tickers & biased_tickers
    
    st.write(f"**Common tickers for comparison:** {len(common_tickers)}")
    
    # Aggregate metrics comparison
    st.markdown("### Key Metrics Comparison")
    
    metrics_comparison = []
    
    # Welch MSE
    robust_welch = robust_df["Welch_MSE"].mean()
    biased_welch = biased_df["Welch_MSE"].mean()
    diff_welch = ((biased_welch - robust_welch) / robust_welch) * 100
    metrics_comparison.append({
        "Metric": "Mean Welch MSE",
        "Robust (31)": f"{robust_welch:.6f}",
        "Biased (20)": f"{biased_welch:.6f}",
        "Difference": f"{diff_welch:+.2f}%",
        "Interpretation": "Higher in Biased = Overoptimistic" if diff_welch < 0 else "Lower in Biased = Expected"
    })
    
    # Best ML MSE
    robust_ml = robust_df["Best_ML_MSE"].mean()
    biased_ml = biased_df["Best_ML_MSE"].mean()
    diff_ml = ((biased_ml - robust_ml) / robust_ml) * 100
    metrics_comparison.append({
        "Metric": "Mean Best ML MSE",
        "Robust (31)": f"{robust_ml:.6f}",
        "Biased (20)": f"{biased_ml:.6f}",
        "Difference": f"{diff_ml:+.2f}%",
        "Interpretation": "Lower in Biased = Survivorship bias effect"
    })
    
    # Improvement %
    robust_imp = robust_df["Improvement_vs_Welch_%"].mean()
    biased_imp = biased_df["Improvement_vs_Welch_%"].mean()
    diff_imp = biased_imp - robust_imp
    metrics_comparison.append({
        "Metric": "Mean ML Improvement",
        "Robust (31)": f"{robust_imp:+.2f}%",
        "Biased (20)": f"{biased_imp:+.2f}%",
        "Difference": f"{diff_imp:+.2f}pp",
        "Interpretation": "Higher in Biased = Overestimated ML benefit"
    })
    
    # Significant results
    robust_sig = robust_df["Is_Significant"].mean() * 100
    biased_sig = biased_df["Is_Significant"].mean() * 100
    diff_sig = biased_sig - robust_sig
    metrics_comparison.append({
        "Metric": "% Statistically Significant",
        "Robust (31)": f"{robust_sig:.1f}%",
        "Biased (20)": f"{biased_sig:.1f}%",
        "Difference": f"{diff_sig:+.1f}pp",
        "Interpretation": "Higher in Biased = False confidence"
    })
    
    comparison_df = pd.DataFrame(metrics_comparison)
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    
    # Visual comparison
    st.markdown("### Visual Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # MSE Distribution comparison
        fig_mse = go.Figure()
        fig_mse.add_trace(go.Box(
            y=robust_df["Welch_MSE"], name="Robust", marker_color="#28a745"
        ))
        fig_mse.add_trace(go.Box(
            y=biased_df["Welch_MSE"], name="Biased", marker_color="#dc3545"
        ))
        fig_mse.update_layout(
            title="Welch MSE Distribution",
            yaxis_title="MSE",
            height=400
        )
        st.plotly_chart(fig_mse, use_container_width=True)
    
    with col2:
        # Improvement Distribution comparison
        fig_imp = go.Figure()
        fig_imp.add_trace(go.Box(
            y=robust_df["Improvement_vs_Welch_%"], name="Robust", marker_color="#28a745"
        ))
        fig_imp.add_trace(go.Box(
            y=biased_df["Improvement_vs_Welch_%"], name="Biased", marker_color="#dc3545"
        ))
        fig_imp.update_layout(
            title="ML Improvement Distribution",
            yaxis_title="Improvement (%)",
            height=400
        )
        st.plotly_chart(fig_imp, use_container_width=True)
    
    # Best ML Model distribution
    st.markdown("### Best ML Model Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        robust_models = robust_df["Best_ML_Model"].value_counts()
        fig_r = px.pie(values=robust_models.values, names=robust_models.index, 
                       title="Robust (31 tickers)")
        fig_r.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_r, use_container_width=True)
    
    with col2:
        biased_models = biased_df["Best_ML_Model"].value_counts()
        fig_b = px.pie(values=biased_models.values, names=biased_models.index,
                       title="Biased (20 tickers)")
        fig_b.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_b, use_container_width=True)
    
    # Ticker-by-ticker comparison for common tickers
    if len(common_tickers) > 0:
        st.markdown("### Ticker-by-Ticker Comparison (Common Tickers)")
        
        robust_common = robust_df[robust_df["Ticker"].isin(common_tickers)].set_index("Ticker")
        biased_common = biased_df[biased_df["Ticker"].isin(common_tickers)].set_index("Ticker")
        
        ticker_comparison = []
        for ticker in common_tickers:
            if ticker in robust_common.index and ticker in biased_common.index:
                r_mse = robust_common.loc[ticker, "Welch_MSE"]
                b_mse = biased_common.loc[ticker, "Welch_MSE"]
                r_imp = robust_common.loc[ticker, "Improvement_vs_Welch_%"]
                b_imp = biased_common.loc[ticker, "Improvement_vs_Welch_%"]
                
                ticker_comparison.append({
                    "Ticker": ticker,
                    "Company": get_company_name(ticker),
                    "Robust MSE": r_mse,
                    "Biased MSE": b_mse,
                    "MSE Diff %": ((b_mse - r_mse) / r_mse) * 100 if r_mse > 0 else 0,
                    "Robust Imp%": r_imp,
                    "Biased Imp%": b_imp,
                })
        
        ticker_df = pd.DataFrame(ticker_comparison)
        st.dataframe(
            ticker_df.style.format({
                "Robust MSE": "{:.6f}",
                "Biased MSE": "{:.6f}",
                "MSE Diff %": "{:+.2f}%",
                "Robust Imp%": "{:+.2f}%",
                "Biased Imp%": "{:+.2f}%",
            }),
            use_container_width=True
        )
    
    # Key insights
    st.markdown("### 💡 Key Insights")
    
    # Calculate bias impact
    mse_bias = ((biased_welch - robust_welch) / robust_welch) * 100
    imp_bias = biased_imp - robust_imp
    
    if mse_bias < -5:
        st.error(f"""
        **⚠️ Survivorship Bias Detected!**
        
        The Biased analysis (current SMI only) shows **{abs(mse_bias):.1f}% lower MSE** than the Robust analysis.
        This suggests that excluding failed companies leads to **overoptimistic** performance estimates.
        
        ML improvement appears **{abs(imp_bias):.1f} percentage points higher** in the biased analysis,
        which could mislead investment decisions.
        """)
    elif mse_bias < 0:
        st.warning(f"""
        **⚡ Moderate Survivorship Bias**
        
        Small but measurable difference: Biased MSE is **{abs(mse_bias):.1f}% lower**.
        Consider using the Robust analysis for more reliable conclusions.
        """)
    else:
        st.success("""
        **✓ No Significant Survivorship Bias Detected**
        
        The Robust and Biased analyses show similar results. This could indicate:
        - Historical members had similar risk characteristics
        - The time period analyzed doesn't capture major corporate failures
        """)


# ==============================================================================
# Main Application
# ==============================================================================

def main():
    """Main Streamlit application."""
    
    st.markdown('<p class="main-header">📈 Beta Dynamics Dashboard</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        st.subheader("Date Range")
        start_date = st.date_input("Start Date", value=pd.to_datetime("2015-01-01"))
        end_date = st.date_input("End Date", value=pd.to_datetime("2024-12-31"))
        
        st.markdown("---")
        st.subheader("Stock Selection")
        
        all_tickers = SMI_TICKERS + [t for t in HISTORICAL_TICKERS if t not in SMI_TICKERS]
        selected_ticker = st.selectbox(
            "Select Stock",
            options=all_tickers,
            format_func=lambda x: f"{x} - {get_company_name(x)}" + (" (Historical)" if x in HISTORICAL_TICKERS else "")
        )
        
        st.markdown("---")
        st.subheader("Parameters")
        volatility_threshold = st.slider("Volatility Threshold", 15.0, 30.0, 20.0, 1.0)
        run_ml = st.checkbox("Run ML Models", value=False, help="Enable full ML analysis")
        
        st.markdown("---")
        st.info("This dashboard uses the **same code** as `main.py` for consistency.")
    
    # Load data
    with st.spinner("Loading data..."):
        aligned_data, real_data = load_data(str(start_date), str(end_date), include_historical=True)
    
    ticker_col = f"{selected_ticker}_Return"
    if ticker_col not in aligned_data.columns:
        st.error(f"Data for {selected_ticker} not available.")
        return
    
    with st.spinner("Computing betas..."):
        betas = compute_betas_for_display(aligned_data, ticker_col)
        regimes = detect_regimes(aligned_data, volatility_threshold)
    
    # Tabs - Added Global Results
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Visualization", 
        "📈 Comparison", 
        "🔬 Full Analysis",
        "🌍 Global Results",
        "ℹ️ About"
    ])
    
    with tab1:
        st.header(f"Beta Time Series: {selected_ticker}")
        
        if selected_ticker in HISTORICAL_TICKERS:
            st.warning(f"⚠️ {selected_ticker} is a historical member (no longer in SMI).")
        
        fig = go.Figure()
        colors = {"Realized": "#1f77b4", "Welch_BSWA": "#ff7f0e", "Kalman": "#2ca02c", "Naive": "#d62728"}
        
        for name, values in betas.items():
            fig.add_trace(go.Scatter(
                x=aligned_data.index, y=values, mode='lines', name=name,
                line=dict(color=colors.get(name), width=1.5 if name == "Realized" else 1),
                opacity=1.0 if name == "Realized" else 0.7
            ))
        
        # Add crisis shading
        crisis_mask = regimes == "High Volatility"
        in_crisis = False
        for i, (date, is_crisis) in enumerate(zip(aligned_data.index, crisis_mask)):
            if is_crisis and not in_crisis:
                start = date
                in_crisis = True
            elif not is_crisis and in_crisis:
                fig.add_vrect(x0=start, x1=aligned_data.index[i-1], fillcolor="red", opacity=0.1, line_width=0)
                in_crisis = False
        
        fig.update_layout(title=f"Beta Estimation - {selected_ticker}", xaxis_title="Date", 
                         yaxis_title="Beta", hovermode="x unified", height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.header("Model Performance")
        
        realized = betas["Realized"]
        metrics_data = []
        
        for name, preds in betas.items():
            if name == "Realized":
                continue
            valid = ~(np.isnan(preds) | np.isnan(realized))
            if np.any(valid):
                mse = np.mean((preds[valid] - realized[valid]) ** 2)
                mae = np.mean(np.abs(preds[valid] - realized[valid]))
                calm_mask = (regimes.values == "Calm") & valid
                crisis_mask = (regimes.values == "High Volatility") & valid
                mse_calm = np.mean((preds[calm_mask] - realized[calm_mask]) ** 2) if np.any(calm_mask) else np.nan
                mse_crisis = np.mean((preds[crisis_mask] - realized[crisis_mask]) ** 2) if np.any(crisis_mask) else np.nan
                metrics_data.append({"Model": name, "MSE": mse, "MAE": mae, "MSE (Calm)": mse_calm, "MSE (Crisis)": mse_crisis})
        
        metrics_df = pd.DataFrame(metrics_data)
        st.dataframe(metrics_df.style.format({"MSE": "{:.6f}", "MAE": "{:.6f}", "MSE (Calm)": "{:.6f}", "MSE (Crisis)": "{:.6f}"})
                    .highlight_min(subset=["MSE"], color="lightgreen"), use_container_width=True)
        
        # Bar chart
        fig_bar = go.Figure()
        for col, label in [("MSE", "All"), ("MSE (Calm)", "Calm"), ("MSE (Crisis)", "Crisis")]:
            fig_bar.add_trace(go.Bar(name=label, x=metrics_df["Model"], y=metrics_df[col]))
        fig_bar.update_layout(barmode='group', title="MSE by Regime", height=400)
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with tab3:
        st.header("Full ML Analysis")
        st.markdown("Runs the **complete pipeline** (same as `main.py`).")
        
        if not run_ml:
            st.info("Enable 'Run ML Models' in sidebar.")
        else:
            with st.spinner(f"Analyzing {selected_ticker}..."):
                result = run_full_analysis(aligned_data, selected_ticker, regimes)
            
            if result is None:
                st.error("Analysis failed.")
            else:
                st.success(f"✓ Complete for {selected_ticker}")
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Samples", f"{result.n_samples:,}")
                col2.metric("Welch MSE", f"{result.welch_mse:.6f}")
                col3.metric("Best ML", result.best_ml_model)
                col4.metric("vs Welch", f"{result.improvement_vs_welch_pct:+.2f}%")
                
                st.subheader("Statistical Test (DM)")
                if not np.isnan(result.dm_p_value):
                    st.write(f"DM Statistic: {result.dm_statistic:.3f}, p-value: {result.dm_p_value:.4f}")
                    if result.is_significant:
                        st.success("✓ Improvement is statistically significant (p < 0.05)")
                    else:
                        st.warning("Improvement is not statistically significant")
    
    with tab4:
        render_global_results_tab()
    
    with tab5:
        st.header("About")
        st.markdown("""
        ### Beta Dynamics & Portfolio Resilience
        
        **Key Features**:
        - Uses `process_single_ticker()` from `main.py` for consistency
        - **Global Results** tab compares Robust vs Biased batch analyses
        - Quantifies survivorship bias impact on model performance
        
        **Models**: Naive, Welch BSWA, Kalman, Ridge, Random Forest, XGBoost, MLP
        
        **References**: Welch (2021), Diebold & Mariano (1995)
        
        ---
        
        **Usage for Global Results**:
        ```bash
        # Generate Robust report
        python main.py
        cp results/final_report.csv results/final_report_robust.csv
        
        # Generate Biased report  
        python main.py --no-historical
        cp results/final_report.csv results/final_report_biased.csv
        ```
        
        ---
        HEC Lausanne - Master in Finance
        """)
        
        st.info(f"Data: {aligned_data.index.min().date()} to {aligned_data.index.max().date()} ({len(aligned_data):,} days)")


if __name__ == "__main__":
    main()
