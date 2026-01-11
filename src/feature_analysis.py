"""
Feature Importance Analysis Module
===================================

HEC Lausanne - Data Science and Advanced Programming
Master in Finance

This module analyzes which features are most important for predicting beta.
It works with tree models (RF, XGBoost) and linear models (Ridge).
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

# Seaborn makes nicer plots but it's optional
try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
    logger.warning("Seaborn not installed. Using basic matplotlib plots.")

# SHAP is optional (more advanced feature importance)
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.info("SHAP not installed. Feature importance will use basic methods only.")
    logger.info("Install with: pip install shap")


class FeatureImportanceAnalyzer:
    """
    Analyze feature importance for ML models.
    
    We support:
    - Tree models: use the built-in feature_importances_
    - Linear models: use absolute value of coefficients
    - SHAP values: if the library is installed
    
    Attributes:
        feature_names: Names of the features
    
    Example:
        >>> analyzer = FeatureImportanceAnalyzer(['Vol_21d', 'Mom_63d', 'Beta_Lag1'])
        >>> importance_df = analyzer.get_importance(rf_model, 'RandomForest')
        >>> analyzer.plot_importance({'RF': importance_df})
    """
    
    def __init__(self, feature_names: List[str]) -> None:
        """
        Initialize analyzer.
        
        Args:
            feature_names: List of feature column names
        """
        self.feature_names = feature_names
    
    def get_importance(
        self, 
        model,
        model_name: str = "Model"
    ) -> pd.DataFrame:
        """
        Extract feature importance from any sklearn-compatible model.
        
        Args:
            model: Fitted sklearn model or pipeline
            model_name: Name for display
            
        Returns:
            DataFrame with features sorted by importance
            
        Raises:
            ValueError: If model doesn't support importance extraction
        """
        # Handle pipelines
        if hasattr(model, 'named_steps'):
            # sklearn Pipeline - get the regressor step
            if 'regressor' in model.named_steps:
                model = model.named_steps['regressor']
            else:
                # Try to get the last step
                model = list(model.named_steps.values())[-1]
        
        # Get importance scores based on model type
        if hasattr(model, 'feature_importances_'):
            # Tree-based models (RandomForest, XGBoost, GradientBoosting)
            importances = model.feature_importances_
            method = "tree_importance"
        elif hasattr(model, 'coef_'):
            # Linear models (Ridge, Lasso, ElasticNet)
            importances = np.abs(model.coef_)
            if importances.ndim > 1:
                importances = importances.flatten()
            method = "abs_coefficient"
        else:
            raise ValueError(
                f"{model_name} ({type(model).__name__}) does not have "
                "feature_importances_ or coef_ attribute"
            )
        
        # Validate dimensions
        if len(importances) != len(self.feature_names):
            logger.warning(
                f"Mismatch: {len(importances)} importances vs "
                f"{len(self.feature_names)} features. Using available data."
            )
            n = min(len(importances), len(self.feature_names))
            importances = importances[:n]
            features = self.feature_names[:n]
        else:
            features = self.feature_names
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'Feature': features,
            'Importance': importances,
            'Model': model_name,
            'Method': method
        }).sort_values('Importance', ascending=False).reset_index(drop=True)
        
        # Normalize to percentages
        total = importance_df['Importance'].sum()
        if total > 0:
            importance_df['Importance_Pct'] = (importance_df['Importance'] / total) * 100
        else:
            importance_df['Importance_Pct'] = 0.0
        
        return importance_df
    
    def plot_importance(
        self,
        importance_dict: Dict[str, pd.DataFrame],
        top_n: int = 10,
        figsize: Tuple[int, int] = (12, 6),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot feature importance comparison across models.
        
        Args:
            importance_dict: Dict mapping model names to importance DataFrames
            top_n: Number of top features to show
            figsize: Figure size (width, height)
            save_path: Path to save figure (optional)
            
        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Combine all importances
        all_importance = pd.concat(importance_dict.values(), ignore_index=True)
        
        # Get top N features (by max importance across models)
        top_features = (
            all_importance
            .groupby('Feature')['Importance']
            .max()
            .nlargest(top_n)
            .index
            .tolist()
        )
        
        # Filter to top features
        plot_data = all_importance[
            all_importance['Feature'].isin(top_features)
        ].copy()
        
        # Order features by average importance
        feature_order = (
            plot_data
            .groupby('Feature')['Importance']
            .mean()
            .sort_values(ascending=True)
            .index
            .tolist()
        )
        
        # Plot
        if SEABORN_AVAILABLE:
            sns.barplot(
                data=plot_data,
                x='Importance',
                y='Feature',
                hue='Model',
                order=feature_order,
                ax=ax,
                palette='Set2'
            )
        else:
            # Fallback to basic matplotlib
            models = plot_data['Model'].unique()
            n_models = len(models)
            bar_height = 0.8 / n_models
            
            for i, model in enumerate(models):
                model_data = plot_data[plot_data['Model'] == model]
                y_pos = np.arange(len(feature_order))
                values = [
                    model_data[model_data['Feature'] == f]['Importance'].values[0]
                    if f in model_data['Feature'].values else 0
                    for f in feature_order
                ]
                ax.barh(y_pos + i * bar_height, values, height=bar_height, label=model)
            
            ax.set_yticks(np.arange(len(feature_order)) + 0.4)
            ax.set_yticklabels(feature_order)
            ax.legend()
        
        ax.set_title(
            f'Top {top_n} Feature Importance - Model Comparison', 
            fontsize=14, 
            fontweight='bold'
        )
        ax.set_xlabel('Importance Score')
        ax.set_ylabel('Feature')
        
        if SEABORN_AVAILABLE:
            ax.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature importance plot saved: {save_path}")
        
        return fig
    
    def get_shap_importance(
        self,
        model,
        X: np.ndarray,
        model_name: str = "Model",
        max_samples: int = 1000
    ) -> Optional[pd.DataFrame]:
        """
        Compute SHAP-based feature importance (if SHAP is available).
        
        Args:
            model: Fitted model
            X: Feature matrix
            model_name: Name for display
            max_samples: Maximum samples for SHAP computation
            
        Returns:
            DataFrame with SHAP importance, or None if SHAP unavailable
        """
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not available. Install with: pip install shap")
            return None
        
        try:
            # Subsample for performance
            if len(X) > max_samples:
                idx = np.random.choice(len(X), max_samples, replace=False)
                X_sample = X[idx]
            else:
                X_sample = X
            
            # Remove NaN
            valid_mask = ~np.any(np.isnan(X_sample), axis=1)
            X_sample = X_sample[valid_mask]
            
            # Create explainer
            if hasattr(model, 'feature_importances_'):
                # Tree-based
                explainer = shap.TreeExplainer(model)
            else:
                # Use KernelExplainer as fallback
                explainer = shap.KernelExplainer(
                    model.predict,
                    shap.sample(X_sample, 100)
                )
            
            # Compute SHAP values
            shap_values = explainer.shap_values(X_sample)
            
            # Mean absolute SHAP values
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            
            mean_shap = np.abs(shap_values).mean(axis=0)
            
            # Create DataFrame
            importance_df = pd.DataFrame({
                'Feature': self.feature_names,
                'Importance': mean_shap,
                'Model': model_name,
                'Method': 'shap'
            }).sort_values('Importance', ascending=False).reset_index(drop=True)
            
            return importance_df
            
        except Exception as e:
            logger.error(f"SHAP computation failed: {e}")
            return None


def analyze_all_models(
    ml_results: Dict,
    feature_names: List[str],
    output_dir: str = "results"
) -> Dict[str, pd.DataFrame]:
    """
    Analyze feature importance for all ML models.
    
    Args:
        ml_results: Dict from MLModelPipeline.train_all_models()
                   {model_name: (pipeline, cv_results)}
        feature_names: List of feature names
        output_dir: Directory to save plots and CSV
        
    Returns:
        Dict mapping model names to importance DataFrames
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    analyzer = FeatureImportanceAnalyzer(feature_names)
    importance_dict = {}
    
    print("\n" + "-" * 50)
    print("FEATURE IMPORTANCE BY MODEL")
    print("-" * 50)
    
    # Extract importance from each model
    for model_name, model_data in ml_results.items():
        try:
            # Handle tuple (pipeline, cv_results) or just model
            if isinstance(model_data, tuple):
                pipeline = model_data[0]
            else:
                pipeline = model_data
            
            # Get importance
            importance_df = analyzer.get_importance(pipeline, model_name)
            importance_dict[model_name] = importance_df
            
            # Print top 5
            print(f"\n{model_name.upper()} - Top 5 Features:")
            top5 = importance_df.head(5)[['Feature', 'Importance', 'Importance_Pct']]
            for _, row in top5.iterrows():
                print(f"  {row['Feature']:20s} {row['Importance']:.4f} ({row['Importance_Pct']:.1f}%)")
            
        except Exception as e:
            logger.warning(f"Could not extract importance for {model_name}: {e}")
    
    # Plot comparison
    if importance_dict:
        print("\n" + "-" * 50)
        print("Generating feature importance visualization...")
        
        fig = analyzer.plot_importance(
            importance_dict,
            top_n=10,
            save_path=str(output_path / "feature_importance.png")
        )
        plt.close(fig)
        
        # Save importance to CSV
        all_importance = pd.concat(importance_dict.values(), ignore_index=True)
        csv_path = output_path / "feature_importance.csv"
        all_importance.to_csv(csv_path, index=False)
        print(f"✓ Feature importance saved: {csv_path}")
        
        # Generate consensus ranking
        print("\n" + "-" * 50)
        print("CONSENSUS RANKING (Average across models)")
        print("-" * 50)
        
        consensus = (
            all_importance
            .groupby('Feature')['Importance_Pct']
            .agg(['mean', 'std', 'count'])
            .sort_values('mean', ascending=False)
            .head(10)
        )
        consensus.columns = ['Avg_Importance_%', 'Std', 'N_Models']
        print(consensus.round(2).to_string())
    
    return importance_dict


def main() -> None:
    """Example usage of feature importance analysis."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge
    
    print("=" * 70)
    print("FEATURE IMPORTANCE ANALYSIS DEMO")
    print("=" * 70)
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 500
    
    feature_names = ['Vol_21d', 'Vol_63d', 'Mom_21d', 'Mom_63d', 
                     'Beta_Lag1', 'Market_Vol_21d', 'Regime_HighVol']
    n_features = len(feature_names)
    
    X = np.random.randn(n_samples, n_features)
    # Beta_Lag1 is most important (index 4)
    y = 0.1 * X[:, 0] + 0.05 * X[:, 1] + 0.8 * X[:, 4] + np.random.randn(n_samples) * 0.1
    
    # Train models
    rf = RandomForestRegressor(n_estimators=50, random_state=42)
    rf.fit(X, y)
    
    ridge = Ridge(alpha=1.0)
    ridge.fit(X, y)
    
    # Analyze
    ml_results = {
        'RandomForest': (rf, {}),
        'Ridge': (ridge, {}),
    }
    
    importance_dict = analyze_all_models(
        ml_results=ml_results,
        feature_names=feature_names,
        output_dir="results"
    )
    
    print("\n✓ Analysis complete!")


if __name__ == "__main__":
    main()
