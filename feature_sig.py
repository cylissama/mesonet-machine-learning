import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
import shap
import warnings

warnings.filterwarnings('ignore')


class MesonetFeatureSignificance:
    """
    Feature significance analysis for Kentucky Mesonet data using Random Forest and SHAP
    """

    def __init__(self, data, target_variable, site_column='NetSiteAbbrev'):
        """
        Initialize with your mesonet dataset

        Parameters:
        - data: pandas DataFrame with your mesonet data
        - target_variable: 'TAIR_VT20' or 'VT90_VT20'
        - site_column: column name for site identification
        """
        self.data = data.copy()
        self.target_variable = target_variable
        self.site_column = site_column

        # Define feature columns (exclude target, identifiers, and timestamps)
        exclude_cols = [target_variable, site_column, 'County']
        # Add any timestamp columns you might have
        if 'datetime' in data.columns:
            exclude_cols.append('datetime')
        if 'timestamp' in data.columns:
            exclude_cols.append('timestamp')

        self.feature_columns = [col for col in data.columns if col not in exclude_cols]

        print(f"Target variable: {target_variable}")
        print(f"Number of features: {len(self.feature_columns)}")
        print(f"Features: {self.feature_columns}")
        print(f"Sites in dataset: {data[site_column].unique()}")
        print(f"Dataset shape: {data.shape}")

        # Initialize storage for results
        self.results = {}
        self.models = {}

    def prepare_data(self, site=None, remove_outliers=True, outlier_threshold=3):
        """
        Prepare data for analysis

        Parameters:
        - site: specific site to analyze ('CRRL', 'RFSM', or None for both)
        - remove_outliers: whether to remove statistical outliers
        - outlier_threshold: number of standard deviations for outlier detection
        """
        # Filter by site if specified
        if site:
            data_subset = self.data[self.data[self.site_column] == site].copy()
            print(f"\nAnalyzing site: {site}")
        else:
            data_subset = self.data.copy()
            print(f"\nAnalyzing combined data from all sites")

        # Remove rows with missing target values
        data_subset = data_subset.dropna(subset=[self.target_variable])

        # Get features and target
        X = data_subset[self.feature_columns]
        y = data_subset[self.target_variable]

        # Handle missing values in features (forward fill then backward fill)
        X = X.fillna(method='ffill').fillna(method='bfill')

        # Remove outliers if requested
        if remove_outliers:
            # Calculate z-scores for target variable
            z_scores = np.abs((y - y.mean()) / y.std())
            outlier_mask = z_scores < outlier_threshold

            X = X[outlier_mask]
            y = y[outlier_mask]

            print(f"Removed {(~outlier_mask).sum()} outliers")

        print(f"Final dataset shape: X={X.shape}, y={y.shape}")

        return X, y, data_subset

    def random_forest_analysis(self, X, y, site_name="", n_estimators=100,
                               test_size=0.2, random_state=42):
        """
        Perform Random Forest feature importance analysis

        Parameters:
        - X: feature matrix
        - y: target variable
        - site_name: name for results storage
        - n_estimators: number of trees in the forest
        - test_size: proportion of data for testing
        - random_state: for reproducibility
        """
        print(f"\n=== Random Forest Analysis {site_name} ===")

        # Split data (using last 20% for testing to respect time order)
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        print(f"Training set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")

        # Train Random Forest
        rf = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
            max_depth=10,  # Prevent overfitting
            min_samples_split=5,
            min_samples_leaf=2
        )

        rf.fit(X_train, y_train)

        # Make predictions and evaluate
        y_pred = rf.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        print(f"Model Performance:")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  R²: {r2:.4f}")

        # Feature importance (built-in)
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)

        # Permutation importance (more reliable)
        print("Calculating permutation importance...")
        perm_importance = permutation_importance(
            rf, X_test, y_test,
            n_repeats=10,
            random_state=random_state,
            n_jobs=-1
        )

        feature_importance['perm_importance_mean'] = perm_importance.importances_mean
        feature_importance['perm_importance_std'] = perm_importance.importances_std

        # Store results
        key = f"rf_{site_name}" if site_name else "rf_combined"
        self.results[key] = {
            'feature_importance': feature_importance,
            'model_performance': {'rmse': rmse, 'r2': r2},
            'predictions': {'y_test': y_test, 'y_pred': y_pred}
        }
        self.models[key] = rf

        print(f"\nTop 10 Features by Built-in Importance:")
        print(feature_importance.head(10)[['feature', 'importance']])

        print(f"\nTop 10 Features by Permutation Importance:")
        print(feature_importance.head(10)[['feature', 'perm_importance_mean']])

        return rf, feature_importance

    def shap_analysis(self, model, X, y, site_name="", max_samples=1000):
        """
        Perform SHAP analysis for feature importance and interactions

        Parameters:
        - model: trained Random Forest model
        - X: feature matrix
        - y: target variable
        - site_name: name for results storage
        - max_samples: maximum samples for SHAP calculation (for performance)
        """
        print(f"\n=== SHAP Analysis {site_name} ===")

        # Subsample data if too large (SHAP can be slow)
        if len(X) > max_samples:
            sample_idx = np.random.choice(len(X), max_samples, replace=False)
            X_sample = X.iloc[sample_idx]
            print(f"Using {max_samples} samples for SHAP analysis")
        else:
            X_sample = X
            print(f"Using all {len(X)} samples for SHAP analysis")

        # Create SHAP explainer
        print("Creating SHAP explainer...")
        explainer = shap.TreeExplainer(model)

        # Calculate SHAP values
        print("Calculating SHAP values...")
        shap_values = explainer.shap_values(X_sample)

        # Calculate feature importance based on mean absolute SHAP values
        shap_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'shap_importance': np.abs(shap_values).mean(axis=0)
        }).sort_values('shap_importance', ascending=False)

        # Store results
        key = f"shap_{site_name}" if site_name else "shap_combined"
        self.results[key] = {
            'shap_values': shap_values,
            'shap_importance': shap_importance,
            'explainer': explainer,
            'X_sample': X_sample
        }

        print(f"\nTop 10 Features by SHAP Importance:")
        print(shap_importance.head(10))

        return shap_values, shap_importance

    def compare_methods(self, site_name=""):
        """
        Compare Random Forest and SHAP feature importance rankings
        """
        rf_key = f"rf_{site_name}" if site_name else "rf_combined"
        shap_key = f"shap_{site_name}" if site_name else "shap_combined"

        if rf_key not in self.results or shap_key not in self.results:
            print("Please run both Random Forest and SHAP analysis first")
            return None

        # Get importance rankings
        rf_importance = self.results[rf_key]['feature_importance'].copy()
        rf_importance['rf_rank'] = range(1, len(rf_importance) + 1)

        shap_importance = self.results[shap_key]['shap_importance'].copy()
        shap_importance['shap_rank'] = range(1, len(shap_importance) + 1)

        # Merge rankings
        comparison = rf_importance[['feature', 'importance', 'perm_importance_mean', 'rf_rank']].merge(
            shap_importance[['feature', 'shap_importance', 'shap_rank']],
            on='feature'
        )

        # Calculate rank correlation
        rank_correlation = comparison['rf_rank'].corr(comparison['shap_rank'], method='spearman')

        # Calculate average rank
        comparison['avg_rank'] = (comparison['rf_rank'] + comparison['shap_rank']) / 2
        comparison = comparison.sort_values('avg_rank')

        print(f"\n=== Method Comparison {site_name} ===")
        print(f"Rank correlation (Spearman): {rank_correlation:.3f}")
        print(f"\nTop 15 features by average ranking:")
        print(comparison.head(15)[['feature', 'rf_rank', 'shap_rank', 'avg_rank']])

        # Store comparison
        key = f"comparison_{site_name}" if site_name else "comparison_combined"
        self.results[key] = comparison

        return comparison

    def plot_importance_comparison(self, site_name="", top_n=15):
        """
        Create visualization comparing RF and SHAP importance
        """
        comparison_key = f"comparison_{site_name}" if site_name else "comparison_combined"

        if comparison_key not in self.results:
            print("Please run compare_methods() first")
            return

        comparison = self.results[comparison_key].head(top_n)

        fig, axes = plt.subplots(1, 3, figsize=(18, 8))
        fig.suptitle(f'Feature Importance Comparison - {self.target_variable} {site_name}', fontsize=16)

        # Plot 1: Random Forest Importance
        axes[0].barh(range(len(comparison)), comparison['importance'])
        axes[0].set_yticks(range(len(comparison)))
        axes[0].set_yticklabels(comparison['feature'], fontsize=10)
        axes[0].set_xlabel('RF Feature Importance')
        axes[0].set_title('Random Forest (Built-in)')
        axes[0].invert_yaxis()

        # Plot 2: SHAP Importance
        axes[1].barh(range(len(comparison)), comparison['shap_importance'])
        axes[1].set_yticks(range(len(comparison)))
        axes[1].set_yticklabels(comparison['feature'], fontsize=10)
        axes[1].set_xlabel('Mean |SHAP Value|')
        axes[1].set_title('SHAP Importance')
        axes[1].invert_yaxis()

        # Plot 3: Rank Comparison
        axes[2].scatter(comparison['rf_rank'], comparison['shap_rank'], alpha=0.7)
        axes[2].plot([1, top_n], [1, top_n], 'r--', alpha=0.5)
        axes[2].set_xlabel('RF Rank')
        axes[2].set_ylabel('SHAP Rank')
        axes[2].set_title('Rank Correlation')

        # Add labels for top features
        for i, row in comparison.head(8).iterrows():
            axes[2].annotate(row['feature'],
                             (row['rf_rank'], row['shap_rank']),
                             fontsize=8, alpha=0.7)

        plt.tight_layout()
        plt.show()

    def plot_shap_summary(self, site_name="", max_display=20):
        """
        Create SHAP summary plots
        """
        shap_key = f"shap_{site_name}" if site_name else "shap_combined"

        if shap_key not in self.results:
            print("Please run SHAP analysis first")
            return

        shap_values = self.results[shap_key]['shap_values']
        X_sample = self.results[shap_key]['X_sample']

        # Summary plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_sample,
                          feature_names=self.feature_columns,
                          max_display=max_display, show=False)
        plt.title(f'SHAP Summary Plot - {self.target_variable} {site_name}')
        plt.tight_layout()
        plt.show()

        # Bar plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_sample,
                          feature_names=self.feature_columns,
                          plot_type="bar", max_display=max_display, show=False)
        plt.title(f'SHAP Feature Importance - {self.target_variable} {site_name}')
        plt.tight_layout()
        plt.show()

    def analyze_site(self, site, **kwargs):
        """
        Complete analysis workflow for a specific site
        """
        print(f"\n{'=' * 60}")
        print(f"ANALYZING SITE: {site}")
        print(f"{'=' * 60}")

        # Prepare data
        X, y, _ = self.prepare_data(site=site)

        # Random Forest analysis
        rf_model, rf_importance = self.random_forest_analysis(X, y, site_name=site, **kwargs)

        # SHAP analysis
        shap_values, shap_importance = self.shap_analysis(rf_model, X, y, site_name=site)

        # Compare methods
        comparison = self.compare_methods(site_name=site)

        # Create visualizations
        self.plot_importance_comparison(site_name=site)
        self.plot_shap_summary(site_name=site)

        return {
            'rf_importance': rf_importance,
            'shap_importance': shap_importance,
            'comparison': comparison,
            'model': rf_model
        }

    def analyze_combined(self, **kwargs):
        """
        Analyze combined data from both sites
        """
        print(f"\n{'=' * 60}")
        print(f"ANALYZING COMBINED DATA")
        print(f"{'=' * 60}")

        # Prepare data
        X, y, _ = self.prepare_data(site=None)

        # Random Forest analysis
        rf_model, rf_importance = self.random_forest_analysis(X, y, **kwargs)

        # SHAP analysis
        shap_values, shap_importance = self.shap_analysis(rf_model, X, y)

        # Compare methods
        comparison = self.compare_methods()

        # Create visualizations
        self.plot_importance_comparison()
        self.plot_shap_summary()

        return {
            'rf_importance': rf_importance,
            'shap_importance': shap_importance,
            'comparison': comparison,
            'model': rf_model
        }


# Example workflow
def run_feature_significance_analysis():
    """
    Example workflow for running the analysis
    """

    # Step 1: Load your data
    print("Step 1: Loading data...")
    # Replace with your actual file paths
    crrl_data = pd.read_csv('/Users/cylis/Work/mes_summer25/CRRL mods/CRRL_mesonet_data_filled_v2.csv')
    rfsm_data = pd.read_csv('/Users/cylis/Work/mes_summer25/RFSM mods/RFSM_mesonet_data_filled.csv')

    # Combine datasets
    combined_data = pd.concat([crrl_data, rfsm_data], ignore_index=True)

    # Step 2: Create target variables
    print("Step 2: Creating target variables...")
    combined_data['TAIR_VT20'] = combined_data['TAIR'] - combined_data['VT20']
    combined_data['VT90_VT20'] = combined_data['VT90'] - combined_data['VT20']

    # Step 3: Analyze TAIR-VT20
    print("\n" + "=" * 80)
    print("ANALYZING TARGET: TAIR-VT20 (Surface-20ft Temperature Difference)")
    print("=" * 80)

    analyzer_tair = MesonetFeatureSignificance(combined_data, 'TAIR_VT20')

    # Analyze each site separately
    crrl_results_tair = analyzer_tair.analyze_site('CRRL')
    rfsm_results_tair = analyzer_tair.analyze_site('RFSM')

    # Analyze combined data
    combined_results_tair = analyzer_tair.analyze_combined()

    # Step 4: Analyze VT90-VT20
    print("\n" + "=" * 80)
    print("ANALYZING TARGET: VT90-VT20 (90ft-20ft Temperature Difference)")
    print("=" * 80)

    analyzer_vt = MesonetFeatureSignificance(combined_data, 'VT90_VT20')

    # Analyze each site separately
    crrl_results_vt = analyzer_vt.analyze_site('CRRL')
    rfsm_results_vt = analyzer_vt.analyze_site('RFSM')

    # Analyze combined data
    combined_results_vt = analyzer_vt.analyze_combined()

    return {
        'tair_analyzer': analyzer_tair,
        'vt_analyzer': analyzer_vt,
        'results': {
            'tair': {
                'crrl': crrl_results_tair,
                'rfsm': rfsm_results_tair,
                'combined': combined_results_tair
            },
            'vt': {
                'crrl': crrl_results_vt,
                'rfsm': rfsm_results_vt,
                'combined': combined_results_vt
            }
        }
    }


# Run the analysis
if __name__ == "__main__":
    results = run_feature_significance_analysis()