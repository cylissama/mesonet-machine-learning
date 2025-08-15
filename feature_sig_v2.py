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
import pickle
import json
import os
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


class MesonetFeatureSignificance:
    """
    Feature significance analysis for Kentucky Mesonet data using Random Forest and SHAP
    """

    def __init__(self, data, target_variable, site_column='NetSiteAbbrev', output_dir=None):
        """
        Initialize with your mesonet dataset

        Parameters:
        - data: pandas DataFrame with your mesonet data
        - target_variable: 'TAIR_VT20' or 'VT90_VT20'
        - site_column: column name for site identification
        - output_dir: directory to save results (if None, creates 'feature_significance_results')
        """
        self.data = data.copy()
        self.target_variable = target_variable
        self.site_column = site_column

        # Setup output directory
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = f"feature_significance_results_{target_variable}_{timestamp}"
        else:
            self.output_dir = output_dir

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'figures'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'data'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'models'), exist_ok=True)

        print(f"Output directory: {self.output_dir}")

        # Handle timestamp conversion
        if 'UTCTimestampCollected' in data.columns:
            # Convert timestamp to datetime if it's not already
            data['UTCTimestampCollected'] = pd.to_datetime(data['UTCTimestampCollected'])

            # Create float timestamp (seconds since epoch)
            data['timestamp_float'] = data['UTCTimestampCollected'].astype('int64') // 10 ** 9

            # Create additional time-based features
            data['hour'] = data['UTCTimestampCollected'].dt.hour
            data['day_of_year'] = data['UTCTimestampCollected'].dt.dayofyear
            data['month'] = data['UTCTimestampCollected'].dt.month
            data['is_daytime'] = ((data['hour'] >= 6) & (data['hour'] <= 18)).astype(int)

            print(f"Converted timestamps to float format")
            print(f"Added temporal features: hour, day_of_year, month, is_daytime")

        # Define feature columns (exclude target, identifiers, and timestamps)
        exclude_cols = [target_variable, site_column, 'County', 'UTCTimestampCollected']
        # Include the new temporal features but exclude the original timestamp columns
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

    # Replace these methods in your code to fix the errors:

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

        # Create temporal features if timestamp exists and they don't already exist
        if 'UTCTimestampCollected' in data_subset.columns:
            if 'hour' not in data_subset.columns:
                data_subset['UTCTimestampCollected'] = pd.to_datetime(data_subset['UTCTimestampCollected'])
                data_subset['timestamp_float'] = data_subset['UTCTimestampCollected'].astype('int64') // 10 ** 9
                data_subset['hour'] = data_subset['UTCTimestampCollected'].dt.hour
                data_subset['day_of_year'] = data_subset['UTCTimestampCollected'].dt.dayofyear
                data_subset['month'] = data_subset['UTCTimestampCollected'].dt.month
                data_subset['is_daytime'] = ((data_subset['hour'] >= 6) & (data_subset['hour'] <= 18)).astype(int)

        # Update feature columns to only include columns that actually exist
        available_features = [col for col in self.feature_columns if col in data_subset.columns]

        # Add temporal features if they were created
        temporal_features = ['timestamp_float', 'hour', 'day_of_year', 'month', 'is_daytime']
        for tf in temporal_features:
            if tf in data_subset.columns and tf not in available_features:
                available_features.append(tf)

        # Update the feature columns list
        self.feature_columns = available_features

        # Get features and target
        X = data_subset[self.feature_columns]
        y = data_subset[self.target_variable]

        # Handle missing values in features (forward fill then backward fill)
        # But first, let's handle the timestamp-based ordering
        if 'UTCTimestampCollected' in data_subset.columns:
            data_subset = data_subset.sort_values('UTCTimestampCollected')
            X = data_subset[self.feature_columns]
            y = data_subset[self.target_variable]

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

        # Simple temporal split - use last 20% for testing
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

        # Save results to file
        self._save_rf_results(key, feature_importance, rmse, r2)

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

        # Save SHAP results
        self._save_shap_results(key, shap_importance)

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

        # Save comparison results
        self._save_comparison_results(key, comparison, rank_correlation)

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

        # Save figure
        filename = f"importance_comparison_{site_name}" if site_name else "importance_comparison_combined"
        filepath = os.path.join(self.output_dir, 'figures', f"{filename}.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved figure: {filepath}")

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

        # Save summary plot
        filename = f"shap_summary_{site_name}" if site_name else "shap_summary_combined"
        filepath = os.path.join(self.output_dir, 'figures', f"{filename}.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved figure: {filepath}")

        plt.show()

        # Bar plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_sample,
                          feature_names=self.feature_columns,
                          plot_type="bar", max_display=max_display, show=False)
        plt.title(f'SHAP Feature Importance - {self.target_variable} {site_name}')
        plt.tight_layout()

        # Save bar plot
        filename = f"shap_bar_{site_name}" if site_name else "shap_bar_combined"
        filepath = os.path.join(self.output_dir, 'figures', f"{filename}.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved figure: {filepath}")

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

    def finalize_analysis(self):
        """
        Finalize the analysis by creating summary reports and comprehensive figures
        """
        print(f"\n{'=' * 60}")
        print("FINALIZING ANALYSIS")
        print(f"{'=' * 60}")

        # Create comprehensive figure
        self.create_comprehensive_figure()

        # Save summary report
        report = self.save_summary_report()

        # Print final summary
        print(f"\nAnalysis complete! All results saved to: {self.output_dir}")
        print(f"Directory structure:")
        print(f"  ├── figures/     - All plots and visualizations")
        print(f"  ├── data/        - CSV files and numerical results")
        print(f"  ├── models/      - Trained Random Forest models")
        print(f"  ├── summary_report_{self.target_variable}.json")
        print(f"  └── summary_{self.target_variable}.txt")

        return report

    def _save_rf_results(self, key, feature_importance, rmse, r2):
        """Save Random Forest results to files"""
        # Save feature importance CSV
        filename = os.path.join(self.output_dir, 'data', f"{key}_feature_importance.csv")
        feature_importance.to_csv(filename, index=False)
        print(f"Saved RF results: {filename}")

        # Save model performance
        performance = {
            'model_type': 'Random Forest',
            'target_variable': self.target_variable,
            'rmse': float(rmse),
            'r2': float(r2),
            'n_features': len(self.feature_columns),
            'timestamp': datetime.now().isoformat()
        }

        filename = os.path.join(self.output_dir, 'data', f"{key}_performance.json")
        with open(filename, 'w') as f:
            json.dump(performance, f, indent=2)

        # Save model
        model_filename = os.path.join(self.output_dir, 'models', f"{key}_model.pkl")
        with open(model_filename, 'wb') as f:
            pickle.dump(self.models[key], f)

    def _save_shap_results(self, key, shap_importance):
        """Save SHAP results to files"""
        # Save SHAP importance CSV
        filename = os.path.join(self.output_dir, 'data', f"{key}_importance.csv")
        shap_importance.to_csv(filename, index=False)
        print(f"Saved SHAP results: {filename}")

        # Save SHAP values (numpy array)
        shap_values_filename = os.path.join(self.output_dir, 'data', f"{key}_values.npy")
        np.save(shap_values_filename, self.results[key]['shap_values'])

    def _save_comparison_results(self, key, comparison, rank_correlation):
        """Save method comparison results"""
        # Save comparison CSV
        filename = os.path.join(self.output_dir, 'data', f"{key}.csv")
        comparison.to_csv(filename, index=False)
        print(f"Saved comparison results: {filename}")

        # Save correlation info
        correlation_info = {
            'target_variable': self.target_variable,
            'spearman_rank_correlation': float(rank_correlation),
            'timestamp': datetime.now().isoformat()
        }

        filename = os.path.join(self.output_dir, 'data', f"{key}_correlation.json")
        with open(filename, 'w') as f:
            json.dump(correlation_info, f, indent=2)

    def save_summary_report(self):
        """Create and save a comprehensive summary report"""
        print("\nGenerating summary report...")

        report = {
            'analysis_info': {
                'target_variable': self.target_variable,
                'n_features': len(self.feature_columns),
                'feature_list': self.feature_columns,
                'sites_analyzed': list(self.data[self.site_column].unique()),
                'dataset_shape': list(self.data.shape),
                'timestamp': datetime.now().isoformat()
            },
            'results_summary': {}
        }

        # Add results for each analysis
        for key, result in self.results.items():
            if 'comparison' in key:
                # Get top 10 features from comparison
                top_features = result.head(10)[['feature', 'avg_rank']].to_dict('records')
                report['results_summary'][key] = {
                    'top_10_features': top_features,
                    'method': 'Combined Ranking'
                }
            elif 'rf' in key and 'feature_importance' in result:
                # Get top 10 from RF
                top_features = result['feature_importance'].head(10)[
                    ['feature', 'importance', 'perm_importance_mean']].to_dict('records')
                report['results_summary'][key] = {
                    'top_10_features': top_features,
                    'model_performance': result['model_performance'],
                    'method': 'Random Forest'
                }
            elif 'shap' in key and 'shap_importance' in result:
                # Get top 10 from SHAP
                top_features = result['shap_importance'].head(10)[['feature', 'shap_importance']].to_dict('records')
                report['results_summary'][key] = {
                    'top_10_features': top_features,
                    'method': 'SHAP'
                }

        # Save report
        report_filename = os.path.join(self.output_dir, f"summary_report_{self.target_variable}.json")
        with open(report_filename, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"Saved summary report: {report_filename}")

        # Also create a readable text summary
        self._create_text_summary(report)

        return report

    def _create_text_summary(self, report):
        """Create a human-readable text summary"""
        text_summary = []
        text_summary.append("=" * 80)
        text_summary.append(f"FEATURE SIGNIFICANCE ANALYSIS SUMMARY")
        text_summary.append(f"Target Variable: {self.target_variable}")
        text_summary.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        text_summary.append("=" * 80)
        text_summary.append("")

        text_summary.append(f"Dataset Information:")
        text_summary.append(f"  - Number of features: {len(self.feature_columns)}")
        text_summary.append(f"  - Sites analyzed: {', '.join(self.data[self.site_column].unique())}")
        text_summary.append(f"  - Dataset shape: {self.data.shape}")
        text_summary.append("")

        # Add results for each analysis
        for key, result in report['results_summary'].items():
            if 'comparison' in key:
                site_name = key.replace('comparison_', '') or 'Combined'
                text_summary.append(f"TOP 10 FEATURES - {site_name.upper()} (Combined Ranking):")
                text_summary.append("-" * 50)
                for i, feature in enumerate(result['top_10_features'], 1):
                    text_summary.append(f"  {i:2d}. {feature['feature']:<15} (Avg Rank: {feature['avg_rank']:.1f})")
                text_summary.append("")

        # Save text summary
        summary_filename = os.path.join(self.output_dir, f"summary_{self.target_variable}.txt")
        with open(summary_filename, 'w') as f:
            f.write('\n'.join(text_summary))

        print(f"Saved text summary: {summary_filename}")

    def create_comprehensive_figure(self):
        """Create a comprehensive figure showing all results"""
        sites = ['CRRL', 'RFSM', 'Combined']

        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        fig.suptitle(f'Comprehensive Feature Significance Analysis - {self.target_variable}', fontsize=16)

        for i, site in enumerate(sites):
            site_suffix = site.lower() if site != 'Combined' else 'combined'
            comparison_key = f"comparison_{site_suffix}" if site != 'Combined' else "comparison_combined"

            if comparison_key in self.results:
                comparison = self.results[comparison_key].head(15)

                # RF Importance
                axes[i, 0].barh(range(len(comparison)), comparison['importance'])
                axes[i, 0].set_yticks(range(len(comparison)))
                axes[i, 0].set_yticklabels(comparison['feature'], fontsize=8)
                axes[i, 0].set_xlabel('RF Importance')
                axes[i, 0].set_title(f'{site} - Random Forest')
                axes[i, 0].invert_yaxis()

                # SHAP Importance
                axes[i, 1].barh(range(len(comparison)), comparison['shap_importance'])
                axes[i, 1].set_yticks(range(len(comparison)))
                axes[i, 1].set_yticklabels(comparison['feature'], fontsize=8)
                axes[i, 1].set_xlabel('SHAP Importance')
                axes[i, 1].set_title(f'{site} - SHAP')
                axes[i, 1].invert_yaxis()

                # Rank comparison
                axes[i, 2].scatter(comparison['rf_rank'], comparison['shap_rank'], alpha=0.7)
                axes[i, 2].plot([1, 15], [1, 15], 'r--', alpha=0.5)
                axes[i, 2].set_xlabel('RF Rank')
                axes[i, 2].set_ylabel('SHAP Rank')
                axes[i, 2].set_title(f'{site} - Method Comparison')

                # Add correlation text
                rf_key = f"rf_{site_suffix}" if site != 'Combined' else "rf_combined"
                if rf_key in self.results:
                    corr = comparison['rf_rank'].corr(comparison['shap_rank'], method='spearman')
                    axes[i, 2].text(0.05, 0.95, f'ρ = {corr:.3f}',
                                    transform=axes[i, 2].transAxes,
                                    verticalalignment='top',
                                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()

        # Save comprehensive figure
        filename = os.path.join(self.output_dir, 'figures', f"comprehensive_analysis_{self.target_variable}.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved comprehensive figure: {filename}")

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


# Add this complete function definition to your code

def run_feature_significance_analysis(data_path_crrl=None, data_path_rfsm=None,
                                      output_base_dir="feature_significance_output"):
    """
    Complete workflow for running the feature significance analysis with Kentucky Mesonet data

    Parameters:
    - data_path_crrl: path to CRRL CSV file (or combined file if data_path_rfsm is None)
    - data_path_rfsm: path to RFSM CSV file (optional)
    - output_base_dir: base directory for all outputs
    """

    # Step 1: Load your data
    print("Step 1: Loading data...")

    if data_path_crrl and data_path_rfsm:
        print(f"Loading CRRL data from: {data_path_crrl}")
        print(f"Loading RFSM data from: {data_path_rfsm}")

        crrl_data = pd.read_csv(data_path_crrl)
        rfsm_data = pd.read_csv(data_path_rfsm)

        # Combine datasets
        combined_data = pd.concat([crrl_data, rfsm_data], ignore_index=True)

        print(f"CRRL data shape: {crrl_data.shape}")
        print(f"RFSM data shape: {rfsm_data.shape}")
        print(f"Combined data shape: {combined_data.shape}")

    elif data_path_crrl:  # Single file with both sites
        print(f"Loading combined data from: {data_path_crrl}")
        combined_data = pd.read_csv(data_path_crrl)
        print(f"Data shape: {combined_data.shape}")

    else:
        print("Please provide path(s) to your data file(s)")
        print("Usage examples:")
        print("  - Single file: run_feature_significance_analysis('path/to/combined_data.csv')")
        print("  - Two files: run_feature_significance_analysis('path/to/crrl.csv', 'path/to/rfsm.csv')")
        return None

    # Print basic data info
    print(f"\nData Info:")
    print(f"Columns: {list(combined_data.columns)}")

    if 'NetSiteAbbrev' in combined_data.columns:
        print(f"Sites: {combined_data['NetSiteAbbrev'].unique()}")

    if 'UTCTimestampCollected' in combined_data.columns:
        print(
            f"Date range: {combined_data['UTCTimestampCollected'].min()} to {combined_data['UTCTimestampCollected'].max()}")

    print(f"Missing values per column:")
    missing_info = combined_data.isnull().sum()
    for col, missing in missing_info.items():
        if missing > 0:
            print(f"  {col}: {missing} ({missing / len(combined_data) * 100:.1f}%)")

    # Step 2: Create target variables
    print("\nStep 2: Creating target variables...")

    # Check if required columns exist
    required_cols = ['TAIR', 'VT20', 'VT90']
    missing_cols = [col for col in required_cols if col not in combined_data.columns]
    if missing_cols:
        print(f"Error: Missing required columns: {missing_cols}")
        print(f"Available columns: {list(combined_data.columns)}")
        return None

    # Create target variables
    combined_data['TAIR_VT20'] = combined_data['TAIR'] - combined_data['VT20']
    combined_data['VT90_VT20'] = combined_data['VT90'] - combined_data['VT20']

    print(f"Created target variables:")
    print(
        f"  TAIR_VT20 (Surface-20ft): mean={combined_data['TAIR_VT20'].mean():.2f}, std={combined_data['TAIR_VT20'].std():.2f}")
    print(
        f"  VT90_VT20 (90ft-20ft): mean={combined_data['VT90_VT20'].mean():.2f}, std={combined_data['VT90_VT20'].std():.2f}")

    # Step 3: Remove rows with missing target variables
    print(f"\nStep 3: Cleaning data...")
    initial_size = len(combined_data)

    # Remove rows where target variables cannot be calculated
    combined_data = combined_data.dropna(subset=['TAIR_VT20', 'VT90_VT20'])

    final_size = len(combined_data)
    print(f"Removed {initial_size - final_size} rows with missing target values")
    print(f"Final dataset size: {final_size} rows")

    if final_size == 0:
        print("Error: No valid data remaining after removing missing values")
        return None

    # Create base output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = f"{output_base_dir}_{timestamp}"
    os.makedirs(base_output_dir, exist_ok=True)

    # Step 4: Analyze TAIR-VT20
    print("\n" + "=" * 80)
    print("ANALYZING TARGET: TAIR-VT20 (Surface-20ft Temperature Difference)")
    print("=" * 80)

    analyzer_tair = MesonetFeatureSignificance(
        combined_data,
        'TAIR_VT20',
        output_dir=os.path.join(base_output_dir, 'TAIR_VT20_analysis')
    )

    # Analyze each site separately if both sites exist
    sites_available = combined_data['NetSiteAbbrev'].unique() if 'NetSiteAbbrev' in combined_data.columns else ['All']
    results_tair = {}

    for site in sites_available:
        if site == 'All':
            # Analyze all data together
            results_tair['combined'] = analyzer_tair.analyze_combined()
        else:
            site_data_count = len(combined_data[combined_data['NetSiteAbbrev'] == site])
            print(f"\nSite {site}: {site_data_count} data points")

            if site_data_count > 100:  # Only analyze if sufficient data
                results_tair[site] = analyzer_tair.analyze_site(site)
            else:
                print(f"Skipping {site}: insufficient data (< 100 points)")

    # Analyze combined data if multiple sites
    if len(sites_available) > 1:
        results_tair['combined'] = analyzer_tair.analyze_combined()

    # Finalize TAIR analysis
    analyzer_tair.finalize_analysis()

    # Step 5: Analyze VT90-VT20
    print("\n" + "=" * 80)
    print("ANALYZING TARGET: VT90-VT20 (90ft-20ft Temperature Difference)")
    print("=" * 80)

    analyzer_vt = MesonetFeatureSignificance(
        combined_data,
        'VT90_VT20',
        output_dir=os.path.join(base_output_dir, 'VT90_VT20_analysis')
    )

    results_vt = {}

    for site in sites_available:
        if site == 'All':
            # Analyze all data together
            results_vt['combined'] = analyzer_vt.analyze_combined()
        else:
            site_data_count = len(combined_data[combined_data['NetSiteAbbrev'] == site])

            if site_data_count > 100:  # Only analyze if sufficient data
                results_vt[site] = analyzer_vt.analyze_site(site)
            else:
                print(f"Skipping {site}: insufficient data (< 100 points)")

    # Analyze combined data if multiple sites
    if len(sites_available) > 1:
        results_vt['combined'] = analyzer_vt.analyze_combined()

    # Finalize VT analysis
    analyzer_vt.finalize_analysis()

    # Step 6: Create overall summary
    print(f"\n{'=' * 80}")
    print("CREATING OVERALL SUMMARY")
    print(f"{'=' * 80}")

    overall_summary = {
        'analysis_timestamp': timestamp,
        'data_info': {
            'total_records': len(combined_data),
            'sites': list(sites_available),
            'target_variables': {
                'TAIR_VT20': {
                    'mean': float(combined_data['TAIR_VT20'].mean()),
                    'std': float(combined_data['TAIR_VT20'].std()),
                    'min': float(combined_data['TAIR_VT20'].min()),
                    'max': float(combined_data['TAIR_VT20'].max())
                },
                'VT90_VT20': {
                    'mean': float(combined_data['VT90_VT20'].mean()),
                    'std': float(combined_data['VT90_VT20'].std()),
                    'min': float(combined_data['VT90_VT20'].min()),
                    'max': float(combined_data['VT90_VT20'].max())
                }
            }
        },
        'output_directories': {
            'base': base_output_dir,
            'tair_analysis': os.path.join(base_output_dir, 'TAIR_VT20_analysis'),
            'vt_analysis': os.path.join(base_output_dir, 'VT90_VT20_analysis')
        }
    }

    # Add date range if available
    if 'UTCTimestampCollected' in combined_data.columns:
        overall_summary['data_info']['date_range'] = {
            'start': str(combined_data['UTCTimestampCollected'].min()),
            'end': str(combined_data['UTCTimestampCollected'].max())
        }

    # Save overall summary
    summary_file = os.path.join(base_output_dir, 'overall_analysis_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(overall_summary, f, indent=2)

    print(f"Overall summary saved to: {summary_file}")
    print(f"\nAnalysis complete! Check the following directories:")
    print(f"  📁 Main results: {base_output_dir}")
    print(f"  📁 TAIR-VT20 analysis: {os.path.join(base_output_dir, 'TAIR_VT20_analysis')}")
    print(f"  📁 VT90-VT20 analysis: {os.path.join(base_output_dir, 'VT90_VT20_analysis')}")

    return {
        'tair_analyzer': analyzer_tair,
        'vt_analyzer': analyzer_vt,
        'results': {
            'tair': results_tair,
            'vt': results_vt
        },
        'summary': overall_summary,
        'output_dir': base_output_dir
    }


# Convenience function for single file analysis
def analyze_mesonet_file(file_path, output_dir=None):
    """
    Convenience function to analyze a single mesonet file

    Parameters:
    - file_path: path to your CSV file
    - output_dir: custom output directory (optional)
    """
    return run_feature_significance_analysis(
        data_path_crrl=file_path,
        output_base_dir=output_dir or "mesonet_feature_analysis"
    )

# Run the analysis
if __name__ == "__main__":
    # Example usage:
    # results = run_feature_significance_analysis('path/to/your/data.csv')
    #
    # Or for separate files:
    # results = run_feature_significance_analysis('path/to/crrl.csv', 'path/to/rfsm.csv')
    results = run_feature_significance_analysis(
        data_path_crrl='/Users/cylis/Work/mes_summer25/CRRL mods/CRRL_mesonet_data_filled.csv',
        data_path_rfsm='/Users/cylis/Work/mes_summer25/RFSM mods/RFSM_mesonet_data_filled.csv'
    )

    print("Feature Significance Analysis Tool Ready!")
    print("Usage:")
    print("  results = run_feature_significance_analysis('path/to/your/data.csv')")
    print("  # or")
    print("  results = analyze_mesonet_file('path/to/your/data.csv')")