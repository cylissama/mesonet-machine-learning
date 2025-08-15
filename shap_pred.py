import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import json
import os
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# SHAP import with error handling
try:
    import shap

    print(f"SHAP version: {shap.__version__}")
    SHAP_AVAILABLE = True
except ImportError:
    print("ERROR: SHAP is not installed. Install it with: pip install shap")
    SHAP_AVAILABLE = False


class CRRLSHAPAnalysis:
    """
    SHAP-focused feature significance analysis for CRRL site only
    """

    def __init__(self, data, target_variable, output_dir=None):
        """
        Initialize with CRRL mesonet dataset for SHAP analysis

        Parameters:
        - data: pandas DataFrame with CRRL mesonet data
        - target_variable: 'TAIR_VT20' or 'VT90_VT20'
        - output_dir: directory to save results
        """
        self.data = data.copy()
        self.target_variable = target_variable

        # Filter to CRRL only if site column exists
        if 'NetSiteAbbrev' in self.data.columns:
            self.data = self.data[self.data['NetSiteAbbrev'] == 'CRRL'].copy()
            print(f"Filtered to CRRL site only: {len(self.data)} records")

        # Drop unnecessary columns for simplicity
        columns_to_drop = ['UTCTimestampCollected', 'NetSiteAbbrev', 'County']
        existing_drops = [col for col in columns_to_drop if col in self.data.columns]
        if existing_drops:
            self.data = self.data.drop(columns=existing_drops)
            print(f"Dropped columns: {existing_drops}")

        # Setup output directory
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = f"CRRL_SHAP_{target_variable}_analysis_{timestamp}"
        else:
            self.output_dir = output_dir

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'figures'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'data'), exist_ok=True)

        print(f"Output directory: {self.output_dir}")

        # Define feature columns (exclude target and any unnamed columns)
        exclude_cols = [target_variable]
        self.feature_columns = [col for col in self.data.columns
                                if col not in exclude_cols and not col.startswith('Unnamed')]

        print(f"Target variable: {target_variable}")
        print(f"Number of features: {len(self.feature_columns)}")
        print(f"Features: {self.feature_columns}")
        print(f"Dataset shape: {self.data.shape}")

        # Initialize storage for results
        self.results = {}
        self.model = None
        self.explainer = None
        self.shap_values = None

    def prepare_data(self, remove_outliers=True, outlier_threshold=3):
        """
        Prepare CRRL data for SHAP analysis
        """
        print("\nPreparing CRRL data for SHAP analysis...")
        print(f"Starting with {len(self.data)} rows")

        # Check if target variable exists
        if self.target_variable not in self.data.columns:
            print(f"ERROR: Target variable '{self.target_variable}' not found in data!")
            print(f"Available columns: {list(self.data.columns)}")
            return None, None

        # Remove rows with missing target values
        initial_count = len(self.data)
        clean_data = self.data.dropna(subset=[self.target_variable]).copy()
        after_target_drop = len(clean_data)

        print(f"After removing rows with missing {self.target_variable}: {after_target_drop} rows")
        print(f"Removed {initial_count - after_target_drop} rows due to missing target")

        if len(clean_data) == 0:
            print("ERROR: No data remaining after removing missing target values!")
            return None, None

        # Check feature columns exist
        missing_features = [col for col in self.feature_columns if col not in clean_data.columns]
        if missing_features:
            print(f"WARNING: Missing feature columns: {missing_features}")
            self.feature_columns = [col for col in self.feature_columns if col in clean_data.columns]
            print(f"Updated feature columns: {self.feature_columns}")

        if len(self.feature_columns) == 0:
            print("ERROR: No feature columns available!")
            return None, None

        # Get features and target
        X = clean_data[self.feature_columns]
        y = clean_data[self.target_variable]

        print(f"Before handling missing values: X={X.shape}, y={y.shape}")

        # Check for missing values in features
        missing_counts = X.isnull().sum()
        if missing_counts.sum() > 0:
            print("Missing values in features:")
            for col, count in missing_counts.items():
                if count > 0:
                    print(f"  {col}: {count} missing values")

        # Handle missing values in features
        X = X.fillna(method='ffill').fillna(method='bfill')

        # Remove any rows that still have NaN (all NaN columns)
        before_nan_drop = len(X)
        nan_mask = ~X.isnull().any(axis=1)
        X = X[nan_mask]
        y = y[nan_mask]
        after_nan_drop = len(X)

        if after_nan_drop < before_nan_drop:
            print(f"Removed {before_nan_drop - after_nan_drop} rows that still had NaN values")

        if len(X) == 0:
            print("ERROR: No data remaining after handling missing values!")
            return None, None

        # Remove outliers if requested
        if remove_outliers and len(y) > 0:
            y_mean = y.mean()
            y_std = y.std()

            if y_std == 0:
                print("WARNING: Target variable has zero standard deviation - skipping outlier removal")
            else:
                z_scores = np.abs((y - y_mean) / y_std)
                outlier_mask = z_scores < outlier_threshold

                X = X[outlier_mask]
                y = y[outlier_mask]

                removed_count = (~outlier_mask).sum()
                print(f"Removed {removed_count} outliers (z-score > {outlier_threshold})")

        if len(X) == 0:
            print("ERROR: No data remaining after outlier removal!")
            return None, None

        print(f"Final dataset shape: X={X.shape}, y={y.shape}")
        print(f"Target variable stats: mean={y.mean():.3f}, std={y.std():.3f}, min={y.min():.3f}, max={y.max():.3f}")

        return X, y

    def train_base_model(self, X_train, y_train, n_estimators=100, random_state=42):
        """
        Train a Random Forest model for SHAP analysis
        """
        print("\nTraining Random Forest model for SHAP analysis...")

        rf = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
            max_depth=10,
            min_samples_split=max(2, min(5, len(X_train) // 10)),
            min_samples_leaf=max(1, min(2, len(X_train) // 20))
        )

        try:
            rf.fit(X_train, y_train)
            self.model = rf
            print("Random Forest model trained successfully")
            return rf
        except Exception as e:
            print(f"ERROR: Random Forest training failed: {str(e)}")
            return None

    def run_shap_analysis(self, max_samples=1000, n_estimators=100, test_size=0.2, random_state=42):
        """
        Run comprehensive SHAP feature importance analysis
        """
        print(f"\n{'=' * 60}")
        print(f"CRRL SHAP ANALYSIS - {self.target_variable}")
        print(f"{'=' * 60}")

        # Prepare data
        X, y = self.prepare_data()

        # Check if data preparation was successful
        if X is None or y is None:
            print("ERROR: Data preparation failed. Cannot proceed with SHAP analysis.")
            return None

        if len(X) < 10:
            print(f"ERROR: Insufficient data for analysis. Only {len(X)} samples available.")
            print("Need at least 10 samples for meaningful SHAP analysis.")
            return None

        # Split data (simple split since we don't have timestamps)
        split_idx = int(len(X) * (1 - test_size))

        if split_idx < 1:
            print("ERROR: Not enough data for train/test split.")
            return None

        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        print(f"\nData split:")
        print(f"  Training set: {X_train.shape}")
        print(f"  Test set: {X_test.shape}")

        # Train base model
        model = self.train_base_model(X_train, y_train, n_estimators, random_state)
        if model is None:
            return None

        # Evaluate model performance
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        print(f"\nBase Model Performance:")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  R²: {r2:.4f}")

        # Prepare data for SHAP (subsample if too large)
        if len(X) > max_samples:
            print(f"\nSubsampling data for SHAP analysis: {max_samples} samples")
            sample_idx = np.random.choice(len(X), max_samples, replace=False)
            X_shap = X.iloc[sample_idx]
        else:
            X_shap = X
            print(f"\nUsing all {len(X)} samples for SHAP analysis")

        # Create SHAP explainer
        print("Creating SHAP TreeExplainer...")
        try:
            self.explainer = shap.TreeExplainer(model)
            print("SHAP explainer created successfully")
        except Exception as e:
            print(f"ERROR: Failed to create SHAP explainer: {str(e)}")
            return None

        # Calculate SHAP values
        print("Calculating SHAP values...")
        try:
            if hasattr(self.explainer, 'shap_values'):
                # Older SHAP versions
                self.shap_values = self.explainer.shap_values(X_shap)
            else:
                # Newer SHAP versions
                shap_explanation = self.explainer(X_shap)
                if hasattr(shap_explanation, 'values'):
                    self.shap_values = shap_explanation.values
                else:
                    self.shap_values = shap_explanation

            print(f"SHAP values calculated: shape {self.shap_values.shape}")
        except Exception as e:
            print(f"ERROR: Failed to calculate SHAP values: {str(e)}")
            print("Trying alternative calculation method...")
            try:
                # Alternative method
                self.shap_values = self.explainer(X_shap.values)
                if hasattr(self.shap_values, 'values'):
                    self.shap_values = self.shap_values.values
                print(f"SHAP values calculated with alternative method: shape {self.shap_values.shape}")
            except Exception as e2:
                print(f"ERROR: Alternative method also failed: {str(e2)}")
                return None

        # Calculate SHAP-based feature importance
        shap_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'shap_importance': np.abs(self.shap_values).mean(axis=0)
        }).sort_values('shap_importance', ascending=False)

        # Add rank
        shap_importance['shap_rank'] = range(1, len(shap_importance) + 1)

        # Store results
        self.results = {
            'shap_importance': shap_importance,
            'shap_values': self.shap_values,
            'X_sample': X_shap,
            'model_performance': {'rmse': rmse, 'r2': r2},
            'predictions': {'y_test': y_test, 'y_pred': y_pred},
            'data_info': {
                'n_samples': len(X),
                'n_features': len(self.feature_columns),
                'n_shap_samples': len(X_shap),
                'train_size': len(X_train),
                'test_size': len(X_test)
            }
        }

        print(f"\nTop 15 Features by SHAP Importance:")
        print(shap_importance.head(15)[['feature', 'shap_importance', 'shap_rank']])

        return shap_importance

    def create_shap_visualizations(self, max_display=20):
        """
        Create comprehensive SHAP visualizations
        """
        if self.shap_values is None or 'X_sample' not in self.results:
            print("ERROR: SHAP analysis must be run first!")
            return

        X_sample = self.results['X_sample']

        print(f"\nCreating SHAP visualizations...")

        # 1. SHAP Summary Plot (dot plot)
        plt.figure(figsize=(12, 8))
        shap.summary_plot(self.shap_values, X_sample,
                          feature_names=self.feature_columns,
                          max_display=max_display, show=False)
        plt.title(f'SHAP Summary Plot - {self.target_variable} (CRRL Site)', fontsize=14, pad=20)
        plt.tight_layout()

        # Save summary plot
        summary_path = os.path.join(self.output_dir, 'figures', f'CRRL_SHAP_summary_{self.target_variable}.png')
        plt.savefig(summary_path, dpi=300, bbox_inches='tight')
        print(f"Saved SHAP summary plot: {summary_path}")
        plt.show()

        # 2. SHAP Bar Plot (feature importance)
        plt.figure(figsize=(12, 8))
        shap.summary_plot(self.shap_values, X_sample,
                          feature_names=self.feature_columns,
                          plot_type="bar", max_display=max_display, show=False)
        plt.title(f'SHAP Feature Importance - {self.target_variable} (CRRL Site)', fontsize=14, pad=20)
        plt.tight_layout()

        # Save bar plot
        bar_path = os.path.join(self.output_dir, 'figures', f'CRRL_SHAP_bar_{self.target_variable}.png')
        plt.savefig(bar_path, dpi=300, bbox_inches='tight')
        print(f"Saved SHAP bar plot: {bar_path}")
        plt.show()

        # 3. SHAP Waterfall Plot (for a single prediction)
        plt.figure(figsize=(12, 8))
        # Choose a random sample for waterfall plot
        sample_idx = np.random.choice(len(X_sample))
        shap.plots.waterfall(shap.Explanation(values=self.shap_values[sample_idx],
                                              base_values=self.explainer.expected_value,
                                              data=X_sample.iloc[sample_idx],
                                              feature_names=self.feature_columns),
                             max_display=15, show=False)
        plt.title(f'SHAP Waterfall Plot - Single Prediction Example\n{self.target_variable} (CRRL Site)',
                  fontsize=14, pad=20)
        plt.tight_layout()

        # Save waterfall plot
        waterfall_path = os.path.join(self.output_dir, 'figures', f'CRRL_SHAP_waterfall_{self.target_variable}.png')
        plt.savefig(waterfall_path, dpi=300, bbox_inches='tight')
        print(f"Saved SHAP waterfall plot: {waterfall_path}")
        plt.show()

        # 4. Custom comprehensive analysis plot
        self._create_comprehensive_plot()

    def _create_comprehensive_plot(self):
        """
        Create a comprehensive analysis plot combining SHAP importance with model performance
        """
        if not self.results:
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'CRRL SHAP Comprehensive Analysis - {self.target_variable}', fontsize=16)

        shap_importance = self.results['shap_importance']

        # 1. SHAP Importance Bar Plot
        top_features = shap_importance.head(15)
        axes[0, 0].barh(range(len(top_features)), top_features['shap_importance'])
        axes[0, 0].set_yticks(range(len(top_features)))
        axes[0, 0].set_yticklabels(top_features['feature'], fontsize=10)
        axes[0, 0].set_xlabel('Mean |SHAP Value|')
        axes[0, 0].set_title('SHAP Feature Importance')
        axes[0, 0].invert_yaxis()

        # 2. SHAP Values Distribution (box plot)
        # Select top 10 features for box plot
        top_10_features = top_features.head(10)['feature'].tolist()

        try:
            top_10_indices = [self.feature_columns.index(f) for f in top_10_features]
            shap_data = self.shap_values[:, top_10_indices]

            # Create box plot with proper dimensions
            box_data = [shap_data[:, i] for i in range(shap_data.shape[1])]
            box_labels = [f[:8] for f in top_10_features]

            axes[0, 1].boxplot(box_data, labels=box_labels)
            axes[0, 1].set_xlabel('Top 10 Features')
            axes[0, 1].set_ylabel('SHAP Values')
            axes[0, 1].set_title('SHAP Values Distribution')
            axes[0, 1].tick_params(axis='x', rotation=45)

        except Exception as e:
            print(f"Warning: Boxplot creation failed: {e}")
            # Create a simple scatter plot instead
            axes[0, 1].scatter(range(len(top_10_features)),
                               [shap_importance.loc[shap_importance['feature'] == f, 'shap_importance'].iloc[0]
                                for f in top_10_features])
            axes[0, 1].set_xticks(range(len(top_10_features)))
            axes[0, 1].set_xticklabels([f[:8] for f in top_10_features], rotation=45)
            axes[0, 1].set_xlabel('Top 10 Features')
            axes[0, 1].set_ylabel('SHAP Importance')
            axes[0, 1].set_title('SHAP Importance (Fallback)')

        # 3. Model Performance (Actual vs Predicted)
        y_test = self.results['predictions']['y_test']
        y_pred = self.results['predictions']['y_pred']

        axes[1, 0].scatter(y_test, y_pred, alpha=0.6)
        axes[1, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', alpha=0.8)
        axes[1, 0].set_xlabel('Actual')
        axes[1, 0].set_ylabel('Predicted')
        axes[1, 0].set_title('Model Predictions vs Actual')

        # Add performance metrics
        rmse = self.results['model_performance']['rmse']
        r2 = self.results['model_performance']['r2']
        axes[1, 0].text(0.05, 0.95, f'RMSE = {rmse:.3f}\nR² = {r2:.3f}',
                        transform=axes[1, 0].transAxes, fontsize=12,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # 4. Feature Importance Ranking
        feature_names = [f[:12] for f in top_features.head(15)['feature']]  # Truncate names
        y_pos = range(len(feature_names))

        axes[1, 1].barh(y_pos, top_features.head(15)['shap_importance'])
        axes[1, 1].set_yticks(y_pos)
        axes[1, 1].set_yticklabels(feature_names, fontsize=9)
        axes[1, 1].set_xlabel('SHAP Importance')
        axes[1, 1].set_title('Top 15 Features (Detailed)')
        axes[1, 1].invert_yaxis()

        plt.tight_layout()

        # Save comprehensive plot
        comp_path = os.path.join(self.output_dir, 'figures', f'CRRL_SHAP_comprehensive_{self.target_variable}.png')
        plt.savefig(comp_path, dpi=300, bbox_inches='tight')
        print(f"Saved comprehensive plot: {comp_path}")
        plt.show()

    def save_results(self):
        """
        Save all SHAP results to files
        """
        if not self.results:
            print("ERROR: No results to save. Run SHAP analysis first!")
            return None

        print("\nSaving SHAP results...")

        # Save SHAP importance CSV
        importance_file = os.path.join(self.output_dir, 'data', f'CRRL_SHAP_{self.target_variable}_importance.csv')
        self.results['shap_importance'].to_csv(importance_file, index=False)
        print(f"Saved SHAP importance: {importance_file}")

        # Save SHAP values (numpy array)
        shap_values_file = os.path.join(self.output_dir, 'data', f'CRRL_SHAP_{self.target_variable}_values.npy')
        np.save(shap_values_file, self.shap_values)
        print(f"Saved SHAP values: {shap_values_file}")

        # Save sample data used for SHAP
        sample_file = os.path.join(self.output_dir, 'data', f'CRRL_SHAP_{self.target_variable}_sample_data.csv')
        self.results['X_sample'].to_csv(sample_file, index=False)
        print(f"Saved sample data: {sample_file}")

        # Save model performance and summary
        summary = {
            'analysis_info': {
                'method': 'SHAP (SHapley Additive exPlanations)',
                'site': 'CRRL',
                'target_variable': self.target_variable,
                'analysis_timestamp': datetime.now().isoformat(),
                'n_samples': self.results['data_info']['n_samples'],
                'n_features': self.results['data_info']['n_features'],
                'n_shap_samples': self.results['data_info']['n_shap_samples']
            },
            'model_performance': self.results['model_performance'],
            'top_10_features': {
                'by_shap_importance': self.results['shap_importance'].head(10)[['feature', 'shap_importance']].to_dict(
                    'records')
            },
            'feature_rankings': {
                'shap_ranking': self.results['shap_importance'][['feature', 'shap_importance', 'shap_rank']].to_dict(
                    'records')
            }
        }

        summary_file = os.path.join(self.output_dir, f'CRRL_SHAP_{self.target_variable}_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Saved summary: {summary_file}")

        # Save trained model and explainer
        model_file = os.path.join(self.output_dir, f'CRRL_SHAP_{self.target_variable}_model.pkl')
        with open(model_file, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"Saved model: {model_file}")

        explainer_file = os.path.join(self.output_dir, f'CRRL_SHAP_{self.target_variable}_explainer.pkl')
        with open(explainer_file, 'wb') as f:
            pickle.dump(self.explainer, f)
        print(f"Saved SHAP explainer: {explainer_file}")

        # Create readable text summary
        self._create_text_summary()

        return summary

    def _create_text_summary(self):
        """Create a human-readable text summary"""
        text_lines = []
        text_lines.append("=" * 70)
        text_lines.append(f"CRRL SHAP FEATURE SIGNIFICANCE ANALYSIS")
        text_lines.append(f"Target Variable: {self.target_variable}")
        text_lines.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        text_lines.append("=" * 70)
        text_lines.append("")

        # Method info
        text_lines.append("Analysis Method: SHAP (SHapley Additive exPlanations)")
        text_lines.append("Base Model: Random Forest Regressor")
        text_lines.append("")

        # Data info
        text_lines.append("Dataset Information:")
        text_lines.append(f"  Site: CRRL")
        text_lines.append(f"  Total samples: {self.results['data_info']['n_samples']}")
        text_lines.append(f"  SHAP analysis samples: {self.results['data_info']['n_shap_samples']}")
        text_lines.append(f"  Number of features: {self.results['data_info']['n_features']}")
        text_lines.append(f"  Training samples: {self.results['data_info']['train_size']}")
        text_lines.append(f"  Test samples: {self.results['data_info']['test_size']}")
        text_lines.append("")

        # Model performance
        rmse = self.results['model_performance']['rmse']
        r2 = self.results['model_performance']['r2']
        text_lines.append("Base Model Performance:")
        text_lines.append(f"  RMSE: {rmse:.4f}")
        text_lines.append(f"  R²: {r2:.4f}")
        text_lines.append("")

        # Top features
        text_lines.append("TOP 15 FEATURES (SHAP Importance):")
        text_lines.append("-" * 60)
        text_lines.append(f"{'Rank':<5} {'Feature':<20} {'SHAP Importance':<15}")
        text_lines.append("-" * 60)

        top_features = self.results['shap_importance'].head(15)
        for i, (_, row) in enumerate(top_features.iterrows(), 1):
            text_lines.append(f"{i:<5} {row['feature']:<20} {row['shap_importance']:<15.6f}")

        text_lines.append("")
        text_lines.append("SHAP Analysis Interpretation:")
        text_lines.append("- Higher SHAP importance = greater impact on predictions")
        text_lines.append("- SHAP values show both magnitude and direction of feature effects")
        text_lines.append("- Individual SHAP values sum to prediction minus expected value")
        text_lines.append("- Features with high variability in SHAP values have complex relationships")

        # Save text summary
        text_file = os.path.join(self.output_dir, f'CRRL_SHAP_{self.target_variable}_summary.txt')
        with open(text_file, 'w') as f:
            f.write('\n'.join(text_lines))
        print(f"Saved text summary: {text_file}")


def analyze_crrl_shap(data_file, target_variable, output_dir=None, max_samples=1000):
    """
    Main function to run SHAP analysis on CRRL site

    Parameters:
    - data_file: path to CSV file containing mesonet data
    - target_variable: 'TAIR_VT20' or 'VT90_VT20'
    - output_dir: optional output directory
    - max_samples: maximum samples for SHAP calculation (for performance)
    """
    print(f"Loading data from: {data_file}")
    data = pd.read_csv(data_file)

    print(f"Original data shape: {data.shape}")
    print(f"Sites in data: {data['NetSiteAbbrev'].unique() if 'NetSiteAbbrev' in data.columns else 'No site column'}")

    # Create target variables if they don't exist
    if target_variable not in data.columns:
        if target_variable == 'TAIR_VT20':
            data['TAIR_VT20'] = data['TAIR'] - data['VT20']
            print("Created TAIR_VT20 target variable")
        elif target_variable == 'VT90_VT20':
            data['VT90_VT20'] = data['VT90'] - data['VT20']
            print("Created VT90_VT20 target variable")

    # Initialize SHAP analyzer
    analyzer = CRRLSHAPAnalysis(data, target_variable, output_dir)

    # Run SHAP analysis
    shap_importance = analyzer.run_shap_analysis(max_samples=max_samples)

    if shap_importance is not None:
        # Create visualizations
        analyzer.create_shap_visualizations()

        # Save results
        summary = analyzer.save_results()

        print(f"\n{'=' * 70}")
        print("SHAP ANALYSIS COMPLETE!")
        print(f"Results saved to: {analyzer.output_dir}")
        print("Key outputs:")
        print("  📊 SHAP summary plots (dot plot showing feature effects)")
        print("  📈 SHAP importance bar chart")
        print("  🌊 SHAP waterfall plot (single prediction explanation)")
        print("  📋 Feature importance rankings (CSV)")
        print("  💾 SHAP values and trained model (for reuse)")
        print(f"{'=' * 70}")

        return analyzer, shap_importance, summary
    else:
        print("SHAP analysis failed!")
        return None, None, None


# Example usage
if __name__ == "__main__":
    # Analyze TAIR-VT20 for CRRL using SHAP
    print("Running SHAP analysis for TAIR-VT20 at CRRL site...")
    analyzer1, importance1, summary1 = analyze_crrl_shap(
        '/Users/cylis/Work/mes_summer25/CRRL mods/CRRL_mesonet_data_filled_v2.csv',  # Replace with your file path
        'TAIR_VT20',
        max_samples=1000  # Adjust based on your data size and computational resources
    )

    # Analyze VT90-VT20 for CRRL using SHAP
    print("\nRunning SHAP analysis for VT90-VT20 at CRRL site...")
    analyzer2, importance2, summary2 = analyze_crrl_shap(
        '/Users/cylis/Work/mes_summer25/CRRL mods/CRRL_mesonet_data_filled_v2.csv',  # Replace with your file path
        'VT90_VT20',
        max_samples=1000  # Adjust based on your data size and computational resources
    )

    print("\nBoth SHAP analyses complete!")
    print("\nSHAP Analysis Summary:")
    print("=" * 50)

    if analyzer1 and importance1 is not None:
        print(f"TAIR-VT20 Analysis:")
        print(f"  Top 3 features: {', '.join(importance1.head(3)['feature'].tolist())}")
        print(f"  Model R²: {analyzer1.results['model_performance']['r2']:.3f}")
        print(f"  Output: {analyzer1.output_dir}")

    if analyzer2 and importance2 is not None:
        print(f"\nVT90-VT20 Analysis:")
        print(f"  Top 3 features: {', '.join(importance2.head(3)['feature'].tolist())}")
        print(f"  Model R²: {analyzer2.results['model_performance']['r2']:.3f}")
        print(f"  Output: {analyzer2.output_dir}")

    print("\n" + "=" * 50)
    print("SHAP analysis provides insights into:")
    print("✓ Which features have the strongest impact on predictions")
    print("✓ How feature values affect predictions (positive/negative)")
    print("✓ Feature interactions and complex relationships")
    print("✓ Individual prediction explanations")
    print("✓ Model behavior across the entire dataset")