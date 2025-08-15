import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
import pickle
import json
import os
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


class RFSMFeatureAnalysis:
    """
    Simplified Random Forest feature significance analysis for RFSM site only
    """

    def __init__(self, data, target_variable, output_dir=None):
        """
        Initialize with RFSM mesonet dataset

        Parameters:
        - data: pandas DataFrame with RFSM mesonet data
        - target_variable: 'TAIR_VT20' or 'VT90_VT20'
        - output_dir: directory to save results
        """
        self.data = data.copy()
        self.target_variable = target_variable

        # Filter to RFSM only if site column exists
        # if 'NetSiteAbbrev' in self.data.columns:
        #     self.data = self.data[self.data['NetSiteAbbrev'] == 'RFSM'].copy()
        #     print(f"Filtered to RFSM site only: {len(self.data)} records")

        # Drop unnecessary columns for simplicity
        columns_to_drop = ['UTCTimestampCollected', 'NetSiteAbbrev', 'County']
        existing_drops = [col for col in columns_to_drop if col in self.data.columns]
        if existing_drops:
            self.data = self.data.drop(columns=existing_drops)
            print(f"Dropped columns: {existing_drops}")

        # Setup output directory
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = f"RFSM_{target_variable}_analysis_{timestamp}"
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

    def prepare_data(self, remove_outliers=True, outlier_threshold=3):
        """
        Prepare RFSM data for analysis
        """
        print("\nPreparing RFSM data...")

        # Remove rows with missing target values
        clean_data = self.data.dropna(subset=[self.target_variable]).copy()

        # Get features and target
        X = clean_data[self.feature_columns]
        y = clean_data[self.target_variable]

        # Handle missing values in features
        X = X.fillna(method='ffill').fillna(method='bfill')

        # Remove outliers if requested
        if remove_outliers:
            z_scores = np.abs((y - y.mean()) / y.std())
            outlier_mask = z_scores < outlier_threshold

            X = X[outlier_mask]
            y = y[outlier_mask]

            removed_count = (~outlier_mask).sum()
            print(f"Removed {removed_count} outliers")

        print(f"Final dataset shape: X={X.shape}, y={y.shape}")
        print(f"Target variable stats: mean={y.mean():.3f}, std={y.std():.3f}")

        return X, y

    def run_random_forest_analysis(self, n_estimators=100, test_size=0.2, random_state=42):
        """
        Run Random Forest feature importance analysis
        """
        print(f"\n{'=' * 60}")
        print(f"RFSM RANDOM FOREST ANALYSIS - {self.target_variable}")
        print(f"{'=' * 60}")

        # Prepare data
        X, y = self.prepare_data()

        # Split data temporally (last 20% for testing)
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        print(f"\nData split:")
        print(f"  Training set: {X_train.shape}")
        print(f"  Test set: {X_test.shape}")

        # Train Random Forest
        print("\nTraining Random Forest...")
        rf = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2
        )

        rf.fit(X_train, y_train)
        self.model = rf

        # Make predictions and evaluate
        y_pred = rf.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        print(f"\nModel Performance:")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  R²: {r2:.4f}")

        # Feature importance (built-in)
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'rf_importance': rf.feature_importances_
        }).sort_values('rf_importance', ascending=False)

        # Permutation importance (more reliable)
        print("\nCalculating permutation importance...")
        perm_importance = permutation_importance(
            rf, X_test, y_test,
            n_repeats=10,
            random_state=random_state,
            n_jobs=-1
        )

        feature_importance['perm_importance_mean'] = perm_importance.importances_mean
        feature_importance['perm_importance_std'] = perm_importance.importances_std

        # Add rank columns
        feature_importance['rf_rank'] = range(1, len(feature_importance) + 1)
        feature_importance_perm = feature_importance.sort_values('perm_importance_mean', ascending=False)
        feature_importance_perm['perm_rank'] = range(1, len(feature_importance_perm) + 1)

        # Merge back
        feature_importance = feature_importance.merge(
            feature_importance_perm[['feature', 'perm_rank']],
            on='feature'
        )
        feature_importance = feature_importance.sort_values('rf_importance', ascending=False)

        # Store results
        self.results = {
            'feature_importance': feature_importance,
            'model_performance': {'rmse': rmse, 'r2': r2},
            'predictions': {'y_test': y_test, 'y_pred': y_pred},
            'data_info': {
                'n_samples': len(X),
                'n_features': len(self.feature_columns),
                'train_size': len(X_train),
                'test_size': len(X_test)
            }
        }

        print(f"\nTop 15 Features by Random Forest Importance:")
        print(feature_importance.head(15)[['feature', 'rf_importance', 'rf_rank']])

        print(f"\nTop 15 Features by Permutation Importance:")
        perm_sorted = feature_importance.sort_values('perm_importance_mean', ascending=False)
        print(perm_sorted.head(15)[['feature', 'perm_importance_mean', 'perm_rank']])

        return feature_importance

    def create_visualizations(self, top_n=20):
        """
        Create visualizations of feature importance
        """
        if not self.results:
            print("Run random forest analysis first!")
            return

        feature_importance = self.results['feature_importance']

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'RFSM Feature Importance Analysis - {self.target_variable}', fontsize=16)

        # 1. RF Importance Bar Plot
        top_rf = feature_importance.head(top_n)
        axes[0, 0].barh(range(len(top_rf)), top_rf['rf_importance'])
        axes[0, 0].set_yticks(range(len(top_rf)))
        axes[0, 0].set_yticklabels(top_rf['feature'], fontsize=10)
        axes[0, 0].set_xlabel('Random Forest Importance')
        axes[0, 0].set_title('Built-in Feature Importance')
        axes[0, 0].invert_yaxis()

        # 2. Permutation Importance Bar Plot
        top_perm = feature_importance.sort_values('perm_importance_mean', ascending=False).head(top_n)
        axes[0, 1].barh(range(len(top_perm)), top_perm['perm_importance_mean'])
        axes[0, 1].set_yticks(range(len(top_perm)))
        axes[0, 1].set_yticklabels(top_perm['feature'], fontsize=10)
        axes[0, 1].set_xlabel('Permutation Importance')
        axes[0, 1].set_title('Permutation Feature Importance')
        axes[0, 1].invert_yaxis()

        # 3. Rank Comparison
        axes[1, 0].scatter(feature_importance['rf_rank'], feature_importance['perm_rank'], alpha=0.7)
        axes[1, 0].plot([1, top_n], [1, top_n], 'r--', alpha=0.5)
        axes[1, 0].set_xlabel('RF Importance Rank')
        axes[1, 0].set_ylabel('Permutation Importance Rank')
        axes[1, 0].set_title('Rank Comparison')

        # Add correlation
        rank_corr = feature_importance['rf_rank'].corr(feature_importance['perm_rank'], method='spearman')
        axes[1, 0].text(0.05, 0.95, f'Spearman ρ = {rank_corr:.3f}',
                        transform=axes[1, 0].transAxes, fontsize=12,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # 4. Model Performance
        y_test = self.results['predictions']['y_test']
        y_pred = self.results['predictions']['y_pred']

        axes[1, 1].scatter(y_test, y_pred, alpha=0.6)
        axes[1, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', alpha=0.8)
        axes[1, 1].set_xlabel('Actual')
        axes[1, 1].set_ylabel('Predicted')
        axes[1, 1].set_title('Model Predictions vs Actual')

        # Add performance metrics
        rmse = self.results['model_performance']['rmse']
        r2 = self.results['model_performance']['r2']
        axes[1, 1].text(0.05, 0.95, f'RMSE = {rmse:.3f}\nR² = {r2:.3f}',
                        transform=axes[1, 1].transAxes, fontsize=12,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()

        # Save figure
        fig_path = os.path.join(self.output_dir, 'figures', f'RFSM_{self.target_variable}_analysis.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure: {fig_path}")

        plt.show()

    def save_results(self):
        """
        Save all results to files
        """
        print("\nSaving results...")

        # Save feature importance CSV
        importance_file = os.path.join(self.output_dir, 'data', f'RFSM_{self.target_variable}_feature_importance.csv')
        self.results['feature_importance'].to_csv(importance_file, index=False)
        print(f"Saved feature importance: {importance_file}")

        # Save model performance and summary
        summary = {
            'analysis_info': {
                'site': 'RFSM',
                'target_variable': self.target_variable,
                'analysis_timestamp': datetime.now().isoformat(),
                'n_samples': self.results['data_info']['n_samples'],
                'n_features': self.results['data_info']['n_features']
            },
            'model_performance': self.results['model_performance'],
            'top_10_features': {
                'by_rf_importance': self.results['feature_importance'].head(10)[['feature', 'rf_importance']].to_dict(
                    'records'),
                'by_perm_importance':
                    self.results['feature_importance'].sort_values('perm_importance_mean', ascending=False).head(10)[
                        ['feature', 'perm_importance_mean']].to_dict('records')
            }
        }

        summary_file = os.path.join(self.output_dir, f'RFSM_{self.target_variable}_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Saved summary: {summary_file}")

        # Save trained model
        model_file = os.path.join(self.output_dir, f'RFSM_{self.target_variable}_model.pkl')
        with open(model_file, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"Saved model: {model_file}")

        # Create readable text summary
        self._create_text_summary()

        return summary

    def _create_text_summary(self):
        """Create a human-readable text summary"""
        text_lines = []
        text_lines.append("=" * 70)
        text_lines.append(f"RFSM FEATURE SIGNIFICANCE ANALYSIS")
        text_lines.append(f"Target Variable: {self.target_variable}")
        text_lines.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        text_lines.append("=" * 70)
        text_lines.append("")

        # Data info
        text_lines.append("Dataset Information:")
        text_lines.append(f"  Site: RFSM")
        text_lines.append(f"  Total samples: {self.results['data_info']['n_samples']}")
        text_lines.append(f"  Number of features: {self.results['data_info']['n_features']}")
        text_lines.append(f"  Training samples: {self.results['data_info']['train_size']}")
        text_lines.append(f"  Test samples: {self.results['data_info']['test_size']}")
        text_lines.append("")

        # Model performance
        rmse = self.results['model_performance']['rmse']
        r2 = self.results['model_performance']['r2']
        text_lines.append("Model Performance:")
        text_lines.append(f"  RMSE: {rmse:.4f}")
        text_lines.append(f"  R²: {r2:.4f}")
        text_lines.append("")

        # Top features
        text_lines.append("TOP 15 FEATURES (Random Forest Importance):")
        text_lines.append("-" * 50)
        top_features = self.results['feature_importance'].head(15)
        for i, (_, row) in enumerate(top_features.iterrows(), 1):
            text_lines.append(f"  {i:2d}. {row['feature']:<20} {row['rf_importance']:.4f}")

        text_lines.append("")
        text_lines.append("TOP 15 FEATURES (Permutation Importance):")
        text_lines.append("-" * 50)
        top_perm = self.results['feature_importance'].sort_values('perm_importance_mean', ascending=False).head(15)
        for i, (_, row) in enumerate(top_perm.iterrows(), 1):
            text_lines.append(f"  {i:2d}. {row['feature']:<20} {row['perm_importance_mean']:.4f}")

        # Save text summary
        text_file = os.path.join(self.output_dir, f'RFSM_{self.target_variable}_summary.txt')
        with open(text_file, 'w') as f:
            f.write('\n'.join(text_lines))
        print(f"Saved text summary: {text_file}")


def analyze_RFSM_site(data_file, target_variable, output_dir=None):
    """
    Main function to analyze RFSM site

    Parameters:
    - data_file: path to CSV file containing mesonet data
    - target_variable: 'TAIR_VT20' or 'VT90_VT20'
    - output_dir: optional output directory
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

    # Initialize analyzer
    analyzer = RFSMFeatureAnalysis(data, target_variable, output_dir)

    # Run analysis
    feature_importance = analyzer.run_random_forest_analysis()

    # Create visualizations
    analyzer.create_visualizations()

    # Save results
    summary = analyzer.save_results()

    print(f"\n{'=' * 70}")
    print("ANALYSIS COMPLETE!")
    print(f"Results saved to: {analyzer.output_dir}")
    print(f"{'=' * 70}")

    return analyzer, feature_importance, summary


# Example usage
if __name__ == "__main__":
    # Analyze TAIR-VT20 for RFSM
    print("Analyzing TAIR-VT20 for RFSM site...")
    analyzer1, importance1, summary1 = analyze_RFSM_site(
        '/Users/cylis/Work/mes_summer25/RFSM mods/RFSM_mesonet_data_filled.csv',  # Replace with your file path
        'TAIR_VT20'
    )

    # Analyze VT90-VT20 for RFSM
    print("\nAnalyzing VT90-VT20 for RFSM site...")
    analyzer2, importance2, summary2 = analyze_RFSM_site(
        '/Users/cylis/Work/mes_summer25/RFSM mods/RFSM_mesonet_data_filled.csv',  # Replace with your file path
        'VT90_VT20'
    )

    print("\nBoth analyses complete!")