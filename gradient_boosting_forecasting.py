#!/usr/bin/env python3
"""
Gradient Boosting Models for Kentucky Mesonet Forecasting
Predicting VT20, VT90, and VT90-VT20 difference during inversion conditions

Uses XGBoost, LightGBM, CatBoost, and Stacked Ensemble
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import warnings

# Aggressive warning suppression
warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None
pd.set_option('mode.copy_on_write', True)

# Suppress specific pandas warnings
import sys

if not sys.warnoptions:
    warnings.simplefilter("ignore")

# Redirect pandas warnings to null
import logging

logging.getLogger('pandas').setLevel(logging.ERROR)

# Gradient Boosting Libraries
import xgboost as xgb
import lightgbm as lgb
import catboost as cb


class MesonetGradientBoostingForecaster:
    """
    Gradient boosting forecasting system for VT20, VT90, and their difference
    Focuses on strong inversion periods where VT90 - VT20 > 0
    """

    def __init__(self, enhanced_data_path=None, df=None, site_name=None):
        """
        Initialize with enhanced feature-engineered dataset

        Parameters:
        - enhanced_data_path: Path to feature-engineered CSV
        - df: Pre-loaded DataFrame with features
        - site_name: Site identifier (RFSM or CRRL)
        """
        if df is not None:
            self.df = df.copy()
        elif enhanced_data_path:
            self.df = pd.read_csv(enhanced_data_path)
        else:
            raise ValueError("Must provide either enhanced_data_path or df")

        self.site_name = site_name or self.df['NetSiteAbbrev'].iloc[
            0] if 'NetSiteAbbrev' in self.df.columns else 'Unknown'
        self.models = {}
        self.results = {}

        print(f"Initialized forecaster for {self.site_name}")
        print(f"Dataset shape: {self.df.shape}")

        self.prepare_data()

    def prepare_data(self):
        """Prepare data for modeling"""
        # Convert timestamp if it's not already datetime
        if 'UTCTimestampCollected' in self.df.columns:
            self.df['UTCTimestampCollected'] = pd.to_datetime(self.df['UTCTimestampCollected'])
            self.df = self.df.sort_values('UTCTimestampCollected').reset_index(drop=True)

        # Define prediction targets
        self.target_variables = {
            'VT20': 'VT20_target_1h',  # Predict VT20 1 hour ahead
            'VT90': 'VT90_target_1h',  # Predict VT90 1 hour ahead
            'VT_diff': 'VT_inversion_90_20_target_1h'  # Predict VT90-VT20 difference 1 hour ahead
        }

        # Check if target columns exist, create if needed
        for var_name, target_col in self.target_variables.items():
            if target_col not in self.df.columns:
                base_var = target_col.replace('_target_1h', '')
                if base_var in self.df.columns:
                    self.df[target_col] = self.df[base_var].shift(-1)  # 1 hour ahead
                else:
                    print(f"Warning: {base_var} not found in dataset")

        # Identify strong inversion periods
        if 'strong_inversion' in self.df.columns:
            self.inversion_mask = self.df['strong_inversion'] == 1
            print(
                f"Strong inversion periods: {self.inversion_mask.sum()} / {len(self.df)} ({self.inversion_mask.mean() * 100:.1f}%)")
        else:
            # Create inversion mask if not exists
            if 'VT_inversion_90_20' in self.df.columns:
                self.inversion_mask = self.df['VT_inversion_90_20'] > 0
            else:
                self.inversion_mask = pd.Series([True] * len(self.df))  # Use all data

    def select_features(self, exclude_future_leak=True):
        """
        Select relevant features for modeling

        Parameters:
        - exclude_future_leak: Remove features that would cause data leakage
        """
        # Features to exclude (target variables and future-looking features)
        exclude_features = [
            'UTCTimestampCollected', 'NetSiteAbbrev', 'County'
        ]

        # Add target variables to exclusion list
        exclude_features.extend(list(self.target_variables.values()))

        if exclude_future_leak:
            # Exclude any features that end with target indicators
            exclude_features.extend([col for col in self.df.columns if 'target_' in col])

        # Exclude categorical string columns that cause issues
        categorical_string_cols = []
        for col in self.df.columns:
            if self.df[col].dtype == 'object':
                # Keep if it's already dummy encoded (starts with common prefixes)
                if not any(col.startswith(prefix) for prefix in ['inversion_type_', 'season_']):
                    categorical_string_cols.append(col)

        exclude_features.extend(categorical_string_cols)

        # Also exclude the original categorical inversion_strength if it exists
        if 'inversion_strength' in self.df.columns:
            exclude_features.append('inversion_strength')

        # Select features
        feature_columns = [col for col in self.df.columns if col not in exclude_features]

        # Remove features with too many missing values
        missing_threshold = 0.5  # Remove features with >50% missing
        feature_columns = [col for col in feature_columns
                           if self.df[col].isna().mean() < missing_threshold]

        # Convert any remaining categorical columns to numeric
        for col in feature_columns:
            if self.df[col].dtype == 'object':
                try:
                    # Try to convert to numeric
                    self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                except:
                    # If conversion fails, remove the column
                    feature_columns.remove(col)
                    print(f"Removed problematic categorical column: {col}")

        print(f"Selected {len(feature_columns)} features for modeling")
        print(f"Excluded {len(exclude_features)} features")

        return feature_columns

    def prepare_train_test_split(self, feature_columns, target_variable, test_size=0.2):
        """
        Prepare time-series aware train/test split

        Parameters:
        - feature_columns: List of feature column names
        - target_variable: Target variable name
        - test_size: Fraction for test set
        """
        # Remove rows with missing targets
        valid_mask = self.df[target_variable].notna()

        # Combine with any other masks (like inversion periods)
        if hasattr(self, 'focus_inversion') and self.focus_inversion:
            valid_mask = valid_mask & self.inversion_mask

        # Get valid data
        X = self.df.loc[valid_mask, feature_columns].copy()
        y = self.df.loc[valid_mask, target_variable].copy()

        # Handle missing values in features more carefully
        print(f"Handling missing values in {len(feature_columns)} features...")

        # For each column, handle missing values appropriately
        for col in X.columns:
            if X[col].isna().any():
                if X[col].dtype in ['object', 'category']:
                    # For categorical columns, fill with mode or drop
                    if col in X.columns:  # Double check it still exists
                        print(f"Dropping categorical column with missing values: {col}")
                        X = X.drop(columns=[col])
                else:
                    # For numeric columns, fill with mean
                    X[col] = X[col].fillna(X[col].mean())

        # Ensure all remaining columns are numeric
        non_numeric_cols = []
        for col in X.columns:
            if X[col].dtype == 'object' or str(X[col].dtype) == 'category':
                non_numeric_cols.append(col)

        if non_numeric_cols:
            print(f"Dropping non-numeric columns: {non_numeric_cols}")
            X = X.drop(columns=non_numeric_cols)

        # Convert any remaining non-numeric data
        for col in X.columns:
            if X[col].dtype == 'object':
                # Try to convert to numeric, drop if it fails
                try:
                    X[col] = pd.to_numeric(X[col], errors='coerce')
                    if X[col].isna().all():
                        X = X.drop(columns=[col])
                        print(f"Dropped column {col} - could not convert to numeric")
                    else:
                        X[col] = X[col].fillna(X[col].mean())
                except:
                    X = X.drop(columns=[col])
                    print(f"Dropped problematic column: {col}")

        # Final check - ensure everything is numeric
        X = X.select_dtypes(include=[np.number])

        # CRITICAL: Clean infinite and extreme values
        print("Cleaning infinite and extreme values...")

        # Replace infinite values with NaN
        X = X.replace([np.inf, -np.inf], np.nan)

        # Check for remaining infinite values
        inf_cols = []
        for col in X.columns:
            if np.isinf(X[col]).any():
                inf_cols.append(col)

        if inf_cols:
            print(f"Found infinite values in columns: {inf_cols}")
            for col in inf_cols:
                # Replace infinite values with column mean
                col_mean = X[col].replace([np.inf, -np.inf], np.nan).mean()
                X[col] = X[col].replace([np.inf, -np.inf], col_mean)

        # Handle extremely large values (beyond float32 range)
        large_value_threshold = 1e30
        for col in X.columns:
            # Cap extremely large values
            max_val = X[col].max()
            min_val = X[col].min()

            if abs(max_val) > large_value_threshold or abs(min_val) > large_value_threshold:
                print(f"Capping extreme values in column {col}: max={max_val:.2e}, min={min_val:.2e}")
                # Cap at 99.9th percentile for positive, 0.1th percentile for negative
                upper_cap = X[col].quantile(0.999)
                lower_cap = X[col].quantile(0.001)
                X[col] = X[col].clip(lower=lower_cap, upper=upper_cap)

        # Fill any remaining NaN values with column means
        for col in X.columns:
            if X[col].isna().any():
                X[col] = X[col].fillna(X[col].mean())

        # Final validation - check for any remaining problematic values
        problem_cols = []
        for col in X.columns:
            if np.isinf(X[col]).any() or np.isnan(X[col]).any():
                problem_cols.append(col)

        if problem_cols:
            print(f"Warning: Still have problematic values in columns: {problem_cols}")
            # Drop these columns as last resort
            X = X.drop(columns=problem_cols)
            print(f"Dropped {len(problem_cols)} problematic columns")

        # Time series split (no random shuffling)
        split_idx = int(len(X) * (1 - test_size))

        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        print(f"Train set: {len(X_train)} samples, {len(X_train.columns)} features")
        print(f"Test set: {len(X_test)} samples")

        # Final check on training data
        if np.isinf(X_train.values).any() or np.isnan(X_train.values).any():
            print("❌ Warning: Training data still contains inf/nan values")
        else:
            print("✅ Training data is clean (no inf/nan values)")

        return X_train, X_test, y_train, y_test

    def train_individual_models(self, X_train, y_train, X_test, y_test, target_name):
        """
        Train XGBoost, LightGBM, and CatBoost models
        """
        models = {}
        predictions = {}

        print(f"\n🔄 Training models for {target_name}...")

        # XGBoost
        print("Training XGBoost...")
        xgb_model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        xgb_model.fit(X_train, y_train)
        xgb_pred = xgb_model.predict(X_test)

        models['XGBoost'] = xgb_model
        predictions['XGBoost'] = xgb_pred

        # LightGBM
        print("Training LightGBM...")
        lgb_model = lgb.LGBMRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        lgb_model.fit(X_train, y_train)
        lgb_pred = lgb_model.predict(X_test)

        models['LightGBM'] = lgb_model
        predictions['LightGBM'] = lgb_pred

        # CatBoost
        print("Training CatBoost...")
        cat_model = cb.CatBoostRegressor(
            iterations=200,
            depth=6,
            learning_rate=0.1,
            random_state=42,
            verbose=False
        )
        cat_model.fit(X_train, y_train)
        cat_pred = cat_model.predict(X_test)

        models['CatBoost'] = cat_model
        predictions['CatBoost'] = cat_pred

        # Stacked Ensemble
        print("Training Stacked Ensemble...")
        stacking_model = StackingRegressor(
            estimators=[
                ('xgb', xgb_model),
                ('lgb', lgb_model),
                ('cat', cat_model)
            ],
            final_estimator=LinearRegression(),
            cv=5
        )
        stacking_model.fit(X_train, y_train)
        stacked_pred = stacking_model.predict(X_test)

        models['Stacked'] = stacking_model
        predictions['Stacked'] = stacked_pred

        return models, predictions

    def evaluate_models(self, predictions, y_test, target_name):
        """
        Evaluate model performance with multiple metrics
        """
        results = {}

        print(f"\n📊 Evaluation Results for {target_name}:")
        print("-" * 60)
        print(f"{'Model':<12} {'RMSE':<8} {'MAE':<8} {'R²':<8}")
        print("-" * 60)

        for model_name, y_pred in predictions.items():
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            results[model_name] = {
                'RMSE': rmse,
                'MAE': mae,
                'R²': r2,
                'predictions': y_pred
            }

            print(f"{model_name:<12} {rmse:<8.3f} {mae:<8.3f} {r2:<8.3f}")

        print("-" * 60)

        # Find best model
        best_model = min(results.keys(), key=lambda x: results[x]['RMSE'])
        print(f"🏆 Best model: {best_model} (RMSE: {results[best_model]['RMSE']:.3f})")

        return results

    def train_and_evaluate_all_targets(self, focus_inversion=True):
        """
        Train models for all target variables (VT20, VT90, VT90-VT20)

        Parameters:
        - focus_inversion: If True, only use strong inversion periods for training
        """
        self.focus_inversion = focus_inversion

        if focus_inversion:
            print(f"\n🎯 Focusing on strong inversion periods only")
            print(f"Using {self.inversion_mask.sum()} / {len(self.df)} data points")

        # Select features
        feature_columns = self.select_features()

        all_results = {}
        all_models = {}

        # Train models for each target variable
        for target_name, target_column in self.target_variables.items():
            if target_column not in self.df.columns:
                print(f"Skipping {target_name} - target column {target_column} not found")
                continue

            print(f"\n{'=' * 60}")
            print(f"TRAINING MODELS FOR {target_name.upper()}")
            print(f"{'=' * 60}")

            # Prepare data
            X_train, X_test, y_train, y_test = self.prepare_train_test_split(
                feature_columns, target_column
            )

            # Train models
            models, predictions = self.train_individual_models(
                X_train, y_train, X_test, y_test, target_name
            )

            # Evaluate models
            results = self.evaluate_models(predictions, y_test, target_name)

            # Store results
            all_models[target_name] = models
            all_results[target_name] = results
            all_results[target_name]['y_test'] = y_test
            all_results[target_name]['feature_columns'] = feature_columns

        self.models = all_models
        self.results = all_results

        return all_results

    def plot_prediction_results(self, target_name, model_name='Stacked', n_points=200, save_path=None):
        """
        Plot prediction vs actual values

        Parameters:
        - target_name: Target variable name ('VT20', 'VT90', 'VT_diff')
        - model_name: Model to plot ('XGBoost', 'LightGBM', 'CatBoost', 'Stacked')
        - n_points: Number of points to plot (for cleaner visualization)
        - save_path: If provided, save plot to this path instead of showing
        """
        print(f"Creating prediction plot for {target_name} using {model_name} model...")

        if target_name not in self.results:
            print(f"No results found for {target_name}")
            available_targets = list(self.results.keys())
            print(f"Available targets: {available_targets}")
            return None

        if model_name not in self.results[target_name]:
            print(f"Model {model_name} not found for {target_name}")
            available_models = [k for k in self.results[target_name].keys() if
                                isinstance(self.results[target_name][k], dict)]
            print(f"Available models: {available_models}")
            return None

        try:
            y_test = self.results[target_name]['y_test']
            y_pred = self.results[target_name][model_name]['predictions']

            # Limit points for cleaner visualization
            if len(y_test) > n_points:
                indices = np.linspace(0, len(y_test) - 1, n_points, dtype=int)
                y_test_plot = y_test.iloc[indices]
                y_pred_plot = y_pred[indices]
            else:
                y_test_plot = y_test
                y_pred_plot = y_pred

            # Create figure
            fig = plt.figure(figsize=(12, 5))

            # Time series plot
            plt.subplot(1, 2, 1)
            plt.plot(range(len(y_test_plot)), y_test_plot, label='Actual', alpha=0.7, linewidth=1)
            plt.plot(range(len(y_pred_plot)), y_pred_plot, label='Predicted', alpha=0.7, linewidth=1)
            plt.xlabel('Time')
            plt.ylabel(target_name)
            plt.title(f'{target_name} Predictions - {model_name} Model\n{self.site_name} Site')
            plt.legend()
            plt.grid(True, alpha=0.3)

            # Scatter plot
            plt.subplot(1, 2, 2)
            plt.scatter(y_test, y_pred, alpha=0.6, s=10)
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            plt.xlabel('Actual')
            plt.ylabel('Predicted')
            r2_score = self.results[target_name][model_name]["R²"]
            plt.title(f'Actual vs Predicted\nR² = {r2_score:.3f}')
            plt.grid(True, alpha=0.3)

            plt.tight_layout()

            # Save or show
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"Plot saved to: {save_path}")
            else:
                plt.show()

            return fig

        except Exception as e:
            print(f"Error creating plot for {target_name}: {str(e)}")
            import traceback
            print(f"Full error: {traceback.format_exc()}")
            return None

    def get_feature_importance(self, target_name, model_name='XGBoost', top_n=15, save_path=None):
        """
        Get and plot feature importance

        Parameters:
        - target_name: Target variable name
        - model_name: Model name
        - top_n: Number of top features to show
        - save_path: If provided, save plot to this path
        """
        print(f"Analyzing feature importance for {target_name} using {model_name}...")

        if target_name not in self.models or model_name not in self.models[target_name]:
            print(f"Model {model_name} for {target_name} not found")
            available_models = list(self.models.get(target_name, {}).keys()) if target_name in self.models else []
            print(f"Available models for {target_name}: {available_models}")
            return None

        try:
            model = self.models[target_name][model_name]
            feature_columns = self.results[target_name]['feature_columns']

            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            elif hasattr(model, 'get_feature_importance'):
                importances = model.get_feature_importance()
            else:
                print(f"Feature importance not available for {model_name}")
                return None

            # Create feature importance DataFrame
            importance_df = pd.DataFrame({
                'feature': feature_columns,
                'importance': importances
            }).sort_values('importance', ascending=False).head(top_n)

            # Create plot
            fig = plt.figure(figsize=(10, 8))
            plt.barh(range(len(importance_df)), importance_df['importance'])
            plt.yticks(range(len(importance_df)), importance_df['feature'])
            plt.xlabel('Feature Importance')
            plt.title(f'Top {top_n} Features - {model_name} Model\n{target_name} Prediction ({self.site_name})')
            plt.gca().invert_yaxis()
            plt.tight_layout()

            # Save or show
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"Feature importance plot saved to: {save_path}")
            else:
                plt.show()

            return importance_df

        except Exception as e:
            print(f"Error analyzing feature importance: {str(e)}")
            import traceback
            print(f"Full error: {traceback.format_exc()}")
            return None

    def save_results(self, output_folder):
        """
        Save model results and predictions
        """
        import os
        from datetime import datetime

        os.makedirs(output_folder, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save summary results
        summary_results = []
        for target_name in self.results:
            for model_name in ['XGBoost', 'LightGBM', 'CatBoost', 'Stacked']:
                if model_name in self.results[target_name]:
                    summary_results.append({
                        'site': self.site_name,
                        'target': target_name,
                        'model': model_name,
                        'RMSE': self.results[target_name][model_name]['RMSE'],
                        'MAE': self.results[target_name][model_name]['MAE'],
                        'R²': self.results[target_name][model_name]['R²']
                    })

        summary_df = pd.DataFrame(summary_results)
        summary_path = os.path.join(output_folder, f"{self.site_name}_model_results_{timestamp}.csv")
        summary_df.to_csv(summary_path, index=False)

        print(f"✅ Results saved to: {summary_path}")
        return summary_path


# Example usage
if __name__ == "__main__":
    print("=== Kentucky Mesonet Gradient Boosting Forecasting ===")
    print()
    print("Example usage:")
    print("# Load enhanced dataset")
    print("forecaster = MesonetGradientBoostingForecaster(")
    print(
        "    enhanced_data_path='/Users/cylis/Work/mes_summer25/codes/feature_engineered_data/RFSM_enhanced_features.csv',")
    print("    site_name='RFSM'")
    print(")")
    print()
    print("# Train models for VT20, VT90, and VT90-VT20 difference")
    print("results = forecaster.train_and_evaluate_all_targets(focus_inversion=True)")
    print()
    print("# Plot results")
    print("forecaster.plot_prediction_results('VT20', 'Stacked')")
    print("forecaster.plot_prediction_results('VT90', 'Stacked')")
    print("forecaster.plot_prediction_results('VT_diff', 'Stacked')")
    print()
    print("# Get feature importance")
    print("forecaster.get_feature_importance('VT_diff', 'XGBoost')")
    print()
    print("# Save results")
    print("forecaster.save_results('/Users/cylis/Work/mes_summer25/codes/model_results/')")