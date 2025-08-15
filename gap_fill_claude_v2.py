import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


class MesonetGapFiller:
    """
    Gap filling for Kentucky Mesonet observational data
    Implements multiple methods for different variable types and gap lengths
    """

    def __init__(self, df):
        """
        Initialize with mesonet dataframe

        Parameters:
        df (pd.DataFrame): Mesonet data with columns as specified
        """
        self.df = df.copy()
        self.original_df = df.copy()  # Keep original for comparison

        # Define variable groups for special handling
        self.continuous_vars = ['TAIR', 'DWPT', 'PRES', 'RELH', 'SRAD']
        self.wind_vars = ['WSPD', 'WSSD', 'WDIR', 'WDSD']
        self.precip_vars = ['PRCP']
        self.soil_vars = ['SM02', 'SM04', 'ST02', 'ST04']
        self.voltage_vars = ['VT05', 'VT20', 'VT90', 'VR05', 'VR20', 'VR90']

        # All numeric variables
        self.all_numeric_vars = (self.continuous_vars + self.wind_vars +
                                 self.precip_vars + self.soil_vars + self.voltage_vars)

        # Ensure datetime column
        if 'UTCTimestampCollected' in self.df.columns:
            try:
                # Try to parse datetime with various formats
                self.df['UTCTimestampCollected'] = pd.to_datetime(self.df['UTCTimestampCollected'], errors='coerce')

                # Check if parsing failed
                if self.df['UTCTimestampCollected'].isna().all():
                    print("Warning: Could not parse datetime column. Trying alternative formats...")
                    # Try common datetime formats
                    for fmt in ['%Y-%m-%d %H:%M:%S', '%Y/%m/%d %H:%M:%S', '%m/%d/%Y %H:%M:%S',
                                '%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y']:
                        try:
                            self.df['UTCTimestampCollected'] = pd.to_datetime(self.df['UTCTimestampCollected'],
                                                                              format=fmt)
                            print(f"Successfully parsed datetime with format: {fmt}")
                            break
                        except:
                            continue

                # Remove any rows where datetime parsing failed
                valid_datetime_mask = ~self.df['UTCTimestampCollected'].isna()
                if not valid_datetime_mask.all():
                    print(f"Removing {(~valid_datetime_mask).sum()} rows with invalid datetime")
                    self.df = self.df[valid_datetime_mask]

                self.df = self.df.sort_values('UTCTimestampCollected')
                self.df = self.df.set_index('UTCTimestampCollected')
            except Exception as e:
                print(f"Error processing datetime column: {e}")
                print("Proceeding without datetime index...")
        else:
            print("Warning: 'UTCTimestampCollected' column not found")

    def _ensure_numeric_columns(self, df, columns):
        """
        Ensure specified columns are numeric, converting if possible

        Parameters:
        df (pd.DataFrame): DataFrame to process
        columns (list): List of column names to ensure are numeric

        Returns:
        pd.DataFrame: DataFrame with numeric columns
        """
        df_copy = df.copy()

        for col in columns:
            if col in df_copy.columns:
                # Try to convert to numeric, replacing non-numeric values with NaN
                df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')

        return df_copy

    def _get_valid_feature_vars(self, exclude_var):
        """
        Get list of valid numeric feature variables, excluding specified variable

        Parameters:
        exclude_var (str): Variable to exclude from features

        Returns:
        list: List of valid numeric column names
        """
        feature_vars = []

        for var in self.all_numeric_vars:
            if var != exclude_var and var in self.df.columns:
                # Check if column is actually numeric
                try:
                    pd.to_numeric(self.df[var], errors='raise')
                    feature_vars.append(var)
                except (ValueError, TypeError):
                    print(f"Warning: Skipping non-numeric column {var}")
                    continue

        return feature_vars

    def analyze_gaps(self, var_name):
        """Analyze missing data patterns for a variable"""
        if var_name not in self.df.columns:
            return None

        missing_mask = self.df[var_name].isna()
        total_missing = missing_mask.sum()
        total_points = len(self.df)
        missing_pct = (total_missing / total_points) * 100

        # Find consecutive gaps
        gaps = []
        gap_start = None

        for i, is_missing in enumerate(missing_mask):
            if is_missing and gap_start is None:
                gap_start = i
            elif not is_missing and gap_start is not None:
                gap_length = i - gap_start
                gaps.append(gap_length)
                gap_start = None

        if gap_start is not None:  # Handle gap at end
            gaps.append(len(missing_mask) - gap_start)

        return {
            'variable': var_name,
            'total_missing': total_missing,
            'missing_percentage': missing_pct,
            'number_of_gaps': len(gaps),
            'gap_lengths': gaps,
            'max_gap_length': max(gaps) if gaps else 0,
            'avg_gap_length': np.mean(gaps) if gaps else 0
        }

    def linear_interpolation(self, var_name, limit=12):
        """
        Linear interpolation for continuous variables

        Parameters:
        var_name (str): Variable name to interpolate
        limit (int): Maximum number of consecutive NaNs to fill (default 12 = 1 hour)
        """
        if var_name not in self.df.columns:
            return

        # Ensure column is numeric first
        self.df[var_name] = pd.to_numeric(self.df[var_name], errors='coerce')

        if var_name in self.wind_vars and 'DIR' in var_name:
            # Special handling for wind direction (circular)
            self._circular_interpolation(var_name, limit)
        elif var_name in self.precip_vars:
            # Don't interpolate precipitation - fill with zeros
            self.df[var_name].fillna(0, inplace=True)
        else:
            # Standard linear interpolation
            self.df[var_name] = self.df[var_name].interpolate(
                method='linear',
                limit=limit,
                limit_direction='both'
            )

    def _circular_interpolation(self, var_name, limit=12):
        """Handle circular interpolation for wind direction"""
        # Convert to radians
        rad_values = np.deg2rad(self.df[var_name])

        # Convert to cartesian coordinates
        x = np.cos(rad_values)
        y = np.sin(rad_values)

        # Interpolate x and y separately
        x_interp = pd.Series(x, index=self.df.index).interpolate(
            method='linear', limit=limit, limit_direction='both'
        )
        y_interp = pd.Series(y, index=self.df.index).interpolate(
            method='linear', limit=limit, limit_direction='both'
        )

        # Convert back to degrees
        self.df[var_name] = np.rad2deg(np.arctan2(y_interp, x_interp)) % 360

    def forward_backward_fill(self, var_name, limit=6):
        """
        Forward/backward fill for slowly changing variables

        Parameters:
        var_name (str): Variable name to fill
        limit (int): Maximum number of consecutive NaNs to fill
        """
        if var_name not in self.df.columns:
            return

        # Ensure column is numeric first
        self.df[var_name] = pd.to_numeric(self.df[var_name], errors='coerce')

        if var_name in self.precip_vars:
            # Precipitation: forward fill only
            self.df[var_name].fillna(method='ffill', limit=limit, inplace=True)
        else:
            # Forward fill then backward fill
            self.df[var_name].fillna(method='ffill', limit=limit, inplace=True)
            self.df[var_name].fillna(method='bfill', limit=limit, inplace=True)

    def _create_temporal_features(self, df):
        """Create temporal features for ML models"""
        df = df.copy()

        # Extract time-based features
        df['hour'] = df.index.hour
        df['day_of_year'] = df.index.dayofyear
        df['month'] = df.index.month

        # Cyclical encoding
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)

        return df

    def random_forest_imputation(self, var_name, n_estimators=100, max_depth=10):
        """
        Random Forest imputation for complex patterns with improved error handling

        Parameters:
        var_name (str): Variable name to impute
        n_estimators (int): Number of trees in the forest
        max_depth (int): Maximum depth of trees
        """
        if var_name not in self.df.columns:
            return

        # Ensure target variable is numeric
        self.df[var_name] = pd.to_numeric(self.df[var_name], errors='coerce')

        # Get valid numeric feature variables
        feature_vars = self._get_valid_feature_vars(var_name)

        if len(feature_vars) == 0:
            print(f"Warning: No valid numeric features found for {var_name}. Skipping Random Forest imputation.")
            return

        # Add temporal features
        df_features = self._create_temporal_features(self.df)
        temporal_features = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos']

        # Ensure all feature columns are numeric
        all_features = feature_vars + temporal_features
        df_features = self._ensure_numeric_columns(df_features, all_features)

        # Prepare data
        X = df_features[all_features].copy()
        y = self.df[var_name].copy()

        # Split into train (non-missing) and predict (missing)
        missing_mask = y.isna()
        X_train = X[~missing_mask]
        y_train = y[~missing_mask]
        X_predict = X[missing_mask]

        if len(X_train) == 0 or len(X_predict) == 0:
            print(f"Warning: Insufficient data for Random Forest imputation of {var_name}")
            return

        # Handle any remaining NaNs in features using median for robustness
        numeric_columns = X_train.select_dtypes(include=[np.number]).columns

        if len(numeric_columns) == 0:
            print(f"Warning: No numeric columns available for {var_name} imputation")
            return

        # Fill NaNs with median (more robust than mean for outliers)
        X_train_clean = X_train[numeric_columns].copy()
        X_predict_clean = X_predict[numeric_columns].copy()

        # Calculate medians only from numeric columns
        medians = X_train_clean.median()

        X_train_clean = X_train_clean.fillna(medians)
        X_predict_clean = X_predict_clean.fillna(medians)

        try:
            # Train Random Forest
            rf = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42,
                n_jobs=-1
            )

            rf.fit(X_train_clean, y_train)

            # Predict missing values
            y_pred = rf.predict(X_predict_clean)

            # Fill missing values
            self.df.loc[missing_mask, var_name] = y_pred

            print(f"Successfully filled {missing_mask.sum()} missing values for {var_name} using Random Forest")

        except Exception as e:
            print(f"Error in Random Forest imputation for {var_name}: {e}")
            print("Falling back to linear interpolation...")
            self.linear_interpolation(var_name)

    def gradient_boosting_imputation(self, var_name, n_estimators=200, learning_rate=0.1):
        """
        Gradient Boosting imputation with improved error handling

        Parameters:
        var_name (str): Variable name to impute
        n_estimators (int): Number of boosting iterations
        learning_rate (float): Learning rate
        """
        if var_name not in self.df.columns:
            return

        # Ensure target variable is numeric
        self.df[var_name] = pd.to_numeric(self.df[var_name], errors='coerce')

        # Get valid numeric feature variables
        feature_vars = self._get_valid_feature_vars(var_name)

        if len(feature_vars) == 0:
            print(f"Warning: No valid numeric features found for {var_name}. Falling back to simpler method.")
            self.linear_interpolation(var_name)
            return

        # Add temporal features
        df_features = self._create_temporal_features(self.df)
        temporal_features = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos']

        # Ensure all feature columns are numeric
        all_features = feature_vars + temporal_features
        df_features = self._ensure_numeric_columns(df_features, all_features)

        # Add lag features (previous 1-2 time steps) - limit to top features
        top_features = feature_vars[:3]  # Reduced to prevent feature explosion
        for lag in [1, 2]:
            for var in top_features:
                if var in df_features.columns:
                    df_features[f'{var}_lag{lag}'] = df_features[var].shift(lag)

        lag_features = [col for col in df_features.columns if 'lag' in col]

        # Prepare data
        final_features = all_features + lag_features
        X = df_features[final_features].copy()
        y = self.df[var_name].copy()

        # Remove rows with too many NaN values in features
        # Keep rows where at least 70% of features are non-null
        threshold = int(0.7 * len(final_features))
        valid_idx = X.count(axis=1) >= threshold

        X = X[valid_idx]
        y = y[valid_idx]

        if len(X) == 0:
            print(f"Warning: No valid data remaining for Gradient Boosting of {var_name}")
            self.linear_interpolation(var_name)
            return

        # Split into train and predict
        missing_mask = y.isna()
        X_train = X[~missing_mask]
        y_train = y[~missing_mask]
        X_predict = X[missing_mask]

        if len(X_train) == 0 or len(X_predict) == 0:
            print(f"Warning: Insufficient data for Gradient Boosting of {var_name}")
            self.linear_interpolation(var_name)
            return

        # Handle remaining NaNs with median
        numeric_columns = X_train.select_dtypes(include=[np.number]).columns

        if len(numeric_columns) == 0:
            print(f"Warning: No numeric features for {var_name}")
            self.linear_interpolation(var_name)
            return

        X_train_clean = X_train[numeric_columns].copy()
        X_predict_clean = X_predict[numeric_columns].copy()

        medians = X_train_clean.median()
        X_train_clean = X_train_clean.fillna(medians)
        X_predict_clean = X_predict_clean.fillna(medians)

        try:
            # Train Gradient Boosting
            gb = GradientBoostingRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=5,
                random_state=42
            )

            gb.fit(X_train_clean, y_train)

            # Predict missing values
            y_pred = gb.predict(X_predict_clean)

            # Fill missing values (need to align indices properly)
            pred_indices = X_predict.index
            original_indices = self.df.index[self.df.index.isin(pred_indices)]

            if len(original_indices) > 0:
                self.df.loc[original_indices, var_name] = y_pred[:len(original_indices)]
                print(
                    f"Successfully filled {len(original_indices)} missing values for {var_name} using Gradient Boosting")

        except Exception as e:
            print(f"Error in Gradient Boosting imputation for {var_name}: {e}")
            print("Falling back to linear interpolation...")
            self.linear_interpolation(var_name)

    def fill_gaps(self, method='auto', variables=None):
        """
        Main method to fill gaps using specified method

        Parameters:
        method (str): 'linear', 'ffill', 'random_forest', 'gradient_boosting', or 'auto'
        variables (list): List of variables to fill, or None for all
        """
        if variables is None:
            variables = self.all_numeric_vars

        # Filter to only existing columns
        variables = [v for v in variables if v in self.df.columns]

        for var in variables:
            print(f"Filling gaps for {var}...")

            if method == 'auto':
                # Auto-select method based on variable type and gap analysis
                gap_info = self.analyze_gaps(var)
                if gap_info and gap_info['max_gap_length'] > 0:
                    if gap_info['max_gap_length'] <= 12:  # Short gaps
                        self.linear_interpolation(var)
                    elif gap_info['max_gap_length'] <= 48:  # Medium gaps
                        self.random_forest_imputation(var)
                    else:  # Long gaps
                        self.gradient_boosting_imputation(var)
            elif method == 'linear':
                self.linear_interpolation(var)
            elif method == 'ffill':
                self.forward_backward_fill(var)
            elif method == 'random_forest':
                self.random_forest_imputation(var)
            elif method == 'gradient_boosting':
                self.gradient_boosting_imputation(var)

    def evaluate_imputation(self, var_name, test_fraction=0.1):
        """
        Evaluate imputation quality by creating artificial gaps

        Parameters:
        var_name (str): Variable to evaluate
        test_fraction (float): Fraction of data to use as test set
        """
        if var_name not in self.original_df.columns:
            return None

        # Get complete data points
        complete_mask = ~self.original_df[var_name].isna()
        complete_indices = self.original_df[complete_mask].index

        # Sample test indices
        n_test = int(len(complete_indices) * test_fraction)
        test_indices = np.random.choice(complete_indices, n_test, replace=False)

        # Create test dataset with artificial gaps
        test_df = self.original_df.copy()
        test_df.loc[test_indices, var_name] = np.nan

        # Apply imputation
        filler = MesonetGapFiller(test_df)
        filler.fill_gaps(method='auto', variables=[var_name])

        # Calculate metrics
        y_true = self.original_df.loc[test_indices, var_name]
        y_pred = filler.df.loc[test_indices, var_name]

        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)

        return {
            'variable': var_name,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'n_test_points': n_test
        }

    def plot_comparison(self, var_name, start_date=None, end_date=None):
        """Plot original vs filled data for visual comparison"""
        if var_name not in self.df.columns:
            return

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        # Select date range
        if start_date and end_date:
            mask = (self.df.index >= start_date) & (self.df.index <= end_date)
            plot_df = self.df[mask]
            orig_df = self.original_df[mask]
        else:
            plot_df = self.df
            orig_df = self.original_df

        # Plot original with gaps
        ax1.plot(orig_df.index, orig_df[var_name], 'b-', label='Original', alpha=0.7)
        ax1.set_ylabel(var_name)
        ax1.set_title(f'Original Data with Gaps - {var_name}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot filled data
        ax2.plot(plot_df.index, plot_df[var_name], 'g-', label='Gap-filled', alpha=0.7)

        # Highlight filled points
        filled_mask = orig_df[var_name].isna() & ~plot_df[var_name].isna()
        if filled_mask.any():
            ax2.scatter(plot_df[filled_mask].index, plot_df[filled_mask][var_name],
                        c='red', s=10, label='Filled values', alpha=0.6)

        ax2.set_ylabel(var_name)
        ax2.set_xlabel('Time')
        ax2.set_title(f'Gap-filled Data - {var_name}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def get_filled_dataframe(self):
        """Return the gap-filled dataframe"""
        # Ensure the index is reset if it's datetime
        if isinstance(self.df.index, pd.DatetimeIndex):
            result_df = self.df.reset_index()
        else:
            result_df = self.df.copy()
        return result_df

    def summary_report(self):
        """Generate summary report of gap filling"""
        print("Gap Filling Summary Report")
        print("=" * 50)

        for var in self.all_numeric_vars:
            if var in self.original_df.columns:
                original_missing = self.original_df[var].isna().sum()
                current_missing = self.df[var].isna().sum()
                filled = original_missing - current_missing

                print(f"\n{var}:")
                print(f"  Original missing: {original_missing} ({original_missing / len(self.df) * 100:.1f}%)")
                print(f"  Current missing: {current_missing} ({current_missing / len(self.df) * 100:.1f}%)")
                print(f"  Filled: {filled} points")


def load_and_prepare_data(filepath):
    """
    Helper function to load and prepare mesonet data

    Parameters:
    filepath (str): Path to the CSV file

    Returns:
    pd.DataFrame: Prepared dataframe
    """
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)

    # Display basic info
    print(f"Data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    # Check for datetime column issues
    if 'UTCTimestampCollected' in df.columns:
        # Check first few values
        print(f"\nFirst few timestamp values:")
        print(df['UTCTimestampCollected'].head())

        # Try to identify the datetime format
        sample_timestamp = df['UTCTimestampCollected'].iloc[0] if len(df) > 0 else None
        if pd.isna(sample_timestamp):
            # Find first non-null timestamp
            non_null_mask = df['UTCTimestampCollected'].notna()
            if non_null_mask.any():
                sample_timestamp = df.loc[non_null_mask, 'UTCTimestampCollected'].iloc[0]

        print(f"\nSample timestamp: {sample_timestamp}")
        print(f"Timestamp data type: {type(sample_timestamp)}")

    return df


# Example usage
if __name__ == "__main__":
    # Load your data
    df = load_and_prepare_data('/Users/cylis/Work/mes_summer25/original/CRRL.csv')

    # Initialize gap filler
    filler = MesonetGapFiller(df)

    # Fill gaps
    filler.fill_gaps(method='auto')

    # Get summary
    filler.summary_report()

    # Save results
    filled_df = filler.get_filled_dataframe()
    filled_df.to_csv('CRRL_mesonet_data_filled.csv', index=False)

    print("Gap filling complete!")