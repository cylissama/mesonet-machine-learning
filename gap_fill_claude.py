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
        Random Forest imputation for complex patterns

        Parameters:
        var_name (str): Variable name to impute
        n_estimators (int): Number of trees in the forest
        max_depth (int): Maximum depth of trees
        """
        if var_name not in self.df.columns:
            return

        # Create feature matrix
        feature_vars = [v for v in self.all_numeric_vars if v != var_name and v in self.df.columns]

        # Add temporal features
        df_features = self._create_temporal_features(self.df)
        temporal_features = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos']

        # Prepare data
        X = df_features[feature_vars + temporal_features].copy()
        y = self.df[var_name].copy()

        # Split into train (non-missing) and predict (missing)
        missing_mask = y.isna()
        X_train = X[~missing_mask]
        y_train = y[~missing_mask]
        X_predict = X[missing_mask]

        if len(X_train) == 0 or len(X_predict) == 0:
            return

        # Handle any remaining NaNs in features
        X_train = X_train.fillna(X_train.mean())
        X_predict = X_predict.fillna(X_train.mean())

        # Train Random Forest
        rf = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1
        )

        rf.fit(X_train, y_train)

        # Predict missing values
        y_pred = rf.predict(X_predict)

        # Fill missing values
        self.df.loc[missing_mask, var_name] = y_pred

    def gradient_boosting_imputation(self, var_name, n_estimators=200, learning_rate=0.1):
        """
        Gradient Boosting imputation for highest accuracy

        Parameters:
        var_name (str): Variable name to impute
        n_estimators (int): Number of boosting iterations
        learning_rate (float): Learning rate
        """
        if var_name not in self.df.columns:
            return

        # Create feature matrix with lag features
        feature_vars = [v for v in self.all_numeric_vars if v != var_name and v in self.df.columns]

        # Add temporal features
        df_features = self._create_temporal_features(self.df)
        temporal_features = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos']

        # Add lag features (previous 1-2 time steps)
        for lag in [1, 2]:
            for var in feature_vars[:5]:  # Limit lag features to avoid explosion
                df_features[f'{var}_lag{lag}'] = self.df[var].shift(lag)

        lag_features = [col for col in df_features.columns if 'lag' in col]

        # Prepare data
        X = df_features[feature_vars + temporal_features + lag_features].copy()
        y = self.df[var_name].copy()

        # Remove rows with NaN in lag features
        valid_idx = ~X.isna().any(axis=1)
        X = X[valid_idx]
        y = y[valid_idx]

        # Split into train and predict
        missing_mask = y.isna()
        X_train = X[~missing_mask]
        y_train = y[~missing_mask]
        X_predict = X[missing_mask]

        if len(X_train) == 0 or len(X_predict) == 0:
            return

        # Handle any remaining NaNs
        X_train = X_train.fillna(X_train.mean())
        X_predict = X_predict.fillna(X_train.mean())

        # Train Gradient Boosting
        gb = GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=5,
            random_state=42
        )

        gb.fit(X_train, y_train)

        # Predict missing values
        y_pred = gb.predict(X_predict)

        # Fill missing values (need to align indices)
        pred_indices = X_predict.index
        self.df.loc[pred_indices, var_name] = y_pred

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
    df = df.drop(0)

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
def main():
    """Example of how to use the MesonetGapFiller class"""

    # Example 1: Create sample data for testing
    print("Creating sample mesonet data for demonstration...")

    # Create sample data
    import datetime
    dates = pd.date_range(start='2024-01-01', end='2024-01-07', freq='5min')
    n_points = len(dates)

    # Create sample dataframe
    sample_data = {
        'NetSiteAbbrev': ['SITE1'] * n_points,
        'County': ['County1'] * n_points,
        'UTCTimestampCollected': dates,
        'TAIR': np.random.normal(20, 5, n_points),
        'DWPT': np.random.normal(15, 3, n_points),
        'PRCP': np.random.exponential(0.1, n_points),
        'PRES': np.random.normal(1013, 5, n_points),
        'RELH': np.random.normal(70, 10, n_points),
        'SRAD': np.maximum(0, np.random.normal(500, 100, n_points)),
        'WSPD': np.random.exponential(3, n_points),
        'WDIR': np.random.uniform(0, 360, n_points)
    }

    df = pd.DataFrame(sample_data)

    # Introduce some gaps
    gap_indices = np.random.choice(n_points, size=int(n_points * 0.1), replace=False)
    for col in ['TAIR', 'RELH', 'WSPD']:
        df.loc[gap_indices, col] = np.nan

    print(f"Sample data created with {len(df)} records")
    print(f"Introduced gaps in TAIR, RELH, and WSPD (~10% missing)")

    # Initialize gap filler
    print("\nInitializing MesonetGapFiller...")
    filler = MesonetGapFiller(df)

    # Analyze gaps
    print("\nAnalyzing gaps in data...")
    for var in ['TAIR', 'RELH', 'WSPD']:
        if var in df.columns:
            gap_info = filler.analyze_gaps(var)
            if gap_info:
                print(f"\n{var}:")
                print(f"  Missing: {gap_info['missing_percentage']:.1f}%")
                print(f"  Number of gaps: {gap_info['number_of_gaps']}")
                print(f"  Max gap length: {gap_info['max_gap_length']} timesteps")

    # Fill gaps using auto method
    print("\nFilling gaps using 'auto' method...")
    filler.fill_gaps(method='auto', variables=['TAIR', 'RELH', 'WSPD'])

    # Evaluate imputation quality
    print("\nEvaluating imputation quality...")
    for var in ['TAIR', 'RELH', 'WSPD']:
        eval_results = filler.evaluate_imputation(var, test_fraction=0.1)
        if eval_results:
            print(f"\n{var}:")
            print(f"  MAE: {eval_results['mae']:.3f}")
            print(f"  RMSE: {eval_results['rmse']:.3f}")
            print(f"  R²: {eval_results['r2']:.3f}")

    # Get summary report
    print("\n" + "=" * 50)
    filler.summary_report()

    # Plot comparison for one variable
    print("\nGenerating comparison plot for TAIR...")
    filler.plot_comparison('TAIR')

    # Get filled dataframe
    filled_df = filler.get_filled_dataframe()
    print(f"\nGap filling complete! Filled dataframe has {len(filled_df)} records")

    # Example 2: Load actual data from CSV
    print("\n" + "=" * 50)
    print("\nTo use with your actual data:")
    print("1. Load your CSV file:")
    print("   df = pd.read_csv('your_mesonet_data.csv')")
    print("2. Initialize the gap filler:")
    print("   filler = MesonetGapFiller(df)")
    print("3. Fill gaps:")
    print("   filler.fill_gaps(method='auto')")
    print("4. Save results:")
    print("   filled_df = filler.get_filled_dataframe()")
    print("   filled_df.to_csv('mesonet_data_filled.csv')")

    return filler, filled_df


if __name__ == "__main__":
    # Uncomment the line below to run with sample data
    # main()

    # To use with your actual data, uncomment and modify the code below:

    # Load your data
    df = load_and_prepare_data('/Users/cylis/Work/mes_summer25/original/RFSM.csv')

    # Initialize gap filler
    filler = MesonetGapFiller(df)

    # Fill gaps
    filler.fill_gaps(method='auto')

    # Get summary
    filler.summary_report()

    # Save results
    filled_df = filler.get_filled_dataframe()
    filled_df.to_csv('mesonet_data_filled.csv', index=False)

    print("Gap filling implementation ready!")
    print("To use with your data, modify the code in the __main__ section")