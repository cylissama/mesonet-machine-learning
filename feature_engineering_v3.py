#!/usr/bin/env python3
"""
Kentucky Mesonet Feature Engineering for Gradient Boosting Forecasting
Focuses on atmospheric inversion detection and forecasting features
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')


class MesonetFeatureEngineer:
    """
    Feature engineering class for Kentucky Mesonet data
    Specialized for atmospheric inversion analysis and forecasting
    """

    def __init__(self, data_path=None, df=None):
        """
        Initialize with either file path or DataFrame
        """
        if df is not None:
            self.df = df.copy()
        elif data_path:
            self.df = pd.read_csv(data_path)
        else:
            raise ValueError("Must provide either data_path or df")

        self.prepare_datetime()

    def prepare_datetime(self):
        """Convert timestamp and extract basic temporal features"""
        self.df['UTCTimestampCollected'] = pd.to_datetime(self.df['UTCTimestampCollected'])
        self.df = self.df.sort_values('UTCTimestampCollected').reset_index(drop=True)

        # Basic temporal features
        self.df['hour'] = self.df['UTCTimestampCollected'].dt.hour
        self.df['day_of_year'] = self.df['UTCTimestampCollected'].dt.dayofyear
        self.df['month'] = self.df['UTCTimestampCollected'].dt.month
        self.df['day_of_week'] = self.df['UTCTimestampCollected'].dt.dayofweek
        self.df['is_weekend'] = (self.df['day_of_week'] >= 5).astype(int)

        # Seasonal features
        self.df['season'] = self.df['month'].map({
            12: 'winter', 1: 'winter', 2: 'winter',
            3: 'spring', 4: 'spring', 5: 'spring',
            6: 'summer', 7: 'summer', 8: 'summer',
            9: 'fall', 10: 'fall', 11: 'fall'
        })

        # Cyclical encoding for hour and day_of_year
        self.df['hour_sin'] = np.sin(2 * np.pi * self.df['hour'] / 24)
        self.df['hour_cos'] = np.cos(2 * np.pi * self.df['hour'] / 24)
        self.df['doy_sin'] = np.sin(2 * np.pi * self.df['day_of_year'] / 365)
        self.df['doy_cos'] = np.cos(2 * np.pi * self.df['day_of_year'] / 365)

    def create_inversion_features(self):
        """
        Create atmospheric inversion-specific features
        Key for identifying VT90 - VT02 > 0 conditions
        """
        # Virtual temperature differences (inversion indicators)
        self.df['VT_inversion_90_05'] = self.df['VT90'] - self.df['VT05']  # Primary inversion
        self.df['VT_inversion_90_20'] = self.df['VT90'] - self.df['VT20']  # Upper inversion (YOUR TARGET)
        self.df['VT_inversion_20_05'] = self.df['VT20'] - self.df['VT05']  # Lower inversion

        # Note: You mentioned VT90 - VT02, but your data has VT05 as lowest level
        # If you have VT02 data, add: self.df['VT_inversion_90_02'] = self.df['VT90'] - self.df['VT02']

        # Inversion strength categories
        self.df['strong_inversion'] = (self.df['VT_inversion_90_05'] > 0).astype(int)
        self.df['very_strong_inversion'] = (self.df['VT_inversion_90_05'] > 2.0).astype(int)
        self.df['inversion_strength'] = pd.cut(
            self.df['VT_inversion_90_05'],
            bins=[-np.inf, -2, 0, 2, 5, np.inf],
            labels=['strong_lapse', 'weak_lapse', 'neutral', 'weak_inversion', 'strong_inversion']
        )

        # Virtual temperature gradients (per meter)
        self.df['VT_gradient_05_20'] = (self.df['VT20'] - self.df['VT05']) / 15  # °C/m
        self.df['VT_gradient_20_90'] = (self.df['VT90'] - self.df['VT20']) / 70  # °C/m
        self.df['VT_gradient_05_90'] = (self.df['VT90'] - self.df['VT05']) / 85  # °C/m

        # Atmospheric stability indicators
        self.df['stability_index'] = self.df['VT_gradient_05_90'] * 1000  # mK/m
        self.df['boundary_layer_strength'] = np.abs(self.df['VT_inversion_90_05'])

        # Target-specific features for your prediction task
        self.df['VT20_VT90_ratio'] = self.df['VT20'] / self.df['VT90']  # Ratio between levels
        self.df['VT_profile_curvature'] = (self.df['VT90'] - self.df['VT20']) - (
                    self.df['VT20'] - self.df['VT05'])  # Profile shape

    def create_lag_features(self, target_cols, lags=[1, 2, 3, 6, 12, 24]):
        """
        Create lagged features for time series forecasting
        """
        for col in target_cols:
            if col in self.df.columns:
                for lag in lags:
                    self.df[f'{col}_lag_{lag}'] = self.df[col].shift(lag)

    def create_rolling_features(self, target_cols, windows=[3, 6, 12, 24]):
        """
        Create rolling statistics features
        """
        for col in target_cols:
            if col in self.df.columns:
                for window in windows:
                    # Rolling statistics
                    self.df[f'{col}_rolling_mean_{window}'] = self.df[col].rolling(window=window).mean()
                    self.df[f'{col}_rolling_std_{window}'] = self.df[col].rolling(window=window).std()
                    self.df[f'{col}_rolling_min_{window}'] = self.df[col].rolling(window=window).min()
                    self.df[f'{col}_rolling_max_{window}'] = self.df[col].rolling(window=window).max()

                    # Rolling changes
                    self.df[f'{col}_rolling_change_{window}'] = (
                            self.df[col] - self.df[col].shift(window)
                    )

    def create_weather_derivatives(self):
        """
        Create derived meteorological features
        """
        # Temperature-related
        self.df['temp_dewpoint_spread'] = self.df['TAIR'] - self.df['DWPT']
        self.df['temp_change_1h'] = self.df['TAIR'].diff(1)
        self.df['temp_change_3h'] = self.df['TAIR'].diff(3)
        self.df['temp_acceleration'] = self.df['temp_change_1h'].diff(1)

        # Pressure features
        self.df['pressure_tendency_1h'] = self.df['PRES'].diff(1)
        self.df['pressure_tendency_3h'] = self.df['PRES'].diff(3) / 3
        self.df['pressure_acceleration'] = self.df['pressure_tendency_1h'].diff(1)

        # Wind features
        self.df['wind_speed_change'] = self.df['WSPD'].diff(1)
        self.df['wind_direction_change'] = self.df['WDIR'].diff(1)
        # Handle wind direction wrap-around (0° = 360°)
        self.df['wind_direction_change'] = np.where(
            self.df['wind_direction_change'] > 180,
            self.df['wind_direction_change'] - 360,
            self.df['wind_direction_change']
        )
        self.df['wind_direction_change'] = np.where(
            self.df['wind_direction_change'] < -180,
            self.df['wind_direction_change'] + 360,
            self.df['wind_direction_change']
        )

        # Wind components (useful for ML models)
        wind_rad = np.radians(self.df['WDIR'])
        self.df['wind_u'] = -self.df['WSPD'] * np.sin(wind_rad)  # East-West component
        self.df['wind_v'] = -self.df['WSPD'] * np.cos(wind_rad)  # North-South component

        # Solar radiation features
        self.df['solar_rad_change'] = self.df['SRAD'].diff(1)
        self.df['is_daylight'] = (self.df['SRAD'] > 10).astype(int)

        # Humidity features
        self.df['humidity_change'] = self.df['RELH'].diff(1)

    def create_soil_features(self):
        """
        Create soil-related features
        """
        # Soil temperature gradient
        self.df['soil_temp_gradient'] = self.df['ST04'] - self.df['ST02']

        # Soil moisture gradient
        self.df['soil_moisture_gradient'] = self.df['SM04'] - self.df['SM02']

        # Soil-air temperature difference
        self.df['soil_air_temp_diff_02'] = self.df['ST02'] - self.df['TAIR']
        self.df['soil_air_temp_diff_04'] = self.df['ST04'] - self.df['TAIR']

        # Soil thermal inertia indicators
        self.df['soil_temp_change_02'] = self.df['ST02'].diff(1)
        self.df['soil_temp_change_04'] = self.df['ST04'].diff(1)

    def create_interaction_features(self):
        """
        Create interaction features between variables
        """
        # Temperature-humidity interactions
        self.df['temp_humidity_interaction'] = self.df['TAIR'] * self.df['RELH']

        # Wind-temperature interactions (wind chill effect)
        self.df['wind_temp_interaction'] = self.df['WSPD'] * (self.df['TAIR'] - 10)

        # Solar-temperature interactions
        self.df['solar_temp_interaction'] = self.df['SRAD'] * self.df['TAIR']

        # Pressure-temperature interactions
        self.df['pressure_temp_interaction'] = self.df['PRES'] * self.df['TAIR']

    def create_atmospheric_physics_features(self):
        """
        Create physically-based atmospheric features
        """
        # Potential temperature (approximate)
        # θ ≈ T * (1000/P)^0.286
        self.df['potential_temp'] = self.df['TAIR'] * (1000 / self.df['PRES']) ** 0.286

        # Saturation vapor pressure (Tetens formula)
        self.df['sat_vapor_pressure'] = 6.112 * np.exp(17.67 * self.df['TAIR'] / (self.df['TAIR'] + 243.5))

        # Actual vapor pressure
        self.df['vapor_pressure'] = self.df['sat_vapor_pressure'] * self.df['RELH'] / 100

        # Virtual temperature calculation (more accurate than provided VT)
        self.df['virtual_temp_calc'] = self.df['TAIR'] * (1 + 0.61 * self.df['vapor_pressure'] / self.df['PRES'])

    def identify_inversion_events(self, threshold=0.0):
        """
        Identify and label strong inversion events
        Returns DataFrame with inversion event information
        """
        # Create inversion flag
        inversion_mask = self.df['VT_inversion_90_05'] > threshold

        # Group consecutive inversion periods
        inversion_groups = (inversion_mask != inversion_mask.shift()).cumsum()

        inversion_events = []
        for group in inversion_groups.unique():
            group_data = self.df[inversion_groups == group]
            if group_data['strong_inversion'].iloc[0] == 1:  # This is an inversion group
                event_info = {
                    'start_time': group_data['UTCTimestampCollected'].min(),
                    'end_time': group_data['UTCTimestampCollected'].max(),
                    'duration_hours': len(group_data),
                    'max_inversion_strength': group_data['VT_inversion_90_05'].max(),
                    'avg_inversion_strength': group_data['VT_inversion_90_05'].mean(),
                    'site': group_data['NetSiteAbbrev'].iloc[0],
                    'season': group_data['season'].iloc[0]
                }
                inversion_events.append(event_info)

        return pd.DataFrame(inversion_events)

    def create_forecast_targets(self, target_variable, horizons=[1, 3, 6, 12]):
        """
        Create forecast target variables for different time horizons
        """
        for horizon in horizons:
            self.df[f'{target_variable}_target_{horizon}h'] = self.df[target_variable].shift(-horizon)

    def engineer_all_features(self, save_to_folder=None):
        """
        Run all feature engineering steps and optionally save the result

        Parameters:
        - save_to_folder (str): Path to folder where enhanced dataset should be saved
        """
        print("Creating inversion features...")
        self.create_inversion_features()

        print("Creating weather derivatives...")
        self.create_weather_derivatives()

        print("Creating soil features...")
        self.create_soil_features()

        print("Creating atmospheric physics features...")
        self.create_atmospheric_physics_features()

        print("Creating interaction features...")
        self.create_interaction_features()

        # Define core meteorological variables for lag/rolling features
        core_vars = ['TAIR', 'DWPT', 'PRES', 'RELH', 'WSPD', 'SRAD',
                     'VT05', 'VT20', 'VT90', 'VT_inversion_90_05']

        print("Creating lag features...")
        self.create_lag_features(core_vars, lags=[1, 2, 3, 6, 12])

        print("Creating rolling features...")
        self.create_rolling_features(core_vars, windows=[3, 6, 12])

        # Create forecast targets for your specific variables
        print("Creating forecast targets...")

        # Your specific prediction targets
        self.create_forecast_targets('VT20', horizons=[1, 3, 6, 12])  # Predict VT20
        self.create_forecast_targets('VT90', horizons=[1, 3, 6, 12])  # Predict VT90
        self.create_forecast_targets('VT_inversion_90_20', horizons=[1, 3, 6, 12])  # Predict VT90-VT20 difference

        # Also create targets for the main inversion indicator
        self.create_forecast_targets('VT_inversion_90_05', horizons=[1, 3, 6, 12])  # Predict VT90-VT05 difference

        # Optional: Air temperature for comparison
        self.create_forecast_targets('TAIR', horizons=[1, 3, 6, 12])

        print(f"Feature engineering complete. Dataset shape: {self.df.shape}")

        # Save the enhanced dataset if folder is specified
        if save_to_folder:
            self.save_enhanced_dataset(save_to_folder)

        return self.df

    def save_enhanced_dataset(self, output_folder):
        """
        Save the enhanced dataset to the specified folder

        Parameters:
        - output_folder (str): Path to folder where files should be saved
        """
        import os

        # Create the output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)

        # Get site name from the data
        site_name = self.df['NetSiteAbbrev'].iloc[0] if 'NetSiteAbbrev' in self.df.columns else 'unknown_site'

        # Generate filename with timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Main enhanced dataset
        enhanced_filename = f"{site_name}_enhanced_features_{timestamp}.csv"
        enhanced_filepath = os.path.join(output_folder, enhanced_filename)

        print(f"Saving enhanced dataset to: {enhanced_filepath}")
        self.df.to_csv(enhanced_filepath, index=False)

        # Save inversion events summary
        inversion_events = self.identify_inversion_events(threshold=0.0)
        if len(inversion_events) > 0:
            inversion_filename = f"{site_name}_inversion_events_{timestamp}.csv"
            inversion_filepath = os.path.join(output_folder, inversion_filename)
            print(f"Saving inversion events to: {inversion_filepath}")
            inversion_events.to_csv(inversion_filepath, index=False)

            # Print inversion summary
            print(f"\n=== INVERSION SUMMARY FOR {site_name} ===")
            print(f"Total timesteps: {len(self.df)}")
            strong_inversions = self.df[self.df['strong_inversion'] == 1]
            print(f"Strong inversion timesteps: {len(strong_inversions)}")
            print(f"Percentage of time with strong inversions: {len(strong_inversions) / len(self.df) * 100:.1f}%")
            print(f"Number of distinct inversion events: {len(inversion_events)}")
            if len(strong_inversions) > 0:
                print(f"Strongest inversion: {strong_inversions['VT_inversion_90_05'].max():.2f}°C")
                print(f"Average inversion strength: {strong_inversions['VT_inversion_90_05'].mean():.2f}°C")

        # Save feature groups information
        feature_groups = self.get_feature_importance_groups()
        feature_info_filename = f"{site_name}_feature_groups_{timestamp}.txt"
        feature_info_filepath = os.path.join(output_folder, feature_info_filename)

        print(f"Saving feature groups info to: {feature_info_filepath}")
        with open(feature_info_filepath, 'w') as f:
            f.write(f"Feature Engineering Summary for {site_name}\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Original dataset shape: {len(self.df)} rows\n")
            f.write(f"Enhanced dataset shape: {self.df.shape}\n")
            f.write(f"Total features: {self.df.shape[1]}\n\n")

            for group_name, features in feature_groups.items():
                f.write(f"\n{group_name.upper()} FEATURES ({len(features)} features):\n")
                for feature in features:
                    if feature in self.df.columns:
                        f.write(f"  - {feature}\n")

        print(f"\n✅ All files saved to: {output_folder}")
        print(f"✅ Enhanced dataset: {enhanced_filename}")
        if len(inversion_events) > 0:
            print(f"✅ Inversion events: {inversion_filename}")
        print(f"✅ Feature groups: {feature_info_filename}")

        return enhanced_filepath

    def get_feature_importance_groups(self):
        """
        Return feature groups for organized analysis
        """
        feature_groups = {
            'temporal': [col for col in self.df.columns if
                         any(x in col for x in ['hour', 'day', 'month', 'season', 'doy'])],
            'inversion': [col for col in self.df.columns if 'inversion' in col or 'VT_' in col],
            'temperature': [col for col in self.df.columns if 'TAIR' in col or 'temp' in col],
            'pressure': [col for col in self.df.columns if 'PRES' in col or 'pressure' in col],
            'wind': [col for col in self.df.columns if any(x in col for x in ['WSPD', 'WDIR', 'wind'])],
            'humidity': [col for col in self.df.columns if
                         any(x in col for x in ['RELH', 'DWPT', 'humidity', 'vapor'])],
            'solar': [col for col in self.df.columns if 'SRAD' in col or 'solar' in col],
            'soil': [col for col in self.df.columns if any(x in col for x in ['SM', 'ST', 'soil'])],
            'lag_features': [col for col in self.df.columns if 'lag_' in col],
            'rolling_features': [col for col in self.df.columns if 'rolling_' in col],
        }
        return feature_groups


# Example usage
if __name__ == "__main__":
    # Set your specific output folder
    OUTPUT_FOLDER = "/Users/cylis/Work/mes_summer25/codes/feature_engineered_data"
    #
    # print("=== Kentucky Mesonet Feature Engineering ===")
    # print(f"Output folder: {OUTPUT_FOLDER}")
    # print()
    #
    # # Process RFSM site
    # print("🔄 Processing RFSM site...")
    # print("Load your RFSM data with:")
    # print("fe_rfsm = MesonetFeatureEngineer(data_path='path/to/RFSM_data.csv')")
    # print("rfsm_enhanced = fe_rfsm.engineer_all_features(save_to_folder=OUTPUT_FOLDER)")
    # print()
    #
    # # Process CRRL site
    # print("🔄 Processing CRRL site...")
    # print("Load your CRRL data with:")
    # print("fe_crrl = MesonetFeatureEngineer(data_path='path/to/CRRL_data.csv')")
    # print("crrl_enhanced = fe_crrl.engineer_all_features(save_to_folder=OUTPUT_FOLDER)")
    # print()
    #
    # print("📁 Files will be saved to:")
    # print(f"   {OUTPUT_FOLDER}/RFSM_enhanced_features_YYYYMMDD_HHMMSS.csv")
    # print(f"   {OUTPUT_FOLDER}/RFSM_inversion_events_YYYYMMDD_HHMMSS.csv")
    # print(f"   {OUTPUT_FOLDER}/RFSM_feature_groups_YYYYMMDD_HHMMSS.txt")
    # print(f"   {OUTPUT_FOLDER}/CRRL_enhanced_features_YYYYMMDD_HHMMSS.csv")
    # print(f"   {OUTPUT_FOLDER}/CRRL_inversion_events_YYYYMMDD_HHMMSS.csv")
    # print(f"   {OUTPUT_FOLDER}/CRRL_feature_groups_YYYYMMDD_HHMMSS.txt")
    # print()

    # Example with actual usage (uncomment when you have data files)

    # Actual usage example:

    # Process RFSM
    fe_rfsm = MesonetFeatureEngineer(data_path='/RFSM mods/RFSM_mesonet_data_filled.csv')
    rfsm_enhanced = fe_rfsm.engineer_all_features(save_to_folder=OUTPUT_FOLDER)

    # Process CRRL
    fe_crrl = MesonetFeatureEngineer(data_path='/CRRL mods/CRRL_mesonet_data_filled_v2.csv')
    crrl_enhanced = fe_crrl.engineer_all_features(save_to_folder=OUTPUT_FOLDER)

    print("✅ Feature engineering complete for both sites!")
    print("✅ Enhanced datasets saved and ready for gradient boosting!")


    print("Next steps:")
    print("1. Run feature engineering on your RFSM and CRRL data")
    print("2. Load the enhanced datasets from the output folder")
    print("3. Train XGBoost, LightGBM, CatBoost, and stacked models")
    print("4. Focus on strong inversion periods (where strong_inversion == 1)")
    print("5. Compare forecasting performance across different conditions")