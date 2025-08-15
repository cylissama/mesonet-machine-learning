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
        self.df['VT_inversion_90_20'] = self.df['VT90'] - self.df['VT20']  # Upper inversion
        self.df['VT_inversion_20_05'] = self.df['VT20'] - self.df['VT05']  # Lower inversion

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

    def engineer_all_features(self):
        """
        Run all feature engineering steps
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

        # Create forecast targets (example for temperature)
        print("Creating forecast targets...")
        self.create_forecast_targets('TAIR', horizons=[1, 3, 6])
        self.create_forecast_targets('VT_inversion_90_05', horizons=[1, 3, 6])

        print(f"Feature engineering complete. Dataset shape: {self.df.shape}")
        return self.df

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
    # Example with synthetic data structure
    # Replace with actual data loading

    # Load RFSM data
    fe_rfsm = MesonetFeatureEngineer(data_path='/Users/cylis/Work/mes_summer25/RFSM mods/RFSM_mesonet_data_filled.csv')
    rfsm_features = fe_rfsm.engineer_all_features()

    # Load CRRL data
    fe_crrl = MesonetFeatureEngineer(data_path='/Users/cylis/Work/mes_summer25/CRRL mods/CRRL_mesonet_data_filled_v2.csv')
    crrl_features = fe_crrl.engineer_all_features()

    rfsm_inversions = fe_rfsm.identify_inversion_events(threshold=0.0)
    crrl_inversions = fe_crrl.identify_inversion_events(threshold=0.0)

    # Create sample data structure (replace with your actual data loading)
    # print("=== Kentucky Mesonet Feature Engineering Example ===")
    # print("Load your data with:")
    # print("fe_rfsm = MesonetFeatureEngineer(data_path='RFSM_data.csv')")
    # print("fe_crrl = MesonetFeatureEngineer(data_path='CRRL_data.csv')")
    # print()
    # print("Then run feature engineering:")
    # print("rfsm_features = fe_rfsm.engineer_all_features()")
    # print("crrl_features = fe_crrl.engineer_all_features()")
    # print()
    # print("Identify inversion events:")
    # print("rfsm_inversions = fe_rfsm.identify_inversion_events(threshold=0.0)")
    # print("print(f'Found {len(rfsm_inversions)} inversion events at RFSM')")