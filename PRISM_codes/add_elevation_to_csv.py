#!/usr/bin/env python
"""
CSV Elevation Enrichment Script

This script reads a CSV file containing latitude and longitude coordinates,
queries elevation data using a Digital Elevation Model (DEM), and adds the
elevation values as a new column to the original CSV file.

Dependencies:
- The DEM elevation lookup functionality from the dem-elevation-lookup script
"""

import os
import time
import numpy as np
import pandas as pd
import scipy.io
from scipy.spatial import cKDTree

def load_dem_data(file_path):
    """
    Load DEM data from a MATLAB file.

    Args:
        file_path (str): Path to the MATLAB file containing DEM data

    Returns:
        tuple: (latitudes, longitudes, elevations) arrays
    """
    mat_data = scipy.io.loadmat(file_path)
    dem_tiles = mat_data['DEMtiles']

    latitudes = dem_tiles[0, 0]["lat"].flatten()
    longitudes = dem_tiles[0, 0]["lon"].flatten()
    elevations = dem_tiles[0, 0]["z"]

    return latitudes, longitudes, elevations

def downsample_dem_data(latitudes, longitudes, elevations, factor):
    """
    Downsample DEM data to reduce computational complexity.

    Args:
        latitudes (np.ndarray): Array of latitude values
        longitudes (np.ndarray): Array of longitude values
        elevations (np.ndarray): 2D array of elevation values
        factor (int): Downsampling factor (take every Nth point)

    Returns:
        tuple: (grid_points, elevation_values)
    """
    # Downsample the grid
    ds_latitudes = latitudes[::factor]
    ds_longitudes = longitudes[::factor]
    ds_elevations = elevations[::factor, ::factor]

    # Create coordinate grid
    lon_grid, lat_grid = np.meshgrid(ds_longitudes, ds_latitudes)
    grid_points = np.column_stack((lat_grid.ravel(), lon_grid.ravel()))
    elevation_values = ds_elevations.ravel()

    return grid_points, elevation_values


def create_elevation_lookup(grid_points, elevation_values):
    """
    Create a function to look up elevations for given coordinates.

    Args:
        grid_points (np.ndarray): Array of (lat, lon) coordinates
        elevation_values (np.ndarray): Corresponding elevation values

    Returns:
        function: Lookup function that takes (lat, lon) and returns elevation
    """
    # Create KD-tree for efficient spatial lookups
    tree = cKDTree(grid_points)

    def get_elevation(lat, lon):
        """Get the elevation for a specific latitude/longitude pair."""
        _, index = tree.query([lat, lon])
        return elevation_values[index]

    return get_elevation


def enrich_csv_with_elevation(input_csv_path, output_csv_path, get_elevation_func,
                              lat_col='Latitude', lon_col='Longitude', elev_col='Elevation'):
    """
    Add elevation data to a CSV file based on latitude and longitude values.

    Args:
        input_csv_path (str): Path to the input CSV file
        output_csv_path (str): Path to save the enriched CSV file
        get_elevation_func (function): Function to get elevation for lat/lon pairs
        lat_col (str): Name of the latitude column in the CSV
        lon_col (str): Name of the longitude column in the CSV
        elev_col (str): Name to use for the new elevation column

    Returns:
        pd.DataFrame: The enriched DataFrame with elevation data
    """
    print(f"Reading CSV file: {input_csv_path}")
    df = pd.read_csv(input_csv_path)

    # Validate the columns exist
    if lat_col not in df.columns or lon_col not in df.columns:
        raise ValueError(f"Input CSV must contain {lat_col} and {lon_col} columns")

    # Add elevation column using our lookup function
    print(f"Processing {len(df)} coordinates and adding elevation data...")

    # Create progress tracker for large files
    total_rows = len(df)
    report_interval = max(1, total_rows // 10)  # Report progress every 10%

    # Process elevations
    elevations = []
    for i, (_, row) in enumerate(df.iterrows()):
        # Get elevation
        elevation = get_elevation_func(row[lat_col], row[lon_col])
        elevations.append(elevation)

        # Report progress
        if (i + 1) % report_interval == 0:
            progress = (i + 1) / total_rows * 100
            print(f"Progress: {progress:.1f}% ({i + 1}/{total_rows} coordinates processed)")

    # Add the elevations as a new column
    df[elev_col] = elevations

    # Save the enriched dataframe
    print(f"Saving enriched CSV to: {output_csv_path}")
    df.to_csv(output_csv_path, index=False)

    return df


def main():
    """Main execution function."""
    # Start timer for performance measurement
    start_time = time.time()

    # Configuration
    dem_file_path = "/Volumes/Mesonet/spring_ml/DEMdata/DEMdata.mat"
    input_csv_path = input("Enter path to input CSV file: ")

    # Generate output path based on input path
    base_name = os.path.splitext(os.path.basename(input_csv_path))[0]
    output_dir = os.path.dirname(input_csv_path)
    output_csv_path = os.path.join(output_dir, f"{base_name}_with_elevation.csv")

    # Allow customizing column names
    lat_col = input("Enter name of latitude column [Latitude]: ") or "Latitude"
    lon_col = input("Enter name of longitude column [Longitude]: ") or "Longitude"
    elev_col = input("Enter name for new elevation column [Elevation]: ") or "Elevation"

    # Downsampling factor for DEM data - can be adjusted based on accuracy/speed needs
    subset_factor = int(input("Enter downsampling factor for DEM grid [20]: ") or "20")

    # Load and process DEM data
    print("Loading DEM data...")
    latitudes, longitudes, elevations = load_dem_data(dem_file_path)

    print(f"Downsampling DEM data with factor {subset_factor}...")
    grid_points, elevation_values = downsample_dem_data(
        latitudes, longitudes, elevations, subset_factor
    )

    # Create elevation lookup function
    get_elevation = create_elevation_lookup(grid_points, elevation_values)

    # Process the CSV file and add elevations
    enriched_df = enrich_csv_with_elevation(
        input_csv_path,
        output_csv_path,
        get_elevation,
        lat_col,
        lon_col,
        elev_col
    )

    # Display summary
    print("\nSummary of elevation data added:")
    print(f"  Min elevation: {enriched_df[elev_col].min():.1f} meters")
    print(f"  Max elevation: {enriched_df[elev_col].max():.1f} meters")
    print(f"  Mean elevation: {enriched_df[elev_col].mean():.1f} meters")

    # Calculate and display execution time
    end_time = time.time()
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")
    print(f"Successfully saved enriched data to: {output_csv_path}")


if __name__ == "__main__":
    main()