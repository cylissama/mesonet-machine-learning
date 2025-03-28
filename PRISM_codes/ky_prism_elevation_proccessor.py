#!/usr/bin/env python3
"""
Kentucky PRISM Elevation Processor

This script adds elevation data to processed PRISM CSV files for Kentucky.
It reads existing CSV files, queries elevation data for each point using the
DEM data, and adds the elevation as a new column or updates existing altitude values.

Features:
- Works with processed PRISM CSV files
- Uses KD-Tree for efficient elevation lookup
- Processes files in bulk or individually
- Preserves all existing data and metadata
- Saves modified files to original location or new folder

Author: For KY MESONET @ WKU
"""

import os
import time
import argparse
import glob
import numpy as np
import pandas as pd
import scipy.io
from scipy.spatial import cKDTree
from tqdm import tqdm

# Default path for DEM data file
DEFAULT_DEM_PATH = "/Volumes/Mesonet/spring_ml/DEMdata/DEMdata.mat"


def load_dem_data(file_path):
    """
    Load DEM data from a MATLAB file.

    Args:
        file_path (str): Path to the MATLAB file containing DEM data

    Returns:
        tuple: (latitudes, longitudes, elevations) arrays
    """
    print(f"Loading DEM data from {file_path}...")

    try:
        mat_data = scipy.io.loadmat(file_path)
        dem_tiles = mat_data['DEMtiles']

        latitudes = dem_tiles[0, 0]["lat"].flatten()
        longitudes = dem_tiles[0, 0]["lon"].flatten()
        elevations = dem_tiles[0, 0]["z"]

        print(f"DEM data loaded: {len(latitudes)} x {len(longitudes)} grid")
        return latitudes, longitudes, elevations

    except Exception as e:
        print(f"Error loading DEM data: {str(e)}")
        raise


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
    print(f"Downsampling DEM data (factor: {factor})...")

    # Downsample the grid
    ds_latitudes = latitudes[::factor]
    ds_longitudes = longitudes[::factor]
    ds_elevations = elevations[::factor, ::factor]

    print(f"Downsampled grid size: {len(ds_latitudes)} x {len(ds_longitudes)}")

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
    print("Creating KD-tree for elevation lookup...")

    # Create KD-tree for efficient spatial lookups
    tree = cKDTree(grid_points)

    def get_elevation(lat, lon):
        """Get the elevation for a specific latitude/longitude pair."""
        _, index = tree.query([lat, lon])
        return elevation_values[index]

    return get_elevation


def initialize_elevation_lookup(dem_file_path=None, subset_factor=20):
    """
    Initialize the elevation lookup functionality.

    Args:
        dem_file_path (str, optional): Path to the DEM data file
        subset_factor (int, optional): Downsampling factor for the DEM data

    Returns:
        function: get_elevation function to lookup elevations
    """
    if dem_file_path is None:
        dem_file_path = DEFAULT_DEM_PATH

    # Load and prepare DEM data
    latitudes, longitudes, elevations = load_dem_data(dem_file_path)

    # Downsample the DEM data for improved performance
    grid_points, elevation_values = downsample_dem_data(
        latitudes, longitudes, elevations, subset_factor
    )

    # Create and return the lookup function
    return create_elevation_lookup(grid_points, elevation_values)


def add_elevation_to_dataframe(df, get_elevation, lat_col='latitude', lon_col='longitude', elev_col='altitude'):
    """
    Add elevation data to a dataframe based on latitude and longitude.

    Args:
        df (pd.DataFrame): DataFrame containing latitude and longitude columns
        get_elevation (function): Function to lookup elevation for given coordinates
        lat_col (str): Name of the latitude column
        lon_col (str): Name of the longitude column
        elev_col (str): Name for the elevation column

    Returns:
        pd.DataFrame: DataFrame with elevation column updated
    """
    # Check if the required columns exist
    if lat_col not in df.columns or lon_col not in df.columns:
        raise ValueError(f"DataFrame must contain {lat_col} and {lon_col} columns")

    # Create a copy to avoid modifying the original
    result_df = df.copy()

    # Process in batches with progress bar
    print(f"Adding elevation data to {len(df)} points...")
    elevations = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing elevations"):
        elevation = get_elevation(row[lat_col], row[lon_col])
        elevations.append(elevation)

    # Add or update the elevation column
    result_df[elev_col] = elevations

    # Calculate elevation statistics
    min_elev = result_df[elev_col].min()
    max_elev = result_df[elev_col].max()
    mean_elev = result_df[elev_col].mean()

    print(f"Elevation statistics: Min={min_elev:.1f}m, Max={max_elev:.1f}m, Mean={mean_elev:.1f}m")

    return result_df


def process_csv_file(csv_path, get_elevation, output_dir=None, overwrite=False):
    """
    Process a single CSV file: add elevation data and save the result.

    Args:
        csv_path (str): Path to the CSV file
        get_elevation (function): Function to lookup elevation for given coordinates
        output_dir (str, optional): Directory to save the output CSV
        overwrite (bool): Whether to overwrite the original file

    Returns:
        str: Path to the output CSV file
    """
    print(f"Processing {csv_path}...")

    # Load the CSV file
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded CSV with {len(df)} rows and {len(df.columns)} columns")
    except Exception as e:
        print(f"Error loading CSV file: {str(e)}")
        return None

    # Add elevation data
    try:
        result_df = add_elevation_to_dataframe(df, get_elevation)
    except Exception as e:
        print(f"Error adding elevation data: {str(e)}")
        return None

    # Determine output path
    if overwrite:
        output_path = csv_path
    else:
        base_name = os.path.basename(csv_path)
        if output_dir:
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, base_name)
        else:
            # Add suffix to filename in the same directory
            dir_name = os.path.dirname(csv_path)
            name, ext = os.path.splitext(base_name)
            output_path = os.path.join(dir_name, f"{name}_with_elevation{ext}")

    # Save the result
    try:
        result_df.to_csv(output_path, index=False)
        print(f"Saved result to {output_path}")
        return output_path
    except Exception as e:
        print(f"Error saving result: {str(e)}")
        return None


def process_directory(input_dir, get_elevation, pattern="*.csv", output_dir=None, overwrite=False):
    """
    Process all CSV files in a directory.

    Args:
        input_dir (str): Directory containing CSV files
        get_elevation (function): Function to lookup elevation for given coordinates
        pattern (str): Glob pattern to match CSV files
        output_dir (str, optional): Directory to save the output CSV files
        overwrite (bool): Whether to overwrite the original files

    Returns:
        tuple: (processed_count, failed_count, output_files)
    """
    # Find all matching CSV files
    input_pattern = os.path.join(input_dir, pattern)
    csv_files = glob.glob(input_pattern)

    if not csv_files:
        print(f"No files matching {input_pattern} found")
        return 0, 0, []

    print(f"Found {len(csv_files)} CSV files to process")

    # Process each file
    processed_count = 0
    failed_count = 0
    output_files = []

    start_time = time.time()

    for csv_file in csv_files:
        output_path = process_csv_file(csv_file, get_elevation, output_dir, overwrite)

        if output_path:
            processed_count += 1
            output_files.append(output_path)
        else:
            failed_count += 1

    elapsed_time = time.time() - start_time

    # Print summary
    print("\n" + "=" * 50)
    print(f"Processing completed in {elapsed_time:.2f} seconds")
    print(f"Successfully processed {processed_count} files")
    if failed_count > 0:
        print(f"Failed to process {failed_count} files")
    print("=" * 50)

    return processed_count, failed_count, output_files


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Add elevation data to PRISM CSV files")
    parser.add_argument("input", help="Input CSV file or directory")
    parser.add_argument("--output-dir", "-o", default=None,
                        help="Directory to save output files (default: same as input)")
    parser.add_argument("--pattern", "-p", default="*.csv",
                        help="File pattern to match when input is a directory (default: *.csv)")
    parser.add_argument("--dem-file", "-d", default=DEFAULT_DEM_PATH,
                        help=f"Path to DEM data file (default: {DEFAULT_DEM_PATH})")
    parser.add_argument("--downsample", "-s", type=int, default=20,
                        help="Downsample factor for DEM data (default: 20)")
    parser.add_argument("--overwrite", "-w", action="store_true",
                        help="Overwrite input files instead of creating new ones")

    args = parser.parse_args()

    # Initialize elevation lookup
    try:
        get_elevation = initialize_elevation_lookup(args.dem_file, args.downsample)
    except Exception as e:
        print(f"Failed to initialize elevation lookup: {str(e)}")
        return 1

    # Process input
    if os.path.isdir(args.input):
        # Process directory
        process_directory(
            args.input,
            get_elevation,
            args.pattern,
            args.output_dir,
            args.overwrite
        )
    else:
        # Process single file
        process_csv_file(
            args.input,
            get_elevation,
            args.output_dir,
            args.overwrite
        )

    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        exit(exit_code)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        exit(1)
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        exit(1)