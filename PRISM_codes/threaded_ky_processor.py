#!/usr/bin/env python3
"""
Kentucky PRISM Data Processor

This script processes PRISM BIL files by:
1. Subsetting them to the Kentucky region
2. Converting to CSV format
3. Adding elevation data
4. Preparing for pyMICA analysis

!!Bottleneck here is still getting the elevation!!

Author: Based on code by Cy Dixon, KY MESONET @ WKU
"""

import os
import time
import argparse
import re
import traceback
from datetime import datetime
import concurrent.futures
import threading
from queue import Queue

import numpy as np
import pandas as pd
import rasterio
import scipy.io
from scipy.spatial import cKDTree

# Kentucky bounding box (WGS84 coordinates)
KY_MIN_LON, KY_MAX_LON = -89.813, -81.688
KY_MIN_LAT, KY_MAX_LAT = 36.188, 39.438

# Global variables for elevation lookup
ELEVATION_LOOKUP = None
ELEVATION_GRID_POINTS = None
ELEVATION_VALUES = None

# Thread safety for elevation lookup initialization
ELEVATION_LOCK = threading.Lock()


def load_dem_data(file_path):
    """
    Load DEM data from a MATLAB file.

    Args:
        file_path (str): Path to the MATLAB file containing DEM data

    Returns:
        tuple: (latitudes, longitudes, elevations) arrays
    """
    print(f"Loading DEM data from {file_path}...")
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


def initialize_elevation_lookup(dem_file_path, factor=20):
    """
    Initialize the global elevation lookup function if not already initialized.
    Thread-safe implementation to avoid multiple initializations in parallel processing.

    Args:
        dem_file_path (str): Path to the DEM data file
        factor (int): Downsampling factor for the DEM data

    Returns:
        function: Elevation lookup function
    """
    global ELEVATION_LOOKUP, ELEVATION_GRID_POINTS, ELEVATION_VALUES

    # Check if we need to initialize (outside the lock for better performance)
    if ELEVATION_LOOKUP is None and dem_file_path:
        # Use a lock to prevent multiple threads from initializing simultaneously
        with ELEVATION_LOCK:
            # Check again in case another thread initialized while we were waiting
            if ELEVATION_LOOKUP is None:
                start_time = time.time()
                print("Initializing elevation lookup table (one-time operation)...")

                # Load and downsample DEM data
                latitudes, longitudes, elevations = load_dem_data(dem_file_path)
                ELEVATION_GRID_POINTS, ELEVATION_VALUES = downsample_dem_data(
                    latitudes, longitudes, elevations, factor
                )

                # Create lookup function
                ELEVATION_LOOKUP = create_elevation_lookup(ELEVATION_GRID_POINTS, ELEVATION_VALUES)

                elapsed = time.time() - start_time
                print(f"Elevation lookup table initialized in {elapsed:.2f} seconds")

    return ELEVATION_LOOKUP


def subset_bil_to_kentucky(bil_file, var_type, timestamp, subsample=1):
    """
    Extracts Kentucky region data from a BIL file and returns as DataFrame.

    Parameters:
    - bil_file (str): Path to the BIL file
    - var_type (str): Column name for the extracted data values
    - timestamp (str): Start date in the format "YYYY-MM-DD HH:MM:SS"
    - subsample (int): Subsampling factor to reduce data size

    Returns:
    - pd.DataFrame: DataFrame containing Kentucky points with coordinates, values, and timestamps
    """
    start_time = time.time()
    print(f"Subsetting {bil_file} to Kentucky region...")

    with rasterio.open(bil_file) as src:
        # Get metadata
        transform = src.transform
        nodata = src.nodata

        # Define Kentucky window in pixel coordinates
        window = src.window(
            left=KY_MIN_LON,
            bottom=KY_MIN_LAT,
            right=KY_MAX_LON,
            top=KY_MAX_LAT
        )

        # Read just the Kentucky subset
        subset = src.read(1, window=window)
        subset_transform = src.window_transform(window)

        # Generate coordinates for each pixel in the subset
        rows, cols = np.indices(subset.shape)
        lons, lats = rasterio.transform.xy(
            subset_transform,
            rows.flatten(),
            cols.flatten()
        )

        # Create timestamps array (same timestamp for all points)
        timestamps = [datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
                      for _ in range(len(lons[::subsample]))]

        # Create DataFrame with subsampling
        df = pd.DataFrame({
            'longitude': lons[::subsample],
            'latitude': lats[::subsample],
            var_type: subset.flatten()[::subsample],
            'timestamp': timestamps
        })

        # Filter out NoData values
        if nodata is not None:
            df = df[df[var_type] != nodata]

        elapsed = time.time() - start_time
        print(f"Extracted {len(df)} Kentucky points in {elapsed:.2f} seconds")

    return df


def prep_station_data(station_data, random_state=None):
    """
    Prepares station data for pyMICA by adding required columns and reordering.

    Parameters:
    - station_data (pd.DataFrame): DataFrame containing station data
    - random_state (int): Seed for reproducible random numbers

    Returns:
    - pd.DataFrame: Processed DataFrame ready for pyMICA
    """
    start_time = time.time()

    # Add required columns
    station_data['key'] = range(1, len(station_data) + 1)
    station_data['dist'] = 0.0
    station_data['hr'] = 0.0

    # If 'altitude' column doesn't exist, initialize it
    if 'altitude' not in station_data.columns:
        station_data['altitude'] = 0.0

    # Define column order for pyMICA
    base_columns = ['key', 'altitude', 'dist', 'hr', 'latitude', 'longitude']

    # Add the variable column (could be 'tmean', 'ppt', etc.)
    value_columns = [col for col in station_data.columns
                     if col not in base_columns + ['timestamp', 'key', 'altitude', 'dist', 'hr']]

    # Reorder columns
    ordered_columns = base_columns + value_columns + ['timestamp']
    existing_columns = [col for col in ordered_columns if col in station_data.columns]
    station_data = station_data.reindex(columns=existing_columns)

    elapsed = time.time() - start_time
    print(f"Prepared station data in {elapsed:.2f} seconds")

    return station_data


def add_elevation_to_dataframe(df, get_elevation, num_threads=4):
    """
    Add elevation data to a dataframe based on latitude and longitude.
    Uses parallel processing for faster elevation lookups.

    Parameters:
    - df (pd.DataFrame): DataFrame containing 'latitude' and 'longitude' columns
    - get_elevation (function): Function to lookup elevation for given coordinates
    - num_threads (int): Number of threads to use for parallel processing

    Returns:
    - pd.DataFrame: DataFrame with 'altitude' column updated
    """
    start_time = time.time()
    print(f"Adding elevation data using {num_threads} threads...")

    # Set up progress tracking
    total_rows = len(df)
    progress_lock = threading.Lock()
    processed_count = 0
    report_interval = max(1, total_rows // 10)  # Report progress every 10%

    # Function to process a chunk of rows
    def process_chunk(chunk, result_queue, chunk_id):
        nonlocal processed_count
        chunk_elevations = []

        for i, (_, row) in enumerate(chunk.iterrows()):
            elevation = get_elevation(row['latitude'], row['longitude'])
            chunk_elevations.append((row.name, elevation))

            # Update progress counter (thread-safe)
            with progress_lock:
                processed_count += 1
                if processed_count % report_interval == 0:
                    progress = processed_count / total_rows * 100
                    print(f"Progress: {progress:.1f}% ({processed_count}/{total_rows} coordinates processed)")

        # Put results in the queue
        result_queue.put((chunk_id, chunk_elevations))

    # Split the dataframe into chunks
    chunks = np.array_split(df, num_threads)
    result_queue = Queue()
    threads = []

    # Start threads to process chunks
    for i, chunk in enumerate(chunks):
        thread = threading.Thread(target=process_chunk, args=(chunk, result_queue, i))
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    # Collect results and update the dataframe
    elevations_dict = {}
    while not result_queue.empty():
        _, elevations = result_queue.get()
        for row_idx, elev in elevations:
            elevations_dict[row_idx] = elev

    # Create a Series from the dictionary and assign to the dataframe
    df['altitude'] = pd.Series(elevations_dict)

    elapsed = time.time() - start_time
    print(f"Added elevation data in {elapsed:.2f} seconds")

    return df


def process_bil_file(bil_file_path, output_folder, subsample=1, random_state=None, dem_file_path=None,
                     elevation_threads=4):
    """
    Process a single BIL file: subset to Kentucky, add elevation, and prepare for pyMICA.

    Parameters:
    - bil_file_path (str): Path to the BIL file
    - output_folder (str): Base folder to save output CSV files (will create subfolders)
    - subsample (int): Subsampling factor to reduce data size
    - random_state (int): Random seed for reproducible processing
    - dem_file_path (str): Path to the DEM data file
    - elevation_threads (int): Number of threads to use for elevation lookups

    Returns:
    - str: Path to the output CSV file
    """
    file_start_time = time.time()

    # Extract filename from path
    filename = os.path.basename(bil_file_path)

    # Parse the filename to extract metadata
    pattern = r"^(\w+)_(\w+)_(\w+)_(\w+)_(\d{8})\.bil$"
    match = re.match(pattern, filename)

    if match:
        product = match.group(1)
        variable = match.group(2)
        region = match.group(3)
        resolution = match.group(4)
        date_str = match.group(5)

        # Format date for timestamp
        formatted_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]} 00:00:00"
        year = date_str[:4]

        # Create structured output path: /output_folder/variable/year/
        var_year_folder = os.path.join(output_folder, variable, year)
        os.makedirs(var_year_folder, exist_ok=True)

        # Create output CSV filename
        base_name = filename.replace('.bil', '')
        csv_output = os.path.join(var_year_folder, f"{base_name}_KY_CONV.csv")

        # Process the BIL file - subset to Kentucky
        ky_data_df = subset_bil_to_kentucky(
            bil_file_path,
            variable,
            formatted_date,
            subsample
        )

        # If DEM file path is provided, add elevation data
        if dem_file_path:
            # Get the global elevation lookup function (initialize if needed)
            get_elevation = initialize_elevation_lookup(dem_file_path)

            # Add elevation data with specified number of threads
            ky_data_df = add_elevation_to_dataframe(ky_data_df, get_elevation, num_threads=elevation_threads)

        # Prepare data for pyMICA
        processed_df = prep_station_data(ky_data_df, random_state=random_state)

        # Save to CSV
        save_start_time = time.time()
        processed_df.to_csv(csv_output, index=False)
        save_elapsed = time.time() - save_start_time
        print(f"Saved {len(processed_df)} points to {csv_output} in {save_elapsed:.2f} seconds")

        file_elapsed = time.time() - file_start_time

        # Print summary
        print(f"Processed: {filename} in {file_elapsed:.2f} seconds")
        print(f"├─ Product: {product}")
        print(f"├─ Variable: {variable}")
        print(f"├─ Region: Kentucky (subset from {region})")
        print(f"├─ Resolution: {resolution}")
        print(f"└─ Date: {formatted_date[:10]}\n")

        return csv_output
    else:
        print(f"Skipped non-matching file: {filename}")
        return None


def process_file_task(args):
    """
    Helper function for parallel processing. Processes a single BIL file.

    Parameters:
    - args: Tuple containing (file_path, output_dir, file_index, total_files, subsample, random_state, dem_file_path, elevation_threads)

    Returns:
    - Tuple: (output_file, file_processing_time, file_index, error_message)
    """
    bil_file_path, output_dir, file_index, total_files, subsample, random_state, dem_file_path, elevation_threads = args
    filename = os.path.basename(bil_file_path)

    try:
        print(f"Processing {file_index + 1}/{total_files}: {filename}")
        file_start_time = time.time()

        # Process the file
        output_file = process_bil_file(
            bil_file_path,
            output_dir,
            subsample,
            random_state,
            dem_file_path,
            elevation_threads=elevation_threads  # Pass the correct parameter
        )

        file_elapsed = time.time() - file_start_time
        print(f"File {filename} processed in {file_elapsed:.2f} seconds")

        return (output_file, file_elapsed, file_index, None)
    except Exception as e:
        error_message = f"Error processing {filename}: {str(e)}"
        print(error_message)
        # Print the full traceback for debugging
        traceback.print_exc()
        return (None, 0, file_index, error_message)


def process_directory(input_dir, output_dir, pattern=None, subsample=1, random_state=None, dem_file_path=None,
                      max_workers=None, elevation_threads=4):
    """
    Process all BIL files in a directory in parallel, subsetting each to Kentucky.

    Parameters:
    - input_dir (str): Directory containing BIL files to process
    - output_dir (str): Directory where CSV files will be saved
    - pattern (str, optional): Only process files matching this pattern
    - subsample (int): Subsampling factor for data reduction
    - random_state (int): Random seed for reproducible processing
    - dem_file_path (str): Path to the DEM data file
    - max_workers (int, optional): Maximum number of parallel workers. If None, uses CPU count.
    - elevation_threads (int): Number of threads to use for elevation lookups within each file

    Returns:
    - tuple: (Number of files processed, List of output files)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Record start time
    total_start_time = time.time()

    print(f"Starting to process BIL files in: {input_dir}")
    print(f"Output will be saved to: {output_dir}")
    print(f"Subsample factor: {subsample}")
    print(f"Kentucky bounding box: Lon ({KY_MIN_LON}, {KY_MAX_LON}), Lat ({KY_MIN_LAT}, {KY_MAX_LAT})")

    # Determine the number of workers
    if max_workers is None:
        import multiprocessing
        max_workers = max(1, multiprocessing.cpu_count() - 1)  # Leave one CPU free

    print(f"Using {max_workers} parallel workers")
    print(f"Using {elevation_threads} threads for elevation lookups within each file")
    print("-" * 50)

    # Initialize elevation lookup if DEM file is provided (do this before threading)
    if dem_file_path:
        initialize_elevation_lookup(dem_file_path)

    # Get all BIL files and sort them alphanumerically
    bil_files = [f for f in os.listdir(input_dir) if f.endswith(".bil")]

    # Natural sort for filenames (handles numbers properly)
    def natural_sort_key(s):
        import re
        return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

    # Sort files alphanumerically
    bil_files.sort(key=natural_sort_key)

    # Filter files by pattern if specified
    if pattern:
        bil_files = [f for f in bil_files if pattern in f]

    print(f"Found {len(bil_files)} BIL files to process")

    # Process files in parallel
    output_files = []
    error_files = []
    file_times = []

    # Prepare arguments for each file
    file_args = [
        (os.path.join(input_dir, filename), output_dir, i, len(bil_files), subsample, random_state, dem_file_path,
         elevation_threads)
        for i, filename in enumerate(bil_files)
    ]

    # Thread-safe collections for results
    results_lock = threading.Lock()

    # Process files using ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_file_task, args) for args in file_args]

        for future in concurrent.futures.as_completed(futures):
            try:
                output_file, file_elapsed, file_index, error_message = future.result()

                with results_lock:
                    if output_file:
                        output_files.append(output_file)
                        file_times.append(file_elapsed)
                    elif error_message:
                        error_files.append(bil_files[file_index])
            except Exception as e:
                print(f"Error in future processing: {str(e)}")
                traceback.print_exc()

    # Calculate processing time
    total_elapsed = time.time() - total_start_time
    processed_count = len(output_files)

    # Print summary
    print("\n" + "=" * 50)
    print(f"Parallel processing completed in {total_elapsed:.2f} seconds")
    if file_times:
        avg_time = sum(file_times) / len(file_times)
        print(f"Average processing time per file: {avg_time:.2f} seconds")
        print(f"Fastest file: {min(file_times):.2f} seconds")
        print(f"Slowest file: {max(file_times):.2f} seconds")
        print(f"Throughput: {len(file_times) / total_elapsed:.2f} files/second")
    print(f"Processed {processed_count} files")
    print(f"Generated {len(output_files)} CSV files")
    if error_files:
        print(f"Failed to process {len(error_files)} files")
        for err_file in error_files:
            print(f" - {err_file}")
    print("=" * 50)

    return processed_count, output_files


def generate_pymica_sample(csv_path, sample_size=10, output_dir=None):
    """
    Generate sample data for pyMICA from a processed CSV file.

    Parameters:
    - csv_path (str): Path to processed CSV file
    - sample_size (int): Number of sample points to generate
    - output_dir (str): Directory to save the sample data (defaults to same directory as CSV)

    Returns:
    - list: List of dictionaries in pyMICA format
    """
    start_time = time.time()

    # Read the CSV file
    station_data = pd.read_csv(csv_path)

    # Get variable column (not one of the standard columns)
    standard_cols = ['key', 'altitude', 'dist', 'hr', 'latitude', 'longitude', 'timestamp']
    var_cols = [col for col in station_data.columns if col not in standard_cols]

    if not var_cols:
        print("Error: No variable column found in the CSV file")
        return []

    var_col = var_cols[0]  # Take the first variable column

    # Extract metadata for folder structure
    csv_basename = os.path.basename(csv_path)
    match = re.match(r"^(\w+)_(\w+)_(\w+)_(\w+)_(\d{8}).*\.csv$", csv_basename)

    if match:
        variable = match.group(2)
        date_str = match.group(5)
        year = date_str[:4]
    else:
        variable = var_col
        year = "unknown"

    # Generate sample data
    sample_size = min(sample_size, len(station_data))
    sample_data = []

    for key in station_data['key'].head(sample_size):
        df_data = station_data[station_data['key'] == key]

        sample_data.append({
            'id': int(key),
            'longitude': float(df_data['longitude'].iloc[0]),
            'latitude': float(df_data['latitude'].iloc[0]),
            'value': float(df_data[var_col].iloc[0]),
            'altitude': float(df_data['altitude'].iloc[0]),
            'dist': float(df_data['dist'].iloc[0])
        })

    elapsed = time.time() - start_time
    print(f"Generated pyMICA sample in {elapsed:.2f} seconds")

    # If output_dir is provided, create a structured path for the sample data
    if output_dir:
        sample_dir = os.path.join(output_dir, variable, year)
        os.makedirs(sample_dir, exist_ok=True)
        return sample_data, os.path.join(sample_dir, f"pymica_sample_{variable}_{year}.json")

    return sample_data


def main():
    """Main execution function."""
    total_start_time = time.time()

    parser = argparse.ArgumentParser(description="Process PRISM BIL files for Kentucky region")
    parser.add_argument("input_dir", help="Directory containing BIL files to process")
    parser.add_argument("--output_dir", default=None,
                        help="Directory where CSV files will be saved (default: /Volumes/PRISMdata/output_data/)")
    parser.add_argument("--pattern", default=None,
                        help="Only process files matching this pattern (e.g., '2021')")
    parser.add_argument("--subsample", type=int, default=5,
                        help="Subsampling factor to reduce data size (default: 5)")
    parser.add_argument("--random_state", type=int, default=42,
                        help="Random seed for reproducible processing (default: 42)")
    parser.add_argument("--dem_file", default=None,
                        help="Path to the DEM data file for elevation data")
    parser.add_argument("--pymica_sample", action="store_true",
                        help="Generate pyMICA sample data from the first processed file")
    parser.add_argument("--threads", type=int, default=None,
                        help="Number of threads to use for parallel processing (default: CPU count - 1)")
    parser.add_argument("--elevation_threads", type=int, default=4,
                        help="Number of threads to use for elevation lookups (default: 4)")

    args = parser.parse_args()

    # Use default output directory if not specified
    if args.output_dir is None:
        args.output_dir = "/Volumes/PRISMdata/output_data/"

    # Print the runtime configuration
    print("\nKentucky PRISM Data Processor")
    print(f"Run at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    if args.pattern:
        print(f"Processing files matching: {args.pattern}")
    print(f"Subsample factor: {args.subsample}")
    print(f"Random state: {args.random_state}")
    if args.threads:
        print(f"Using {args.threads} threads for file processing")
    print(f"Using {args.elevation_threads} threads for elevation lookups")
    print("\n")

    # Process the directory
    dir_start_time = time.time()
    try:
        _, output_files = process_directory(
            args.input_dir,
            args.output_dir,
            args.pattern,
            args.subsample,
            args.random_state,
            args.dem_file,
            max_workers=args.threads,
            elevation_threads=args.elevation_threads
        )
        dir_elapsed = time.time() - dir_start_time
        print(f"Directory processing completed in {dir_elapsed:.2f} seconds")

        # Generate pyMICA sample if requested
        if args.pymica_sample and output_files:
            sample_start_time = time.time()
            first_file = output_files[0]
            print(f"\nGenerating pyMICA sample data from {first_file}...")
            sample_data, sample_path = generate_pymica_sample(first_file, sample_size=10, output_dir=args.output_dir)

            print("\nSample data for pyMICA:")
            for i, sample in enumerate(sample_data, 1):
                print(f"Sample {i}:", sample)

            # Save sample data to a JSON file
            import json
            with open(sample_path, 'w') as f:
                json.dump(sample_data, f, indent=2)
            sample_elapsed = time.time() - sample_start_time
            print(f"\nSaved pyMICA sample data to: {sample_path} in {sample_elapsed:.2f} seconds")
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        traceback.print_exc()

    # Total runtime
    total_elapsed = time.time() - total_start_time
    print(f"\nTotal script runtime: {total_elapsed:.2f} seconds")


if __name__ == "__main__":
    main()