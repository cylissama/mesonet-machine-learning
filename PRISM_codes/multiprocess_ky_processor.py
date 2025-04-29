#!/usr/bin/env python3
"""
Kentucky PRISM Data Processor - Optimized for Apple M2 Chip

This script processes PRISM BIL files by:
1. Subsetting them to the Kentucky region
2. Converting to CSV format
3. Adding elevation data
4. Preparing for pyMICA analysis

Enhanced to process multiple years for a single variable.
Performance-optimized for Apple Silicon (M2) processors.
Author: Based on code by Cy Dixon, KY MESONET @ WKU
"""

import os
import time
import argparse
import re
import traceback
import multiprocessing
from multiprocessing import Pool, Lock, Manager, cpu_count
import threading
from queue import Queue
from datetime import datetime
from functools import partial

import numpy as np
import pandas as pd
import rasterio
import scipy.io
from scipy.spatial import cKDTree

# Kentucky bounding box (WGS84 coordinates)
KY_MIN_LON, KY_MAX_LON = -89.813, -81.688
KY_MIN_LAT, KY_MAX_LAT = 36.188, 39.438

# Global variables for elevation lookup
# We'll initialize these in each process as needed due to multiprocessing
ELEVATION_LOOKUP = None
ELEVATION_GRID_POINTS = None
ELEVATION_VALUES = None

# Thread safety for elevation lookup initialization
ELEVATION_LOCK = threading.Lock()

# Multiprocessing print lock for clean console output
PRINT_LOCK = None  # Will be initialized in main()


def safe_print(*args, **kwargs):
    """Thread and process safe printing to prevent output interleaving"""
    if PRINT_LOCK is not None:
        with PRINT_LOCK:
            print(*args, **kwargs)
    else:
        print(*args, **kwargs)


def load_dem_data(file_path):
    """
    Load DEM data from a MATLAB file.

    Args:
        file_path (str): Path to the MATLAB file containing DEM data

    Returns:
        tuple: (latitudes, longitudes, elevations) arrays
    """
    safe_print(f"Loading DEM data from {file_path}...")
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
    # KD-tree is particularly efficient on M2 chips due to their vector processing capabilities
    tree = cKDTree(grid_points)

    def get_elevation(lat, lon):
        """Get the elevation for a specific latitude/longitude pair."""
        _, index = tree.query([lat, lon])
        return elevation_values[index]

    return get_elevation


def initialize_elevation_lookup(dem_file_path, factor=20):
    """
    Initialize the elevation lookup function, optimized for multiprocessing.

    This function is process-aware: each process needs its own copy of the lookup function
    since multiprocessing doesn't share memory like threading does.

    Args:
        dem_file_path (str): Path to the DEM data file
        factor (int): Downsampling factor for the DEM data

    Returns:
        function: Elevation lookup function
    """
    global ELEVATION_LOOKUP, ELEVATION_GRID_POINTS, ELEVATION_VALUES

    # Each process needs its own copy of the lookup
    if ELEVATION_LOOKUP is None and dem_file_path:
        # We still use thread locking within each process for safety
        with ELEVATION_LOCK:
            if ELEVATION_LOOKUP is None:
                start_time = time.time()
                process_id = multiprocessing.current_process().name
                safe_print(f"{process_id}: Initializing elevation lookup table...")

                # Load and downsample DEM data
                latitudes, longitudes, elevations = load_dem_data(dem_file_path)
                ELEVATION_GRID_POINTS, ELEVATION_VALUES = downsample_dem_data(
                    latitudes, longitudes, elevations, factor
                )

                # Create lookup function
                ELEVATION_LOOKUP = create_elevation_lookup(ELEVATION_GRID_POINTS, ELEVATION_VALUES)

                elapsed = time.time() - start_time
                safe_print(f"{process_id}: Elevation lookup table initialized in {elapsed:.2f} seconds")

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
    safe_print(f"Subsetting {bil_file} to Kentucky region...")

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
        safe_print(f"Extracted {len(df)} Kentucky points in {elapsed:.2f} seconds")

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

    # Reorder columns - avoid using swapaxes/reindex which may call deprecated functions
    ordered_columns = base_columns + value_columns + ['timestamp']
    existing_columns = [col for col in ordered_columns if col in station_data.columns]

    # Use a more direct approach to reorder columns instead of reindex
    # This avoids the deprecated swapaxes warning
    station_data = station_data[existing_columns].copy()

    elapsed = time.time() - start_time
    safe_print(f"Prepared station data in {elapsed:.2f} seconds")

    return station_data


def add_elevation_to_dataframe(df, get_elevation, num_threads=2):
    """
    Add elevation data to a dataframe based on latitude and longitude.
    Uses threading for faster elevation lookups (I/O bound task).

    Optimized for M2 chip by limiting threads to 2-3 for elevation lookups
    since this task doesn't benefit from too many threads and we want to
    preserve resources for other parallel processes.

    Parameters:
    - df (pd.DataFrame): DataFrame containing 'latitude' and 'longitude' columns
    - get_elevation (function): Function to lookup elevation for given coordinates
    - num_threads (int): Number of threads to use for parallel processing

    Returns:
    - pd.DataFrame: DataFrame with 'altitude' column updated
    """
    start_time = time.time()
    process_id = multiprocessing.current_process().name
    safe_print(f"{process_id}: Adding elevation data using {num_threads} threads...")

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
                    safe_print(
                        f"{process_id}: Progress: {progress:.1f}% ({processed_count}/{total_rows} coordinates processed)")

        # Put results in the queue
        result_queue.put((chunk_id, chunk_elevations))

    # Split the dataframe into chunks
    # For M2, smaller chunks work better due to cache size and efficiency cores
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
    safe_print(f"{process_id}: Added elevation data in {elapsed:.2f} seconds")

    return df


def process_bil_file(params):
    """
    Process a single BIL file: subset to Kentucky, add elevation, and prepare for pyMICA.

    Modified to accept a single params dict for multiprocessing support.

    Returns:
    - dict: Result information including output path, processing time, etc.
    """
    # Unpack parameters
    bil_file_path = params['bil_file_path']
    output_folder = params['output_folder']
    file_index = params['file_index']
    total_files = params['total_files']
    subsample = params['subsample']
    random_state = params['random_state']
    dem_file_path = params['dem_file_path']
    elevation_threads = params['elevation_threads']

    try:
        process_id = multiprocessing.current_process().name
        filename = os.path.basename(bil_file_path)
        safe_print(f"{process_id}: Processing {file_index + 1}/{total_files}: {filename}")

        file_start_time = time.time()

        # Parse the filename to extract metadata
        pattern = r"^(\w+)_(\w+)_(\w+)_(\w+)_(\d{8})\.bil$"
        match = re.match(pattern, filename)

        if not match:
            safe_print(f"{process_id}: Skipped non-matching file: {filename}")
            return {
                'file_index': file_index,
                'output_file': None,
                'processing_time': 0,
                'success': False,
                'error': "Filename did not match expected pattern"
            }

        # Extract metadata from filename
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
            # Get the elevation lookup function (initialize in this process if needed)
            get_elevation = initialize_elevation_lookup(dem_file_path)

            # Add elevation data with specified number of threads
            # For M2 processors, 2-3 threads is optimal for this task
            ky_data_df = add_elevation_to_dataframe(ky_data_df, get_elevation, num_threads=elevation_threads)

        # Prepare data for pyMICA
        processed_df = prep_station_data(ky_data_df, random_state=random_state)

        # Save to CSV
        save_start_time = time.time()
        processed_df.to_csv(csv_output, index=False)
        save_elapsed = time.time() - save_start_time
        safe_print(f"{process_id}: Saved {len(processed_df)} points to {csv_output} in {save_elapsed:.2f} seconds")

        file_elapsed = time.time() - file_start_time

        # Print summary
        safe_print(f"{process_id}: Processed: {filename} in {file_elapsed:.2f} seconds")
        safe_print(f"{process_id}: ├─ Product: {product}")
        safe_print(f"{process_id}: ├─ Variable: {variable}")
        safe_print(f"{process_id}: ├─ Region: Kentucky (subset from {region})")
        safe_print(f"{process_id}: ├─ Resolution: {resolution}")
        safe_print(f"{process_id}: └─ Date: {formatted_date[:10]}")

        return {
            'file_index': file_index,
            'output_file': csv_output,
            'processing_time': file_elapsed,
            'success': True,
            'error': None,
            'year': year
        }

    except Exception as e:
        error_message = f"Error processing {os.path.basename(bil_file_path)}: {str(e)}"
        safe_print(error_message)
        # Print the full traceback for debugging
        safe_print(traceback.format_exc())

        return {
            'file_index': file_index,
            'output_file': None,
            'processing_time': 0,
            'success': False,
            'error': error_message,
            'year': None
        }


def process_directory(input_dir, output_dir, pattern=None, subsample=1, random_state=None, dem_file_path=None,
                      max_workers=None, elevation_threads=2):
    """
    Process all BIL files in a directory in parallel using multiprocessing.

    Multiprocessing is more efficient than threading for CPU-bound tasks on M2 chips
    because it avoids Python's Global Interpreter Lock (GIL).

    Parameters:
    - input_dir (str): Directory containing BIL files to process
    - output_dir (str): Directory where CSV files will be saved
    - pattern (str, optional): Only process files matching this pattern
    - subsample (int): Subsampling factor for data reduction
    - random_state (int): Random seed for reproducible processing
    - dem_file_path (str): Path to the DEM data file
    - max_workers (int, optional): Maximum number of parallel processes.
      For M2 chip, 4-6 is optimal (4 performance cores + a couple efficiency cores)
    - elevation_threads (int): Number of threads to use for elevation lookups within each process.
      For M2 chip, 2-3 is optimal.

    Returns:
    - tuple: (Number of files processed, List of output files)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Record start time
    total_start_time = time.time()

    safe_print(f"Starting to process BIL files in: {input_dir}")
    safe_print(f"Output will be saved to: {output_dir}")
    safe_print(f"Subsample factor: {subsample}")
    safe_print(f"Kentucky bounding box: Lon ({KY_MIN_LON}, {KY_MAX_LON}), Lat ({KY_MIN_LAT}, {KY_MAX_LAT})")

    # Determine the optimal number of workers for M2 chip
    if max_workers is None:
        # M2 chip typically has 8 cores (4 performance, 4 efficiency)
        # Using 4-6 processes is optimal to leave resources for the OS
        m2_optimal = min(6, cpu_count() - 1)
        max_workers = max(1, m2_optimal)

    safe_print(f"Using {max_workers} parallel processes for file processing")
    safe_print(f"Using {elevation_threads} threads for elevation lookups within each process")
    safe_print("-" * 50)

    # Get all BIL files and sort them alphanumerically
    bil_files = [f for f in os.listdir(input_dir) if f.endswith(".bil")]

    # Natural sort for filenames (handles numbers properly)
    def natural_sort_key(s):
        return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

    # Sort files alphanumerically
    bil_files.sort(key=natural_sort_key)

    # Filter files by pattern if specified
    if pattern:
        bil_files = [f for f in bil_files if pattern in f]

    safe_print(f"Found {len(bil_files)} BIL files to process")

    # Prepare parameters for each file
    file_params = []
    for i, filename in enumerate(bil_files):
        file_params.append({
            'bil_file_path': os.path.join(input_dir, filename),
            'output_folder': output_dir,
            'file_index': i,
            'total_files': len(bil_files),
            'subsample': subsample,
            'random_state': random_state,
            'dem_file_path': dem_file_path,
            'elevation_threads': elevation_threads
        })

    # Process files using multiprocessing Pool
    # This is more efficient than threading for CPU-bound tasks on M2 chips
    # because it avoids Python's Global Interpreter Lock (GIL)
    output_files = []
    error_files = []
    file_times = []

    # Results by year for summary
    results_by_year = {}

    with Pool(processes=max_workers) as pool:
        results = pool.map(process_bil_file, file_params)

    # Process results
    for result in results:
        if result['success']:
            output_files.append(result['output_file'])
            file_times.append(result['processing_time'])

            # Track files by year for summary
            year = result['year']
            if year not in results_by_year:
                results_by_year[year] = {'count': 0, 'time': 0}
            results_by_year[year]['count'] += 1
            results_by_year[year]['time'] += result['processing_time']
        else:
            error_files.append(bil_files[result['file_index']])

    # Calculate processing time
    total_elapsed = time.time() - total_start_time
    processed_count = len(output_files)

    # Print summary
    safe_print("\n" + "=" * 50)
    safe_print(f"Parallel processing completed in {total_elapsed:.2f} seconds")
    if file_times:
        avg_time = sum(file_times) / len(file_times)
        safe_print(f"Average processing time per file: {avg_time:.2f} seconds")
        safe_print(f"Fastest file: {min(file_times):.2f} seconds")
        safe_print(f"Slowest file: {max(file_times):.2f} seconds")
        safe_print(f"Throughput: {len(file_times) / total_elapsed:.2f} files/second")
    safe_print(f"Processed {processed_count} files")
    safe_print(f"Generated {len(output_files)} CSV files")

    # Print results by year
    if results_by_year:
        safe_print("\nResults by year:")
        for year, stats in sorted(results_by_year.items()):
            if year is not None:  # Skip None years (errors)
                safe_print(f"  Year {year}: {stats['count']} files processed in {stats['time']:.2f} seconds")

    if error_files:
        safe_print(f"\nFailed to process {len(error_files)} files")
        for err_file in error_files:
            safe_print(f" - {err_file}")
    safe_print("=" * 50)

    return processed_count, output_files


def process_variable_across_years(base_dir, variable, years, output_dir, subsample=1, random_state=None,
                                  dem_file_path=None, max_workers=None, elevation_threads=2):
    """
    Process all data for a specific variable across multiple years.

    Parameters:
    - base_dir (str): Base directory containing PRISM data
    - variable (str): Variable to process (e.g., 'ppt', 'tmean')
    - years (list): List of years to process
    - output_dir (str): Directory where CSV files will be saved
    - subsample (int): Subsampling factor for data reduction
    - random_state (int): Random seed for reproducible processing
    - dem_file_path (str): Path to the DEM data file
    - max_workers (int): Maximum number of parallel processes
    - elevation_threads (int): Number of threads to use for elevation lookups

    Returns:
    - dict: Summary of processed files by year
    """
    total_start_time = time.time()

    safe_print(f"\n{'=' * 70}")
    safe_print(f"PROCESSING VARIABLE: {variable} ACROSS {len(years)} YEARS")
    safe_print(f"{'=' * 70}\n")

    summary = {}
    total_files_processed = 0

    # Process each year
    for year in years:
        year_dir = os.path.join(base_dir, variable, 'daily', str(year))

        if not os.path.exists(year_dir):
            safe_print(f"WARNING: Directory not found for {variable}/{year}: {year_dir}")
            continue

        safe_print(f"\n{'-' * 70}")
        safe_print(f"Processing {variable} data for year {year}")
        safe_print(f"Source directory: {year_dir}")
        safe_print(f"{'-' * 70}\n")

        try:
            year_start_time = time.time()

            # Process all files in this year's directory
            processed_count, output_files = process_directory(
                year_dir,
                output_dir,
                pattern=str(year),
                subsample=subsample,
                random_state=random_state,
                dem_file_path=dem_file_path,
                max_workers=max_workers,
                elevation_threads=elevation_threads
            )

            year_elapsed = time.time() - year_start_time

            # Add to summary
            summary[year] = {
                'files_processed': processed_count,
                'processing_time': year_elapsed,
                'output_files': len(output_files)
            }

            total_files_processed += processed_count

            safe_print(f"\nCompleted processing {variable} data for year {year}")
            safe_print(f"Processed {processed_count} files in {year_elapsed:.2f} seconds")

        except Exception as e:
            safe_print(f"ERROR processing {variable} for year {year}: {str(e)}")
            traceback.print_exc()
            summary[year] = {
                'files_processed': 0,
                'processing_time': 0,
                'output_files': 0,
                'error': str(e)
            }

    # Print overall summary
    total_elapsed = time.time() - total_start_time

    safe_print(f"\n{'#' * 70}")
    safe_print(f"VARIABLE {variable} PROCESSING COMPLETE")
    safe_print(f"Total files processed: {total_files_processed}")
    safe_print(f"Total processing time: {total_elapsed:.2f} seconds")
    safe_print(f"Years processed: {', '.join(str(y) for y in years)}")
    safe_print(f"{'#' * 70}\n")

    return summary


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
        safe_print("Error: No variable column found in the CSV file")
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
    safe_print(f"Generated pyMICA sample in {elapsed:.2f} seconds")

    # If output_dir is provided, create a structured path for the sample data
    if output_dir:
        sample_dir = os.path.join(output_dir, variable, year)
        os.makedirs(sample_dir, exist_ok=True)
        return sample_data, os.path.join(sample_dir, f"pymica_sample_{variable}_{year}.json")

    return sample_data


def detect_available_years(base_dir, variable):
    """
    Detect available years for a specific variable in the directory structure.

    Parameters:
    - base_dir (str): Base directory containing PRISM data
    - variable (str): Variable to check (e.g., 'ppt', 'tmean')

    Returns:
    - list: List of available years (as strings)
    """
    var_dir = os.path.join(base_dir, variable, 'daily')

    # Check if the variable directory exists
    if not os.path.exists(var_dir):
        safe_print(f"WARNING: Directory not found for {variable}: {var_dir}")
        return []

    # List all subdirectories (years)
    try:
        year_dirs = [d for d in os.listdir(var_dir) if os.path.isdir(os.path.join(var_dir, d))]

        # Filter to ensure they are valid years
        valid_years = [y for y in year_dirs if y.isdigit() and len(y) == 4]
        valid_years.sort()  # Sort chronologically

        return valid_years
    except Exception as e:
        safe_print(f"Error detecting years for {variable}: {str(e)}")
        return []


def main():
    """Main execution function."""
    # Initialize multiprocessing with 'spawn' method for better compatibility
    # on macOS with M2 chip - this prevents potential issues with fork() on macOS
    multiprocessing.set_start_method('spawn', force=True)

    # Setup manager for shared resources across processes
    manager = Manager()
    global PRINT_LOCK
    PRINT_LOCK = manager.Lock()

    total_start_time = time.time()

    parser = argparse.ArgumentParser(description="Process PRISM BIL files for Kentucky region - Optimized for M2 Mac")
    parser.add_argument("input_dir",
                        help="Base directory containing PRISM data with structure: /base/variable/daily/year")
    parser.add_argument("--variable", required=True,
                        help="Variable to process (e.g., 'ppt', 'tmean')")
    parser.add_argument("--years", default=None,
                        help="Years to process (comma-separated, e.g., '2019,2020,2021'). If omitted, all available years will be processed.")
    parser.add_argument("--output_dir", default=None,
                        help="Directory where CSV files will be saved (default: /Volumes/PRISMdata/pymica_PRISM_data/)")
    parser.add_argument("--subsample", type=int, default=1,
                        help="Subsampling factor to reduce data size (default: 1)")
    parser.add_argument("--random_state", type=int, default=42,
                        help="Random seed for reproducible processing (default: 42)")
    parser.add_argument("--dem_file", default=None,
                        help="Path to the DEM data file for elevation data")
    parser.add_argument("--pymica_sample", action="store_true",
                        help="Generate pyMICA sample data from the first processed file of each year")
    parser.add_argument("--processes", type=int, default=None,
                        help="Number of processes to use (default: optimal for M2 chip, typically 4-6)")
    parser.add_argument("--elevation_threads", type=int, default=2,
                        help="Number of threads to use for elevation lookups within each process (default: 2)")

    args = parser.parse_args()

    # Use default output directory if not specified
    if args.output_dir is None:
        args.output_dir = "/Volumes/PRISMdata/pymica_PRISM_data/"

    # Determine years to process
    if args.years:
        years_to_process = [y.strip() for y in args.years.split(",")]
    else:
        # Auto-detect available years
        years_to_process = detect_available_years(args.input_dir, args.variable)
        if not years_to_process:
            safe_print(f"ERROR: No valid years found for {args.variable} in {args.input_dir}")
            return

    # Print the runtime configuration
    safe_print("\nKentucky PRISM Data Processor - M2 Mac Optimized with Multi-Year Support")
    safe_print(f"Run at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    safe_print(f"Base directory: {args.input_dir}")
    safe_print(f"Variable: {args.variable}")
    safe_print(f"Years to process: {', '.join(years_to_process)}")
    safe_print(f"Output directory: {args.output_dir}")
    safe_print(f"Subsample factor: {args.subsample}")
    safe_print(f"Random state: {args.random_state}")

    # M2-specific configuration information
    process_count = args.processes if args.processes else min(6, cpu_count() - 1)
    safe_print(f"M2 Optimization: Using {process_count} processes (optimal for M2 chip)")
    safe_print(f"M2 Optimization: Using {args.elevation_threads} threads per process for elevation lookups")
    safe_print(f"M2 Optimization: Using 'spawn' multiprocessing method for macOS compatibility")
    safe_print("\n")

    try:
        # Process the variable across all specified years
        summary = process_variable_across_years(
            args.input_dir,
            args.variable,
            years_to_process,
            args.output_dir,
            subsample=args.subsample,
            random_state=args.random_state,
            dem_file_path=args.dem_file,
            max_workers=args.processes,
            elevation_threads=args.elevation_threads
        )

        # Generate pyMICA samples if requested
        if args.pymica_sample:
            safe_print("\nGenerating pyMICA sample data from each year...")

            for year in years_to_process:
                year_dir = os.path.join(args.output_dir, args.variable, year)
                if os.path.exists(year_dir):
                    # Find the first CSV file for this year
                    csv_files = [f for f in os.listdir(year_dir) if f.endswith(".csv")]
                    if csv_files:
                        first_file = os.path.join(year_dir, csv_files[0])
                        safe_print(f"\nGenerating pyMICA sample data from {year}: {first_file}")

                        try:
                            sample_data, sample_path = generate_pymica_sample(
                                first_file,
                                sample_size=10,
                                output_dir=args.output_dir
                            )

                            # Save sample data to a JSON file
                            import json
                            with open(sample_path, 'w') as f:
                                json.dump(sample_data, f, indent=2)
                            safe_print(f"Saved pyMICA sample data for {year} to: {sample_path}")
                        except Exception as e:
                            safe_print(f"Error generating pyMICA sample for {year}: {str(e)}")

    except Exception as e:
        safe_print(f"Error in main execution: {str(e)}")
        traceback.print_exc()

    # Total runtime
    total_elapsed = time.time() - total_start_time
    safe_print(f"\nTotal script runtime: {total_elapsed:.2f} seconds")


if __name__ == "__main__":
    main()