#!/usr/bin/env python3
# Driver code for processing multiple BIL files
# Cy Dixon 2025, KY MESONET @ WKU

'''
run with
python master_conversion_driver.py /Volumes/PRISMdata/PRISM_data/an/ppt/daily/2019
'''

import argparse
import os
import time
from datetime import datetime
from convert_bil_to_csv import prep_station_data, save_dataframe_to_csv
from dem_kd_tree import load_dem_data, downsample_dem_data, create_elevation_lookup


def initialize_elevation_lookup(dem_file_path=None, subset_factor=20):
    """
    Initialize the elevation lookup functionality.

    Parameters:
    - dem_file_path (str, optional): Path to the DEM data file
    - subset_factor (int, optional): Downsampling factor for the DEM data

    Returns:
    - function: get_elevation function to lookup elevations
    """
    if dem_file_path is None:
        dem_file_path = "/Volumes/Mesonet/spring_ml/DEMdata/DEMdata.mat"

    # Load and prepare DEM data
    print(f"Loading DEM data from {dem_file_path}...")
    latitudes, longitudes, elevations = load_dem_data(dem_file_path)

    # Downsample the DEM data for improved performance
    print(f"Downsampling DEM data (factor: {subset_factor})...")
    _, _, _, grid_points, elevation_values = downsample_dem_data(
        latitudes, longitudes, elevations, subset_factor
    )

    # Create and return the lookup function
    print("Creating elevation lookup function...")
    get_elevation, _ = create_elevation_lookup(grid_points, elevation_values)

    return get_elevation


def add_elevation_to_dataframe(df, get_elevation):
    """
    Add elevation data to a dataframe based on latitude and longitude.

    Parameters:
    - df (pd.DataFrame): DataFrame containing 'latitude' and 'longitude' columns
    - get_elevation (function): Function to lookup elevation for given coordinates

    Returns:
    - pd.DataFrame: DataFrame with 'altitude' column updated
    """
    # Update the altitude column directly
    df['altitude'] = df.apply(
        lambda row: get_elevation(row['latitude'], row['longitude']),
        axis=1
    )

    print(f"Added elevation data to {len(df)} points")
    return df


def process_directory(input_dir, output_dir, pattern=None, subsample=1, random_state=None, dem_file_path=None):
    """
    Process all BIL files in a directory in alphanumeric order.

    Parameters:
    - input_dir (str): Directory containing BIL files to process
    - output_dir (str): Directory where CSV files will be saved
    - pattern (str, optional): Only process files matching this pattern (e.g., "2021")
    - subsample (int, optional): Subsampling factor for data reduction
    - random_state (int, optional): Random seed for reproducible processing
    - dem_file_path (str, optional): Path to the DEM data file

    Returns:
    - tuple: (Number of files processed, List of output files)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Track files processed and output files generated
    processed_count = 0
    output_files = []
    error_files = []

    # Record start time
    start_time = time.time()

    print(f"Starting to process BIL files in: {input_dir}")
    print(f"Output will be saved to: {output_dir}")
    print(f"Subsample factor: {subsample}")
    print("-" * 50)

    # Initialize elevation lookup function (do this once for all files)
    get_elevation = initialize_elevation_lookup(dem_file_path)

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

    # Process files in order
    for file_index, filename in enumerate(bil_files):
        # Full path to the file
        bil_file_path = os.path.join(input_dir, filename)

        try:
            # Parse the filename to extract metadata (similar logic as in convert_bil_to_csv.py)
            import re
            pattern = r"^(\w+)_(\w+)_(\w+)_(\w+)_(\d{8})\.bil$"
            match = re.match(pattern, os.path.basename(bil_file_path))

            if not match:
                print(f"Skipped non-matching file: {filename}")
                continue

            product = match.group(1)
            variable = match.group(2)
            region = match.group(3)
            resolution = match.group(4)
            date_str = match.group(5)

            # Format date for timestamp
            formatted_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]} 00:00:00"

            # Create output CSV filename
            base_name = os.path.basename(bil_file_path).replace('.bil', '')
            csv_output = os.path.join(output_dir, f"{base_name}_CONV.csv")

            # Use the process_bil_file function from convert_bil_to_csv but intercept before saving
            from convert_bil_to_csv import read_bil_to_dataframe

            print(f"Processing {file_index + 1}/{len(bil_files)}: {filename}")

            # Read the BIL file to a DataFrame
            bil_data_df = read_bil_to_dataframe(bil_file_path, variable, formatted_date, subsample)

            # Prepare the station data
            processed_df = prep_station_data(bil_data_df, random_state=random_state)

            # Add elevation data
            print("Adding elevation data...")
            enhanced_df = add_elevation_to_dataframe(processed_df, get_elevation)

            # Save the enhanced data to CSV
            save_dataframe_to_csv(enhanced_df, csv_output)

            output_files.append(csv_output)
            processed_count += 1

            print(f"Processed: {filename}")
            print(f"├─ Product: {product}")
            print(f"├─ Variable: {variable}")
            print(f"├─ Region: {region}")
            print(f"├─ Resolution: {resolution}")
            print(f"└─ Date: {formatted_date[:10]}\n")

        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            error_files.append(filename)

    # Calculate processing time
    elapsed_time = time.time() - start_time

    # Print summary
    print("\n" + "=" * 50)
    print(f"Processing completed in {elapsed_time:.2f} seconds")
    print(f"Processed {processed_count} files")
    print(f"Generated {len(output_files)} CSV files")
    if error_files:
        print(f"Failed to process {len(error_files)} files")
        for err_file in error_files:
            print(f" - {err_file}")
    print("=" * 50)

    return processed_count, output_files


def main():
    """
    Main function to parse command line arguments and process BIL files.
    """
    parser = argparse.ArgumentParser(description="Process multiple BIL files in a directory")
    parser.add_argument("input_dir", help="Directory containing BIL files to process")
    parser.add_argument("--output_dir", default=None,
                        help="Directory where CSV files will be saved (default: /Volumes/Mesonet/spring_ml/output_data/)")
    parser.add_argument("--pattern", default=None, help="Only process files matching this pattern (e.g., '2021')")
    parser.add_argument("--subsample", type=int, default=1,
                        help="Subsampling factor to reduce data size (default: 1, no subsampling)")
    parser.add_argument("--random_state", type=int, default=42,
                        help="Random seed for reproducible processing (default: 42)")
    parser.add_argument("--dem_file", default=None,
                        help="Path to the DEM data file (default: /Volumes/Mesonet/spring_ml/DEMdata/DEMdata.mat)")

    args = parser.parse_args()

    # Use default output directory if not specified
    if args.output_dir is None:
        args.output_dir = "/Volumes/Mesonet/spring_ml/output_data/"

    # Print the runtime configuration
    print("\nBIL File Processing Script")
    print(f"Run at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    if args.pattern:
        print(f"Processing files matching: {args.pattern}")
    print(f"Random state: {args.random_state}")
    print("\n")

    # Process the directory
    process_directory(
        args.input_dir,
        args.output_dir,
        args.pattern,
        args.subsample,
        args.random_state,
        args.dem_file
    )


if __name__ == "__main__":
    main()