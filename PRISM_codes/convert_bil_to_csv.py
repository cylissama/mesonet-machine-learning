# Cy Dixon 2025, KY MESONET @ WKU
# convert bil file to csv file then add timestamps for the day to the csv file
# prep data function also included

import rasterio
import numpy as np
import pandas as pd
from datetime import datetime
import os
import re


def prep_station_data(station_data, random_state=None):
    """
    Prepares station data for analysis by adding synthetic columns and reordering columns

    Parameters:
    station_data (pd.DataFrame): DataFrame containing station data
    random_state (int): Seed for reproducible random numbers (default: None)

    Returns:
    pd.DataFrame: Processed DataFrame with added columns and reordered columns
    """
    # Add synthetic columns with optional reproducibility
    rng = np.random.default_rng(random_state)

    station_data['key'] = range(1, len(station_data) + 1)
    station_data['dist'] = 0.0
    station_data['hr'] = 0.0
    station_data['altitude'] = 0.0

    # Define column order
    base_columns = ['key', 'altitude', 'dist', 'hr', 'latitude', 'longitude', 'tmean']
    other_columns = [col for col in station_data.columns if col not in base_columns and col in station_data.columns]

    # Reorder columns (only include columns that exist in the DataFrame)
    existing_columns = [col for col in base_columns if col in station_data.columns] + other_columns
    station_data = station_data.reindex(columns=existing_columns)

    # data prep for pymica
    data = []
    for key in station_data['key'].head(10):
        df_data = station_data[station_data['key'] == key]
        value_col = 'tmean' if 'tmean' in station_data.columns else station_data.columns[0]

        data.append(
            {
                'id': key,
                'longitude': float(df_data['longitude'].iloc[0]),
                'latitude': float(df_data['latitude'].iloc[0]),
                'value': float(df_data[value_col].iloc[0]),
                'altitude': float(df_data['altitude'].iloc[0]),
                'dist': float(df_data['dist'].iloc[0])
            }
        )

    # print('Sample data: ', data[0])
    # print('Number of points: ', len(data))

    return station_data


def read_bil_to_dataframe(bil_file, var_type, timestamp, subsample=1):
    """
    Reads a BIL file and returns a Pandas DataFrame with subsampled (because of nodata) longitude, latitude, variable data, and timestamps.

    Parameters:
    - bil_file (str): Path to the BIL file.
    - var_type (str): Column name for the extracted data values.
    - timestamp (str): Start date in the format "%Y-%m-%d %H:%M:%S".
    - subsample (int, optional): Subsampling factor to reduce data size. Default is 1 (no subsampling).

    Returns:
    - pd.DataFrame: DataFrame containing 'longitude', 'latitude', the variable column, and 'timestamp'.
    """
    with rasterio.open(bil_file) as src:
        # Read first band
        data = src.read(1)
        nodata = src.nodata
        transform = src.transform

        # Generate coordinates
        xs, ys = np.meshgrid(np.arange(src.width), np.arange(src.height))
        xs, ys = rasterio.transform.xy(transform, ys, xs)
        xs = np.array(xs).flatten()
        ys = np.array(ys).flatten()
        data = data.flatten()

        # Create timestamps
        timestamps = [datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S") for _ in range(len(xs))]

        # Create DataFrame with subsampling
        df = pd.DataFrame({
            'longitude': xs[::subsample],
            'latitude': ys[::subsample],
            var_type: data[::subsample],
            'timestamp': [timestamps[i] for i in range(0, len(timestamps), subsample)]
        })

        # Filter out NoData values
        if nodata is not None:
            df = df[df[var_type] != nodata]

    return df


def save_dataframe_to_csv(df, output_path):
    """
    Saves a Pandas DataFrame to a CSV file.

    Parameters:
    - df (pd.DataFrame): The DataFrame to save.
    - output_path (str): The file path where the CSV should be saved.

    Returns:
    - None
    """
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} points to {output_path}")


def main(bil_file_path, output_folder=None, subsample=1, random_state=None):
    """
    Process a single BIL file, convert it to CSV with integrated data preparation.

    Parameters:
    - bil_file_path (str): Path to the BIL file to process
    - output_folder (str, optional): Folder to save output CSV files.
      If None, uses '/Volumes/Mesonet/spring_ml/output_data/'
    - subsample (int, optional): Subsampling factor to reduce data size. Default is 1 (no subsampling).
    - random_state (int, optional): Random seed for reproducible processing

    Returns:
    - str: Path to the output CSV file
    """
    if output_folder is None:
        output_folder = "/Volumes/Mesonet/spring_ml/output_data/"

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

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

        # Create output CSV filename
        base_name = filename.replace('.bil', '')
        csv_output = os.path.join(output_folder, f"{base_name}_CONV.csv")

        # Process the BIL file
        print(f"Reading BIL file: {bil_file_path}")
        bil_data_df = read_bil_to_dataframe(bil_file_path, variable, formatted_date, subsample)

        # Apply the prep_station_data function to further process the data
        print(f"Preparing station data...")
        processed_df = prep_station_data(bil_data_df, random_state=random_state)

        # Save the processed data to CSV
        save_dataframe_to_csv(processed_df, csv_output)

        print(f"Processed: {filename}")
        print(f"├─ Product: {product}")
        print(f"├─ Variable: {variable}")
        print(f"├─ Region: {region}")
        print(f"├─ Resolution: {resolution}")
        print(f"└─ Date: {formatted_date[:10]}\n")

        return csv_output
    else:
        print(f"Skipped non-matching file: {filename}")
        return None


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        print("Usage: python convert_bil_to_csv.py <path_to_bil_file>")