# Cy Dixon 2025, KY MESONET @ WKU
# convert bil file to csv file then add timestamps for the day to the csv file
# prep data function also included

import rasterio
import numpy as np
import pandas as pd
from datetime import datetime
import os
import re

def prep_station_data(file_path, random_state=None): #  UNTESTED
    """
    Prepares station data for analysis by adding synthetic columns (for now) and reordering columns

    Parameters:
    file_path (str): Path to the input CSV file
    random_state (int): Seed for reproducible random numbers (default: None)

    Returns:
    pd.DataFrame: Processed DataFrame with added columns and reordered columns
    """
    # Read raw data
    station_data = pd.read_csv(file_path)

    # Add synthetic columns with optional reproducibility
    rng = np.random.default_rng(random_state)

    station_data['key'] = range(1, len(station_data) + 1)
    station_data['dist'] = rng.uniform(1, 100, len(station_data))
    station_data['hr'] = rng.integers(0, 24, len(station_data))
    station_data['altitude'] = rng.uniform(50, 500, len(station_data))

    # Define column order
    base_columns = ['key', 'altitude', 'dist', 'hr', 'latitude', 'longitude', 'tmean']
    other_columns = [col for col in station_data.columns if col not in base_columns]

    # Reorder columns
    station_data = station_data.reindex(columns=base_columns + other_columns)

    # data prep for pymica
    data = []
    for key in station_data['key'].head(10):
        df_data = station_data[station_data['key'] == key]
        # df_meta = metadata[metadata['key'] == key]
        data.append(
            {
                'id': key,
                'longitude': float(df_data['longitude'].iloc[0]),
                'latitude': float(df_data['latitude'].iloc[0]),
                'value': float(df_data['tmean'].iloc[0]),  # value for 'tmean' is manually placed
                'altitude': float(df_data['altitude'].iloc[0]),
                'dist': float(df_data['dist'].iloc[0])
            }
        )

    print('Sample data: ', data[0])
    print('Number of points: ', len(data))

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

def main():
    # Configuration
    base_data_path = "/Volumes/Mesonet/PRISMdata_2021/tmean/2021/prism_tmean_us_30s_20210101.bil" # could loop over

    bil_file = "prism_tmean_us_30s_20210101.bil"
    bil_file = base_data_path + bil_file

    path_without_extension = base_data_path.split(".")[0]
    parts = path_without_extension.split("_")

    product = parts[0]  # "prism"
    variable = parts[1]  # "tmean"
    region = parts[2]  # "us"
    resolution = parts[3]  # "30s"
    date_str = parts[4]  # "20210101" (YYYYMMDD format)


    # Extract the base filename from bil_file and create CSV output path
    base_name = bil_file.split('/')[-1].replace('.bil', '')
    csv_output = f"/Volumes/Mesonet/spring_ml/output_data/{base_name}_CONV.csv"
    subsample = 1  # Process every Nth pixel to reduce file size (set to 1 for all data)
    timestamp = "2021-01-01 00:00:00"  # Start date for timestamps

    bil_data_df = read_bil_to_dataframe(bil_file, variable, timestamp, subsample)
    save_dataframe_to_csv(bil_data_df, csv_output)

# example loop
'''
    # Path to your data folder
    data_folder = "/Volumes/Mesonet/spring_ml/PRISM_data/PRISM_Tmean2021/"

    # Regex pattern to parse filenames
    pattern = r"^(\w+)_(\w+)_(\w+)_(\w+)_(\d{8})\.bil$"

    n = 0
    # Loop through all files in directory
    for filename in os.listdir(data_folder):
        if filename.endswith(".bil"):
            match = re.match(pattern, filename)
            if match:
                # Extract components
                product = match.group(1)
                variable = match.group(2)
                region = match.group(3)
                resolution = match.group(4)
                date_str = match.group(5)

                # Format date as YYYY-MM-DD
                formatted_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"

                # Print results
                print(f"File: {filename}")
                print(f"├─ Product: {product}")
                print(f"├─ Variable: {variable}")
                print(f"├─ Region: {region}")
                print(f"├─ Resolution: {resolution}")
                print(f"└─ Date: {formatted_date}\n")
                n += 1
            else:
                print(f"Skipped non-matching file: {filename}")
'''

if __name__ == "__main__":
    main()