# Cy Dixon 2025, MESONET WKU
# convert bil file to csv file then add timestamps for the day to the csv file

import rasterio
import numpy as np
import pandas as pd
from datetime import datetime

def read_bil_to_dataframe(bil_file, var_type, timestamp, subsample=1):
    """
    Reads a BIL file and returns a Pandas DataFrame with subsampled longitude, latitude, variable data, and timestamps.

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
    base_data_path = "/Volumes/Mesonet/PRISMdata_2021/tmean/2021/"
    var_type = base_data_path.rstrip('/').split('/')[-2]  # gets variable name
    year = base_data_path.rstrip('/').split('/')[-1]  # gets year, if needed
    base_data_dir = base_data_path
    bil_file = "prism_tmean_us_30s_20210101.bil"
    bil_file = base_data_dir + bil_file
    # Extract the base filename from bil_file and create CSV output path
    base_name = bil_file.split('/')[-1].replace('.bil', '')
    csv_output = f"/Volumes/Mesonet/spring_ml/output_data/{base_name}_CONV.csv"
    subsample = 1  # Process every Nth pixel to reduce file size (set to 1 for all data)
    timestamp = "2021-01-01 00:00:00"  # Start date for timestamps

    bil_data_df = read_bil_to_dataframe(bil_file, var_type, timestamp, subsample)
    save_dataframe_to_csv(bil_data_df, csv_output)

if __name__ == "__main__":
    main()