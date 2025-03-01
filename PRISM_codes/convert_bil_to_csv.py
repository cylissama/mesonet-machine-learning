# Cy Dixon 2025, MESONET WKU
# convert bil file to csv file then add timestamps for the day to the csv file

import rasterio
import numpy as np
import pandas as pd
from datetime import datetime

# Configuration
BASE_DATA_PATH = "/Volumes/Mesonet/PRISMdata_2021/tmean/2021/"
var_type = BASE_DATA_PATH.rstrip('/').split('/')[-2]  # gets variable name
year = BASE_DATA_PATH.rstrip('/').split('/')[-1]      # gets year, if needed
BASE_DATA_DIR = BASE_DATA_PATH
BIL_FILE = "prism_tmean_us_30s_20210101.bil"
BIL_FILE = BASE_DATA_DIR + BIL_FILE
# Extract the base filename from BIL_FILE and create CSV output path
base_name = BIL_FILE.split('/')[-1].replace('.bil', '')
CSV_OUTPUT = f"/Volumes/Mesonet/spring_ml/output_data/{base_name}_CONV.csv"
SUBSAMPLE = 1  # Process every Nth pixel to reduce file size (set to 1 for all data)
START_DATE = "2021-01-01 00:00:00"  # Start date for timestamps

# Read BIL file
with rasterio.open(BIL_FILE) as src:
    # Get raster data
    data = src.read(1)  # Read first band
    nodata = src.nodata
    transform = src.transform

    # Generate coordinates
    xs, ys = np.meshgrid(np.arange(src.width), np.arange(src.height))
    xs, ys = rasterio.transform.xy(transform, ys, xs)
    xs = np.array(xs).flatten()
    ys = np.array(ys).flatten()

# Create DataFrame
bil_data_df = pd.DataFrame({
    'longitude': xs[::SUBSAMPLE],
    'latitude': ys[::SUBSAMPLE],
    var_type: data.flatten()[::SUBSAMPLE]
})

# add a column for timestamps
timestamps = [datetime.strptime(START_DATE, "%Y-%m-%d %H:%M:%S") for _ in range(len(bil_data_df))]
timestamps_df = pd.DataFrame(timestamps, columns=["timestamp"])
timestamped_data = pd.concat([bil_data_df, timestamps_df], axis=1)

# Filter out nodata values
# this was why it appeared that the timestamps were incorrect, i understand it now
if nodata is not None:
    timestamped_data = timestamped_data[timestamped_data[var_type] != nodata]

# THIS TOOK 2m 15s TO RUN AND GENERATED A 1.2 GB CSV FILE
# THIS HAD BEEN FIXED, RATHER AVOIDED

# Save to CSV
timestamped_data.to_csv(CSV_OUTPUT, index=False)
print(f"Saved {len(timestamped_data)} points to {CSV_OUTPUT}")