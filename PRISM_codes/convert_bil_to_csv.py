# convert bil file to csv file

import rasterio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Configuration
BIL_FILE = "/Volumes/Mesonet/spring_ml/PRISM_data/PRISM_Tmean2021/prism_tmean_us_30s_20210101.bil"
CSV_OUTPUT = "test.csv"  # Output CSV
SUBSAMPLE = 10  # Process every Nth pixel to reduce file size (set to 1 for all data)
START_DATE = "2021-01-01"  # Start date for timestamps
TIME_INTERVAL = 30  # Time interval in seconds

# Read BIL file
with rasterio.open(BIL_FILE) as src:
    # Get raster data
    data = src.read(1)  # Read first band
    nodata = src.nodata
    transform = src.transform

    print("Name: ", src.name)
    print("Width: ", src.width)
    print("Height: ", src.height)
    print("Bounds: ", src.bounds)
    print({i: dtype for i, dtype in zip(src.indexes, src.dtypes)})

    # Generate coordinates
    xs, ys = np.meshgrid(np.arange(src.width), np.arange(src.height))
    xs, ys = rasterio.transform.xy(transform, ys, xs)
    xs = np.array(xs).flatten()
    ys = np.array(ys).flatten()

# Generate timestamps
start_date = datetime.strptime(START_DATE, "%Y-%m-%d")
num_points = len(xs[::SUBSAMPLE])
timestamps = [start_date + timedelta(seconds=i * TIME_INTERVAL) for i in range(num_points)]

# Create DataFrame
df = pd.DataFrame({
    'timestamp': timestamps,
    'longitude': xs[::SUBSAMPLE],
    'latitude': ys[::SUBSAMPLE],
    'value': data.flatten()[::SUBSAMPLE]
})

# Filter out nodata values
if nodata is not None:
    df = df[df['value'] != nodata]

# Save to CSV
df.to_csv(CSV_OUTPUT, index=False)
print(f"Saved {len(df)} points to {CSV_OUTPUT}")