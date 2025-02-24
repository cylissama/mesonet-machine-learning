# convert bil file to csv file

import rasterio
import numpy as np
import pandas as pd

# Configuration
BIL_FILE = "/Volumes/Mesonet/spring_ml/PRISM_data/prism_2020_ppt_data/PRISM_ppt_stable_4kmM3_2020_bil.bil"  
# Your input file
CSV_OUTPUT = "precipitation_2020.csv"                  # Output CSV
SUBSAMPLE = 10  # Process every Nth pixel to reduce file size (set to 1 for all data)

# Read BIL file
with rasterio.open(BIL_FILE) as src:
    # Get raster data and metadata
    data = src.read(1)  # Read first band
    transform = src.transform
    nodata = src.nodata

    # Generate coordinate grids
    rows, cols = np.indices(data.shape)
    xs, ys = rasterio.transform.xy(transform, rows.flatten(), cols.flatten())

# Create DataFrame
df = pd.DataFrame({
    'longitude': xs[::SUBSAMPLE],
    'latitude': ys[::SUBSAMPLE],
    'precipitation': data.flatten()[::SUBSAMPLE]
})

# Filter out nodata values
df = df[df['precipitation'] != nodata]

# Save to CSV
df.to_csv(CSV_OUTPUT, index=False)
print(f"Saved {len(df)} points to {CSV_OUTPUT}")