# convert bil file to csv file

import rasterio
import numpy as np
import pandas as pd

# Configuration
BIL_FILE = "/Volumes/Mesonet/spring_ml/PRISM_data/an/ppt/daily/2019/prism_ppt_us_30s_20190101.bil"  
# /Volumes/Mesonet/spring_ml/PRISM_data/4km/ppt/2020/PRISM_ppt_stable_4kmM3_2020_all_bil/PRISM_ppt_stable_4kmM3_2020_bil.bil
# Your input file
CSV_OUTPUT = "prism_ppt_us_30s_20190101.csv"                  # Output CSV
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