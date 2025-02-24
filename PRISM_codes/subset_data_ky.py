import rasterio
import numpy as np
import pandas as pd

# Configuration
BIL_FILE = "/Volumes/Mesonet/spring_ml/PRISM_data/prism_2020_ppt_data/PRISM_ppt_stable_4kmM3_2020_bil.bil"
CSV_OUTPUT = "kentucky_precipitation_2020.csv"
SUBSAMPLE = 5  # Use 1 for all points, 5=20% of data

# Kentucky bounding box (WGS84)
MIN_LON, MAX_LON = -89.813, -81.688
MIN_LAT, MAX_LAT = 36.188, 39.438

with rasterio.open(BIL_FILE) as src:
    # Get transform and array data
    transform = src.transform
    data = src.read(1)
    nodata = src.nodata

    # Convert geographic coordinates to pixel coordinates
    window = src.window(
        left=MIN_LON,
        bottom=MIN_LAT,
        right=MAX_LON,
        top=MAX_LAT
    )
    
    # Read just the Kentucky subset
    subset = src.read(1, window=window)
    subset_transform = src.window_transform(window)

    # Generate coordinates for subset
    rows, cols = np.indices(subset.shape)
    lons, lats = rasterio.transform.xy(
        subset_transform,
        rows.flatten(),
        cols.flatten()
    )

# Create DataFrame with subsampling
df = pd.DataFrame({
    'longitude': lons[::SUBSAMPLE],
    'latitude': lats[::SUBSAMPLE],
    'precipitation': subset.flatten()[::SUBSAMPLE]
})

# Filter valid values and save
df = df[df['precipitation'] != nodata]
df.to_csv(CSV_OUTPUT, index=False)
print(f"Saved {len(df):,} Kentucky points to {CSV_OUTPUT}")