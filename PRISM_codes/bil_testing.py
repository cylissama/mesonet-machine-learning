# convert bil file to csv file

import rasterio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Configuration
BIL_FILE = "/Volumes/Mesonet/spring_ml/PRISM_data/PRISM_Tmean2021/prism_tmean_us_30s_20210101.bil"
CSV_OUTPUT = "/Volumes/Mesonet/spring_ml/PRISM_data/converted_data/prism_tmean_us_30s_20210101.csv"  # Output CSV
SUBSAMPLE = 1  # Process every Nth pixel to reduce file size (set to 1 for all data)
START_DATE = "2021-01-01 00:00:00"  # Start date for timestamps
TIME_INTERVAL = 30  # Time interval in seconds

# Read BIL file
with rasterio.open(BIL_FILE) as data:

    # Get raster data
    band1 = data.read(1)  # Read first band, returns numpy.ndarry

    print(data.count)
    print(data.width)
    print(data.height)
    print(data.bounds)
    print("Indexes: ", data.indexes)

    print("NO DATA: ", data.nodata)
    msk = data.read_masks(1)
    print("Mask Shape: ", msk.shape)
    print("MSK: ", msk)

    plt.figure(figsize=(10, 8))
    plt.imshow(msk, cmap='gray')
    plt.colorbar(label='Mask Value')
    plt.title('PRISM Data Mask')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.show()