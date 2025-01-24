import matplotlib.pyplot as plot
import numpy as np
from osgeo import gdal

path = "/Volumes/Mesonet/winter_break/PRISM_data/4km_tmax_12/PRISM_tmax_30yr_normal_4kmM5_12_bil/PRISM_tmax_30yr_normal_4kmM5_12_bil.bil"

data = gdal.Open(path)

band = data.GetRasterBand(1)

array = band.ReadAsArray().astype(np.float32)

array[array == -9999] = np.nan

data = None

plot.imshow(array, cmap='viridis')
plot.show()