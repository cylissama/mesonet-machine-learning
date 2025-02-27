import numpy as np
import rasterio
from matplotlib import pyplot


src = rasterio.open("/Volumes/Mesonet/spring_ml/PRISM_data/PRISM_Tmean2021/prism_tmean_us_30s_20210101.bil")
pyplot.imshow(src.read(1), cmap='viridis')
pyplot.show()