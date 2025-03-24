import rasterio
from rasterio.transform import from_origin
from statsmodels.compat import scipy


# Convert your .mat data to GeoTIFF once
def convert_mat_to_geotiff(mat_file, output_tiff):
    mat_data = scipy.io.loadmat(mat_file)
    dem_tiles = mat_data['DEMtiles']

    latitudes = dem_tiles[0, 0]["lat"].flatten()
    longitudes = dem_tiles[0, 0]["lon"].flatten()
    elevations = dem_tiles[0, 0]["z"]

    # Calculate resolution and define transform
    res_x = abs(longitudes[1] - longitudes[0])
    res_y = abs(latitudes[1] - latitudes[0])
    transform = from_origin(longitudes.min(), latitudes.max(), res_x, res_y)

    # Write to GeoTIFF
    with rasterio.open(
            output_tiff,
            'w',
            driver='GTiff',
            height=elevations.shape[0],
            width=elevations.shape[1],
            count=1,
            dtype=elevations.dtype,
            crs='+proj=longlat +datum=WGS84 +no_defs',
            transform=transform,
    ) as dst:
        dst.write(elevations, 1)


# Then for lookups, much faster
def create_fast_elevation_lookup(tiff_file):
    dataset = rasterio.open(tiff_file)

    def get_elevation(lat, lon):
        # Convert coordinates to pixel indices
        row, col = dataset.index(lon, lat)
        # Read the single pixel value
        elevation = dataset.read(1, window=((row, row + 1), (col, col + 1)))
        return elevation[0, 0]

    return get_elevation