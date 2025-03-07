import scipy.io
from scipy.spatial import cKDTree
import numpy as np
import pandas as pd
import time

#start timer testing purposes for optimization 

start_time = time.time()

#load matlab file 
mat_data = scipy.io.loadmat("/Users/kelceegabbard/Downloads/DEMdata.mat")


test_prism_data = pd.DataFrame({
    "Latitude": [37.5, 38.0, 37.8],
    "Longitude": [-85.0, -84.5, -85.2]
})
#extract DEMtiles 
dem_tiles = mat_data['DEMtiles']
#print(mat_data.keys())

#print to check data type 
'''
print(type(dem_tiles))
print(dem_tiles.shape) #checks dimensions if an array 
print(dem_tiles.dtype) #checks data type of array
'''

latitudes = dem_tiles[0, 0]["lat"]
longitudes = dem_tiles[0,0]["lon"]
elevations = dem_tiles[0, 0]["z"]

#check type and shapes
'''
print(type(latitudes), type(longitudes), type(elevations))
print(latitudes.shape, longitudes.shape, elevations.shape)
'''

latitudes = latitudes.flatten()
longitudes = longitudes.flatten()  

### ''' SUBSET DEM GRID (DOWNSAMPLING FOR SPEED ''' ###
subset_factor = 20 #downsample factor

latitudes = latitudes[::subset_factor] #every Nth latitude (10)
longitudes = longitudes[::subset_factor] #every Nth longitude (10)
elevations = elevations[::subset_factor, ::subset_factor] #donwsampled elevation grid


lon_grid, lat_grid = np.meshgrid(longitudes, latitudes) #grid
grid_points = np.column_stack((lat_grid.ravel(), lon_grid.ravel()))
elevation_vals = elevations.ravel() #flattened elevation values

tree = cKDTree(grid_points) #creates tree from grid points


def get_nearest_elevation(lat, lon):
    _, index = tree.query([lat, lon]) # find nearest index in DEM dataset
    return elevation_vals[index] #returns corresponding elevation


def get_nearest_debug(lat, lon):
    dist, index = tree.query([lat, lon]) #find nearest point
    nearest_lat, nearest_lon = grid_points[index] #get nearest lat/lon 
    nearest_z = elevation_vals[index] #get nearest elevation

    print(f"Input: {lat}, {lon} -> Nearest Grid Point: {nearest_lat}, {nearest_lon} -> Elevation: {nearest_z} (Distance: {dist})")

'''
test_prism_data["Elevation"] = test_prism_data.apply(
    lambda row: get_nearest_debug(row["Latitude"], row["Longitude"]), axis=1
    )
'''

test_prism_data["Elevation"] = test_prism_data.apply(
    lambda row: get_nearest_elevation(row["Latitude"], row["Longitude"]), axis=1
    )

print(test_prism_data)

end_time = time.time()

print(f"Time taken: {end_time - start_time:.2f} seconds")