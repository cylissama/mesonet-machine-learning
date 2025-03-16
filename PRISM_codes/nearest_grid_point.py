#!/usr/bin/env python
# Kelcee Gabbard

"""
DEM Elevation Lookup Utility
(DEM Data not included in repo)

This script loads Digital Elevation Model (DEM) data from a MATLAB (.mat) file,
creates a spatial index, and provides efficient lookup of elevation values
for given latitude/longitude coordinates.

The script demonstrates:
1. Loading and processing DEM data
2. Spatial indexing with KD-trees for efficient nearest-neighbor searches
3. Lookup functionality for elevation data
"""

import time
import numpy as np
import pandas as pd
import scipy.io
from scipy.spatial import cKDTree


def load_dem_data(file_path):
    """
    Load DEM data from a MATLAB file.

    Args:
        file_path (str): Path to the MATLAB file containing DEM data

    Returns:
        tuple: (latitudes, longitudes, elevations) arrays
    """
    mat_data = scipy.io.loadmat(file_path)
    dem_tiles = mat_data['DEMtiles']

    latitudes = dem_tiles[0, 0]["lat"].flatten()
    longitudes = dem_tiles[0, 0]["lon"].flatten()
    elevations = dem_tiles[0, 0]["z"]

    return latitudes, longitudes, elevations


def downsample_dem_data(latitudes, longitudes, elevations, factor):
    """
    Downsample DEM data to reduce computational complexity.

    Args:
        latitudes (np.ndarray): Array of latitude values
        longitudes (np.ndarray): Array of longitude values
        elevations (np.ndarray): 2D array of elevation values
        factor (int): Downsampling factor (take every Nth point)

    Returns:
        tuple: (downsampled_latitudes, downsampled_longitudes, downsampled_elevations,
                grid_points, elevation_values)
    """
    # Downsample the grid
    ds_latitudes = latitudes[::factor]
    ds_longitudes = longitudes[::factor]
    ds_elevations = elevations[::factor, ::factor]

    # Create coordinate grid
    lon_grid, lat_grid = np.meshgrid(ds_longitudes, ds_latitudes)
    grid_points = np.column_stack((lat_grid.ravel(), lon_grid.ravel()))
    elevation_values = ds_elevations.ravel()

    return ds_latitudes, ds_longitudes, ds_elevations, grid_points, elevation_values


def create_elevation_lookup(grid_points, elevation_values):
    """
    Create a function to look up elevations for given coordinates.

    Args:
        grid_points (np.ndarray): Array of (lat, lon) coordinates
        elevation_values (np.ndarray): Corresponding elevation values

    Returns:
        function: Lookup function that takes (lat, lon) and returns elevation
    """
    # Create KD-tree for efficient spatial lookups
    tree = cKDTree(grid_points)

    def get_elevation(lat, lon):
        """
        Get the elevation for a specific latitude/longitude pair.

        Args:
            lat (float): Latitude
            lon (float): Longitude

        Returns:
            float: Elevation value at the nearest grid point
        """
        _, index = tree.query([lat, lon])
        return elevation_values[index]

    def get_elevation_with_details(lat, lon):
        """
        Get elevation and detailed information for a specific latitude/longitude pair.

        Args:
            lat (float): Latitude
            lon (float): Longitude

        Returns:
            tuple: (elevation, nearest_lat, nearest_lon, distance)
        """
        dist, index = tree.query([lat, lon])
        nearest_lat, nearest_lon = grid_points[index]
        elevation = elevation_values[index]

        print(f"Input: {lat}, {lon} -> Nearest Grid Point: {nearest_lat}, {nearest_lon} "
              f"-> Elevation: {elevation} (Distance: {dist})")

        return elevation

    return get_elevation, get_elevation_with_details


def main():
    """Main execution function."""
    # Start timer for performance measurement
    start_time = time.time()

    # Load DEM data from MATLAB file
    dem_file_path = "/Volumes/Mesonet/spring_ml/DEMdata/DEMdata.mat"
    latitudes, longitudes, elevations = load_dem_data(dem_file_path)

    # Downsample the DEM data for improved performance
    subset_factor = 20
    _, _, _, grid_points, elevation_values = downsample_dem_data(
        latitudes, longitudes, elevations, subset_factor
    )

    # Create lookup functions
    get_elevation, get_elevation_with_details = create_elevation_lookup(
        grid_points, elevation_values
    )

    # Sample data for testing
    test_data = pd.DataFrame({
        "Latitude": [37.5, 38.0, 37.8],
        "Longitude": [-85.0, -84.5, -85.2]
    })

    # Add elevations to the test data
    test_data["Elevation"] = test_data.apply(
        lambda row: get_elevation(row["Latitude"], row["Longitude"]),
        axis=1
    )

    # Display results
    print("\nElevation lookup results:")
    print(test_data)

    # Calculate and display execution time
    end_time = time.time()
    print(f"\nExecution time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()