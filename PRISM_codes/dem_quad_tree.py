#!/usr/bin/env python
# Kelcee Gabbard

"""
DEM Elevation Lookup Utility with Quadtree
(DEM Data not included in repo)

This script loads Digital Elevation Model (DEM) data from a MATLAB (.mat) file,
creates a spatial index using a quadtree, and provides efficient lookup of elevation values
for given latitude/longitude coordinates.

The script demonstrates:
1. Loading and processing DEM data
2. Spatial indexing with quadtrees for efficient nearest-neighbor searches
3. Lookup functionality for elevation data
"""

import time
import numpy as np
import pandas as pd
import scipy.io


class QuadTree:
    """
    A Quadtree implementation for spatial indexing of 2D points.
    """

    def __init__(self, boundary, capacity=4, max_depth=10):
        """
        Initialize a quadtree node.

        Args:
            boundary (tuple): (min_lat, max_lat, min_lon, max_lon) defining the region
            capacity (int): Maximum number of points per node before splitting
            max_depth (int): Maximum depth of the tree
        """
        self.boundary = boundary
        self.capacity = capacity
        self.max_depth = max_depth
        self.depth = 0
        self.points = []  # [(lat, lon, elevation), ...]
        self.divided = False
        self.northwest = None
        self.northeast = None
        self.southwest = None
        self.southeast = None

    def divide(self):
        """Split the current node into four quadrants."""
        min_lat, max_lat, min_lon, max_lon = self.boundary
        mid_lat = (min_lat + max_lat) / 2
        mid_lon = (min_lon + max_lon) / 2

        # Create boundaries for the four quadrants
        nw_boundary = (mid_lat, max_lat, min_lon, mid_lon)
        ne_boundary = (mid_lat, max_lat, mid_lon, max_lon)
        sw_boundary = (min_lat, mid_lat, min_lon, mid_lon)
        se_boundary = (min_lat, mid_lat, mid_lon, max_lon)

        # Create child nodes
        self.northwest = QuadTree(nw_boundary, self.capacity, self.max_depth)
        self.northeast = QuadTree(ne_boundary, self.capacity, self.max_depth)
        self.southwest = QuadTree(sw_boundary, self.capacity, self.max_depth)
        self.southeast = QuadTree(se_boundary, self.capacity, self.max_depth)

        # Set depth for children
        self.northwest.depth = self.depth + 1
        self.northeast.depth = self.depth + 1
        self.southwest.depth = self.depth + 1
        self.southeast.depth = self.depth + 1

        # Move existing points to appropriate children
        for point in self.points:
            self._insert_point_to_children(point)

        self.points = []
        self.divided = True

    def _insert_point_to_children(self, point):
        """Helper method to insert a point into the appropriate child node."""
        lat, lon, _ = point
        if self.northwest.contains_point(lat, lon):
            self.northwest.insert(point)
        elif self.northeast.contains_point(lat, lon):
            self.northeast.insert(point)
        elif self.southwest.contains_point(lat, lon):
            self.southwest.insert(point)
        elif self.southeast.contains_point(lat, lon):
            self.southeast.insert(point)

    def contains_point(self, lat, lon):
        """Check if a point falls within this node's boundary."""
        min_lat, max_lat, min_lon, max_lon = self.boundary
        return (min_lat <= lat <= max_lat) and (min_lon <= lon <= max_lon)

    def insert(self, point):
        """
        Insert a point (lat, lon, elevation) into the quadtree.

        Args:
            point (tuple): (latitude, longitude, elevation)

        Returns:
            bool: True if insertion was successful
        """
        # Check if point is within boundary
        lat, lon, _ = point
        if not self.contains_point(lat, lon):
            return False

        # If we have space and haven't divided yet, add the point here
        if len(self.points) < self.capacity and not self.divided and self.depth < self.max_depth:
            self.points.append(point)
            return True

        # If we're at capacity and haven't divided yet, divide this node
        if not self.divided and self.depth < self.max_depth:
            self.divide()

        # If we've already divided, insert into appropriate child
        if self.divided:
            return self._insert_point_to_children(point)

        # If we're at max depth, we'll just store more points here
        self.points.append(point)
        return True

    def query(self, lat, lon):
        """
        Find the nearest point to the given coordinates.

        Args:
            lat (float): Latitude
            lon (float): Longitude

        Returns:
            tuple: The nearest point (lat, lon, elevation)
        """
        # If node has no points and isn't divided, there's nothing to query
        if not self.points and not self.divided:
            return None

        # If we're at a leaf node or haven't divided, find the closest point from this node
        if not self.divided:
            if not self.points:
                return None

            # Find closest point
            closest_point = None
            min_dist = float('inf')

            for point in self.points:
                p_lat, p_lon, _ = point
                dist = (lat - p_lat) ** 2 + (lon - p_lon) ** 2  # Squared Euclidean distance
                if dist < min_dist:
                    min_dist = dist
                    closest_point = point

            return closest_point

        # If we've divided, first determine which quadrant the point would fall into
        min_lat, max_lat, min_lon, max_lon = self.boundary
        mid_lat = (min_lat + max_lat) / 2
        mid_lon = (min_lon + max_lon) / 2

        # Determine which quadrant should be searched first
        best_child = None
        if lat >= mid_lat:
            if lon >= mid_lon:
                best_child = self.northeast
            else:
                best_child = self.northwest
        else:
            if lon >= mid_lon:
                best_child = self.southeast
            else:
                best_child = self.southwest

        # Query the best child first
        best_point = best_child.query(lat, lon)

        # If we found a point, calculate its distance
        if best_point:
            p_lat, p_lon, _ = best_point
            best_dist = (lat - p_lat) ** 2 + (lon - p_lon) ** 2
        else:
            best_dist = float('inf')

        # Check if we need to search other quadrants
        other_children = [
            self.northwest, self.northeast, self.southwest, self.southeast
        ]
        other_children.remove(best_child)

        # For each other quadrant, check if we might find a closer point
        for child in other_children:
            min_lat, max_lat, min_lon, max_lon = child.boundary

            # Calculate distance to the closest possible point in this quadrant
            dx = max(min_lon - lon, 0, lon - max_lon)
            dy = max(min_lat - lat, 0, lat - max_lat)
            dist_to_quadrant = dx ** 2 + dy ** 2

            # If this quadrant could contain a closer point, search it
            if dist_to_quadrant < best_dist:
                point = child.query(lat, lon)
                if point:
                    p_lat, p_lon, _ = point
                    dist = (lat - p_lat) ** 2 + (lon - p_lon) ** 2
                    if dist < best_dist:
                        best_dist = dist
                        best_point = point

        return best_point


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
        tuple: (downsampled_latitudes, downsampled_longitudes, downsampled_elevations)
    """
    # Downsample the grid
    ds_latitudes = latitudes[::factor]
    ds_longitudes = longitudes[::factor]
    ds_elevations = elevations[::factor, ::factor]

    return ds_latitudes, ds_longitudes, ds_elevations


def build_quadtree(latitudes, longitudes, elevations):
    """
    Build a quadtree from the DEM data.

    Args:
        latitudes (np.ndarray): Array of latitude values
        longitudes (np.ndarray): Array of longitude values 
        elevations (np.ndarray): 2D array of elevation values

    Returns:
        QuadTree: A quadtree containing the DEM data points
    """
    # Create boundary for the entire dataset
    min_lat, max_lat = min(latitudes), max(latitudes)
    min_lon, max_lon = min(longitudes), max(longitudes)
    boundary = (min_lat, max_lat, min_lon, max_lon)

    # Create the quadtree
    quadtree = QuadTree(boundary)

    # Insert points into the quadtree
    print("Building quadtree...")
    total_points = len(latitudes) * len(longitudes)
    points_inserted = 0
    progress_interval = max(1, total_points // 20)  # Show progress in 5% increments

    for i, lat in enumerate(latitudes):
        for j, lon in enumerate(longitudes):
            elevation = elevations[i, j]
            quadtree.insert((lat, lon, elevation))

            points_inserted += 1
            if points_inserted % progress_interval == 0:
                progress = (points_inserted / total_points) * 100
                print(f"Progress: {progress:.1f}% ({points_inserted}/{total_points})")

    return quadtree


def create_elevation_lookup(quadtree):
    """
    Create functions to look up elevations for given coordinates.

    Args:
        quadtree (QuadTree): Quadtree containing the DEM data points

    Returns:
        tuple: (get_elevation, get_elevation_with_details) functions
    """

    def get_elevation(lat, lon):
        """
        Get the elevation for a specific latitude/longitude pair.

        Args:
            lat (float): Latitude
            lon (float): Longitude

        Returns:
            float: Elevation value at the nearest grid point
        """
        result = quadtree.query(lat, lon)
        if result:
            _, _, elevation = result
            return elevation
        return None

    def get_elevation_with_details(lat, lon):
        """
        Get elevation and detailed information for a specific latitude/longitude pair.

        Args:
            lat (float): Latitude
            lon (float): Longitude

        Returns:
            float: Elevation at the nearest grid point
        """
        result = quadtree.query(lat, lon)
        if result:
            nearest_lat, nearest_lon, elevation = result
            dist = np.sqrt((lat - nearest_lat) ** 2 + (lon - nearest_lon) ** 2)

            print(f"Input: {lat}, {lon} -> Nearest Grid Point: {nearest_lat}, {nearest_lon} "
                  f"-> Elevation: {elevation} (Distance: {dist:.6f})")

            return elevation
        return None

    return get_elevation, get_elevation_with_details


def main():
    """Main execution function."""
    # Start timer for performance measurement
    start_time = time.time()

    # Load DEM data from MATLAB file
    dem_file_path = "/Volumes/Mesonet/spring_ml/DEMdata/DEMdata.mat"
    print("Loading DEM data...")
    latitudes, longitudes, elevations = load_dem_data(dem_file_path)

    # Downsample the DEM data for improved performance
    subset_factor = 20
    print(f"Downsampling data by factor of {subset_factor}...")
    ds_latitudes, ds_longitudes, ds_elevations = downsample_dem_data(
        latitudes, longitudes, elevations, subset_factor
    )

    print(f"Original dimensions: {len(latitudes)}x{len(longitudes)}")
    print(f"Downsampled dimensions: {len(ds_latitudes)}x{len(ds_longitudes)}")

    # Build quadtree
    quadtree = build_quadtree(ds_latitudes, ds_longitudes, ds_elevations)
    tree_build_time = time.time()
    print(f"Quadtree built in {tree_build_time - start_time:.2f} seconds")

    # Create lookup functions
    get_elevation, get_elevation_with_details = create_elevation_lookup(quadtree)

    # Sample data for testing
    test_data = pd.DataFrame({
        "Latitude": [37.5, 38.0, 37.8],
        "Longitude": [-85.0, -84.5, -85.2]
    })

    # Add elevations to the test data
    print("\nPerforming lookups...")
    test_data["Elevation"] = test_data.apply(
        lambda row: get_elevation(row["Latitude"], row["Longitude"]),
        axis=1
    )

    # Display more detailed information for each point
    print("\nDetailed lookup results:")
    for _, row in test_data.iterrows():
        get_elevation_with_details(row["Latitude"], row["Longitude"])

    # Display results table
    print("\nElevation lookup results:")
    print(test_data)

    # Calculate and display execution time
    end_time = time.time()
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")
    print(f"Lookup time: {end_time - tree_build_time:.2f} seconds")


if __name__ == "__main__":
    main()