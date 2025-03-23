#!/usr/bin/env python
"""
PRISM Data Processing Master Script

This script automates the processing of PRISM climate data files:
1. Reads BIL files from PRISM data directories
2. Converts them to CSV format
3. Adds timestamps corresponding to the day of the data
4. Adds elevation data from DEM files
5. Saves the processed data in a structured output directory

Usage:
    python prism_data_processor.py [--start_year YEAR] [--end_year YEAR] [--variables VAR1,VAR2]
"""

import os
import re
import time
import argparse
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import scipy.io
from scipy.spatial import cKDTree

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("prism_processing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class PRISMDataProcessor:
    """Class for processing PRISM climate data"""

    def __init__(self, config):
        """
        Initialize the processor with configuration.

        Args:
            config (dict): Configuration dictionary with paths and settings
        """
        self.config = config
        self.dem_lookup = None

        # Create output directories if they don't exist
        for var in config['variables']:
            os.makedirs(os.path.join(config['output_dir'], var), exist_ok=True)

        # Initialize DEM lookup function
        if config['add_elevation']:
            logger.info("Initializing DEM elevation lookup...")
            self.initialize_dem_lookup()

    def initialize_dem_lookup(self):
        """Initialize the DEM elevation lookup function"""
        try:
            # Load DEM data
            latitudes, longitudes, elevations = self.load_dem_data(self.config['dem_file'])

            # Downsample for efficiency
            grid_points, elevation_values = self.downsample_dem_data(
                latitudes, longitudes, elevations, self.config['dem_downsample_factor']
            )

            # Create lookup function
            self.dem_lookup = self.create_elevation_lookup(grid_points, elevation_values)
            logger.info("DEM elevation lookup initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize DEM lookup: {str(e)}")
            if self.config['add_elevation']:
                logger.warning("Continuing without elevation data")
                self.config['add_elevation'] = False

    def load_dem_data(self, file_path):
        """
        Load DEM data from a MATLAB file.

        Args:
            file_path (str): Path to the MATLAB file containing DEM data

        Returns:
            tuple: (latitudes, longitudes, elevations) arrays
        """
        logger.info(f"Loading DEM data from {file_path}")
        mat_data = scipy.io.loadmat(file_path)
        dem_tiles = mat_data['DEMtiles']

        latitudes = dem_tiles[0, 0]["lat"].flatten()
        longitudes = dem_tiles[0, 0]["lon"].flatten()
        elevations = dem_tiles[0, 0]["z"]

        return latitudes, longitudes, elevations

    def downsample_dem_data(self, latitudes, longitudes, elevations, factor):
        """
        Downsample DEM data to reduce computational complexity.

        Args:
            latitudes (np.ndarray): Array of latitude values
            longitudes (np.ndarray): Array of longitude values
            elevations (np.ndarray): 2D array of elevation values
            factor (int): Downsampling factor (take every Nth point)

        Returns:
            tuple: (grid_points, elevation_values)
        """
        logger.info(f"Downsampling DEM data with factor {factor}")
        # Downsample the grid
        ds_latitudes = latitudes[::factor]
        ds_longitudes = longitudes[::factor]
        ds_elevations = elevations[::factor, ::factor]

        # Create coordinate grid
        lon_grid, lat_grid = np.meshgrid(ds_longitudes, ds_latitudes)
        grid_points = np.column_stack((lat_grid.ravel(), lon_grid.ravel()))
        elevation_values = ds_elevations.ravel()

        return grid_points, elevation_values

    def create_elevation_lookup(self, grid_points, elevation_values):
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
            """Get the elevation for a specific latitude/longitude pair."""
            _, index = tree.query([lat, lon])
            return elevation_values[index]

        return get_elevation

    def find_prism_files(self):
        """
        Find all PRISM BIL files for the specified years and variables.

        Returns:
            list: List of dictionaries with file info (path, variable, date)
        """
        files_to_process = []

        for year in range(self.config['start_year'], self.config['end_year'] + 1):
            for var in self.config['variables']:
                # Define the directory pattern based on file organization
                dir_pattern = os.path.join(self.config['data_dir'], f"{var}{year}")

                # Handle both direct files and nested directories
                search_paths = [dir_pattern]
                if os.path.exists(dir_pattern):
                    # If it's a directory, add it for search
                    for root, _, files in os.walk(dir_pattern):
                        search_paths.append(root)

                # Look for BIL files in each search path
                for search_path in search_paths:
                    if os.path.exists(search_path):
                        for filename in os.listdir(search_path):
                            if filename.endswith(".bil"):
                                # Extract date from filename using regex
                                match = re.search(r'(\d{8})\.bil$', filename)
                                if match:
                                    date_str = match.group(1)
                                    formatted_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"

                                    file_path = os.path.join(search_path, filename)

                                    # Check if output file already exists to avoid reprocessing
                                    output_path = self.get_output_path(var, formatted_date)
                                    if os.path.exists(output_path) and not self.config['overwrite']:
                                        logger.debug(f"Skipping existing file: {output_path}")
                                        continue

                                    files_to_process.append({
                                        'path': file_path,
                                        'variable': var,
                                        'date': formatted_date,
                                        'output_path': output_path
                                    })

        logger.info(f"Found {len(files_to_process)} files to process")
        return files_to_process

    def get_output_path(self, variable, date_str):
        """
        Generate output file path based on variable and date.

        Args:
            variable (str): Climate variable (e.g., 'tmean', 'ppt')
            date_str (str): Date string in YYYY-MM-DD format

        Returns:
            str: Path to output CSV file
        """
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        year_dir = os.path.join(self.config['output_dir'], variable, str(date_obj.year))
        os.makedirs(year_dir, exist_ok=True)

        filename = f"prism_{variable}_{date_str.replace('-', '')}"
        if self.config['add_elevation']:
            filename += "_with_elev"

        return os.path.join(year_dir, f"{filename}.csv")

    def read_bil_to_dataframe(self, bil_file, variable, date_str, subsample=1):
        """
        Reads a BIL file and returns a Pandas DataFrame with coordinates and data.

        Args:
            bil_file (str): Path to the BIL file
            variable (str): Climate variable name for the column
            date_str (str): Date string in YYYY-MM-DD format
            subsample (int): Subsampling factor (1 for all data)

        Returns:
            pd.DataFrame: DataFrame with lat, lon, variable data and timestamp
        """
        logger.info(f"Reading BIL file: {bil_file}")
        try:
            with rasterio.open(bil_file) as src:
                # Read data
                data = src.read(1)
                nodata = src.nodata
                transform = src.transform

                # Generate coordinates for each pixel
                height, width = data.shape
                rows, cols = np.indices((height, width))
                xs, ys = rasterio.transform.xy(transform, rows.flatten(), cols.flatten())

                # Create timestamp for the day
                timestamp = pd.to_datetime(date_str)

                # Create DataFrame with subsampling
                df = pd.DataFrame({
                    'longitude': xs[::subsample],
                    'latitude': ys[::subsample],
                    variable: data.flatten()[::subsample],
                    'timestamp': timestamp
                })

                # Filter out NoData values
                if nodata is not None:
                    df = df[df[variable] != nodata]

                logger.info(f"Processed {len(df)} valid points from {bil_file}")
                return df

        except Exception as e:
            logger.error(f"Error reading BIL file {bil_file}: {str(e)}")
            return None

    def add_elevation_to_dataframe(self, df):
        """
        Add elevation data to DataFrame using DEM lookup.

        Args:
            df (pd.DataFrame): DataFrame with latitude and longitude columns

        Returns:
            pd.DataFrame: DataFrame with added elevation column
        """
        if not self.config['add_elevation'] or self.dem_lookup is None:
            return df

        logger.info(f"Adding elevation data to {len(df)} points")

        # Create progress tracker for large files
        total_rows = len(df)
        report_interval = max(1, total_rows // 10)  # Report progress every 10%

        # Process elevations
        elevations = []
        for i, (_, row) in enumerate(df.iterrows()):
            # Get elevation
            elevation = self.dem_lookup(row['latitude'], row['longitude'])
            elevations.append(elevation)

            # Report progress
            if (i + 1) % report_interval == 0:
                progress = (i + 1) / total_rows * 100
                logger.debug(f"Elevation progress: {progress:.1f}% ({i + 1}/{total_rows})")

        # Add the elevations as a new column
        df['elevation'] = elevations

        return df

    def process_file(self, file_info):
        """
        Process a single PRISM file.

        Args:
            file_info (dict): Dictionary with file information

        Returns:
            bool: True if processing was successful, False otherwise
        """
        try:
            # Extract file information
            bil_file = file_info['path']
            variable = file_info['variable']
            date_str = file_info['date']
            output_path = file_info['output_path']

            logger.info(f"Processing {variable} data for {date_str}")

            # Convert BIL to DataFrame
            df = self.read_bil_to_dataframe(
                bil_file,
                variable,
                date_str,
                subsample=self.config['subsample_factor']
            )

            if df is None or len(df) == 0:
                logger.warning(f"No valid data found in {bil_file}")
                return False

            # Add elevation data if configured
            if self.config['add_elevation'] and self.dem_lookup is not None:
                df = self.add_elevation_to_dataframe(df)

            # Save processed data
            df.to_csv(output_path, index=False)
            logger.info(f"Saved processed data to {output_path}")

            return True

        except Exception as e:
            logger.error(f"Error processing file {file_info['path']}: {str(e)}")
            return False

    def process_all_files(self):
        """
        Process all PRISM files found for the configured years and variables.

        Returns:
            tuple: (processed_count, error_count)
        """
        start_time = time.time()

        # Find all files to process
        files_to_process = self.find_prism_files()

        # Initialize counters
        processed_count = 0
        error_count = 0

        # Process each file
        total_files = len(files_to_process)
        for i, file_info in enumerate(files_to_process):
            logger.info(f"Processing file {i + 1}/{total_files}: {file_info['path']}")

            success = self.process_file(file_info)
            if success:
                processed_count += 1
            else:
                error_count += 1

            # Log progress
            progress = (i + 1) / total_files * 100
            elapsed_time = time.time() - start_time
            estimated_total = elapsed_time * total_files / (i + 1)
            remaining_time = estimated_total - elapsed_time

            logger.info(f"Progress: {progress:.1f}% - Processed: {processed_count}, "
                        f"Errors: {error_count}, Estimated time remaining: {remaining_time / 60:.1f} min")

        # Log summary
        total_time = time.time() - start_time
        logger.info("=" * 50)
        logger.info(f"Processing complete!")
        logger.info(f"Total files: {total_files}")
        logger.info(f"Successfully processed: {processed_count}")
        logger.info(f"Errors: {error_count}")
        logger.info(f"Total processing time: {total_time / 60:.1f} minutes")
        logger.info("=" * 50)

        return processed_count, error_count


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Process PRISM climate data files')

    parser.add_argument('--start_year', type=int, default=2019,
                        help='Start year for processing (default: 2019)')
    parser.