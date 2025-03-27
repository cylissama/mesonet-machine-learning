#!/usr/bin/env python3
"""
PRISM Data US Map Visualizer

This script visualizes a single day of PRISM data on a US map, with
special highlighting for Kentucky. It works with the processed CSV files
created by the Kentucky subsetting processor.

Requirements:
- pandas
- geopandas
- matplotlib
- cartopy
- contextily (optional, for basemaps)
- seaborn (optional, for enhanced colormaps)

Author: For KY MESONET @ WKU
"""

import os
import re
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import geopandas as gpd

# Import cartopy for map projections
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    HAVE_CARTOPY = True
except ImportError:
    print("Warning: cartopy not installed. Using basic map visualization.")
    HAVE_CARTOPY = False

# Optional imports for enhanced visualization
try:
    import contextily as ctx

    HAVE_CONTEXTILY = True
except ImportError:
    HAVE_CONTEXTILY = False

try:
    import seaborn as sns

    HAVE_SEABORN = True
except ImportError:
    HAVE_SEABORN = False


def parse_filename(filename):
    """Extract metadata from the filename."""
    pattern = r"^(\w+)_(\w+)_(\w+)_(\w+)_(\d{8}).*\.csv$"
    match = re.match(pattern, os.path.basename(filename))

    if match:
        product = match.group(1)
        variable = match.group(2)
        region = match.group(3)
        resolution = match.group(4)
        date_str = match.group(5)

        # Format date
        formatted_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"

        return {
            'product': product,
            'variable': variable,
            'region': region,
            'resolution': resolution,
            'date': formatted_date
        }
    else:
        return None


def load_csv_data(csv_path):
    """Load PRISM data from a CSV file."""
    print(f"Loading data from {csv_path}...")

    # Read the CSV file
    df = pd.read_csv(csv_path)

    # Check required columns
    required_cols = ['latitude', 'longitude']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"CSV file must contain columns: {required_cols}")

    # Get the variable column (not one of the standard columns)
    standard_cols = ['key', 'altitude', 'dist', 'hr', 'latitude', 'longitude', 'timestamp']
    var_cols = [col for col in df.columns if col not in standard_cols]

    if not var_cols:
        raise ValueError("No variable column found in the CSV file")

    var_col = var_cols[0]  # Take the first variable column
    print(f"Found variable column: {var_col}")

    # Basic data information
    print(f"Data shape: {df.shape}")
    print(f"Value range: {df[var_col].min()} to {df[var_col].max()}")

    return df, var_col


def get_custom_colormap(variable):
    """Get a custom colormap based on the variable type."""
    if variable.lower() in ['tmean', 'tmax', 'tmin']:
        # Temperature colormap (blue to red)
        if HAVE_SEABORN:
            return sns.color_palette("RdBu_r", as_cmap=True)
        else:
            return plt.cm.RdBu_r
    elif variable.lower() in ['ppt', 'precipitation']:
        # Precipitation colormap (white to blue)
        if HAVE_SEABORN:
            return sns.color_palette("Blues", as_cmap=True)
        else:
            return plt.cm.Blues
    else:
        # Default colormap
        return plt.cm.viridis


def get_kentucky_shape():
    """Get the Kentucky state boundary as a GeoDataFrame."""
    # Download US states data
    states = gpd.read_file(
        "https://www2.census.gov/geo/tiger/TIGER2021/STATE/tl_2021_us_state.zip"
    )

    # Extract Kentucky
    kentucky = states[states.NAME == "Kentucky"]

    return kentucky


def create_us_map_with_cartopy(df, var_col, metadata, output_path=None, highlight_kentucky=True):
    """Create a US map visualization with cartopy."""
    print("Creating US map visualization with cartopy...")

    # Set up the figure and projection
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.AlbersEqualArea(central_longitude=-96, central_latitude=38))

    # Add US states
    ax.add_feature(cfeature.STATES, linewidth=0.5, edgecolor='black')
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)

    # Set extent to continental US
    ax.set_extent([-125, -66, 24, 50], crs=ccrs.PlateCarree())

    # Get custom colormap based on variable
    cmap = get_custom_colormap(metadata['variable'])

    # Plot the data
    sc = ax.scatter(
        df['longitude'],
        df['latitude'],
        c=df[var_col],
        cmap=cmap,
        s=5,  # Point size
        alpha=0.7,
        transform=ccrs.PlateCarree()
    )

    # Highlight Kentucky if requested
    if highlight_kentucky:
        try:
            # Get Kentucky state boundary
            kentucky = get_kentucky_shape()

            # Add Kentucky boundary with special styling
            ax.add_geometries(
                kentucky.geometry,
                crs=ccrs.PlateCarree(),
                edgecolor='red',
                facecolor='none',
                linewidth=2
            )

            # Add label for Kentucky (using centroid of the state)
            centroid = kentucky.geometry.centroid.iloc[0]
            txt = ax.text(
                centroid.x, centroid.y,
                "KENTUCKY",
                transform=ccrs.PlateCarree(),
                fontsize=12,
                ha='center',
                va='center',
                color='red',
                fontweight='bold'
            )
            txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='white')])

        except Exception as e:
            print(f"Could not add Kentucky boundary: {e}")
            print("Falling back to bounding box...")

            # Kentucky bounding box as fallback
            ky_min_lon, ky_max_lon = -89.813, -81.688
            ky_min_lat, ky_max_lat = 36.188, 39.438

            # Draw rectangle around Kentucky
            ky_box = plt.Rectangle(
                (ky_min_lon, ky_min_lat),
                ky_max_lon - ky_min_lon,
                ky_max_lat - ky_min_lat,
                linewidth=2,
                edgecolor='red',
                facecolor='none',
                transform=ccrs.PlateCarree()
            )
            ax.add_patch(ky_box)

            # Add label for Kentucky
            txt = ax.text(
                (ky_min_lon + ky_max_lon) / 2,
                (ky_min_lat + ky_max_lat) / 2,
                "KENTUCKY",
                transform=ccrs.PlateCarree(),
                fontsize=12,
                ha='center',
                va='center',
                color='red',
                fontweight='bold'
            )
            txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='white')])

    # Add a colorbar
    cbar = plt.colorbar(sc, ax=ax, orientation='horizontal', pad=0.05, aspect=40)
    variable_name = metadata['variable'].upper()

    # Add appropriate units based on variable
    if variable_name in ['TMEAN', 'TMAX', 'TMIN']:
        cbar.set_label(f"{variable_name} (°C)")
    elif variable_name in ['PPT']:
        cbar.set_label(f"{variable_name} (mm)")
    else:
        cbar.set_label(variable_name)

    # Set title
    plt.title(
        f"PRISM {variable_name} - {metadata['date']}\n{metadata['product']} {metadata['resolution']}",
        fontsize=16
    )

    # Add a small credit text
    plt.figtext(
        0.01, 0.01,
        "Data: PRISM Climate Group | Visualization: KY MESONET @ WKU",
        fontsize=8
    )

    # Save or show the figure
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Map saved to {output_path}")
    else:
        plt.show()

    plt.close()


def create_basic_us_map(df, var_col, metadata, output_path=None, highlight_kentucky=True):
    """Create a basic US map visualization without cartopy."""
    print("Creating basic US map visualization...")

    # Convert to GeoDataFrame for mapping
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df['longitude'], df['latitude']),
        crs="EPSG:4326"  # WGS84
    )

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(15, 10))

    # Get custom colormap based on variable
    cmap = get_custom_colormap(metadata['variable'])

    # Plot the data points
    scatter = gdf.plot(
        ax=ax,
        column=var_col,
        cmap=cmap,
        markersize=5,
        alpha=0.7,
        legend=True
    )

    # Add basemap if contextily is available
    if HAVE_CONTEXTILY:
        try:
            ctx.add_basemap(
                ax,
                crs=gdf.crs,
                source=ctx.providers.CartoDB.Positron
            )
        except Exception as e:
            print(f"Could not add basemap: {e}")

    # Highlight Kentucky if requested
    if highlight_kentucky:
        try:
            # Get Kentucky state boundary
            kentucky = get_kentucky_shape()

            # Make sure the CRS matches
            if kentucky.crs != gdf.crs:
                kentucky = kentucky.to_crs(gdf.crs)

            # Plot Kentucky boundary with special styling
            kentucky.boundary.plot(
                ax=ax,
                edgecolor='red',
                linewidth=2
            )

            # Add label for Kentucky (using centroid of the state)
            centroid = kentucky.geometry.centroid.iloc[0]
            txt = ax.text(
                centroid.x, centroid.y,
                "KENTUCKY",
                fontsize=12,
                ha='center',
                va='center',
                color='red',
                fontweight='bold'
            )
            txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='white')])

        except Exception as e:
            print(f"Could not add Kentucky boundary: {e}")
            print("Falling back to bounding box...")

            # Kentucky bounding box as fallback
            ky_min_lon, ky_max_lon = -89.813, -81.688
            ky_min_lat, ky_max_lat = 36.188, 39.438

            # Draw rectangle around Kentucky
            ky_box = plt.Rectangle(
                (ky_min_lon, ky_min_lat),
                ky_max_lon - ky_min_lon,
                ky_max_lat - ky_min_lat,
                linewidth=2,
                edgecolor='red',
                facecolor='none'
            )
            ax.add_patch(ky_box)

            # Add label for Kentucky
            txt = ax.text(
                (ky_min_lon + ky_max_lon) / 2,
                (ky_min_lat + ky_max_lat) / 2,
                "KENTUCKY",
                fontsize=12,
                ha='center',
                va='center',
                color='red',
                fontweight='bold'
            )
            txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='white')])

    # Set title and labels
    variable_name = metadata['variable'].upper()
    plt.title(
        f"PRISM {variable_name} - {metadata['date']}\n{metadata['product']} {metadata['resolution']}",
        fontsize=16
    )
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")

    # Add a small credit text
    plt.figtext(
        0.01, 0.01,
        "Data: PRISM Climate Group | Visualization: KY MESONET @ WKU",
        fontsize=8
    )

    # Adjust to US extent
    plt.xlim(-125, -66)
    plt.ylim(24, 50)

    # Save or show the figure
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Map saved to {output_path}")
    else:
        plt.show()

    plt.close()


def create_zoomed_kentucky_map(df, var_col, metadata, output_path=None):
    """Create a zoomed-in map of Kentucky."""
    print("Creating zoomed Kentucky map...")

    # Convert to GeoDataFrame for mapping
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df['longitude'], df['latitude']),
        crs="EPSG:4326"  # WGS84
    )

    # Filter for Kentucky region
    ky_min_lon, ky_max_lon = -89.813, -81.688
    ky_min_lat, ky_max_lat = 36.188, 39.438

    ky_gdf = gdf[
        (gdf['longitude'] >= ky_min_lon) &
        (gdf['longitude'] <= ky_max_lon) &
        (gdf['latitude'] >= ky_min_lat) &
        (gdf['latitude'] <= ky_max_lat)
        ]

    if len(ky_gdf) == 0:
        print("No data points found in Kentucky region. Skipping Kentucky map.")
        return

    print(f"Found {len(ky_gdf)} data points in Kentucky region.")

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))

    # Get Kentucky state boundary
    try:
        kentucky = get_kentucky_shape()
        # Make sure the CRS matches
        if kentucky.crs != ky_gdf.crs:
            kentucky = kentucky.to_crs(ky_gdf.crs)

        # Get better bounds from the actual Kentucky shape
        bounds = kentucky.total_bounds  # [minx, miny, maxx, maxy]
        bounds_with_buffer = [
            bounds[0] - 0.1,  # minx
            bounds[1] - 0.1,  # miny
            bounds[2] + 0.1,  # maxx
            bounds[3] + 0.1  # maxy
        ]
    except Exception as e:
        print(f"Could not get Kentucky boundary for zoomed map: {e}")
        # Fall back to the bounding box
        bounds_with_buffer = [
            ky_min_lon - 0.2,
            ky_min_lat - 0.2,
            ky_max_lon + 0.2,
            ky_max_lat + 0.2
        ]

    # Get custom colormap based on variable
    cmap = get_custom_colormap(metadata['variable'])

    # Plot Kentucky boundary first
    try:
        kentucky.boundary.plot(
            ax=ax,
            edgecolor='black',
            linewidth=1.5
        )
    except Exception as e:
        print(f"Could not plot Kentucky boundary: {e}")

    # Plot the data points
    scatter = ky_gdf.plot(
        ax=ax,
        column=var_col,
        cmap=cmap,
        markersize=30,
        alpha=0.7,
        legend=True
    )

    # Add basemap if contextily is available
    if HAVE_CONTEXTILY:
        try:
            ctx.add_basemap(
                ax,
                crs=ky_gdf.crs,
                source=ctx.providers.CartoDB.Positron
            )
        except Exception as e:
            print(f"Could not add basemap: {e}")

    # Set title and labels
    variable_name = metadata['variable'].upper()
    plt.title(
        f"PRISM {variable_name} for Kentucky - {metadata['date']}\n{metadata['product']} {metadata['resolution']}",
        fontsize=16
    )
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")

    # Add a small credit text
    plt.figtext(
        0.01, 0.01,
        "Data: PRISM Climate Group | Visualization: KY MESONET @ WKU",
        fontsize=8
    )

    # Set extent to Kentucky (with buffer)
    plt.xlim(bounds_with_buffer[0], bounds_with_buffer[2])
    plt.ylim(bounds_with_buffer[1], bounds_with_buffer[3])

    # Save or show the figure
    if output_path:
        # Modify filename to indicate Kentucky zoom
        base, ext = os.path.splitext(output_path)
        ky_output_path = f"{base}_kentucky{ext}"

        plt.savefig(ky_output_path, dpi=300, bbox_inches='tight')
        print(f"Kentucky map saved to {ky_output_path}")
    else:
        plt.show()

    plt.close()


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Visualize PRISM data on a US map")
    parser.add_argument("csv_file", help="Path to the processed CSV file")
    parser.add_argument("--output", "-o", default=None,
                        help="Path to save the output image file (e.g., map.png)")
    parser.add_argument("--no-kentucky-highlight", action="store_true",
                        help="Disable Kentucky highlighting on the map")
    parser.add_argument("--create-kentucky-map", action="store_true",
                        help="Create an additional zoomed-in map of Kentucky")
    parser.add_argument("--use-cartopy", action="store_true",
                        help="Use cartopy for better map visualization (if installed)")

    args = parser.parse_args()

    # Determine output path
    output_path = args.output
    if not output_path:
        # Generate output path based on input filename
        base_dir = os.path.dirname(args.csv_file)
        base_name = os.path.splitext(os.path.basename(args.csv_file))[0]
        output_path = os.path.join(base_dir, f"{base_name}_map.png")

    # Load the data
    df, var_col = load_csv_data(args.csv_file)

    # Extract metadata from filename
    metadata = parse_filename(args.csv_file)
    if not metadata:
        # Use default metadata if filename parsing fails
        print("Could not parse metadata from filename. Using default values.")
        metadata = {
            'product': 'PRISM',
            'variable': var_col,
            'region': 'unknown',
            'resolution': 'unknown',
            'date': datetime.now().strftime('%Y-%m-%d')
        }

    # Create the map visualization
    if args.use_cartopy and HAVE_CARTOPY:
        create_us_map_with_cartopy(
            df, var_col, metadata, output_path,
            highlight_kentucky=not args.no_kentucky_highlight
        )
    else:
        create_basic_us_map(
            df, var_col, metadata, output_path,
            highlight_kentucky=not args.no_kentucky_highlight
        )

    # Create additional Kentucky map if requested
    if args.create_kentucky_map:
        create_zoomed_kentucky_map(df, var_col, metadata, output_path)


if __name__ == "__main__":
    main()