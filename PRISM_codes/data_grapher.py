#!/usr/bin/env python3
"""
Kentucky PRISM Data Visualizer

This script creates a detailed visualization of PRISM climate data specific to Kentucky.
It works with the processed CSV files containing PRISM data subsetted to the Kentucky region.

Features:
- Detailed Kentucky state boundary
- County boundaries and labels
- Statistical metrics display
- Custom colormaps for different climate variables
- Optional basemap for geographic context

Requirements:
- pandas
- geopandas
- matplotlib
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
from matplotlib.lines import Line2D
import geopandas as gpd

# Optional imports for enhanced visualization
try:
    import contextily as ctx
    HAVE_CONTEXTILY = True
except ImportError:
    HAVE_CONTEXTILY = False
    print("Warning: contextily not installed. Basemap will not be available.")

try:
    import seaborn as sns
    HAVE_SEABORN = True
except ImportError:
    HAVE_SEABORN = False
    print("Warning: seaborn not installed. Using default matplotlib colormaps.")


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
    print(f"Value range: {df[var_col].min():.2f} to {df[var_col].max():.2f}")

    return df, var_col


def get_kentucky_shape():
    """Get the Kentucky state boundary as a GeoDataFrame."""
    print("Fetching Kentucky state boundary...")

    # Download US states data
    states = gpd.read_file(
        "https://www2.census.gov/geo/tiger/TIGER2021/STATE/tl_2021_us_state.zip"
    )

    # Extract Kentucky
    kentucky = states[states.NAME == "Kentucky"]

    if len(kentucky) == 0:
        raise ValueError("Kentucky boundary not found in the states dataset.")

    return kentucky


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


def filter_to_kentucky(df):
    """Filter data to Kentucky region based on bounding box coordinates."""
    # Kentucky bounding box
    ky_min_lon, ky_max_lon = -89.813, -81.688
    ky_min_lat, ky_max_lat = 36.188, 39.438

    # Filter data
    ky_df = df[
        (df['longitude'] >= ky_min_lon) &
        (df['longitude'] <= ky_max_lon) &
        (df['latitude'] >= ky_min_lat) &
        (df['latitude'] <= ky_max_lat)
    ]

    return ky_df


def calculate_metrics(df, var_col):
    """Calculate statistical metrics for the variable."""
    metrics = {
        'Min': df[var_col].min(),
        'Max': df[var_col].max(),
        'Mean': df[var_col].mean(),
        'Median': df[var_col].median(),
        'Std Dev': df[var_col].std()
    }

    # Add total for precipitation data (if dataset is not too large)
    if var_col.lower() in ['ppt', 'precipitation'] and len(df) < 10000:
        metrics['Total'] = df[var_col].sum()

    return metrics


def create_kentucky_map(df, var_col, metadata, output_path=None, show_counties=True,
                        show_metrics=True, show_basemap=True):
    """
    Create a detailed map of Kentucky with climate data.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the data to visualize
    var_col : str
        Name of the column containing the variable data to plot
    metadata : dict
        Dictionary with metadata about the data (from parse_filename)
    output_path : str, optional
        Path to save the map image
    show_counties : bool, default=True
        Whether to show county boundaries
    show_metrics : bool, default=True
        Whether to show detailed statistics
    show_basemap : bool, default=True
        Whether to show a basemap (requires contextily)
    """
    print("Creating Kentucky map visualization...")

    # Filter data to Kentucky if needed
    ky_df = filter_to_kentucky(df)

    if len(ky_df) == 0:
        print("No data points found in Kentucky region. Cannot create map.")
        return

    print(f"Using {len(ky_df)} data points for Kentucky map.")

    # Convert to GeoDataFrame for mapping
    gdf = gpd.GeoDataFrame(
        ky_df,
        geometry=gpd.points_from_xy(ky_df['longitude'], ky_df['latitude']),
        crs="EPSG:4326"  # WGS84
    )

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 10))

    # Get Kentucky state boundary
    kentucky = get_kentucky_shape()

    # Make sure the CRS matches
    if kentucky.crs != gdf.crs:
        kentucky = kentucky.to_crs(gdf.crs)

    # Get bounds from Kentucky shape
    bounds = kentucky.total_bounds  # [minx, miny, maxx, maxy]

    # Add a buffer to the bounds
    buffer = 0.1
    bounds_with_buffer = [
        bounds[0] - buffer,  # minx
        bounds[1] - buffer,  # miny
        bounds[2] + buffer,  # maxx
        bounds[3] + buffer   # maxy
    ]

    # Get custom colormap based on variable
    cmap = get_custom_colormap(metadata['variable'])

    # Plot Kentucky boundary first
    kentucky.boundary.plot(
        ax=ax,
        edgecolor='black',
        linewidth=1.5
    )

    # Add counties if requested
    if show_counties:
        try:
            print("Fetching and plotting county boundaries...")
            counties = gpd.read_file(
                "https://www2.census.gov/geo/tiger/TIGER2021/COUNTY/tl_2021_us_county.zip"
            )
            ky_counties = counties[counties.STATEFP == '21']  # Kentucky FIPS code is 21

            if ky_counties.crs != gdf.crs:
                ky_counties = ky_counties.to_crs(gdf.crs)

            # Plot county boundaries
            ky_counties.boundary.plot(
                ax=ax,
                edgecolor='gray',
                linewidth=0.5
            )

            # Add county names for larger counties
            for idx, county in ky_counties.iterrows():
                # Only label counties with area larger than threshold
                if county.geometry.area > 0.05:  # Adjust threshold as needed
                    centroid = county.geometry.centroid
                    ax.text(
                        centroid.x, centroid.y,
                        county.NAME,
                        fontsize=8,
                        ha='center',
                        va='center',
                        color='black',
                        alpha=0.7
                    )
        except Exception as e:
            print(f"Could not add county boundaries: {e}")

    # Plot the data points
    scatter = gdf.plot(
        ax=ax,
        column=var_col,
        cmap=cmap,
        markersize=20,
        alpha=0.7,
        legend=True,
        legend_kwds={
            'label': f"{metadata['variable'].upper()} Value",
            'orientation': 'horizontal',
            'shrink': 0.6,
            'pad': 0.05,
            'fraction': 0.046
        }
    )

    # Add basemap if requested and available
    if show_basemap and HAVE_CONTEXTILY:
        try:
            ctx.add_basemap(
                ax,
                crs=gdf.crs,
                source=ctx.providers.CartoDB.Positron
            )
        except Exception as e:
            print(f"Could not add basemap: {e}")

    # Calculate and display metrics if requested
    if show_metrics:
        metrics = calculate_metrics(ky_df, var_col)

        # Format metrics text based on variable type
        variable_name = metadata['variable'].upper()

        if variable_name in ['TMEAN', 'TMAX', 'TMIN']:
            # Temperature metrics
            metrics_text = (f"Kentucky Temperature Metrics (°C):\n"
                          f"Min: {metrics['Min']:.1f}   Max: {metrics['Max']:.1f}   "
                          f"Mean: {metrics['Mean']:.1f}   Median: {metrics['Median']:.1f}   "
                          f"Std Dev: {metrics['Std Dev']:.1f}")
            unit = "°C"

        elif variable_name in ['PPT']:
            # Precipitation metrics
            metrics_text = (f"Kentucky Precipitation Metrics (mm):\n"
                          f"Min: {metrics['Min']:.1f}   Max: {metrics['Max']:.1f}   "
                          f"Mean: {metrics['Mean']:.1f}   Std Dev: {metrics['Std Dev']:.1f}")

            if 'Total' in metrics:
                metrics_text += f"\nTotal Precipitation: {metrics['Total']:.1f} mm"
            unit = "mm"

        else:
            # Generic metrics
            metrics_text = (f"Kentucky {variable_name} Metrics:\n"
                          f"Min: {metrics['Min']:.1f}   Max: {metrics['Max']:.1f}   "
                          f"Mean: {metrics['Mean']:.1f}   Median: {metrics['Median']:.1f}   "
                          f"Std Dev: {metrics['Std Dev']:.1f}")
            unit = ""

        # Add metrics textbox
        plt.figtext(0.5, 0.01, metrics_text, ha='center', fontsize=11,
                   bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

    # Set title and labels
    variable_name = metadata['variable'].upper()
    full_title = f"PRISM {variable_name} for Kentucky - {metadata['date']}\n{metadata['product']} {metadata['resolution']}"
    plt.title(full_title, fontsize=16)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")

    # Add a map legend for boundaries
    legend_elements = []
    legend_elements.append(Line2D([0], [0], color='black', lw=1.5, label='Kentucky Boundary'))

    if show_counties:
        legend_elements.append(Line2D([0], [0], color='gray', lw=0.5, label='County Boundaries'))

    ax.legend(handles=legend_elements, loc='upper right', framealpha=0.7)

    # Add a small credit text
    plt.figtext(
        0.01, 0.01,
        "Data: PRISM Climate Group | Visualization: KY MESONET @ WKU",
        fontsize=8
    )

    # Set extent to Kentucky with buffer
    plt.xlim(bounds_with_buffer[0], bounds_with_buffer[2])
    plt.ylim(bounds_with_buffer[1], bounds_with_buffer[3])

    # Save or show the figure
    if output_path:
        # Make sure the output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Kentucky map saved to {output_path}")
    else:
        plt.show()

    plt.close()

    # Return some information about the visualization
    return {
        'num_points': len(ky_df),
        'metrics': metrics if show_metrics else None,
        'bounds': bounds_with_buffer
    }


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Visualize PRISM data on a Kentucky map")
    parser.add_argument("csv_file", help="Path to the processed CSV file")
    parser.add_argument("--output", "-o", default=None,
                      help="Path to save the output image file (e.g., ky_map.png)")
    parser.add_argument("--no-counties", action="store_true",
                      help="Disable county boundaries on the map")
    parser.add_argument("--no-metrics", action="store_true",
                      help="Disable detailed metrics display")
    parser.add_argument("--no-basemap", action="store_true",
                      help="Disable basemap background")

    args = parser.parse_args()

    # Determine output path
    output_path = args.output
    if not output_path:
        # Generate output path based on input filename
        base_dir = os.path.dirname(args.csv_file)
        base_name = os.path.splitext(os.path.basename(args.csv_file))[0]
        output_path = os.path.join(base_dir, f"{base_name}_ky_map.png")

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
            'region': 'Kentucky',
            'resolution': 'unknown',
            'date': datetime.now().strftime('%Y-%m-%d')
        }

    # Create the Kentucky map
    result = create_kentucky_map(
        df,
        var_col,
        metadata,
        output_path,
        show_counties=not args.no_counties,
        show_metrics=not args.no_metrics,
        show_basemap=not args.no_basemap
    )

    if result:
        print(f"Successfully created Kentucky map with {result['num_points']} data points.")


if __name__ == "__main__":
    main()