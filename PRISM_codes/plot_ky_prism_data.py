import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import contextily as ctx
import seaborn as sns

# Load data
df = pd.read_csv("/Volumes/Mesonet/spring_ml/PRISM_data/converted_data/kentucky_precipitation_2020.csv")

# Create GeoDataFrame
gdf = gpd.GeoDataFrame(
    df,
    geometry=gpd.points_from_xy(df.longitude, df.latitude),
    crs="EPSG:4326"  # WGS84
)

# Get Kentucky boundary
states = gpd.read_file("https://www2.census.gov/geo/tiger/TIGER2021/STATE/tl_2021_us_state.zip")
kentucky = states[states.NAME == "Kentucky"].to_crs(gdf.crs)

# Create plot
fig, ax = plt.subplots(figsize=(12, 8))

# Plot precipitation points
scatter = gdf.plot(ax=ax,
                  column='precipitation',
                  cmap='viridis',
                  markersize=15,
                  legend=True,
                  alpha=0.7,
                  vmin=df.precipitation.min(),
                  vmax=df.precipitation.max())

# Add Kentucky boundary
kentucky.boundary.plot(ax=ax, linewidth=2, color='black')

# Add basemap
ctx.add_basemap(ax, crs=gdf.crs.to_string(), source=ctx.providers.OpenStreetMap.Mapnik)

# Customize plot
ax.set_title("2020 Annual Precipitation in Kentucky (mm)", fontsize=16)
ax.set_axis_off()

# Add scale bar
ctx.add_basemap(ax, crs=gdf.crs.to_string(), source=ctx.providers.OpenStreetMap.Mapnik)
plt.tight_layout()
plt.show()

# --- heatmap version

plt.figure(figsize=(12, 8))
sns.kdeplot(x=df.longitude, y=df.latitude, 
            weights=df.precipitation,
            cmap='viridis', fill=True,
            thresh=0.05, alpha=0.8)
plt.title("Precipitation Density in Kentucky")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()