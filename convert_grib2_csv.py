import os
import xarray as xr
import pandas as pd

# === CONFIGURATION ===

# Root directory with your GRIB2 data
INPUT_DIR = "/Volumes/PRISMdata/HERBIE/2021"

# Directory where you want to save CSVs
OUTPUT_DIR = "/Volumes/PRISMdata/HERBIE/csv_output"

# Variable to extract (adjust to actual variable in your files)
VARIABLE_NAME = "t2m"  # Change to "TMP" if needed

# Whether to mirror the input folder structure in OUTPUT_DIR
MIRROR_STRUCTURE = True

# Create base output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === RECURSIVE PROCESSING ===

for root, _, files in os.walk(INPUT_DIR):
    for file in files:
        if file.endswith(".grib2"):
            grib_path = os.path.join(root, file)
            print(f"📄 Processing: {grib_path}")

            try:
                # Open GRIB2 file using xarray + cfgrib
                ds = xr.open_dataset(grib_path, engine="cfgrib")

                # Validate variable
                if VARIABLE_NAME not in ds.variables:
                    print(f"⚠️ Variable '{VARIABLE_NAME}' not found in: {file}")
                    continue

                # Extract variable
                data = ds[VARIABLE_NAME]
                if VARIABLE_NAME.lower() in ["t2m", "tmp"]:
                    data = data - 273.15  # Convert Kelvin to Celsius

                # Convert to DataFrame
                df = data.to_dataframe().reset_index()

                # === Create Output Path ===
                if MIRROR_STRUCTURE:
                    relative_path = os.path.relpath(grib_path, INPUT_DIR)
                    csv_relative_path = os.path.splitext(relative_path)[0] + ".csv"
                    csv_path = os.path.join(OUTPUT_DIR, csv_relative_path)
                    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
                else:
                    csv_name = os.path.splitext(file)[0] + ".csv"
                    csv_path = os.path.join(OUTPUT_DIR, csv_name)

                # Save to CSV
                df.to_csv(csv_path, index=False)
                print(f"✅ Saved: {csv_path}")

            except Exception as e:
                print(f"❌ Failed to process {file}: {e}")