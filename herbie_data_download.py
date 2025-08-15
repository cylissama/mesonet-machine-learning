from herbie import Herbie
from datetime import datetime, timedelta
import os
from tqdm import tqdm

# Set base directory for all downloads
BASE_DIR = "/Volumes/PRISMdata/HERBIE/2021"

# Define time range for all of 2021
start_date = datetime(2021, 1, 1, 0)
end_date = datetime(2022, 1, 1, 0)
total_hours = int((end_date - start_date).total_seconds() / 3600)

# Loop over every hour with a progress bar
with tqdm(total=total_hours, desc="Downloading HRRR TMP:2m data") as pbar:
    current_date = start_date
    while current_date < end_date:
        try:
            # Determine monthly subfolder path
            month_str = current_date.strftime("%Y-%m")
            save_path = os.path.join(BASE_DIR, month_str)
            os.makedirs(save_path, exist_ok=True)

            # Construct expected filename
            filename = f"hrrr.t{current_date:%H}z.wrfsfcf00.grib2"
            subset_prefix = f"subset_"
            full_path = os.path.join(save_path, f"{subset_prefix}*{filename}")

            # Check if file already exists (glob match)
            if any(f.endswith(filename) for f in os.listdir(save_path)):
                pbar.update(1)
                current_date += timedelta(hours=1)
                continue

            # Initialize Herbie object
            H = Herbie(
                current_date,
                model="hrrr",
                product="sfc",
                fxx=0,
                verbose=False
            )

            # Download only the 2m temperature
            H.download("TMP:2 m above ground", save_dir=save_path)

        except Exception as e:
            print(f"\n⚠️  Failed on {current_date:%Y-%m-%d %H:%M} UTC: {e}")

        # Increment time and progress bar
        current_date += timedelta(hours=1)
        pbar.update(1)