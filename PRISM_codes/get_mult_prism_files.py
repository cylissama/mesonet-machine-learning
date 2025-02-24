# download 

import os
import requests
from datetime import datetime

# Configuration
BASE_URL = "https://services.nacse.org/prism/data/public/4km"
VARIABLES = ['ppt', 'tmean']  # Precipitation and temperature
YEARS = range(2020, 2025)     # 2020-2024 inclusive
OUTPUT_DIR = "/Volumes/Mesonet/spring_ml/PRISM_data/"
STABILIZATION = 6  # Months after which data becomes stable

def get_stability(year):
    """Determine if data is stable or provisional"""
    data_date = datetime(year, 12, 31)  # Year-end cutoff
    elapsed_months = (datetime.now().year - year) * 12 + \
                     datetime.now().month - data_date.month
    return 'stable' if elapsed_months >= STABILIZATION else 'provisional'

def download_prism_data():
    """Main download function"""
    for year in YEARS:
        for var in VARIABLES:
            # Create directory structure
            dir_path = os.path.join(OUTPUT_DIR, var, str(year))
            os.makedirs(dir_path, exist_ok=True)
            
            # Generate filename
            stability = get_stability(year)
            filename = f"PRISM_{var}_{stability}_4kmM3_{year}_all_bil.zip"
            file_path = os.path.join(dir_path, filename)
            
            # Skip existing files
            if os.path.exists(file_path):
                print(f"Exists: {file_path}")
                continue
                
            # Construct URL
            url = f"{BASE_URL}/{var}/{year}"
            
            try:
                print(f"Downloading {var} {year}...")
                response = requests.get(url)
                response.raise_for_status()
                
                # Save with proper filename
                with open(file_path, "wb") as f:
                    f.write(response.content)
                    
                print(f"Saved: {file_path}")
                
            except requests.exceptions.HTTPError as e:
                print(f"Error downloading {var} {year}: {str(e)}")
            except Exception as e:
                print(f"General error with {var} {year}: {str(e)}")

if __name__ == "__main__":
    download_prism_data()
    print("Download complete. Check directory:", os.path.abspath(OUTPUT_DIR))