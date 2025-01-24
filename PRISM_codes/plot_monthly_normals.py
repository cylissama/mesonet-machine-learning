import os
import rasterio
import matplotlib.pyplot as plt
import numpy as np 

def load_all_months(base_path):
    """
    Loops through the 12 monthly directories (4km_tmax_01..12),
    reads each BIL file with rasterio, and returns a dictionary
    keyed by month (e.g., "01", "02", ..., "12").

    Parameters:
        base_path (str): Path to the folder containing the monthly subfolders
    
    Returns:
        dict: { "01": numpy_array, "02": numpy_array, ..., "12": numpy_array }
    """

    monthly_data = {}

    for month in range(1, 13):
        month_str = f"{month:02d}"  # "01", "02", ... "12"
        
        # Construct folder and file names
        folder_name = f"4km_tmax_{month_str}/PRISM_tmax_30yr_normal_4kmM5_{month_str}_bil"
        bil_filename = f"PRISM_tmax_30yr_normal_4kmM5_{month_str}_bil.bil"
        
        # Full path to the .bil file
        bil_path = os.path.join(base_path, folder_name, bil_filename)

        if os.path.exists(bil_path):
            print(f"Loading {bil_path}...")
            with rasterio.open(bil_path) as src:
                data = src.read(1)  # Read the first (and often only) band
                data[data == -9999] = np.nan # Set nodata values to NaN
                monthly_data[month_str] = data
        else:
            print(f"File not found for month {month_str}: {bil_path}")

    return monthly_data

def main():
    # Adjust base_path if necessary
    base_path = "/Volumes/Mesonet/winter_break/PRISM_data"
    
    # Load all 12 months of data
    all_months_data = load_all_months(base_path)
    print(all_months_data)
    
    # Example: Plot one month to verify
    # (Change "08" to another month if desired)
    month_to_plot = "08"  
    if month_to_plot in all_months_data:
        data_to_plot = all_months_data[month_to_plot]

        plt.figure(figsize=(8, 6))
        plt.imshow(data_to_plot, cmap='viridis')
        plt.colorbar(label="PRISM Tmax (units)")
        plt.title(f"PRISM Tmax Data - All Months (30yr Normal)")
        plt.xlabel("Longitude")
        plt.ylabel("Latidude")
        plt.show()

    else:
        print(f"No data found for month {month_to_plot}")

if __name__ == "__main__":
    main()