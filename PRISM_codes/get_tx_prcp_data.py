# Attempts to use FTP to download PRISM data files for a given year and variable.

import os
from datetime import datetime, timedelta
from ftplib import FTP, error_perm

# Configuration
output_dir = '../output_data'
variables = ['tmean', 'ppt']  # PRCP is 'ppt' in PRISM terminology
base_url = 'prism.oregonstate.edu'
stabilization_period = 6  # Months until data becomes stable
scale_version = '4kmM3'    # From filename format documentation

# Calculate date ranges with stabilization buffer
current_date = datetime.now()
end_date = (current_date.replace(day=1) - timedelta(days=1)).replace(day=1)
start_date = end_date - timedelta(days=5*365)

def get_stability_label(data_date):
    """Determine stability label based on PRISM's processing timeline"""
    elapsed_months = (current_date.year - data_date.year) * 12 + current_date.month - data_date.month
    if elapsed_months < 1:
        return 'early'
    elif 1 <= elapsed_months < stabilization_period:
        return 'provisional'
    else:
        return 'stable'

def remote_path_exists(ftp, path):
    """Check if a remote path exists"""
    try:
        ftp.cwd(path)
        return True
    except error_perm:
        return False

def generate_filename(var, stability, date_str):
    """Generate filename according to PRISM naming convention"""
    #return f'PRISM_{var}_{stability}_{scale_version}_{date_str}_bil.zip'
    return f'PRISM_{var}_{stability}_{scale_version}_2020_all_bil.zip'

with FTP(base_url) as ftp:
    ftp.login()
    ftp.set_pasv(True)
    
    current_date = start_date.replace(day=1)
    while current_date <= end_date:
        date_str = current_date.strftime("%Y%m")
        year = current_date.year
        stability = get_stability_label(current_date)
        
        for var in variables:
            # Construct paths according to PRISM directory structure
            remote_dir = f'/monthly/{var}/{year}/{date_str}'
            print(remote_dir)
            filename = generate_filename(var, stability, date_str)
            print(filename)
            local_dir = os.path.join(output_dir, var, date_str)
            local_path = os.path.join(local_dir, filename)
            
            try:
                # Verify remote directory exists
                if not remote_path_exists(ftp, remote_dir):
                    print(f"Directory not found: {remote_dir}, skipping...")
                    continue
                
                # Check file existence
                ftp.cwd(remote_dir)
                try:
                    file_size = ftp.size(filename)
                except error_perm:
                    print(f"File not found: {filename}, skipping...")
                    continue
                
                # Create local directory
                os.makedirs(local_dir, exist_ok=True)
                
                # Check if we need to download
                if os.path.exists(local_path):
                    local_size = os.path.getsize(local_path)
                    if local_size == file_size:
                        print(f"File exists and matches remote size: {filename}")
                        continue
                
                # Download with progress
                print(f"Downloading {filename} ({file_size//1024} KB)")
                with open(local_path, 'wb') as f:
                    ftp.retrbinary(f'RETR {filename}', f.write)
                    
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                continue
        
        # Move to next month
        current_date = current_date + timedelta(days=32)
        current_date = current_date.replace(day=1)

print("Download complete! Important notes:")
print("- Data stability: early (<1 month), provisional (1-6 months), stable (6+ months)")
print("- Re-run monthly to get updated versions of provisional data")
print("- Cite PRISM data according to their terms of use")