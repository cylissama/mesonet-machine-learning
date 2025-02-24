# download a single PRISM file from the NACSE website using direct HTTP requests

import requests

url = "https://services.nacse.org/prism/data/public/4km/ppt/2020"
output_file = "prism_2020_ppt_data.zip"

try:
    response = requests.get(url)
    response.raise_for_status()  # Check for HTTP errors
    
    with open(output_file, "wb") as f:
        f.write(response.content)
        
    print(f"File saved as {output_file}")

except Exception as e:
    print(f"Download failed: {str(e)}")