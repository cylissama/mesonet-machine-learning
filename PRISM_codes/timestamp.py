import pandas as pd
from datetime import datetime, timedelta

# Create start and end dates
start_date = datetime(2020, 1, 1)
end_date = datetime(2021, 1, 1)

# Generate datetime range with 30-second intervals
datetime_range = []
current_date = start_date
while current_date < end_date:
    datetime_range.append(current_date)
    current_date += timedelta(seconds=30)

# Create DataFrame
df = pd.DataFrame({'timestamp': datetime_range})

# Optional: Set timestamp as index
df.set_index('timestamp', inplace=True)

# Print info about the DataFrame
print(f"DataFrame contains {len(df)} rows")
print("\nFirst few rows:")
print(df.head())
print("\nLast few rows:")
print(df.tail())
# Save DataFrame to CSV file
output_path = '/Volumes/Mesonet/spring_ml/output_data/timestamps.csv'
df.to_csv(output_path)
print(f"\nDataFrame saved to: {output_path}")
print("Size of the file: ", df.size)
print("Shape of the file: ", df.shape)