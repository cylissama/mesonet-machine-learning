import pandas as pd

# Read the CSV file (you'll need to specify your CSV file path)
# Only read first 10 rows
df = pd.read_csv('/Volumes/Mesonet/spring_ml/output_data/prism_tmean_us_30s_20210101_TIMESTAMPED.csv', nrows=10)

# Display information about the DataFrame
print("\nFirst 10 rows of the data:")
print(df)

print("\nColumns in the dataset:")
print(df.columns.tolist())

print("\nNumber of rows:", len(df))

print("\nIndex of the dataset:")
print(df.index.tolist())