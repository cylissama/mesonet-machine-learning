import pandas as pd

# Read the CSV file from the output_data directory
df = pd.read_csv('/Volumes/Mesonet/spring_ml/output_data/prism_tmean_us_30s_20210101_CONV.csv')

# Define your range
min_value = -32
max_value = -30

# Get indices where values fall within range
indices = df.index[(df['tmean'] >= min_value) & (df['tmean'] <= max_value)].tolist()

# Print the indices and corresponding values
print("Indices within range:", indices)
print("Values at these indices:")
print(df.loc[indices, 'tmean'])