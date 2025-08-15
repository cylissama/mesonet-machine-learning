import pandas as pd
import matplotlib.pyplot as plt

# Load the two CSV files
file1 = '/Users/cylis/Work/mes_summer25/RFSM mods/RFSM_mesonet_data_filled.csv'

file2 = '/Users/cylis/Work/mes_summer25/original/RFSM.csv'


df1 = pd.read_csv(file1, low_memory=False)
df2 = pd.read_csv(file2, low_memory=False)
# Ensure TAIR and PRCP are numeric (in case of any stray strings or bad data)
df1['TAIR'] = pd.to_numeric(df1['TAIR'], errors='coerce')
df2['TAIR'] = pd.to_numeric(df2['TAIR'], errors='coerce')
df1['PRCP'] = pd.to_numeric(df1['PRCP'], errors='coerce')
df2['PRCP'] = pd.to_numeric(df2['PRCP'], errors='coerce')

# Drop rows with missing values for clean plotting
df1 = df1.dropna(subset=['TAIR', 'PRCP'])
df2 = df2.dropna(subset=['TAIR', 'PRCP'])

# Plot TAIR
plt.figure(figsize=(10, 5))
plt.plot(df1['TAIR'].values, label=f'{file1} - TAIR', color='blue')
plt.plot(df2['TAIR'].values, label=f'{file2} - TAIR', color='red')
plt.title('TAIR Comparison')
plt.xlabel('Index')
plt.ylabel('TAIR')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot PRCP
plt.figure(figsize=(10, 5))
plt.plot(df1['PRCP'].values, label=f'{file1} - PRCP', color='blue')
plt.plot(df2['PRCP'].values, label=f'{file2} - PRCP', color='red')
plt.title('PRCP Comparison')
plt.xlabel('Index')
plt.ylabel('PRCP')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()