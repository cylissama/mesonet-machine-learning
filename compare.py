import pandas as pd
import matplotlib.pyplot as plt


def simple_tair_plot(file_path):
    """
    Simple plot of TAIR vs UTCtimestamp for a single file
    """

    # Read the CSV
    df = pd.read_csv(file_path, low_memory=False)

    # Clean the data
    df['TAIR'] = pd.to_numeric(df['TAIR'], errors='coerce')
    df['UTCTimestampCollected'] = pd.to_datetime(df['UTCTimestampCollected'], errors='coerce')
    #df_clean = df.dropna(subset=['TAIR', 'UTCtimestamp'])

    # Create plot
    plt.figure(figsize=(12, 6))
    plt.plot(df['UTCTimestampCollected'], df['TAIR'], marker='o', markersize=1, color='red')
    plt.xlabel('UTC Timestamp')
    plt.ylabel('TAIR')
    plt.title('TAIR vs Time')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# Usage for single file
simple_tair_plot('/full_mesonet_data_filled.csv')