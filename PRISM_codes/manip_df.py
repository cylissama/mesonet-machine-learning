import pandas as pd


# try using pymica tutorials

file_data='/Volumews/Mesonet/spring_ml/output_data/prism_tmean_us_30s_20210101_CONV.csv'

station_data=pd.read_csv(file_data)

print(station_data.head())

# to prepare data for pymica we do the following

data = []
for key in station_data['key']:
    df_data = station_data[station_data['key'] == key]
    df_meta = metadata[metadata['key'] == key]
    data.append(
        {
            'id': key,
            'lon': float(df_meta['lon'].iloc[0]),
            'lat': float(df_meta['lat'].iloc[0]),
            'value': float(df_data['temp'].iloc[0]),
            'altitude': float(df_meta['altitude'].iloc[0]),
            'dist': float(df_meta['dist'].iloc[0])
        }
    )