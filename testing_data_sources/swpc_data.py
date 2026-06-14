import requests
import pandas as pd
import matplotlib.pyplot as plt

plasma_url = "https://services.swpc.noaa.gov/products/solar-wind/plasma-7-day.json"
mag_url = "https://services.swpc.noaa.gov/products/solar-wind/mag-7-day.json"
k_index_url = "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json"

plasma_response = requests.get(plasma_url)
if plasma_response.status_code == 200:
    plasma_data = plasma_response.json()

    plasma_df = pd.DataFrame(plasma_data[1:], columns=plasma_data[0])

    plasma_df['time_tag'] = pd.to_datetime(plasma_df['time_tag'], utc=True)

    plasma_df['density'] = plasma_df['density'].astype(float)
    plasma_df['speed'] = plasma_df['speed'].astype(float)
    plasma_df['temperature'] = plasma_df['temperature'].astype(float)

    #print(plasma_df.head())
    #print(plasma_df.dtypes)

else:
    print(f"Failed to retrieve plasma data. Status code: {plasma_response.status_code}")

mag_response = requests.get(mag_url)
if mag_response.status_code == 200:
    mag_data = mag_response.json()

    mag_df = pd.DataFrame(mag_data[1:], columns=mag_data[0])

    mag_df['time_tag'] = pd.to_datetime(mag_df['time_tag'], utc=True)

    mag_df['bx_gsm'] = mag_df['bx_gsm'].astype(float)
    mag_df['by_gsm'] = mag_df['by_gsm'].astype(float)
    mag_df['bz_gsm'] = mag_df['bz_gsm'].astype(float)

    #print(mag_df.head())
    #print(mag_df.dtypes)

else:
    print(f"Failed to retrieve solar wind data. Status code: {mag_response.status_code}")


k_index_response = requests.get(k_index_url)
if k_index_response.status_code == 200:
    k_index_data = k_index_response.json()

    k_index_df = pd.DataFrame(k_index_data[1:], columns=k_index_data[0])

    k_index_df['time_tag'] = pd.to_datetime(k_index_df['time_tag'], utc=True)
    k_index_df['time_tag'] = k_index_df['time_tag'] - pd.to_timedelta('3 hours')

    k_index_df['Kp'] = k_index_df['Kp'].astype(float)
    k_index_df['a_running'] = k_index_df['a_running'].astype(float)
    k_index_df['station_count'] = k_index_df['station_count'].astype(float)
    #print(k_index_df.head())
    #print(k_index_df.dtypes)

else:
    print(f"Failed to retrieve k-index data. Status code: {k_index_response.status_code}")

df1 = k_index_df.set_index('time_tag')

df2 = mag_df.set_index('time_tag')
df2 = df2.resample('3h').agg({'bz_gsm': 'min'})

df3 = plasma_df.set_index('time_tag')
df3 = df3.resample('3h').agg({'density': 'mean', 'speed': 'mean'})

merged_df = df1.join(df2, how='inner').join(df3, how='inner')

merged_df = merged_df.sort_index()

merged_df['Target_Kp'] = merged_df['Kp'].shift(-1)

merged_df = merged_df.drop(columns=['station_count'])

print(merged_df.head())
print(merged_df.dtypes)


bz_corr = merged_df['Target_Kp'].corr(merged_df['bz_gsm'])
print(f"Correlation of 'Target_Kp' with 'bz_gsm': {bz_corr}")

plt.scatter(merged_df['bz_gsm'], merged_df['Target_Kp'])
plt.xlabel('Min Bz (nT), window T')
plt.ylabel('Kp, window T+1')
plt.title('Southward Bz vs next-window Kp')
plt.show()