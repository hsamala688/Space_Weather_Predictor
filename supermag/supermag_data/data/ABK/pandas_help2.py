import pandas as pd

df = pd.read_parquet('data_ABK_20191231.parquet')

all_values = df.values

values_list = df.values.tolist()

print(values_list)

[1577793900.0, 60.0, 'ABK', 18.82, 68.349998, 14.305623, 24.26149, 9.161959, 92.545799, Timestamp('2019-12-31 12:05:00+0000', tz='UTC'), 0.719691, 0.990579, -1.760152, -1.62322, 3.769609, 3.769609]