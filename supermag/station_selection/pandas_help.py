import pandas as pd

df = pd.read_parquet('coverage.parquet')

all_values = df.values

values_list = df.values.tolist()

print(values_list)