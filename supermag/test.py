import pandas as pd
from station_selection import select_balanced
cand = pd.read_parquet("station_selection/candidates.parquet")
chosen = select_balanced(cand, coverage_threshold=0.90, bin_width=10)