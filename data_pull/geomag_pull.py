"""Pull Kp from CelesTrak SW-All.txt.

Same file/OBSERVED-block parse as pull_f107; Kp lives in fields 5-12 (the 8
3-hourly values, stored as Kp*10 rounded).

Outputs:
  data/raw/geomag/kp_daily.parquet
      date-indexed daily summary: kp_max, quiet_all_day
  data/raw/geomag/kp_3hourly.parquet
      timestamp-indexed causal 3-hour Kp values: kp
"""
from pathlib import Path

import pandas as pd
import requests

URL = "https://celestrak.org/SpaceData/SW-All.txt"
DAILY_OUT = Path("data/raw/geomag/kp_daily.parquet")
THREE_HOUR_OUT = Path("data/raw/geomag/kp_3hourly.parquet")
START, END = "2000-01-01", "2025-12-31"
KP_FIELDS = slice(5, 13)          # 8 three-hourly Kp*10 values


def main():
    DAILY_OUT.parent.mkdir(parents=True, exist_ok=True)

    resp = requests.get(URL, timeout=60)
    resp.raise_for_status()
    text = resp.text
    if text.lstrip().startswith("<"):
        raise ValueError(f"{URL} returned HTML, not data (check the URL is current)")

    lines = text.splitlines()
    begin = next(i for i, ln in enumerate(lines) if ln.strip() == "BEGIN OBSERVED")
    end = next(i for i, ln in enumerate(lines) if ln.strip() == "END OBSERVED")

    dates, kp_max, quiet = [], [], []
    kp_times, kp_values = [], []
    for ln in lines[begin + 1:end]:
        f = ln.split()
        if len(f) <= 12:
            continue
        kp8 = [int(x) for x in f[KP_FIELDS]]        # Kp*10
        day = pd.Timestamp(f"{f[0]}-{f[1]}-{f[2]}")
        dates.append(day)
        kp_max.append(max(kp8) / 10.0)
        quiet.append(all(k < 10 for k in kp8))       # Kp<1 for every 3-hr window

        for i, kp10 in enumerate(kp8):
            kp_times.append(day + pd.Timedelta(hours=3 * i))
            kp_values.append(kp10 / 10.0)

    df = pd.DataFrame({"date": pd.to_datetime(dates),
                       "kp_max": kp_max, "quiet_all_day": quiet})
    df = df[(df["date"] >= START) & (df["date"] <= END)].set_index("date").sort_index()

    start = pd.Timestamp(START)
    end_exclusive = pd.Timestamp(END) + pd.Timedelta(days=1)
    kp_3h = pd.DataFrame({"timestamp": kp_times, "kp": kp_values})
    kp_3h = (
        kp_3h[(kp_3h["timestamp"] >= start) & (kp_3h["timestamp"] < end_exclusive)]
        .set_index("timestamp")
        .sort_index()
    )

    df.to_parquet(DAILY_OUT)
    kp_3h.to_parquet(THREE_HOUR_OUT)
    print(f"wrote {DAILY_OUT}  ({len(df)} days)")
    print(f"wrote {THREE_HOUR_OUT}  ({len(kp_3h)} rows)")
    print(f"kp_max range: {df['kp_max'].min():.1f}..{df['kp_max'].max():.1f}")
    print(f"kp 3-hour range: {kp_3h['kp'].min():.1f}..{kp_3h['kp'].max():.1f}")
    print(f"storm days (kp_max>=5): {(df['kp_max'] >= 5).sum()}")
    print(f"true-quiet days (Kp<1 all day): {df['quiet_all_day'].sum()}")


if __name__ == "__main__":
    main()
