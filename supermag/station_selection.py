"""
station_selection.py

Find SuperMAG stations that (a) were reporting across the full 2000-2025 span and
(b) can be selected for uniform coverage across geomagnetic latitude, balanced
between magnetic hemispheres.

Run in two stages, in order:
  probe (main):    monthly inventory coverage + each candidate's geomagnetic
                   latitude. Read-only and cheap. Run first, inspect the printout.
  select_balanced: apply a coverage threshold and bin width chosen AFTER looking
                   at the probe output, then pick a symmetric, coverage-uniform set.

Requires: supermag-api, pandas, pyarrow  (pip install supermag-api pandas pyarrow)
Set SUPERMAG_LOGON to your SuperMAG user id before running.

Notes / assumptions:
  - Inventory presence in a one-day probe window is a proxy for "reporting that
    month", not proof of continuous data. True completeness is confirmed at pull
    time. The coverage fraction is printed so the threshold is chosen from data.
  - maglat is derived from SuperMAG's own mcolat (magnetic colatitude) so it
    matches the coordinate system used downstream. Computed at one epoch; pole
    drift over 2000-2025 is small relative to a 10-degree bin.
  - SuperMAGGetInventory is assumed to return a list of IAGA codes. Adjust the
    iteration in probe_months if your client version returns a different shape.
"""

import os
import time
import pandas as pd
from supermag_api.supermag_api import SuperMAGGetInventory, SuperMAGGetData

LOGON = "hsamala"
START_YEAR = 2000
END_YEAR = 2025
PROBE_DAY = 15           # mid-month probe instant
PROBE_EXTENT = 86400     # one day, in seconds
SLEEP = 0              # raise if the server pushes back
OUT_DIR = "station_selection"


def probe_months():
    """Monthly inventory probe -> long table of (year, month, iaga) presence."""
    rows = []
    for year in range(START_YEAR, END_YEAR + 1):
        for month in range(1, 13):
            start = [year, month, PROBE_DAY, 0, 0]
            status, stations = SuperMAGGetInventory(LOGON, start, PROBE_EXTENT)
            if status == 0:
                rows.append({"year": year, "month": month, "iaga": None})
            else:
                for s in stations:
                    rows.append({"year": year, "month": month, "iaga": s})
            time.sleep(SLEEP)
    return pd.DataFrame(rows)


def coverage_table(presence):
    """Coverage fraction per station plus endpoint-year presence flags."""
    total_months = (END_YEAR - START_YEAR + 1) * 12
    present = presence.dropna(subset=["iaga"])
    cov = present.groupby("iaga").size().rename("months_present").reset_index()
    cov["coverage_frac"] = cov["months_present"] / total_months
    at_start = present[present.year == START_YEAR]["iaga"].unique()
    at_end = present[present.year == END_YEAR]["iaga"].unique()
    cov["present_start"] = cov["iaga"].isin(at_start)
    cov["present_end"] = cov["iaga"].isin(at_end)
    return cov.sort_values("coverage_frac", ascending=False)


def magnetic_latitude(iaga):
    """One short data pull -> signed geomagnetic latitude (deg), or None."""
    for year in (2012, 2008, 2016, START_YEAR, END_YEAR):
        start = [year, 6, 15, 0, 5]
        status, data = SuperMAGGetData(LOGON, start, 60, "all", iaga)
        if status != 0 and len(data) > 0:
            return 90.0 - float(data["mcolat"].iloc[0])
    return None


def select_balanced(cand, coverage_threshold, bin_width=10, per_bin=1):
    """Pick a coverage-uniform set, symmetric across the magnetic equator.

    cand: DataFrame with iaga, coverage_frac, maglat.
    Bins maglat into bands of bin_width degrees. For each band, keeps a northern
    band only if its conjugate southern band can be filled too, so the final set
    is mirrored about the equator. Within a band, takes the highest-coverage
    stations.
    """
    cand = cand[cand.coverage_frac >= coverage_threshold].copy()
    cand["band"] = (cand.maglat // bin_width).astype(int)
    chosen = []
    for b in sorted({abs(x) for x in cand.band.unique() if x != 0}):
        north = cand[cand.band == b].nlargest(per_bin, "coverage_frac")
        south = cand[cand.band == -b].nlargest(per_bin, "coverage_frac")
        n = min(len(north), len(south))
        if n:
            chosen.append(north.head(n))
            chosen.append(south.head(n))
    return (pd.concat(chosen).sort_values("maglat")
            if chosen else cand.head(0))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    presence = probe_months()
    presence.to_parquet(f"{OUT_DIR}/presence.parquet")

    cov = coverage_table(presence)
    cov.to_parquet(f"{OUT_DIR}/coverage.parquet")
    endpoints = cov[cov.present_start & cov.present_end].copy()
    print(f"{len(cov)} stations seen total, "
          f"{len(endpoints)} present at both endpoints")
    print(endpoints[["iaga", "coverage_frac"]].to_string(index=False))

    endpoints["maglat"] = [magnetic_latitude(s) for s in endpoints.iaga]
    endpoints = endpoints.dropna(subset=["maglat"])
    endpoints.to_parquet(f"{OUT_DIR}/candidates.parquet")

    north = int((endpoints.maglat > 0).sum())
    south = int((endpoints.maglat < 0).sum())
    print(f"\ncandidates with maglat: {north} magnetic-north, "
          f"{south} magnetic-south  (south is the binding constraint)")
    print(endpoints[["iaga", "coverage_frac", "maglat"]]
          .sort_values("maglat").to_string(index=False))


if __name__ == "__main__":
    main()