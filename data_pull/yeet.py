# scratch, run from repo root
import time
from data_interpolation import build_interpolated  # or whatever you named the file
# temporarily point it at one year, OR just time a single year manually:

import csv, time
from pathlib import Path
from data_transformation import read_decompress, parse_ionex
from data_interpolation import interpolate_to_gl

root = Path("data")
days = []
with open(root/"manifests"/"ionex_manifest.csv") as f:
    for r in csv.DictReader(f):
        if r["key"].startswith("2015-") and r["status"] in ("present","downloaded"):
            days.append(r)

t0 = time.time()
n = 0
for r in sorted(days, key=lambda x: x["key"]):
    doy = r["key"].split("-")[1]
    p = root/"raw"/"ionex"/"2015"/doy/r["expected_filename"]
    maps, la, lo = parse_ionex(read_decompress(str(p)))
    tec, ts, _ = interpolate_to_gl(maps, la, lo)
    n += tec.shape[0]
print(f"2015: {n} maps in {time.time()-t0:.1f}s")