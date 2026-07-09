"""Measure the IONEX - IRI vertical-range offset on sample days.

IONEX TEC counts electrons up to GPS orbit (~20000 km); IRI here integrates only
to 2000 km, so IRI under-counts by the plasmaspheric content above 2000 km. This
script measures that offset and asks the one question that decides how to handle it:

  is the offset spatially FLAT (a constant -> absorbed by normalization, harmless),
  or LATITUDE-STRUCTURED (plasmaspheric content is higher at low latitudes -> the
  offset has spatial shape that would enter Saksham's spectrum)?

For each sample day: parse+interpolate one midday IONEX map to the 16x31 GL grid,
compute IRI TEC on the same grid/timestamp, subtract, and report flatness.
"""
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import PyIRI
import PyIRI.main_library as main

from data_transformation import read_decompress, parse_ionex
from data_interpolation import interpolate_map, target_grid

AALT = np.arange(80, 20001, 20).astype(float)     # locked default altitude array
F107 = pd.read_parquet("data/raw/f107/f107_daily.parquet")

# Mix of conditions: solar min quiet, solar max moderate, storm.
SAMPLE_DAYS = [date(2008, 6, 15), date(2012, 7, 15), date(2015, 3, 17)]
UT_HOUR = 12                                       # one midday map per day


def ionex_path(d):
    doy = d.timetuple().tm_yday
    return Path(f"data/raw/ionex/{d.year}/{doy:03d}/codg{doy:03d}0.{d.year % 100:02d}i.Z")


def iri_tec(d, ut_hour, tgt_lats, tgt_lons):
    """IRI vTEC on the GL grid [nlat, nlon] for one date/hour."""
    lat2d, lon2d = np.meshgrid(tgt_lats, tgt_lons, indexing="ij")
    alat, alon = lat2d.ravel(), lon2d.ravel()
    f107 = float(F107.loc[str(d), "f107_obs"])
    *_, edp = main.IRI_density_1day(d.year, d.month, d.day, np.array([ut_hour], float),
                                    alon, alat, AALT, f107, PyIRI.coeff_dir, 0)
    tec = main.edp_to_vtec(edp, AALT)              # [1, n_grid]
    return tec[0].reshape(len(tgt_lats), len(tgt_lons))


def main_run():
    tgt_lats, tgt_lons = target_grid()

    for d in SAMPLE_DAYS:
        p = ionex_path(d)
        if not p.exists():
            print(f"{d}: IONEX file missing (gap?), skip")
            continue

        maps, src_lats, src_lons = parse_ionex(read_decompress(str(p)))
        t, grid = min(maps, key=lambda m: abs(m[0].hour - UT_HOUR))   # nearest midday
        ionex = interpolate_map(grid, src_lats, src_lons, tgt_lats, tgt_lons)
        iri = iri_tec(d, t.hour, tgt_lats, tgt_lons)
        offset = ionex - iri

        equ = np.nanmean(offset[np.abs(tgt_lats) < 20])    # low-latitude rows
        pol = np.nanmean(offset[np.abs(tgt_lats) > 60])    # high-latitude rows
        f107 = float(F107.loc[str(d), "f107_obs"])
        print(f"\n{d}  UT={t.hour}  F10.7={f107:.0f}")
        print(f"  IONEX {np.nanmean(ionex):5.1f} | IRI {np.nanmean(iri):5.1f} "
              f"| offset mean {np.nanmean(offset):+5.1f} TECU")
        print(f"  offset spatial std {np.nanstd(offset):4.1f} TECU  "
              f"(<~2 => flat; large => structured)")
        print(f"  low-lat {equ:+.1f} vs high-lat {pol:+.1f} TECU  "
              f"(gap {equ - pol:+.1f} => plasmaspheric structure if large)")

    print("\nRead: if offset means are similar across days AND spatial std is small "
          "AND low/high-lat gap is small -> flat constant, normalization handles it.\n"
          "If the low/high-lat gap is large -> structured, tell Saksham.")


if __name__ == "__main__":
    main_run()