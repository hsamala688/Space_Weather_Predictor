"""Export a stratified dTEC sample for Saksham's IRI-residual L_max analysis.

dTEC = IONEX - IRI on the NATIVE 71x73 grid (NOT the 16x31 GL grid: the GL grid
bandlimits to degree 15 and would destroy the >15 content whose existence is the
whole question). One midday map per day.

Selection (per Saksham's spec; assumptions flagged):
  - candidate years: solar max 2012-2014, solar min 2008-2009 + 2019-2020
  - stratified ~20 random days per (season x phase) -> ~160
  - guaranteed buckets within it: >=10 storms (kp_max>=5, varied) + true-quiet
    days (Kp<1 all day)
  - activity index per map: that day's kp_max

Output: data/person2_dtec_sample.npz
    dtec [N,71,73] | tec_ionex [N,71,73] | tec_iri [N,71,73]
    timestamps [N] | kp_max [N] | lats [71] | lons [73]
"""
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import PyIRI
import PyIRI.main_library as main

from data_transformation import read_decompress, parse_ionex

AALT = np.arange(80, 2001, 20).astype(float)
UT_HOUR = 12
SEED = 0
N_PER_STRATUM = 20
N_STORMS = 12
N_QUIET = 12

MAX_YEARS = [2012, 2013, 2014]
MIN_YEARS = [2008, 2009, 2019, 2020]
SEASON = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
          6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}

F107 = pd.read_parquet("data/raw/f107/f107_daily.parquet")
KP = pd.read_parquet("data/raw/geomag/kp_daily.parquet")


def ionex_path(d):
    doy = d.timetuple().tm_yday
    return Path(f"data/raw/ionex/{d.year}/{doy:03d}/codg{doy:03d}0.{d.year % 100:02d}i.Z")


def iri_native(d, ut, lats, lons):
    lat2d, lon2d = np.meshgrid(lats, lons, indexing="ij")
    f107 = float(F107.loc[str(d), "f107_obs"])
    *_, edp = main.IRI_density_1day(d.year, d.month, d.day, np.array([ut], float),
                                    lon2d.ravel(), lat2d.ravel(), AALT, f107,
                                    PyIRI.coeff_dir, 0)
    return main.edp_to_vtec(edp, AALT)[0].reshape(len(lats), len(lons))


def select_days():
    """Return an ordered, de-duplicated list of candidate dates."""
    rng = np.random.default_rng(SEED)
    cand = KP[(KP.index.year.isin(MAX_YEARS + MIN_YEARS))].copy()
    cand["season"] = cand.index.month.map(SEASON)
    cand["phase"] = np.where(cand.index.year.isin(MAX_YEARS), "max", "min")

    chosen = set()
    # guaranteed storm bucket: spread across intensity
    storms = cand[cand["kp_max"] >= 5].sort_values("kp_max")
    if len(storms):
        idx = np.linspace(0, len(storms) - 1, min(N_STORMS, len(storms))).round().astype(int)
        chosen.update(storms.index[idx])
    # guaranteed quiet bucket
    quiet = cand[cand["quiet_all_day"]]
    if len(quiet):
        chosen.update(rng.choice(quiet.index, min(N_QUIET, len(quiet)), replace=False))
    # stratified random fill by (phase, season)
    for _, grp in cand.groupby(["phase", "season"]):
        pool = [d for d in grp.index if d not in chosen]
        take = min(N_PER_STRATUM, len(pool))
        chosen.update(rng.choice(pool, take, replace=False))

    return sorted(pd.Timestamp(d).date() for d in chosen)


def main_run():
    lats = np.arange(87.5, -87.6, -2.5)
    lons = np.arange(-180, 180.1, 5)

    dtec, ionex_s, iri_s, ts, kp = [], [], [], [], []
    for d in select_days():
        p = ionex_path(d)
        if not p.exists():
            continue
        maps, sl, so = parse_ionex(read_decompress(str(p)))
        t, grid = min(maps, key=lambda m: abs(m[0].hour - UT_HOUR))
        iri = iri_native(d, t.hour, lats, lons)
        dtec.append((grid - iri).astype(np.float32))
        ionex_s.append(grid.astype(np.float32))
        iri_s.append(iri.astype(np.float32))
        ts.append(int(t.timestamp()))
        kp.append(float(KP.loc[str(d), "kp_max"]))

    out = Path("data/person2_dtec_sample.npz")
    np.savez_compressed(
        out, dtec=np.stack(dtec), tec_ionex=np.stack(ionex_s), tec_iri=np.stack(iri_s),
        timestamps=np.array(ts, np.int64), kp_max=np.array(kp, np.float32),
        lats=lats.astype(np.float32), lons=lons.astype(np.float32))

    print(f"wrote {out}  ({len(ts)} maps)")
    print(f"kp_max span: {min(kp):.1f}..{max(kp):.1f} | storms(>=5): {sum(k>=5 for k in kp)}"
          f" | quiet(<1): {sum(k<1 for k in kp)}")
    print(f"dtec mean {np.nanmean(np.stack(dtec)):+.1f} TECU (nonzero = IONEX-IRI offset)")
    
    d = np.load("data/person2_dtec_sample.npz")
    print({k: d[k].shape for k in d.files})
    print("dtec range:", np.nanmin(d["dtec"]), np.nanmax(d["dtec"]))
    print("timestamps unique:", len(np.unique(d["timestamps"])), "of", len(d["timestamps"]))


if __name__ == "__main__":
    main_run()