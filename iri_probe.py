"""PyIRI feasibility probe.

Answers, before scaling to the DeltaTEC sample:
  - does PyIRI import and run without a Fortran compile?          (import + run)
  - does it produce plausible vTEC (dayside high, nightside low)? (value check)
  - how long does one day on a small grid take?                   (timing -> full-archive estimate)

Assumptions (stated per uncertainty; none silent):
  - F107 is a REQUIRED input (confirmed: no default in signature). Using nominal 100
    here just to prove the path; real F10.7 sourcing is the next task.
  - Altitude array range/resolution affects vTEC accuracy. Using 80-2000 km at 20 km
    steps for the probe; the accuracy-tuning choice is Saksham's, not needed to prove
    feasibility.
  - ccir_or_ursi=0 (CCIR), the documented default.
"""
import time
import numpy as np
import PyIRI
import PyIRI.main_library as main

# --- tiny grid: 4 lons x 4 lats = 16 points, paired-and-flattened per PyIRI API ---
lon2d, lat2d = np.meshgrid(np.array([0., 90., 180., 270.]),
                           np.array([-40., 0., 40., 80.]))
alon = lon2d.ravel()
alat = lat2d.ravel()

aUT = np.arange(0, 24, 3)                 # 8 UT frames (coarse, probe only)
aalt = np.arange(80.0, 2000.0 + 1, 20.0)  # km
F107 = 100.0                              # nominal; real value sourced later
year, mth, day = 2015, 3, 17

t0 = time.time()
f2, f1, e_peak, es_peak, sun, mag, edp = main.IRI_density_1day(
    year, mth, day, aUT, alon, alat, aalt, F107, PyIRI.coeff_dir, 0)
tec = main.edp_to_vtec(edp, aalt)
dt = time.time() - t0

print(f"ran in {dt:.2f}s for {len(aUT)} UT frames x {len(alon)} points")
print(f"tec shape: {tec.shape}   (expect [n_UT, n_grid] = [{len(aUT)}, {len(alon)}])")
print(f"vTEC range: {np.nanmin(tec):.1f} .. {np.nanmax(tec):.1f} TECU")
print(f"vTEC mean:  {np.nanmean(tec):.1f} TECU")

# day/night sanity at one UT: subsolar longitude ~ (12 - UT)*15 mapped to 0-360.
# At UT=12 (index 4), noon is near lon 0; nightside near lon 180.
ut_idx = 4
noon_pts  = tec[ut_idx][alon == 0.0]
night_pts = tec[ut_idx][alon == 180.0]
print(f"UT=12  lon0 (near noon) mean {np.nanmean(noon_pts):.1f}  vs  "
      f"lon180 (near midnight) mean {np.nanmean(night_pts):.1f} TECU")