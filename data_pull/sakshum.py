"""
One-off export for Person 2's L_max derivation.

Selects ~100-200 parsed IONEX TEC maps (2003-2018) on the native 2.5 x 5 grid,
spread across geomagnetic activity deciles using daily minimum SYM-H from the
OMNI HRO 5-min files already on disk. No pipeline code is modified.

ASSUMPTION (unconfirmed): "low to high activity" means a continuous spread
across deciles, not two discrete storm/quiet clusters. Flagging per the
original ask's two possible readings.

Output: data/person2_lmax_sample.npz
    tec           [N, 71, 73]  float32, TECU, NaN where source had fill
    timestamps    [N]          int64 epoch seconds (UTC)
    day_min_symh  [N]          float32, that day's minimum SYM-H (nT)
    lats          [71]         deg, 87.5 .. -87.5
    lons          [73]         deg, -180 .. 180

Run from repo root:
    python data_pull/export_lmax_sample.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from data_transformation import read_decompress, parse_ionex, _OMNI_COLUMNS

DATA = Path("data/raw")
YEARS = range(2003, 2019)
TARGET_MAPS = 200
N_DECILE_DAYS = 40          # candidates spread across deciles; loop stops at TARGET_MAPS
SYMH_FILL = 99999           # I6 integer sentinel


def daily_min_symh() -> pd.Series:
    """Daily minimum SYM-H (nT), indexed by date, over 2003-2018."""
    per_year = []
    for y in YEARS:
        f = DATA / "omni_hro" / str(y) / f"omni_5min{y}.asc"
        df = pd.read_csv(read_decompress(str(f)), sep=r"\s+", header=None,
                         names=_OMNI_COLUMNS, usecols=["year", "day", "sym_h"])
        df.loc[df["sym_h"] >= SYMH_FILL, "sym_h"] = np.nan
        day = pd.to_datetime((df["year"] * 1000 + df["day"]).astype(str),
                             format="%Y%j").dt.date
        per_year.append(df.groupby(day)["sym_h"].min())
    return pd.concat(per_year)


def ionex_path(d) -> Path:
    """Short-name IONEX path (2003-2018 all predate the Aug-2023 rename)."""
    doy = d.timetuple().tm_yday
    fname = f"codg{doy:03d}0.{d.year % 100:02d}i.Z"
    return DATA / "ionex" / str(d.year) / f"{doy:03d}" / fname


def main() -> None:
    symh = daily_min_symh().dropna().sort_values()   # most negative (stormiest) first

    # Decile-spaced candidates across the full activity range, extremes included.
    idx = np.linspace(0, len(symh) - 1, N_DECILE_DAYS).round().astype(int)
    candidates = symh.iloc[idx]

    tec_stack, ts_stack, symh_stack = [], [], []
    for d, day_symh in candidates.items():
        if len(ts_stack) >= TARGET_MAPS:
            break
        p = ionex_path(d)
        if not p.exists():                            # a real archive gap
            continue
        maps, lats, lons = parse_ionex(read_decompress(str(p)))
        for t, grid in maps:
            tec_stack.append(grid.astype(np.float32))
            ts_stack.append(int(t.timestamp()))
            symh_stack.append(day_symh)
        print(f"  {d}: min SYM-H {day_symh:+.0f} nT, {len(maps)} maps "
              f"(total {len(ts_stack)})")

    out = Path("data/person2_lmax_sample.npz")
    np.savez_compressed(
        out,
        tec=np.stack(tec_stack),
        timestamps=np.array(ts_stack, dtype=np.int64),
        day_min_symh=np.array(symh_stack, dtype=np.float32),
        lats=lats.astype(np.float32),
        lons=lons.astype(np.float32),
    )
    print(f"\nWrote {out}  ({len(ts_stack)} maps)")
    print(f"Activity span: {symh_stack[0]:+.0f} to {symh_stack[-1]:+.0f} nT")

    # =========================================================================
    # SPOT-CHECK VISUALIZATION: STORM VS QUIET MAP
    # =========================================================================
    # Load the compressed arrays back from disk
    data_file = np.load(out)
    tec_maps = data_file["tec"]            # Shape: [N, 71, 73]
    symh_array = data_file["day_min_symh"]  # Shape: [N]
    lat_axis = data_file["lats"]           # Shape: [71]
    lon_axis = data_file["lons"]           # Shape: [73]

    # Find the precise maps corresponding to the global extremes
    idx_storm = np.argmin(symh_array)
    idx_quiet = np.argmax(symh_array)
    
    storm_val = symh_array[idx_storm]
    quiet_val = symh_array[idx_quiet]

    # Set up subplots with a shared color scale for direct structural comparison
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    extent = [lon_axis[0], lon_axis[-1], lat_axis[-1], lat_axis[0]]
    
    # Establish a uniform global maximum bound for your colorbar scaling
    vmax_limit = max(np.nanmax(tec_maps[idx_storm]), np.nanmax(tec_maps[idx_quiet]))

    # Panel A: Extreme Storm Configuration
    im_storm = axes[0].imshow(tec_maps[idx_storm], cmap="plasma", extent=extent, 
                             origin="upper", vmin=0, vmax=vmax_limit)
    axes[0].set_title(f"STORM MAP\nMin SYM-H: {storm_val:+.0f} nT (Index {idx_storm})", 
                      color="darkred", fontweight="bold")
    axes[0].set_xlabel("Longitude (deg)")
    axes[0].set_ylabel("Latitude (deg)")
    axes[0].grid(True, alpha=0.3, linestyle=":")

    # Panel B: Quiet Baseline Configuration
    im_quiet = axes[1].imshow(tec_maps[idx_quiet], cmap="plasma", extent=extent, 
                             origin="upper", vmin=0, vmax=vmax_limit)
    axes[1].set_title(f"QUIET MAP\nMin SYM-H: {quiet_val:+.0f} nT (Index {idx_quiet})", 
                      color="darkgreen", fontweight="bold")
    axes[1].set_xlabel("Longitude (deg)")
    axes[1].grid(True, alpha=0.3, linestyle=":")

    # Attach a global visual color scale bar to the right
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
    fig.colorbar(im_storm, cax=cbar_ax, label="Total Electron Content (TECU)")

    plt.suptitle("IONEX Validation: Activity Stratification Check", fontsize=14, fontweight="bold")
    plt.show()





if __name__ == "__main__":
    main()