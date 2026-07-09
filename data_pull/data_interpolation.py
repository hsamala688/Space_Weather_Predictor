"""
Interpolate native IONEX 71x73 TEC maps onto the Gauss-Legendre 23x45 grid
Falisha's SFNO transform expects.

Grid contract:
  - Lmax=22
  - Gauss-Legendre latitudes, nlat=23 (cell-centered; GL nodes never sit at a pole)
  - Equiangular longitudes, nlon=45, 0-360 convention, endpoint=False
  - Poles handled by collapsing the native +-87.5 deg ring to a single averaged
    value (IONEX has no data beyond +-87.5; this is extrapolation, not observation)
"""
from __future__ import annotations

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.interpolate import RegularGridInterpolator

LMAX = 22
NLAT = 23
NLON = 45


def target_grid() -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre latitudes (deg) and equiangular 0-360 longitudes (deg)."""
    roots, _ = leggauss(NLAT)                       # roots = cos(colatitude), interior to (-1,1)
    lats = 90.0 - np.degrees(np.arccos(roots))       # -> latitude, descending N to S
    lons = np.linspace(0.0, 360.0, NLON, endpoint=False)
    return lats, lons


def _to_0_360(lons: np.ndarray, grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert source longitudes from -180..180 to 0..360 and re-sort columns to match."""
    lons_360 = lons % 360.0
    uniq, idx = np.unique(lons_360, return_index=True)
    return uniq, grid[:, idx]


def interpolate_map(grid: np.ndarray, src_lats: np.ndarray, src_lons: np.ndarray,
                     tgt_lats: np.ndarray, tgt_lons: np.ndarray) -> np.ndarray:
    """Interpolate one native [n_lat, n_lon] IONEX map onto the GL target grid."""
    lons_360, grid_360 = _to_0_360(src_lons, grid)

    # Wrap-pad across the 0/360 seam so the interpolator sees continuity there.
    pad_lons = np.concatenate([lons_360[-1:] - 360.0, lons_360, lons_360[:1] + 360.0])
    pad_grid = np.concatenate([grid_360[:, -1:], grid_360, grid_360[:, :1]], axis=1)

    # src_lats runs 87.5 -> -87.5 (descending); RegularGridInterpolator needs ascending.
    lat_order = np.argsort(src_lats)
    interp = RegularGridInterpolator(
        (src_lats[lat_order], pad_lons), pad_grid[lat_order],
        bounds_error=False, fill_value=None,
    )

    in_range = (tgt_lats >= src_lats.min()) & (tgt_lats <= src_lats.max())
    pts = np.array([[la, lo] for la in tgt_lats[in_range] for lo in tgt_lons])
    vals = interp(pts).reshape(in_range.sum(), NLON)

    out = np.empty((NLAT, NLON), dtype=np.float32)
    out[in_range] = vals

    # Collapse-to-point: rows beyond native coverage get the mean of the
    # nearest native edge ring (top edge for northern gap, bottom for southern).
    if not in_range.all():
        top_val = np.nanmean(grid_360[np.argmax(src_lats)])       # +87.5 ring
        bot_val = np.nanmean(grid_360[np.argmin(src_lats)])       # -87.5 ring
        for i, la in enumerate(tgt_lats):
            if in_range[i]:
                continue
            out[i] = top_val if la > 0 else bot_val

    return out


def interpolate_to_gl(maps, src_lats, src_lons):
    """maps: list of (timestamp, [71,73] ndarray) from parse_ionex;
    src_lats, src_lons: the native grid vectors parse_ionex returned for these maps.

    Returns (tec [N,23,45] float32, timestamps [N], and the target lats/lons).
    """
    tgt_lats, tgt_lons = target_grid()
    stack = np.stack([
        interpolate_map(grid, src_lats, src_lons, tgt_lats, tgt_lons)
        for _, grid in maps
    ])
    timestamps = np.array([int(t.timestamp()) for t, _ in maps], dtype=np.int64)
    return stack, timestamps, (tgt_lats, tgt_lons)


# ---------------------------------------------------------------------------
# Batch builder: interpolate all IONEX -> per-year .npz in data/interpolated_gl23x45/
# ---------------------------------------------------------------------------

def build_interpolated(data_root="data", year=None, overwrite=False):
    """Interpolate every present IONEX day to the GL grid, one .npz per year.

    Reads the ionex manifest for present days, skips 404 gaps, and skips years
    whose output already exists (resumable). Writes grid.npz (lats/lons) once.
    """
    import csv
    from collections import defaultdict
    from pathlib import Path
    from data_transformation import read_decompress, parse_ionex

    root = Path(data_root)
    out_dir = root / "interpolated_gl23x45"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Group present IONEX days by year from the manifest.
    manifest = root / "manifests" / "ionex_manifest.csv"
    by_year = defaultdict(list)
    with open(manifest, newline="") as f:
        for r in csv.DictReader(f):
            if r["status"] in ("present", "downloaded"):
                year = int(r["key"].split("-")[0])
                by_year[year].append(r)

    grid_written = (out_dir / "grid.npz").exists()

    for current_year in sorted(by_year):
        if year is not None and current_year != year:
            continue

        dest = out_dir / f"{current_year}.npz"
        if dest.exists() and not overwrite:
            print(f"  {current_year}: skip (exists)")
            continue

        tec_parts, ts_parts = [], []
        for r in sorted(by_year[current_year], key=lambda x: x["key"]):
            doy = r["key"].split("-")[1]
            path = root / "raw" / "ionex" / str(current_year) / doy / r["expected_filename"]
            maps, src_lats, src_lons = parse_ionex(read_decompress(str(path)))
            tec, ts, (tgt_lats, tgt_lons) = interpolate_to_gl(maps, src_lats, src_lons)
            tec_parts.append(tec)
            ts_parts.append(ts)

        tec_all = np.concatenate(tec_parts)
        ts_all = np.concatenate(ts_parts)
        np.savez(dest, tec=tec_all, timestamps=ts_all)
        print(f"  {current_year}: {tec_all.shape[0]} maps -> {dest.name}")

        if not grid_written:
            np.savez(out_dir / "grid.npz", lats=tgt_lats, lons=tgt_lons)
            grid_written = True


def smoke_test(data_root="data"):
    from pathlib import Path
    from data_transformation import read_decompress, parse_ionex

    path = Path(data_root) / "raw" / "ionex" / "2010" / "001" / "codg0010.10i.Z"
    maps, src_lats, src_lons = parse_ionex(read_decompress(str(path)))
    tgt_lats, tgt_lons = target_grid()

    print(f"target lats ({len(tgt_lats)}): {tgt_lats.round(2)}")
    print(f"target lons ({len(tgt_lons)}): {tgt_lons.round(2)}")

    t0, grid0 = maps[0]
    out = interpolate_map(grid0, src_lats, src_lons, tgt_lats, tgt_lons)
    print(f"\n{t0} interpolated shape {out.shape}")
    print(f"value range: {np.nanmin(out):.1f} .. {np.nanmax(out):.1f} TECU")
    print(f"seam check, lon~0 vs lon~last: {out[NLAT // 2, 0]:.3f} vs {out[NLAT // 2, -1]:.3f}")
    print(f"top row (should be identical across lon): {out[0].round(3)}")
    print(f"bottom row (should be identical across lon): {out[-1].round(3)}")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--build", action="store_true",
                        help="Build data/interpolated_gl23x45 yearly cache.")
    parser.add_argument("--year", type=int, help="Build one year only.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite an existing yearly cache.")
    args = parser.parse_args()

    if args.build:
        build_interpolated(args.data_root, year=args.year, overwrite=args.overwrite)
    else:
        smoke_test(args.data_root)


if __name__ == "__main__":
    main()
