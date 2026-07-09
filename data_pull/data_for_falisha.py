"""Build Falisha's GL23x45 cache products for the full training dataset.

Baseline residual definition:
    dTEC = IONEX vTEC interpolated to GL23x45 - IRI vTEC on GL23x45

The plasmaspheric offset in IONEX-minus-IRI is intentionally retained for the
baseline dataset. It is expected to be mostly zonal and representable by the
SFNO m=0 modes; a learned zonal correction can be added later.

This script currently builds the cache layers needed before window assembly:
    data/iri_gl23x45/{year}.npz   -> iri [N,23,45], timestamps [N]
    data/dtec_gl23x45/{year}.npz  -> dtec [N,23,45], timestamps [N]
    data/omni_aligned_gl23x45/{year}.npz -> drivers [N,6], timestamps [N]
    data/falisha_windows_gl23x45/*.npy -> normalized train/val/test windows

Run from repo root:
    python3 data_pull/data_for_falisha.py iri-cache --year 2015
    python3 data_pull/data_for_falisha.py dtec-cache --year 2015
    python3 data_pull/data_for_falisha.py omni-cache --year 2015
    python3 data_pull/data_for_falisha.py windows
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import PyIRI
import PyIRI.main_library as iri_main


LMAX = 22
NLAT = 23
NLON = 45
AALT = np.arange(80, 2001, 20).astype(float)
OMNI_HRO_FEATURES = ["b_magnitude", "by_gsm", "bz_gsm", "flow_speed", "proton_density"]
KP_FEATURE = "kp_3hour"
DRIVER_FEATURES = OMNI_HRO_FEATURES + [KP_FEATURE]
INPUT_STEPS = 6
TARGET_STEPS = 3


def load_f107(data_root: Path) -> pd.Series:
    f107 = pd.read_parquet(data_root / "raw" / "f107" / "f107_daily.parquet")
    if "f107_obs" not in f107.columns:
        raise ValueError(f"F10.7 table missing f107_obs column: {list(f107.columns)}")
    out = f107["f107_obs"].astype(float)
    out.index = pd.to_datetime(out.index).date
    return out


def load_gl_grid(data_root: Path) -> tuple[np.ndarray, np.ndarray]:
    grid_path = data_root / "interpolated_gl23x45" / "grid.npz"
    grid = np.load(grid_path)
    lats = grid["lats"].astype(float)
    lons = grid["lons"].astype(float)
    if lats.shape != (NLAT,) or lons.shape != (NLON,):
        raise ValueError(f"Unexpected GL grid shape: lats={lats.shape}, lons={lons.shape}")
    return lats, lons


def iter_years(data_root: Path, requested_year: int | None) -> list[int]:
    if requested_year is not None:
        return [requested_year]
    source_dir = data_root / "interpolated_gl23x45"
    return sorted(int(path.stem) for path in source_dir.glob("*.npz") if path.stem != "grid")


def timestamps_to_frame(timestamps: np.ndarray) -> pd.DataFrame:
    dt = pd.to_datetime(timestamps, unit="s", utc=True)
    return pd.DataFrame({
        "timestamp": timestamps.astype(np.int64),
        "datetime": dt,
        "date": dt.date,
        "ut": dt.hour + dt.minute / 60.0 + dt.second / 3600.0,
    })


def iri_for_day(day, ut_values: np.ndarray, lats: np.ndarray, lons: np.ndarray,
                f107_value: float) -> np.ndarray:
    lat2d, lon2d = np.meshgrid(lats, lons, indexing="ij")
    alat = lat2d.ravel()
    alon = lon2d.ravel()

    *_, edp = iri_main.IRI_density_1day(
        day.year,
        day.month,
        day.day,
        ut_values.astype(float),
        alon,
        alat,
        AALT,
        f107_value,
        PyIRI.coeff_dir,
        0,
    )
    tec = iri_main.edp_to_vtec(edp, AALT)
    return tec.reshape(len(ut_values), NLAT, NLON).astype(np.float32)


def f107_for_day(f107: pd.Series, day) -> float:
    if day in f107.index:
        return float(f107.loc[day])

    previous_day = day - pd.Timedelta(days=1)
    if previous_day in f107.index:
        print(f"    {day}: missing F10.7; using {previous_day} for boundary timestamp")
        return float(f107.loc[previous_day])

    raise KeyError(f"No F10.7 value found for {day}")


def build_iri_cache(data_root: Path, year: int | None = None, overwrite: bool = False) -> None:
    source_dir = data_root / "interpolated_gl23x45"
    out_dir = data_root / "iri_gl23x45"
    out_dir.mkdir(parents=True, exist_ok=True)

    lats, lons = load_gl_grid(data_root)
    f107 = load_f107(data_root)

    for y in iter_years(data_root, year):
        source_path = source_dir / f"{y}.npz"
        dest_path = out_dir / f"{y}.npz"

        if dest_path.exists() and not overwrite:
            print(f"  {y}: skip IRI (exists)")
            continue
        if not source_path.exists():
            print(f"  {y}: missing {source_path}, skip")
            continue

        source = np.load(source_path)
        timestamps = source["timestamps"].astype(np.int64)
        frame = timestamps_to_frame(timestamps)
        iri = np.empty((len(timestamps), NLAT, NLON), dtype=np.float32)

        for day, group in frame.groupby("date", sort=True):
            ut_values = group["ut"].to_numpy(dtype=float)
            day_iri = iri_for_day(day, ut_values, lats, lons, f107_for_day(f107, day))
            iri[group.index.to_numpy()] = day_iri

        np.savez_compressed(
            dest_path,
            iri=iri,
            timestamps=timestamps,
            lats=lats.astype(np.float32),
            lons=lons.astype(np.float32),
            lmax=np.array(LMAX, dtype=np.int16),
            altitude_km=AALT.astype(np.float32),
            tec_definition="IRI vTEC on GL23x45 integrated from 80 to 2000 km",
        )
        print(f"  {y}: {iri.shape[0]} maps -> {dest_path.name}")


def build_dtec_cache(data_root: Path, year: int | None = None, overwrite: bool = False) -> None:
    ionex_dir = data_root / "interpolated_gl23x45"
    iri_dir = data_root / "iri_gl23x45"
    out_dir = data_root / "dtec_gl23x45"
    out_dir.mkdir(parents=True, exist_ok=True)

    for y in iter_years(data_root, year):
        ionex_path = ionex_dir / f"{y}.npz"
        iri_path = iri_dir / f"{y}.npz"
        dest_path = out_dir / f"{y}.npz"

        if dest_path.exists() and not overwrite:
            print(f"  {y}: skip dTEC (exists)")
            continue
        if not iri_path.exists():
            print(f"  {y}: missing {iri_path}, skip")
            continue

        ionex = np.load(ionex_path)
        iri = np.load(iri_path)

        ionex_timestamps = ionex["timestamps"].astype(np.int64)
        iri_timestamps = iri["timestamps"].astype(np.int64)
        if not np.array_equal(ionex_timestamps, iri_timestamps):
            raise ValueError(f"{y}: IONEX and IRI timestamps do not match")

        dtec = ionex["tec"].astype(np.float32) - iri["iri"].astype(np.float32)
        np.savez_compressed(
            dest_path,
            dtec=dtec,
            timestamps=ionex_timestamps,
            lats=iri["lats"].astype(np.float32),
            lons=iri["lons"].astype(np.float32),
            lmax=np.array(LMAX, dtype=np.int16),
            residual_definition=(
                "dTEC = IONEX vTEC on GL23x45 minus IRI vTEC on GL23x45. "
                "No plasmaspheric correction applied; remaining offset is a "
                "known mostly zonal systematic."
            ),
        )
        print(f"  {y}: {dtec.shape[0]} maps -> {dest_path.name}")


def load_omni_year(data_root: Path, year: int) -> pd.DataFrame:
    from data_transformation import read_decompress, parse_omni_hro

    omni_path = data_root / "raw" / "omni_hro" / str(year) / f"omni_5min{year}.asc"
    if not omni_path.exists():
        raise FileNotFoundError(f"Missing OMNI HRO file: {omni_path}")

    omni = parse_omni_hro(read_decompress(str(omni_path)))
    missing_features = [feature for feature in OMNI_HRO_FEATURES if feature not in omni.columns]
    if missing_features:
        raise ValueError(f"OMNI frame missing columns: {missing_features}")

    omni = omni[OMNI_HRO_FEATURES].sort_index()
    if omni.index.has_duplicates:
        omni = omni.groupby(level=0).mean()
    return omni


def align_omni_to_timestamps(omni: pd.DataFrame, timestamps: np.ndarray) -> np.ndarray:
    target_index = pd.DatetimeIndex(pd.to_datetime(timestamps, unit="s"))
    unique_target_index = pd.DatetimeIndex(target_index.unique()).sort_values()
    combined_index = omni.index.union(unique_target_index).sort_values()
    aligned_unique = (
        omni.reindex(combined_index)
        .interpolate(method="time", limit_direction="both")
        .reindex(unique_target_index)
    )

    if aligned_unique.isna().any().any():
        missing = aligned_unique.isna().sum()
        raise ValueError(f"OMNI alignment still has NaNs:\n{missing.to_string()}")

    indexer = aligned_unique.index.get_indexer(target_index)
    if (indexer < 0).any():
        raise ValueError("OMNI alignment failed to map every dTEC timestamp")

    return aligned_unique.to_numpy(dtype=np.float32)[indexer]


def load_kp_3hourly(data_root: Path) -> pd.DataFrame:
    kp_path = data_root / "raw" / "geomag" / "kp_3hourly.parquet"
    if not kp_path.exists():
        raise FileNotFoundError(
            f"Missing 3-hourly Kp file: {kp_path}. "
            "Run python data_pull/geomag_pull.py first."
        )

    kp = pd.read_parquet(kp_path)
    if "kp" not in kp.columns:
        raise ValueError(f"Kp table missing kp column: {list(kp.columns)}")

    kp = kp[["kp"]].sort_index()
    kp.index = pd.to_datetime(kp.index)
    if kp.index.has_duplicates:
        kp = kp.groupby(level=0).mean()
    return kp


def align_kp_to_timestamps(kp: pd.DataFrame, timestamps: np.ndarray) -> np.ndarray:
    target_index = pd.DatetimeIndex(pd.to_datetime(timestamps, unit="s"))
    unique_target_index = pd.DatetimeIndex(target_index.unique()).sort_values()

    aligned_unique = (
        kp.reindex(kp.index.union(unique_target_index).sort_values())
        .ffill()
        .reindex(unique_target_index)
    )

    if aligned_unique.isna().any().any():
        missing = aligned_unique.isna().sum()
        raise ValueError(f"Kp alignment still has NaNs:\n{missing.to_string()}")

    indexer = aligned_unique.index.get_indexer(target_index)
    if (indexer < 0).any():
        raise ValueError("Kp alignment failed to map every dTEC timestamp")

    return aligned_unique.to_numpy(dtype=np.float32)[indexer]


def build_omni_cache(data_root: Path, year: int | None = None, overwrite: bool = False) -> None:
    dtec_dir = data_root / "dtec_gl23x45"
    out_dir = data_root / "omni_aligned_gl23x45"
    out_dir.mkdir(parents=True, exist_ok=True)
    kp = load_kp_3hourly(data_root)

    for y in iter_years(data_root, year):
        dtec_path = dtec_dir / f"{y}.npz"
        dest_path = out_dir / f"{y}.npz"

        if dest_path.exists() and not overwrite:
            print(f"  {y}: skip OMNI (exists)")
            continue
        if not dtec_path.exists():
            print(f"  {y}: missing {dtec_path}, skip")
            continue

        dtec = np.load(dtec_path)
        timestamps = dtec["timestamps"].astype(np.int64)
        omni = load_omni_year(data_root, y)
        aligned_omni = align_omni_to_timestamps(omni, timestamps)
        aligned_kp = align_kp_to_timestamps(kp, timestamps)
        aligned = np.concatenate([aligned_omni, aligned_kp], axis=1).astype(np.float32)

        np.savez_compressed(
            dest_path,
            omni=aligned,
            timestamps=timestamps,
            features=np.asarray(DRIVER_FEATURES),
            source=(
                "OMNI HRO 5-minute values time-interpolated to dTEC timestamps; "
                "3-hourly Kp causally forward-filled to dTEC timestamps"
            ),
        )
        print(f"  {y}: {aligned.shape[0]} rows -> {dest_path.name}")


def load_aligned_series(data_root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dtec_dir = data_root / "dtec_gl23x45"
    omni_dir = data_root / "omni_aligned_gl23x45"
    years = sorted(int(path.stem) for path in dtec_dir.glob("*.npz"))

    dtec_parts = []
    omni_parts = []
    timestamp_parts = []
    lats = None
    lons = None

    for year in years:
        dtec_path = dtec_dir / f"{year}.npz"
        omni_path = omni_dir / f"{year}.npz"
        if not omni_path.exists():
            print(f"  {year}: missing aligned OMNI, skip")
            continue

        dtec = np.load(dtec_path)
        omni = np.load(omni_path)
        timestamps = dtec["timestamps"].astype(np.int64)
        if not np.array_equal(timestamps, omni["timestamps"].astype(np.int64)):
            raise ValueError(f"{year}: dTEC and OMNI timestamps do not match")

        dtec_parts.append(dtec["dtec"].astype(np.float32))
        omni_parts.append(omni["omni"].astype(np.float32))
        timestamp_parts.append(timestamps)

        if lats is None:
            lats = dtec["lats"].astype(np.float32)
            lons = dtec["lons"].astype(np.float32)

    if not dtec_parts:
        raise ValueError("No aligned dTEC/OMNI yearly caches found")

    dtec_all = np.concatenate(dtec_parts)
    omni_all = np.concatenate(omni_parts)
    timestamps_all = np.concatenate(timestamp_parts)

    order = np.argsort(timestamps_all, kind="stable")
    timestamps_all = timestamps_all[order]
    dtec_all = dtec_all[order]
    omni_all = omni_all[order]

    unique_timestamps, unique_idx = np.unique(timestamps_all, return_index=True)
    dropped = len(timestamps_all) - len(unique_timestamps)
    if dropped:
        print(f"  dropped {dropped} duplicate timestamps before windowing")

    return dtec_all[unique_idx], omni_all[unique_idx], unique_timestamps, lats, lons


def valid_window_starts(dtec: np.ndarray, omni: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
    total_steps = INPUT_STEPS + TARGET_STEPS
    if len(timestamps) < total_steps:
        return np.array([], dtype=np.int64)

    diffs = np.diff(timestamps)
    start_count = len(timestamps) - total_steps + 1
    starts = np.arange(start_count, dtype=np.int64)
    valid = np.ones(start_count, dtype=bool)

    for offset in range(total_steps - 1):
        valid &= diffs[starts + offset] == diffs[starts]
    valid &= diffs[starts] > 0

    finite_time = np.isfinite(omni).all(axis=1) & np.isfinite(dtec).all(axis=(1, 2))
    finite_prefix = np.concatenate([[0], np.cumsum(finite_time.astype(np.int64))])
    finite_count = finite_prefix[starts + total_steps] - finite_prefix[starts]
    valid &= finite_count == total_steps

    return starts[valid]


def split_window_starts(starts: np.ndarray, timestamps: np.ndarray,
                        train_end_year: int, val_end_year: int) -> dict[str, np.ndarray]:
    start_years = pd.to_datetime(timestamps[starts], unit="s", utc=True).year
    return {
        "train": starts[start_years <= train_end_year],
        "val": starts[(start_years > train_end_year) & (start_years <= val_end_year)],
        "test": starts[start_years > val_end_year],
    }


def window_stats(dtec: np.ndarray, omni: np.ndarray, starts: np.ndarray,
                 chunk_size: int) -> tuple[float, float, np.ndarray, np.ndarray]:
    if len(starts) == 0:
        raise ValueError("Train split has zero windows; cannot compute normalization stats")

    input_offsets = np.arange(INPUT_STEPS)
    tec_sum = 0.0
    tec_sumsq = 0.0
    tec_count = 0
    omni_sum = np.zeros(len(DRIVER_FEATURES), dtype=np.float64)
    omni_sumsq = np.zeros(len(DRIVER_FEATURES), dtype=np.float64)
    omni_count = 0

    for begin in range(0, len(starts), chunk_size):
        batch_starts = starts[begin:begin + chunk_size]
        idx = batch_starts[:, None] + input_offsets[None, :]
        tec_chunk = dtec[idx].astype(np.float64)
        omni_chunk = omni[idx].astype(np.float64)

        tec_sum += tec_chunk.sum()
        tec_sumsq += np.square(tec_chunk).sum()
        tec_count += tec_chunk.size
        omni_sum += omni_chunk.sum(axis=(0, 1))
        omni_sumsq += np.square(omni_chunk).sum(axis=(0, 1))
        omni_count += omni_chunk.shape[0] * omni_chunk.shape[1]

    tec_mean = tec_sum / tec_count
    tec_var = max(tec_sumsq / tec_count - tec_mean ** 2, 1e-12)
    omni_mean = omni_sum / omni_count
    omni_var = np.maximum(omni_sumsq / omni_count - np.square(omni_mean), 1e-12)
    return tec_mean, float(np.sqrt(tec_var)), omni_mean, np.sqrt(omni_var)


def remove_existing_outputs(out_dir: Path) -> None:
    for path in out_dir.glob("*.npy"):
        path.unlink()
    metadata = out_dir / "metadata.json"
    if metadata.exists():
        metadata.unlink()


def write_split_windows(out_dir: Path, split_name: str, dtec: np.ndarray, omni: np.ndarray,
                        timestamps: np.ndarray, starts: np.ndarray, tec_mean: float,
                        tec_std: float, omni_mean: np.ndarray, omni_std: np.ndarray,
                        chunk_size: int) -> None:
    n = len(starts)
    input_offsets = np.arange(INPUT_STEPS)
    target_offsets = np.arange(INPUT_STEPS, INPUT_STEPS + TARGET_STEPS)

    tec_out = np.lib.format.open_memmap(
        out_dir / f"{split_name}_tec_input.npy",
        mode="w+",
        dtype=np.float32,
        shape=(n, INPUT_STEPS, NLAT, NLON),
    )
    omni_out = np.lib.format.open_memmap(
        out_dir / f"{split_name}_omni_input.npy",
        mode="w+",
        dtype=np.float32,
        shape=(n, INPUT_STEPS, len(DRIVER_FEATURES)),
    )
    target_out = np.lib.format.open_memmap(
        out_dir / f"{split_name}_target.npy",
        mode="w+",
        dtype=np.float32,
        shape=(n, TARGET_STEPS, NLAT, NLON),
    )

    for begin in range(0, n, chunk_size):
        end = min(begin + chunk_size, n)
        batch_starts = starts[begin:end]
        input_idx = batch_starts[:, None] + input_offsets[None, :]
        target_idx = batch_starts[:, None] + target_offsets[None, :]

        tec_out[begin:end] = (dtec[input_idx] - tec_mean) / tec_std
        omni_out[begin:end] = (omni[input_idx] - omni_mean) / omni_std
        target_out[begin:end] = (dtec[target_idx] - tec_mean) / tec_std

    tec_out.flush()
    omni_out.flush()
    target_out.flush()
    np.save(out_dir / f"{split_name}_window_start_times.npy", timestamps[starts].astype(np.int64))
    print(f"  {split_name}: {n} windows")


def build_windowed_dataset(data_root: Path, out_dir: Path, train_end_year: int,
                           val_end_year: int, overwrite: bool, chunk_size: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if any(out_dir.glob("*.npy")) and not overwrite:
        raise FileExistsError(f"{out_dir} already has .npy outputs; pass --overwrite to rebuild")
    if overwrite:
        remove_existing_outputs(out_dir)

    dtec, omni, timestamps, lats, lons = load_aligned_series(data_root)
    starts = valid_window_starts(dtec, omni, timestamps)
    splits = split_window_starts(starts, timestamps, train_end_year, val_end_year)

    print(f"  total unique frames: {len(timestamps)}")
    print(f"  valid windows: {len(starts)}")

    tec_mean, tec_std, omni_mean, omni_std = window_stats(
        dtec, omni, splits["train"], chunk_size
    )

    for split_name in ("train", "val", "test"):
        write_split_windows(
            out_dir, split_name, dtec, omni, timestamps, splits[split_name],
            tec_mean, tec_std, omni_mean, omni_std, chunk_size
        )

    np.save(out_dir / "lats.npy", lats)
    np.save(out_dir / "lons.npy", lons)

    metadata = {
        "format": "disk-backed normalized numpy arrays",
        "lmax": LMAX,
        "nlat": NLAT,
        "nlon": NLON,
        "input_steps": INPUT_STEPS,
        "target_steps": TARGET_STEPS,
        "omni_features": DRIVER_FEATURES,
        "train_end_year": train_end_year,
        "val_end_year": val_end_year,
        "normalization": {
            "tec_mean": tec_mean,
            "tec_std": tec_std,
            "omni_mean": omni_mean.tolist(),
            "omni_std": omni_std.tolist(),
            "computed_from": "train tec_input and train omni_input only",
            "applied_to": "tec_input, omni_input, and target arrays",
        },
        "splits": {name: int(len(split_starts)) for name, split_starts in splits.items()},
        "residual_definition": (
            "dTEC = IONEX vTEC on GL23x45 minus IRI vTEC on GL23x45. "
            "No plasmaspheric correction applied."
        ),
        "window_rule": (
            "A window is kept only if all 9 timestamps are finite, strictly "
            "increasing, and have identical adjacent spacing."
        ),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"  wrote {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name in ("iri-cache", "dtec-cache", "omni-cache"):
        sub = subparsers.add_parser(name)
        sub.add_argument("--year", type=int)
        sub.add_argument("--overwrite", action="store_true")

    windows = subparsers.add_parser("windows")
    windows.add_argument("--out-dir", default="data/falisha_windows_gl23x45")
    windows.add_argument("--train-end-year", type=int, default=2019)
    windows.add_argument("--val-end-year", type=int, default=2022)
    windows.add_argument("--chunk-size", type=int, default=512)
    windows.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()
    data_root = Path(args.data_root)

    if args.command == "iri-cache":
        build_iri_cache(data_root, year=args.year, overwrite=args.overwrite)
    elif args.command == "dtec-cache":
        build_dtec_cache(data_root, year=args.year, overwrite=args.overwrite)
    elif args.command == "omni-cache":
        build_omni_cache(data_root, year=args.year, overwrite=args.overwrite)
    elif args.command == "windows":
        build_windowed_dataset(
            data_root=data_root,
            out_dir=Path(args.out_dir),
            train_end_year=args.train_end_year,
            val_end_year=args.val_end_year,
            overwrite=args.overwrite,
            chunk_size=args.chunk_size,
        )


if __name__ == "__main__":
    main()
