#!/usr/bin/env python3
"""
Pull multi-year SuperMAG 1-minute data to Parquet, one month (or day) per file.
Resumable, rate-limited, with exponential backoff on failures.

Usage:
    export SUPERMAG_LOGON=your_username
    python3 super_mag_pull.py --start-year 2010 --end-year 2020 --dataset indices --outdir ./supermag_data
    python3 super_mag_pull.py --start-year 2010 --end-year 2020 --dataset data --stations ABK THL --outdir ./supermag_data
"""

import argparse
import calendar
import importlib.util
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

SUPERMAG_LOGON="hsamala"

# ---------------------------------------------------------------------------
# API client loader (filename uses a hyphen, so standard import won't work)
# ---------------------------------------------------------------------------

def _load_api(api_path: str | None):
    # Determine the candidates
    candidates = []
    if api_path:
        candidates.append(Path(api_path))
    
    # Fallback candidate
    candidates.append(Path(__file__).parent / "supermag_client_script.py")

    # Find the first candidate that actually exists
    api_file = None
    for c in candidates:
        if c.is_file():
            api_file = c
            break

    if not api_file:
        print("supermag-api.py not found. Pass --api-path or place it next to this script.", file=sys.stderr)
        sys.exit(1)

    # Load and execute the module safely
    spec = importlib.util.spec_from_file_location("supermag_api", str(api_file))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec for {api_file}")

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod



# ---------------------------------------------------------------------------
# Chunk window generators
# ---------------------------------------------------------------------------

def _month_windows(start_year: int, end_year: int):
    cur = datetime(start_year, 1, 1, tzinfo=timezone.utc)
    end = datetime(end_year, 1, 1, tzinfo=timezone.utc)
    while cur < end:
        days = calendar.monthrange(cur.year, cur.month)[1]
        yield cur, days * 86400
        next_month = cur.month % 12 + 1
        next_year = cur.year + (1 if cur.month == 12 else 0)
        cur = cur.replace(year=next_year, month=next_month)


def _day_windows(start_year: int, end_year: int):
    cur = datetime(start_year, 1, 1, tzinfo=timezone.utc)
    end = datetime(end_year, 1, 1, tzinfo=timezone.utc)
    while cur < end:
        yield cur, 86400
        cur += timedelta(days=1)


# ---------------------------------------------------------------------------
# Flattening (§7 of design doc)
# ---------------------------------------------------------------------------

def _flatten(df: pd.DataFrame) -> pd.DataFrame:
    # tval -> datetime
    if "tval" in df.columns:
        df["datetime"] = pd.to_datetime(df["tval"], unit="s", utc=True)

    # N/E/Z component dicts {nez, geo} -> N_nez, N_geo, etc.
    for comp in ("N", "E", "Z"):
        if comp in df.columns and df[comp].dtype == object:
            try:
                expanded = df[comp].apply(pd.Series)
                df[f"{comp}_nez"] = expanded.get("nez")
                df[f"{comp}_geo"] = expanded.get("geo")
                df = df.drop(columns=[comp])
            except Exception:
                pass

    # Vector field dicts {X, Y, Z} -> bgse_x, bgse_y, bgse_z, etc.
    for field in ("bgse", "bgsm", "vgse", "vgsm"):
        if field in df.columns and df[field].dtype == object:
            try:
                expanded = df[field].apply(pd.Series)
                for axis in ("X", "Y", "Z"):
                    df[f"{field}_{axis.lower()}"] = expanded.get(axis)
                df = df.drop(columns=[field])
            except Exception:
                pass

    # Drop 24-element regional arrays (v1: omit rather than stringify)
    regional = [c for c in df.columns if any(c.startswith(p) for p in ("SMLr", "SMUr", "SMEr"))]
    if regional:
        df = df.drop(columns=regional)

    return df


# ---------------------------------------------------------------------------
# Retry / backoff
# ---------------------------------------------------------------------------

def _fetch_with_retry(fn, *args, max_retries: int):
    """Call fn(*args) which returns (status, payload). status==0 is failure."""
    for attempt in range(max_retries + 1):
        try:
            status, payload = fn(*args)
            if status != 0:
                return payload
            
            # Extract the actual server error from the payload
            err = f"API returned status=0. Server message: {payload}"
            
        except Exception as exc:
            err = f"Python Exception: {str(exc)}"
            payload = None

        if attempt == max_retries:
            raise RuntimeError(f"failed after {max_retries + 1} attempts: {err}")
        sleep = 2 ** attempt
        print(f"    retry {attempt + 1}/{max_retries} in {sleep}s ({err})")
        time.sleep(sleep)


# ---------------------------------------------------------------------------
# Output paths and manifest
# ---------------------------------------------------------------------------

def _chunk_path(outdir: Path, dataset: str, window_start: datetime, station: str | None, daily: bool) -> Path:
    tag = window_start.strftime("%Y%m%d" if daily else "%Y%m")
    if dataset == "indices":
        return outdir / "indices" / f"indices_{tag}.parquet"
    return outdir / "data" / station / f"data_{station}_{tag}.parquet"


def _update_manifest(manifest_path: Path, key: str, entry):
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest.setdefault(key, [])
    manifest[key].append(entry)
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))


# ---------------------------------------------------------------------------
# Core pull loop
# ---------------------------------------------------------------------------

def pull(args):
    api = _load_api(getattr(args, "api_path", None))
    outdir = Path(args.outdir)
    manifest_path = outdir / "_manifest.json"
    daily = args.chunk == "daily"
    windows = list((_day_windows if daily else _month_windows)(args.start_year, args.end_year))

    logon = args.logon or os.environ.get("SUPERMAG_LOGON") or "hsamala"
    if not logon:
        sys.exit("Provide --logon or set SUPERMAG_LOGON env var.")

    for dataset in args.dataset:
        stations = args.stations if dataset == "data" else [None]
        if dataset == "data" and not stations:
            sys.exit("--stations required when dataset includes 'data'")

        for station in stations:
            subdir = (outdir / "data" / station) if station else (outdir / "indices")
            subdir.mkdir(parents=True, exist_ok=True)

            for window_start, extent in windows:
                path = _chunk_path(outdir, dataset, window_start, station, daily)
                label = f"{dataset}{'/' + station if station else ''} {window_start.strftime('%Y-%m-%d' if daily else '%Y-%m')}"

                if path.exists():
                    print(f"  skip  {label}")
                    continue

                print(f"  fetch {label} ...", end=" ", flush=True)
                start_str = window_start.strftime("%Y-%m-%dT%H:%M:%S")

                try:
                    if dataset == "indices":
                        raw = _fetch_with_retry(
                            api.SuperMAGGetIndices,
                            logon, start_str, extent, args.flags_indices,
                            max_retries=args.max_retries,
                        )
                    else:
                        raw = _fetch_with_retry(
                            api.SuperMAGGetData,
                            logon, start_str, extent, args.flags_data, station,
                            max_retries=args.max_retries,
                        )
                except RuntimeError as exc:
                    print(f"FAILED ({exc})")
                    _update_manifest(manifest_path, "failed", {"chunk": label, "error": str(exc)})
                    time.sleep(args.rate_limit_seconds)
                    continue

                if raw is None or (hasattr(raw, "__len__") and len(raw) == 0):
                    print("empty")
                    _update_manifest(manifest_path, "empty", label)
                    time.sleep(args.rate_limit_seconds)
                    continue

                df = raw if isinstance(raw, pd.DataFrame) else pd.DataFrame(raw)
                df = _flatten(df)
                df.to_parquet(path, index=False)
                print(f"ok ({len(df):,} rows)")
                _update_manifest(manifest_path, "completed", label)
                time.sleep(args.rate_limit_seconds)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Pull SuperMAG data to Parquet.")
    p.add_argument("--logon", help="SuperMAG username (or set SUPERMAG_LOGON env var)")
    p.add_argument("--start-year", type=int, default=2000)
    p.add_argument("--end-year", type=int, default=2020, help="exclusive upper bound")
    p.add_argument("--dataset", nargs="+", choices=["indices", "data"], default=["indices"])
    p.add_argument("--stations", nargs="+", help="IAGA station codes; required when dataset includes 'data'")
    p.add_argument("--flags-indices", default="all,swiall,imfall")
    p.add_argument("--flags-data", default="all")
    p.add_argument("--outdir", default="./supermag_data")
    p.add_argument("--chunk", choices=["monthly", "daily"], default="monthly")
    p.add_argument("--rate-limit-seconds", type=float, default=1.0)
    p.add_argument("--max-retries", type=int, default=4)
    p.add_argument("--api-path", help="path to supermag-api.py (auto-discovered if omitted)")
    args = p.parse_args()

    if args.start_year >= args.end_year:
        p.error("--start-year must be less than --end-year")

    pull(args)


if __name__ == "__main__":
    main()
