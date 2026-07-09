"""
Stage 1: raw data extraction.
Sources: CDDIS IONEX (authenticated), SPDF OMNI HRO, SPDF OMNI2 (anonymous).
Usage:
    python data_pull/data_time.py            # full pull
    python data_pull/data_time.py --verify   # coverage report only
"""
from __future__ import annotations

import csv
import os
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterator

import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    center: str = "COD"
    start_date: date = date(2000, 1, 1)
    end_date: date = date(2025, 12, 31)
    data_root: Path = Path(__file__).parent.parent / "data"
    ionex_base: str = "https://cddis.nasa.gov/archive/gnss/products/ionex/"
    omni_hro_base: str = "https://spdf.gsfc.nasa.gov/pub/data/omni/high_res_omni/"
    omni2_base: str = "https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/"


CFG = Config()


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------

def _daterange(start: date, end: date) -> Iterator[date]:
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def _years(start: date, end: date) -> range:
    return range(start.year, end.year + 1)


# ---------------------------------------------------------------------------
# IONEX URL + path builder
# ---------------------------------------------------------------------------

# IGS switched to long-name format on this date (DOY 219, 2023).
_RENAME_BOUNDARY = date(2023, 8, 7)


def _ionex_targets(obs_date: date, center: str) -> list[tuple[str, str]]:
    """Return candidate (relative_url, filename) pairs for one IONEX day.

    The 2023 IGS rename boundary is messy in practice, so we try both the
    preferred name for the date and the alternate naming scheme. The first
    successful download becomes the manifest row for that day.
    """
    doy = obs_date.timetuple().tm_yday
    yyyy = obs_date.year
    yy = yyyy % 100

    legacy = f"{center.lower()}g{doy:03d}0.{yy:02d}i.Z"
    long_name = f"{center.upper()}0OPSFIN_{yyyy}{doy:03d}0000_01D_01H_GIM.INX.gz"

    if obs_date < _RENAME_BOUNDARY:
        names = [legacy, long_name]
    else:
        names = [long_name, legacy]

    return [(f"{yyyy}/{doy:03d}/{fname}", fname) for fname in names]


def _ionex_target(obs_date: date, center: str) -> tuple[str, str]:
    """Return the preferred (relative_url, filename) for one IONEX day."""
    return _ionex_targets(obs_date, center)[0]


def _ionex_dest(data_root: Path, obs_date: date, fname: str) -> Path:
    doy = obs_date.timetuple().tm_yday
    p = data_root / "raw" / "ionex" / str(obs_date.year) / f"{doy:03d}" / fname
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# Manifest (append-mode CSV; last row per key wins on read)
# ---------------------------------------------------------------------------

_COLS = ["source", "key", "expected_filename", "status", "reason", "n_bytes", "checked_at"]


def _manifest_path(data_root: Path, source: str) -> Path:
    p = data_root / "manifests" / f"{source}_manifest.csv"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _read_manifest(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    with open(path, newline="") as f:
        return {r["key"]: r for r in csv.DictReader(f)}


def _append_row(path: Path, manifest: dict, **row) -> None:
    manifest[row["key"]] = row
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_COLS)
        if write_header:
            w.writeheader()
        w.writerow(row)


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

_MIN_BYTES = 2048
_HTML_TELLS = (b"<!DOCTYPE", b"<!doctype", b"<html", b"<HTML")


def _make_session() -> requests.Session:
    s = requests.Session()
    s.trust_env = True  # reads ~/.netrc for CDDIS auth
    return s


def _is_html(data: bytes) -> bool:
    head = data[:256].lstrip()
    return any(head.startswith(t) for t in _HTML_TELLS)


def _download(session: requests.Session, url: str, dest: Path, retries: int = 3) -> dict:
    """Atomic download with integrity check. Returns {status, reason, n_bytes}."""
    part = Path(str(dest) + ".part")
    part.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(retries):
        try:
            resp = session.get(url, timeout=60, allow_redirects=True)
        except requests.exceptions.Timeout:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            return {"status": "failed", "reason": "timeout", "n_bytes": 0}
        except requests.exceptions.RequestException as exc:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            return {"status": "failed", "reason": f"connection:{exc}", "n_bytes": 0}

        if resp.status_code == 404:
            return {"status": "failed", "reason": "404", "n_bytes": 0}
        if resp.status_code in (401, 403):
            return {"status": "failed", "reason": "auth", "n_bytes": 0}
        if resp.status_code >= 500:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            return {"status": "failed", "reason": f"http_{resp.status_code}", "n_bytes": 0}
        if resp.status_code != 200:
            return {"status": "failed", "reason": f"http_{resp.status_code}", "n_bytes": 0}

        data = resp.content
        if len(data) < _MIN_BYTES:
            return {"status": "failed", "reason": "bad_content:too_small", "n_bytes": len(data)}
        if _is_html(data):
            return {"status": "failed", "reason": "bad_content:html_page", "n_bytes": len(data)}

        part.write_bytes(data)
        os.replace(part, dest)
        return {"status": "downloaded", "reason": "", "n_bytes": len(data)}

    return {"status": "failed", "reason": "max_retries", "n_bytes": 0}


# ---------------------------------------------------------------------------
# Resumability
# ---------------------------------------------------------------------------

def _skip(key: str, manifest: dict, dest: Path) -> bool:
    row = manifest.get(key)
    if row is None:
        return False
    if row["status"] in ("present", "downloaded") and dest.exists():
        return True
    if row["status"] == "failed" and row["reason"] == "404":
        return True  # real data gap; do not retry
    return False


# ---------------------------------------------------------------------------
# Auth smoke test
# ---------------------------------------------------------------------------

def _smoke_test(cfg: Config, session: requests.Session) -> None:
    """Download one known-good IONEX day. Aborts immediately on auth failure."""
    test_date = date(2010, 1, 1)
    rel_url, fname = _ionex_target(test_date, cfg.center)
    dest = _ionex_dest(cfg.data_root, test_date, fname)
    if dest.exists():
        print("smoke test: file already on disk, skipping fetch.")
        return
    url = cfg.ionex_base + rel_url
    print(f"smoke test: {url}")
    result = _download(session, url, dest)
    if result["status"] != "downloaded":
        sys.exit(
            f"\nAuth smoke test FAILED: {result['reason']}\n"
            "Check ~/.netrc — must contain:\n"
            "  machine urs.earthdata.nasa.gov\n"
            "  login YOUR_USERNAME\n"
            "  password YOUR_PASSWORD"
        )
    print(f"smoke test passed ({result['n_bytes']:,} bytes).")


# ---------------------------------------------------------------------------
# Per-source orchestrators
# ---------------------------------------------------------------------------

def _pull_ionex(cfg: Config, session: requests.Session) -> None:
    mpath = _manifest_path(cfg.data_root, "ionex")
    manifest = _read_manifest(mpath)

    for obs_date in _daterange(cfg.start_date, cfg.end_date):
        doy = obs_date.timetuple().tm_yday
        key = f"{obs_date.year}-{doy:03d}"
        targets = _ionex_targets(obs_date, cfg.center)
        rel_url, fname = targets[0]
        dest = _ionex_dest(cfg.data_root, obs_date, fname)

        if doy == 1:
            print(f"\n{obs_date.year} ", end="", flush=True)

        row = manifest.get(key)
        if row and row["status"] in ("present", "downloaded"):
            existing = _ionex_dest(cfg.data_root, obs_date, row["expected_filename"])
            if existing.exists():
                continue

        result = None
        used_fname = fname
        for rel_url, candidate_fname in targets:
            candidate_dest = _ionex_dest(cfg.data_root, obs_date, candidate_fname)
            result = _download(session, cfg.ionex_base + rel_url, candidate_dest)
            used_fname = candidate_fname
            if result["status"] == "downloaded" or result["reason"] in ("auth",):
                dest = candidate_dest
                break
            if result["reason"] != "404":
                dest = candidate_dest
                break

        _append_row(mpath, manifest,
            source="ionex", key=key, expected_filename=used_fname,
            status=result["status"], reason=result["reason"],
            n_bytes=result["n_bytes"], checked_at=datetime.utcnow().isoformat())

        if result["reason"] == "auth":
            sys.exit(f"\nAuth failure at {key}. Check ~/.netrc credentials.")

        char = "." if result["status"] == "downloaded" else ("_" if result["reason"] == "404" else "x")
        print(char, end="", flush=True)

    print()


def _pull_omni_hro(cfg: Config) -> None:
    mpath = _manifest_path(cfg.data_root, "omni_hro")
    manifest = _read_manifest(mpath)
    session = requests.Session()
    out_dir = cfg.data_root / "raw" / "omni_hro"

    for year in _years(cfg.start_date, cfg.end_date):
        key = str(year)
        fname = f"omni_5min{year}.asc"
        dest = out_dir / str(year) / fname
        dest.parent.mkdir(parents=True, exist_ok=True)

        if _skip(key, manifest, dest):
            print(f"  {year}: skip")
            continue

        result = _download(session, cfg.omni_hro_base + fname, dest)
        _append_row(mpath, manifest,
            source="omni_hro", key=key, expected_filename=fname,
            status=result["status"], reason=result["reason"],
            n_bytes=result["n_bytes"], checked_at=datetime.utcnow().isoformat())
        label = result["reason"] or f"{result['n_bytes']:,} bytes"
        print(f"  {year}: {result['status']} ({label})")


def _pull_omni2(cfg: Config) -> None:
    mpath = _manifest_path(cfg.data_root, "omni2")
    manifest = _read_manifest(mpath)
    session = requests.Session()
    out_dir = cfg.data_root / "raw" / "omni2"

    for year in _years(cfg.start_date, cfg.end_date):
        key = str(year)
        fname = f"omni2_{year}.dat"
        dest = out_dir / str(year) / fname
        dest.parent.mkdir(parents=True, exist_ok=True)

        if _skip(key, manifest, dest):
            print(f"  {year}: skip")
            continue

        result = _download(session, cfg.omni2_base + fname, dest)
        _append_row(mpath, manifest,
            source="omni2", key=key, expected_filename=fname,
            status=result["status"], reason=result["reason"],
            n_bytes=result["n_bytes"], checked_at=datetime.utcnow().isoformat())
        label = result["reason"] or f"{result['n_bytes']:,} bytes"
        print(f"  {year}: {result['status']} ({label})")


# ---------------------------------------------------------------------------
# Coverage report
# ---------------------------------------------------------------------------

def _report(cfg: Config) -> None:
    print("\n=== Coverage Report ===")

    mpath = _manifest_path(cfg.data_root, "ionex")

    manifest = _read_manifest(mpath)

    total = sum(1 for _ in _daterange(cfg.start_date, cfg.end_date))

    present = [k for k, v in manifest.items() if v["status"] in ("present", "downloaded")]
    gaps = [k for k, v in manifest.items() if v["reason"] == "404"]

    failures = [k for k, v in manifest.items()
                if v["status"] == "failed" and v["reason"] != "404"]
    print(f"IONEX     {len(present):>5}/{total} days  |  {len(gaps)} real gaps (404)"
          f"  |  {len(failures)} other failures")
    
    if failures:
        print(f"           needs attention: {failures[:5]}")

    for source in ("omni_hro", "omni2"):
        mpath = _manifest_path(cfg.data_root, source)
        manifest = _read_manifest(mpath)
        total = len(list(_years(cfg.start_date, cfg.end_date)))
        present = [k for k, v in manifest.items() if v["status"] in ("present", "downloaded")]
        failures = [k for k, v in manifest.items() if v["status"] == "failed"]
        print(f"{source:<10} {len(present):>5}/{total} years |  {len(failures)} failures")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="Stage 1: raw data pull.")
    p.add_argument("--verify", action="store_true", help="print coverage report only, no downloads")
    p.add_argument("--start-date", help="override start date, YYYY-MM-DD")
    p.add_argument("--end-date", help="override end date, YYYY-MM-DD")
    p.add_argument("--ionex-only", action="store_true", help="pull only IONEX")
    args = p.parse_args()

    cfg = CFG
    if args.start_date:
        cfg = Config(**{**cfg.__dict__, "start_date": date.fromisoformat(args.start_date)})
    if args.end_date:
        cfg = Config(**{**cfg.__dict__, "end_date": date.fromisoformat(args.end_date)})

    print(f"=== Stage 1: Raw Data Pull ===")
    print(f"Center: {cfg.center}  |  {cfg.start_date} to {cfg.end_date}")
    print(f"Data root: {cfg.data_root.resolve()}\n")

    if args.verify:
        _report(cfg)
        return

    session = _make_session()

    print("--- Auth smoke test ---")
    _smoke_test(cfg, session)

    print("\n--- IONEX (CDDIS, authenticated) ---")
    _pull_ionex(cfg, session)

    if args.ionex_only:
        _report(cfg)
        return

    print("\n--- OMNI HRO (SPDF, anonymous) ---")
    _pull_omni_hro(cfg)

    print("\n--- OMNI2 (SPDF, anonymous) ---")
    _pull_omni2(cfg)

    _report(cfg)


if __name__ == "__main__":
    main()
