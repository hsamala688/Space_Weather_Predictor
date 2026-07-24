#!/usr/bin/env python3
"""One-shot, idempotent migration of the flat data/ tree into a Medallion layout.

Bronze = raw ingested + ingestion manifests. Silver = cleaned/derived. Gold =
ML-ready windows. Moves are pure os.rename (instant on one filesystem, fully
reversible). Already-migrated items are skipped, so re-running is safe.

Not migrated (left in place as an implicit archive): the bare data/interpolated/
dir (superseded by the _gl23x45 build), data/raw/omni2/ (OMNI2 disqualified),
*.celestrak_backup.parquet, kp_daily.parquet, and sample .npz blobs.

Usage:
    python scripts/migrate_to_medallion.py [--data-root data] [--dry-run]
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

# (source relative to data_root, destination relative to data_root)
MOVES: list[tuple[str, str]] = [
    # bronze: raw, source-shaped
    ("raw/ionex", "bronze/ionex"),
    ("raw/omni_hro", "bronze/omni_hro"),
    ("raw/gfz", "bronze/gfz"),
    ("manifests", "bronze/_manifests"),
    # silver: derived index tables (were mislabeled under raw/)
    ("raw/f107/f107_daily.parquet", "silver/f107_daily.parquet"),
    ("raw/geomag/kp_3hourly.parquet", "silver/kp_3hourly.parquet"),
    # silver: cleaned/conformed grids
    ("interpolated_gl23x45", "silver/tec_gl23x45"),
    ("iri_gl23x45", "silver/iri_gl23x45"),
    ("dtec_gl23x45", "silver/dtec_gl23x45"),
    ("omni_aligned_gl23x45", "silver/omni_aligned_gl23x45"),
    # gold: ML-ready windows (de-named)
    ("falisha_windows_gl23x45", "gold/training_windows"),
]

_LAYER_ROOTS = ("bronze", "silver", "gold")


def migrate(data_root: Path, dry_run: bool) -> None:
    for layer in _LAYER_ROOTS:
        (data_root / layer).mkdir(parents=True, exist_ok=True) if not dry_run else None

    moved = skipped = missing = 0
    for src_rel, dst_rel in MOVES:
        src = data_root / src_rel
        dst = data_root / dst_rel

        if dst.exists():
            print(f"  skip (already at destination): {dst_rel}")
            skipped += 1
            continue
        if not src.exists():
            print(f"  missing source, nothing to move: {src_rel}")
            missing += 1
            continue

        print(f"  move: {src_rel}  ->  {dst_rel}")
        if not dry_run:
            dst.parent.mkdir(parents=True, exist_ok=True)
            os.rename(src, dst)
        moved += 1

    print(f"\nSummary: {moved} moved, {skipped} already-migrated, {missing} missing"
          f"{'  (dry run, nothing changed)' if dry_run else ''}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default="data", type=Path)
    ap.add_argument("--dry-run", action="store_true", help="print moves without doing them")
    args = ap.parse_args()

    root = args.data_root.resolve()
    if not root.exists():
        raise SystemExit(f"data root does not exist: {root}")
    print(f"Migrating data root: {root}\n")
    migrate(root, args.dry_run)


if __name__ == "__main__":
    main()
