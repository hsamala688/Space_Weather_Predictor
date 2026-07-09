"""
Stage 2: parsing and gridding of the raw files produced by Stage 1.

Implemented and verified:
  - read_decompress : .Z (Unix LZW) / .gz (gzip) / .asc (plain) -> text stream
  - parse_ionex     : IONEX v1.0 stream -> list of (datetime, ndarray[n_lat, n_lon])
                      TEC maps in TECU, NaN where the source had the 9999 fill value
  - parse_omni_hro  : OMNI 5-min HRO stream -> timestamped DataFrame of the five
                      IMF/solar-wind driver channels, fill values converted to NaN

Intentionally NOT included yet (each carries an open decision):
  - OMNI2 parsing               (hourly low-res product; DIFFERENT format from HRO)
  - IONEX -> equiangular 128x256 interpolation   (pole handling is a design choice)
  - caching of parsed output    (on-disk format/layout undecided)

Usage:
    python data_pull/data_transformation.py    # smoke-test the parsers
"""
from __future__ import annotations

import gzip
import io
import os
from datetime import datetime

import numpy as np
import pandas as pd
import unlzw3


# ---------------------------------------------------------------------------
# Decompression
# ---------------------------------------------------------------------------

def read_decompress(file_path):
    """Return an in-memory text stream for a raw Stage 1 file.

    Branches on extension: .Z is Unix LZW compress (gzip cannot read it),
    .gz is standard gzip, .asc is uncompressed text.
    """
    with open(file_path, 'rb') as f:          # read as binary, not text
        raw_bytes = f.read()

    ext = os.path.splitext(file_path)[1].lower()

    if ext == '.z':                            # pre-2023 IONEX
        return io.StringIO(unlzw3.unlzw(raw_bytes).decode('utf-8'))
    elif ext == '.gz':                         # post-2023 IONEX
        with gzip.open(io.BytesIO(raw_bytes), 'rt', encoding='utf-8') as gz:
            return io.StringIO(gz.read())
    elif ext == '.asc':                        # OMNI plain text
        return io.StringIO(raw_bytes.decode('utf-8'))
    else:
        raise ValueError(f"Unexpected extension {ext!r} for {file_path}")


# ---------------------------------------------------------------------------
# IONEX v1.0 parser
# ---------------------------------------------------------------------------

# Labels (cols 61-80) that are structural, i.e. NOT value lines.
_LABELS = {
    "EXPONENT", "LAT1 / LAT2 / DLAT", "LON1 / LON2 / DLON",
    "END OF HEADER", "START OF TEC MAP", "EPOCH OF CURRENT MAP",
    "LAT/LON1/LON2/DLON/H", "END OF TEC MAP", "START OF RMS MAP",
}


def parse_ionex(stream):
    """Parse an IONEX v1.0 stream into TEC maps.

    Returns:
        maps : list of (datetime, ndarray[n_lat, n_lon]) in TECU, NaN where missing
        lats : ndarray[n_lat] latitudes  (deg, north -> south)
        lons : ndarray[n_lon] longitudes (deg, -180 -> 180)
    """
    exponent = -1                              # IONEX default; header overrides
    lat1 = lat2 = dlat = None
    lon1 = lon2 = dlon = None

    # ---- header ----
    for line in stream:
        label = line[60:].strip()
        if label == "EXPONENT":
            exponent = int(line[:60].split()[0])
        elif label == "LAT1 / LAT2 / DLAT":
            lat1, lat2, dlat = (float(x) for x in line[:60].split())
        elif label == "LON1 / LON2 / DLON":
            lon1, lon2, dlon = (float(x) for x in line[:60].split())
        elif label == "END OF HEADER":
            break

    if dlat is None or dlon is None:
        raise ValueError("IONEX header missing grid definition")

    lats = np.arange(lat1, lat2 + dlat / 2, dlat)   # inclusive of lat2
    lons = np.arange(lon1, lon2 + dlon / 2, dlon)
    n_lat, n_lon = len(lats), len(lons)
    scale = 10.0 ** exponent

    # ---- body (state machine over TEC map blocks) ----
    maps = []
    timestamp = None
    grid = None
    row = -1
    buf = None

    for line in stream:
        label = line[60:].strip()

        if label == "START OF RMS MAP":
            break                                   # done with TEC; ignore RMS maps
        elif label == "START OF TEC MAP":
            grid = np.full((n_lat, n_lon), np.nan)
            row, buf = -1, None
        elif label == "EPOCH OF CURRENT MAP":
            y, mo, d, h, mi, s = (int(x) for x in line[:60].split())
            timestamp = datetime(y, mo, d, h, mi, s)
        elif label == "LAT/LON1/LON2/DLON/H":
            row += 1                                # bands march in header order
            buf = []
        elif label == "END OF TEC MAP":
            maps.append((timestamp, grid))
            buf = None
        elif label not in _LABELS:                  # a value line
            # value lines fill cols 1-80, so split the WHOLE line, not line[:60]
            if buf is not None and 0 <= row < n_lat:
                buf.extend(int(v) for v in line.split())
                if len(buf) >= n_lon:
                    vals = np.array(buf[:n_lon], dtype=float)
                    vals[vals == 9999] = np.nan     # NaN before scaling
                    grid[row] = vals * scale
                    buf = None

    return maps, lats, lons


# ---------------------------------------------------------------------------
# OMNI 5-min HRO parser
# ---------------------------------------------------------------------------

# Full record per HRO_format.txt: 46 base fields + 3 GOES flux fields (5-min only) = 49.
_OMNI_COLUMNS = [
    "year", "day", "hour", "minute", "id_imf", "id_sw",
    "num_pts_imf", "num_pts_sw", "percent_interp", "timeshift",
    "rms_timeshift", "rms_phase", "time_between_obs", "b_magnitude",
    "bx_gse", "by_gse", "bz_gse", "by_gsm", "bz_gsm", "rms_b_scalar",
    "rms_b_vector", "flow_speed", "vx_gse", "vy_gse", "vz_gse",
    "proton_density", "temperature", "flow_pressure", "e_field",
    "beta", "mach_number", "x_gse", "y_gse", "z_gse",
    "bsn_x_gse", "bsn_y_gse", "bsn_z_gse",
    "ae_index", "al_index", "au_index",
    "sym_d", "sym_h", "asy_d", "asy_h",
    "pc_n_index", "magnetosonic_mach",
    "pr_flux_10", "pr_flux_30", "pr_flux_60",   # 5-min only
]

# The five IMF/solar-wind channels that feed the model (omni_input contract).
_OMNI_KEEP = ["b_magnitude", "by_gsm", "bz_gsm", "flow_speed", "proton_density"]

# Fill sentinel per kept field (the max-magnitude value for that field width).
_OMNI_FILL = {
    "b_magnitude": 9999.99,     # F8.2
    "by_gsm": 9999.99,          # F8.2
    "bz_gsm": 9999.99,          # F8.2
    "flow_speed": 99999.9,      # F8.1
    "proton_density": 999.99,   # F7.2
}


def parse_omni_hro(stream):
    """Parse an OMNI 5-min HRO stream into a timestamped driver frame.

    Parses all 49 fields to keep column alignment honest, then returns only the
    five driver channels with fill values converted to NaN.

    Returns:
        DataFrame indexed by timestamp with columns:
        b_magnitude, by_gsm, bz_gsm, flow_speed, proton_density
    """
    # Fail loud if this file is not the 49-field 5-min product (e.g. a 1-min file).
    n_tokens = len(stream.readline().split())
    stream.seek(0)
    if n_tokens != len(_OMNI_COLUMNS):
        raise ValueError(
            f"OMNI parse: expected {len(_OMNI_COLUMNS)} fields per record, "
            f"got {n_tokens} - column list misaligned with this product."
        )

    df = pd.read_csv(stream, sep=r"\s+", header=None, names=_OMNI_COLUMNS)

    # Timestamp from year + day-of-year + hour + minute (day is DOY, not month/day).
    ts = (
        pd.to_datetime((df["year"] * 1000 + df["day"]).astype(str), format="%Y%j")
        + pd.to_timedelta(df["hour"], unit="h")
        + pd.to_timedelta(df["minute"], unit="m")
    )

    out = df[_OMNI_KEEP].copy()
    out.insert(0, "timestamp", ts)
    out = out.set_index("timestamp")

    # Fills are the max-magnitude sentinel per field; mask by threshold, not ==,
    # to avoid float-equality misses. Real values never approach these magnitudes.
    for col, fill in _OMNI_FILL.items():
        out.loc[out[col] >= fill, col] = np.nan

    return out


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def _smoke_ionex(path):
    maps, lats, lons = parse_ionex(read_decompress(path))
    t0, m0 = maps[0]
    finite = m0[np.isfinite(m0)]
    print(f"{path}")
    print(f"  {len(maps)} maps | first {t0} | shape {m0.shape}")
    print(f"  lats {lats[0]}..{lats[-1]}  lons {lons[0]}..{lons[-1]}")
    print(f"  TEC {finite.min():.1f}..{finite.max():.1f} TECU | "
          f"{np.isnan(m0).sum()} NaN cells\n")


def _smoke_omni(path):
    df = parse_omni_hro(read_decompress(path))
    print(f"{path}")
    print(f"  {len(df)} rows | {df.index[0]} .. {df.index[-1]}")
    print(df.agg(["min", "max"]).round(2).to_string())
    print(f"  NaN per column:\n{df.isna().sum().to_string()}\n")


if __name__ == "__main__":
    _smoke_ionex("data/raw/ionex/2010/001/codg0010.10i.Z")
    _smoke_ionex("data/raw/ionex/2024/001/COD0OPSFIN_20240010000_01D_01H_GIM.INX.gz")
    _smoke_omni("data/raw/omni_hro/2015/omni_5min2015.asc")