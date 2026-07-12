# `data_pull/` Pipeline — Full Walkthrough

This document explains, file by file, how raw space-weather data becomes the
normalized training tensors the SFNO model consumes. It reflects the code as
it actually exists in this folder today, not the original design doc's plan
(`data_pull_design.md`) — the two have diverged in a few places, which are
called out below.

## The one-sentence version

Download raw TEC maps and solar-wind/geomagnetic drivers → parse and
decompress them → interpolate TEC onto a common spherical grid → subtract an
IRI climatology baseline to get a residual field → align drivers to that
field's timestamps → cut everything into fixed-length input/target windows →
normalize and serve via a PyTorch `Dataset`.

```
pull_f107 ─┐
geomag_pull.py ─┤ (drivers, independent of IONEX)
data_time.py ───┼──► raw/ionex, raw/omni_hro, raw/omni2  (Stage 1: download)
                │
                ▼
data_transformation.py   (Stage 2: decompress + parse)
                │
                ▼
data_interpolation.py    (Stage 3: 71x73 native grid -> 23x45 Gauss-Legendre grid)
                │
                ▼
data_for_falisha.py      (Stage 4: IRI baseline, dTEC residual, driver
                │          alignment, windowing, normalization)
                ▼
falisha_dataset.py        (Stage 5: PyTorch Dataset/DataLoader)
```

Everything else in the folder (`export_dtec_sample.py`, `sakshum.py`,
`offset.py`, `test.py`, `yeet.py`) is a one-off analysis or scratch script
built on top of Stages 1-3, for teammates doing spectral/offset analysis
rather than model training. These are covered at the end.

---

## Stage 0: Driver pulls that don't depend on IONEX

Two small scripts pull scalar geomagnetic/solar indices independently of the
main IONEX/OMNI pull in `data_time.py`. Both hit the same source
(`https://celestrak.org/SpaceData/SW-All.txt`, CelesTrak's "Space Weather"
combined file) and parse its `BEGIN OBSERVED` / `END OBSERVED` block so that
predicted (forecast) rows never leak into training data.

### `pull_f107` (repo root, not in `data_pull/`)

Pulls daily observed F10.7 solar radio flux, 2000-2025.

- Field 30 of the whitespace-separated OBSERVED block is `F10.7_obs`.
- Values above 300 sfu are burst-inflated single-day spikes that are not a
  valid daily EUV proxy; they're set to NaN and time-interpolated over,
  rather than clipped to 300.
- PyIRI (the ionospheric climatology model used in Stage 4) requires F10.7 as
  an external input — it ships no index of its own, so this pull exists
  purely to feed it.
- Output: `data/raw/f107/f107_daily.parquet` (date-indexed, column
  `f107_obs`).

### `geomag_pull.py`

Pulls Kp index (planetary geomagnetic activity, 3-hour cadence) from the same
CelesTrak file, fields 5-12 of the OBSERVED block (`Kp*10`, integer-encoded).

Produces two outputs from the same parse:
- `data/raw/geomag/kp_daily.parquet` — one row per day: `kp_max` (the day's
  peak Kp, used for storm-day selection) and `quiet_all_day` (True if Kp < 1
  for all eight 3-hour windows that day).
- `data/raw/geomag/kp_3hourly.parquet` — one row per 3-hour window: `kp`.
  This is the causal (forward-fillable) series that Stage 4 aligns onto the
  TEC timestamps as a model input driver.

**Note on redundancy:** the original design doc (`data_pull_design.md`)
recommended sourcing Kp/Dst/F10.7 from OMNI2 to avoid scattering across
sources. In practice, Kp and F10.7 are both pulled from CelesTrak instead,
and OMNI2 (downloaded by `data_time.py`, see below) is never parsed or
consumed by anything downstream — it sits on disk unused. Dst is not
currently pulled from anywhere.

---

## Stage 1: Raw data extraction — `data_time.py`

This is the implementation of `data_pull_design.md`'s "Stage 1" contract:
get authentic bytes onto disk, verbatim, with an honest manifest of what
succeeded/failed. It does **not** decompress or parse anything.

Run as:
```
python data_pull/data_time.py            # full pull, all three sources
python data_pull/data_time.py --verify   # coverage report only, no downloads
python data_pull/data_time.py --ionex-only
python data_pull/data_time.py --start-date 2015-01-01 --end-date 2015-12-31
```

### What it pulls

| Source | Content | Auth | Iteration unit | Local path |
|---|---|---|---|---|
| CDDIS IONEX | Global TEC maps, 1 file/day | NASA Earthdata Login via `.netrc` | per (year, day-of-year) | `data/raw/ionex/{YYYY}/{DDD}/{filename}` |
| SPDF OMNI HRO | 5-minute IMF + solar wind | anonymous | per year | `data/raw/omni_hro/{YYYY}/omni_5min{YYYY}.asc` |
| SPDF OMNI2 | hourly F10.7/Kp/Dst (bundled) | anonymous | per year | `data/raw/omni2/{YYYY}/omni2_{YYYY}.dat` |

The design doc calls for 1-minute HRO; the implementation actually pulls the
5-minute product (`omni_5min{year}.asc`, 49 fields including the three
GOES proton-flux fields the 1-minute product lacks). This is consistent
throughout — the parser in Stage 2 is written specifically for the 49-field
5-minute layout.

### The IONEX naming problem

IGS renamed its TEC map products on 2023-08-07 (day-of-year 219). Before
that date, files use a short legacy name (`codg0840.15i.Z`, LZW-`.Z`
compressed); on/after, a long name
(`COD0OPSFIN_20232190000_01D_01H_GIM.INX.gz`, gzip-compressed).

`_ionex_targets()` doesn't just pick a name based on the date boundary — it
returns **both** candidate names in preferred order (legacy-first if the date
predates the boundary, long-name-first otherwise) and `_pull_ionex()` tries
them in sequence, falling through to the second name only if the first isn't
a clean 404. This is more defensive than the design doc's plan of picking one
name deterministically per date, and absorbs boundary edge cases without
needing exact confirmation of the long-name tokens (`OPS`/`FIN`/`01D_01H`)
against a live directory listing first.

### Authenticated download mechanics

CDDIS responds to an unauthenticated request with a redirect chain through
Earthdata Login (URS). `_make_session()` builds a `requests.Session` with
`trust_env = True` so it reads credentials from `~/.netrc` and carries
cookies through the redirect chain automatically.

`_download()` enforces, before ever writing the final filename:
- **Retries with exponential backoff** on timeouts, connection errors, and
  5xx — but never on a 404 (that's a real data gap, not a transient fault).
- **Integrity gate**: rejects anything under 2048 bytes, and rejects
  anything whose leading bytes look like an HTML page (`<!DOCTYPE`,
  `<html>`) — the telltale sign of a saved login page instead of real data.
- **Atomic write**: downloads to `dest.part`, then `os.replace()`s into the
  final path only after the integrity gate passes. An interrupted run can
  never leave a half-written file masquerading as complete.

Before running the real loop, `_smoke_test()` downloads one known-good IONEX
day (2010-01-01) and hard-exits with a `.netrc`-pointing error message if it
fails — this turns a broken credential setup into an instant, legible error
instead of thousands of silently-failed downloads.

### Manifest and resumability

Each source gets its own append-only CSV under `data/manifests/`
(`ionex_manifest.csv`, `omni_hro_manifest.csv`, `omni2_manifest.csv`) with
columns `source, key, expected_filename, status, reason, n_bytes,
checked_at`. `key` is `YYYY-DDD` for IONEX, `YYYY` for the OMNI sources.

Re-running the script is safe and cheap:
- A key already marked `present`/`downloaded` **and** whose file still
  exists on disk is skipped outright.
- A key previously `failed` with reason `404` is skipped too — a confirmed
  archive gap is not re-hammered on every run.
- Anything else (timeout, auth, bad_content, 5xx) is retried on the next
  run, since those are transient-failure classes, not confirmed gaps.

If an authenticated IONEX request fails with `auth` (401/403 or an HTML
page) mid-run, the whole run aborts immediately rather than logging
thousands of auth failures across the remaining date range — a broken
credential is a setup problem, not a per-day gap.

### Coverage report

`--verify` (or the end of a full run) prints, per source, how many
present/downloaded units exist against the expected total, how many are
confirmed 404 gaps, and how many are "other failures" that need attention
(with the first 5 keys listed). This report is the thing Stage 4's window
builder implicitly trusts: a window is only built from timestamps backed by
real, on-disk data.

---

## Stage 2: Parsing — `data_transformation.py`

Turns the raw, still-compressed Stage 1 bytes into in-memory arrays. Nothing
here touches disk except reading the input file.

### `read_decompress(file_path)`

Branches on file extension, since IONEX and OMNI use genuinely different
compression:
- `.Z` (pre-2023 IONEX) — Unix LZW "compress" format. Standard `gzip`
  library cannot read this; the `unlzw3` package handles it.
- `.gz` (post-2023 IONEX) — ordinary gzip.
- `.asc` (OMNI HRO) — already plain text, just decoded.

Returns an in-memory `io.StringIO` text stream in all three cases, so the
parsers below don't need to know which compression was used.

### `parse_ionex(stream)`

IONEX v1.0 is a fixed-column text format: a header block defines the grid
(`LAT1/LAT2/DLAT`, `LON1/LON2/DLON`, `EXPONENT` — a power-of-ten scale
factor for the integer TEC values), followed by a sequence of per-epoch TEC
map blocks.

Parsing is a small state machine keyed on the column-61-80 label of each
line: `START OF TEC MAP` resets a fresh `[n_lat, n_lon]` grid filled with
NaN, `EPOCH OF CURRENT MAP` gives that map's timestamp, `LAT/LON1/LON2/DLON/H`
starts a new latitude row, and unlabeled lines are TEC value lines belonging
to the current row. A fill value of `9999` becomes NaN *before* the exponent
scaling is applied. `START OF RMS MAP` ends the loop early — RMS maps
(uncertainty estimates) are present in the file but never parsed, since
nothing downstream uses them.

One subtlety flagged directly in the code: value lines fill the whole
80-column width, so they must be split on the full line, not the `[:60]`
slice used for header/label fields — a value near column 60 would otherwise
be silently truncated.

Returns `(maps, lats, lons)` where `maps` is a list of
`(datetime, ndarray[71, 73])` in TECU, and `lats`/`lons` are the *native*
IONEX grid vectors (2.5° lat x 5° lon spacing, 71x73 = the standard IONEX
resolution — not yet the model's target grid).

### `parse_omni_hro(stream)`

Parses the 49-field 5-minute OMNI HRO record layout (`_OMNI_COLUMNS`). It
first sanity-checks the field count of the first line against the expected
49 — if some other OMNI product (e.g. the 1-minute file, which has a
different field count) ever ends up in this path, it fails loudly instead of
silently misaligning every column.

Of the 49 parsed fields, only five become model driver channels
(`_OMNI_KEEP`): `b_magnitude`, `by_gsm`, `bz_gsm`, `flow_speed`,
`proton_density`. Each has a known fill-value sentinel (e.g. `9999.99` for
the magnetic-field fields) that gets masked to NaN by a `>=` threshold
comparison rather than `==`, since exact float equality against a sentinel
is fragile — real physical values never approach these magnitudes anyway.

Timestamp reconstruction: OMNI records store `year` + `day` (day-of-year,
not month/day) + `hour` + `minute` separately; these are combined via
pandas' `%Y%j` (year + day-of-year) format plus explicit hour/minute
timedeltas.

Returns a timestamp-indexed `DataFrame` with just the five kept columns.

---

## Stage 3: Grid interpolation — `data_interpolation.py`

IONEX's native grid (71 lat x 73 lon, 2.5°x5° equiangular) is not what the
model consumes. The model (an SFNO — spherical Fourier neural operator)
needs values on a **Gauss-Legendre grid**: 23 latitudes at the GL quadrature
nodes (never exactly at the poles) x 45 equiangular longitudes, corresponding
to spherical-harmonic bandlimit `Lmax = 22`.

### `target_grid()`

Computes the 23 GL latitudes from `numpy.polynomial.legendre.leggauss(23)`
(the roots are cos(colatitude); converted to degrees latitude, descending
north to south) and 45 equiangular longitudes on `[0, 360)`.

### `interpolate_map(...)`

For one native `[71, 73]` TEC grid:
1. Converts native longitudes from IONEX's `-180..180` convention to
   `0..360` to match the target convention, re-sorting columns to match.
2. **Wrap-pads across the 0°/360° seam** — appends the last source column
   (shifted -360°) before the first, and the first column (shifted +360°)
   after the last — so `RegularGridInterpolator` sees longitude as
   continuous rather than having a false edge at the date line.
3. Re-sorts latitudes ascending (IONEX gives them descending, 87.5°→-87.5°;
   `RegularGridInterpolator` requires ascending).
4. Interpolates onto the 23x45 target grid via `RegularGridInterpolator`
   with `bounds_error=False`.
5. **Pole handling**: IONEX has no data beyond ±87.5°, so any target GL
   latitude closer to a pole than that is out of the native grid's range.
   Rather than extrapolate numerically, those rows are filled with the
   simple mean of the nearest native edge ring (the ±87.5° ring) — an
   explicit, documented approximation, not silent extrapolation.

### `interpolate_to_gl(maps, ...)` and `build_interpolated(...)`

`interpolate_to_gl` applies `interpolate_map` across every parsed map for a
day and stacks the results with their timestamps. `build_interpolated` is
the batch driver: it reads the Stage 1 IONEX manifest for `present`/
`downloaded` days, groups them by year, and writes one `.npz` per year to
`data/interpolated_gl23x45/{year}.npz` (`tec [N,23,45]`, `timestamps [N]`).
It's resumable — a year whose output file already exists is skipped unless
`--overwrite` is passed — and writes the shared `grid.npz` (target lats/lons)
exactly once.

```
python data_pull/data_interpolation.py --build              # all years
python data_pull/data_interpolation.py --build --year 2015  # one year
```

Run with no `--build` flag, it instead runs `smoke_test()`, a manual
sanity check against one known file (2010-01-01) that prints the target
grid, TEC value range, and checks that the 0°/360° seam and both pole rows
look sane (constant across longitude, as they should for an averaged pole
value).

---

## Stage 4: Baseline, alignment, and windowing — `data_for_falisha.py`

This is where the interpolated TEC becomes an actual training sample: a
climatology baseline is subtracted to produce a residual field, driver
channels are time-aligned to the TEC timestamps, and everything is cut into
fixed-length windows and normalized. Four subcommands, meant to be run in
order:

```
python data_pull/data_for_falisha.py iri-cache  [--year Y] [--overwrite]
python data_pull/data_for_falisha.py dtec-cache [--year Y] [--overwrite]
python data_pull/data_for_falisha.py omni-cache [--year Y] [--overwrite]
python data_pull/data_for_falisha.py windows    [--train-end-year Y] [--val-end-year Y] [--overwrite]
```

### Why a residual at all

Raw TEC has a huge, predictable diurnal/seasonal/solar-cycle baseline that
would dominate a naive regression. The model instead predicts:

```
dTEC = IONEX vTEC (on the GL23x45 grid) - IRI vTEC (on the GL23x45 grid)
```

where IRI (International Reference Ionosphere, via the `PyIRI` package) is a
standard empirical climatology model, not a data source — it's computed
on-the-fly from date, time-of-day, F10.7, and geographic position. IONEX
integrates TEC up to GPS orbit altitude (~20,000 km); PyIRI here is only
integrated to 2000 km (`AALT = arange(80, 2001, 20)` km), so there's a known
systematic offset from the missing plasmaspheric content above 2000 km. This
offset is *intentionally not corrected* in the baseline dataset — it's
expected to be mostly zonal (longitude-independent) and representable by the
SFNO's own m=0 spherical harmonic modes, with a learned correction possible
later. (This exact question — is the offset flat or latitude-structured — is
what `offset.py`, described below, was written to check empirically.)

### `iri-cache`: `build_iri_cache()`

For each year of interpolated IONEX data, groups that year's timestamps by
calendar date, and for each date calls `PyIRI.main_library.IRI_density_1day`
once for all of that date's UT hours together (not once per timestamp) —
batching the relatively expensive IRI call per day rather than per 5-minute
sample. `edp_to_vtec` integrates the returned electron density profile into
vertical TEC. F10.7 is looked up per-date from Stage 0's daily F10.7 table;
if a specific date is missing, it falls back to the previous day's value
(logged) rather than failing the whole year.

Output: `data/iri_gl23x45/{year}.npz` — `iri [N,23,45]`, `timestamps [N]`
matching the interpolated-IONEX timestamps exactly, plus the grid, `lmax`,
altitude array, and a `tec_definition` string documenting the 80-2000km
integration range.

### `dtec-cache`: `build_dtec_cache()`

Straightforward subtraction: loads the matching IONEX and IRI `.npz` for a
year, **asserts the timestamp arrays are identical** (hard failure if not —
this is the point where a silent misalignment would otherwise corrupt every
downstream sample), and writes `dtec = ionex - iri`.

Output: `data/dtec_gl23x45/{year}.npz` — `dtec [N,23,45]`, `timestamps [N]`,
plus a `residual_definition` string baked into the file for provenance.

### `omni-cache`: `build_omni_cache()`

Aligns the two driver sources — 5-minute OMNI HRO and 3-hourly Kp — onto the
exact dTEC timestamps for a year, using two different interpolation
strategies appropriate to each source's physical update cadence:

- **OMNI HRO** (`align_omni_to_timestamps`): time-interpolated
  (`method="time"`, `limit_direction="both"`) onto the union of its own
  index and the target timestamps, then reindexed down to just the targets.
  This is a continuous physical quantity sampled every 5 minutes, so
  interpolating between adjacent real samples is legitimate.
- **Kp** (`align_kp_to_timestamps`): **forward-filled**, not interpolated —
  Kp is a step function that's only known 3-hourly and stays constant within
  each 3-hour window (it isn't smoothly continuous, and forward-fill is also
  the causally-correct choice: at any point in time you only know the
  most-recently-completed 3-hour Kp value, not one interpolated from the
  future).

Both alignment functions raise loudly if any timestamp ends up with a NaN
after alignment, or if any target timestamp fails to map into the aligned
index — a silent gap here would otherwise produce a driver row of NaNs that
downstream code might not catch.

The two aligned arrays are concatenated into one `[N, 6]` driver array
(5 OMNI HRO channels + 1 Kp channel — `DRIVER_FEATURES` = `b_magnitude,
by_gsm, bz_gsm, flow_speed, proton_density, kp_3hour`).

Output: `data/omni_aligned_gl23x45/{year}.npz` — `omni [N,6]`,
`timestamps [N]`, `features` (the column-name array), `source` (a
provenance string).

### `windows`: `build_windowed_dataset()`

This is the final assembly step, and the only one that operates across all
years at once rather than per-year.

1. **`load_aligned_series`**: concatenates every year's dTEC + aligned-OMNI
   `.npz` (skipping years missing an OMNI-aligned counterpart), sorts by
   timestamp, and **drops duplicate timestamps** if any exist across
   year-boundary concatenation.
2. **`valid_window_starts`**: a window is `INPUT_STEPS=6` history steps +
   `TARGET_STEPS=3` forecast steps = 9 consecutive frames. A candidate start
   index is only kept if:
   - all 9 timestamps are **strictly increasing with identical adjacent
     spacing** (i.e. no gap in the middle of the 9-frame window — a window
     spanning a data outage would otherwise silently mix two different
     cadences or jump across missing days), and
   - every one of the 9 frames is fully finite in both dTEC and all 6
     driver channels (no NaN anywhere in the window).

   The finite-check uses a cumulative-sum "prefix sum" trick
   (`finite_prefix`) to test "are all 9 consecutive frames finite" for every
   candidate start in one vectorized pass, rather than a per-window Python
   loop over a plausibly-large number of candidate windows.
3. **`split_window_starts`**: splits by the **calendar year of each window's
   start timestamp** — `train_end_year` (default 2019) and `val_end_year`
   (default 2022) are inclusive boundaries; everything after `val_end_year`
   is test. This is a strict chronological split (no shuffling across the
   boundary), appropriate for a forecasting model where leaking future
   information into training would be a genuine evaluation bug.
4. **`window_stats`**: computes mean/std for dTEC and each of the 6 driver
   channels **from the train split's input frames only** (never touching
   val/test, and never touching the target frames — the model's targets are
   normalized using the same stats as its inputs, but the stats themselves
   are computed only from what the model would see as input during
   training). Processed in chunks (`chunk_size`, default 512) to bound peak
   memory rather than materializing every train window at once.
5. **`write_split_windows`**: writes each split's `tec_input`,
   `omni_input`, and `target` directly to disk via
   `np.lib.format.open_memmap` (a writable memory-mapped `.npy`) rather than
   building the full array in RAM first — necessary given the train split
   alone is ~110k windows of `[6,23,45]` float32 tensors. Both dTEC and
   target are normalized with the *same* `tec_mean`/`tec_std` (target is
   just a later slice of the same physical field).

Output layout: `data/falisha_windows_gl23x45/` —
`{split}_tec_input.npy [N,6,23,45]`, `{split}_omni_input.npy [N,6,6]`,
`{split}_target.npy [N,3,23,45]`, `{split}_window_start_times.npy [N]` for
each of `train`/`val`/`test`, plus shared `lats.npy`, `lons.npy`, and a
`metadata.json` recording the grid contract, normalization stats, split
sizes, and the residual/window-rule definitions in plain text for anyone
consuming the dataset without reading this code.

(A prior handoff of this dataset recorded split sizes of 110,124 /
25,406 / 26,270 windows for train/val/test — useful as a sanity-check
magnitude if you rebuild and want to confirm nothing silently changed.)

---

## Stage 5: PyTorch Dataset — `falisha_dataset.py`

A thin, memory-mapped `torch.utils.data.Dataset` (`FalishaDTECDataset`) over
Stage 4's output directory. All four `.npy` arrays are opened with
`mmap_mode="r"` — nothing is loaded into RAM until an item is actually
indexed — and `_validate_shapes()` runs once at construction time to fail
fast if the on-disk shapes don't match what `metadata.json` claims (sample
count consistency across all four arrays, and the hardcoded `(6,23,45)` /
`(6, n_driver_features)` / `(3,23,45)` shape contract).

`__getitem__` copies just the one indexed slice out of the memmap into a
real `torch.Tensor` (`np.array(..., copy=True)` — necessary since a torch
tensor can't be built directly from a read-only memmap view without an
explicit copy). Returns a dict: `tec_input`, `omni_input`, `target` (all
float32), `timestamp` (int64 epoch seconds).

`make_falisha_dataloader()` wraps this in a `DataLoader`, defaulting
`shuffle=True` only for the `train` split.

Run directly (`python data_pull/falisha_dataset.py`), it loads the train
split and prints one sample's shapes as a smoke test.

---

## One-off analysis and export scripts

These build on Stages 1-3 but are not part of the main model-training
pipeline — they were written to answer specific analysis questions for
other people on the project, and are not resumable/production-hardened the
way Stages 1-4 are.

### `offset.py` — is the IONEX/IRI offset flat or structured?

Directly motivates the design decision in Stage 4 to leave the
plasmaspheric IONEX-minus-IRI offset uncorrected. For three sample days
(solar min quiet, solar max moderate, a storm day), it interpolates one
midday IONEX map to the GL grid, computes IRI on the same grid, and reports
the offset's spatial mean, spatial standard deviation, and a low-latitude
(|lat|<20°) vs. high-latitude (|lat|>60°) comparison. A small spatial std
and a small low/high-latitude gap would support "flat constant, safe to
ignore"; a large gap would mean the plasmaspheric content is
latitude-structured and should be flagged rather than absorbed silently by
normalization.

### `export_dtec_sample.py` and `sakshum.py` — samples for a teammate's L_max analysis

Both scripts export stratified TEC samples for what the docstrings call
"Saksham's" / "Person 2's" spectral (`L_max`) analysis — the question of
how many spherical-harmonic degrees are actually needed to represent the
TEC field, which is a separate research question from model training. They
deliberately export on the **native 71x73 grid**, not the 23x45 GL grid,
because the GL grid bandlimits to degree 15 and would destroy exactly the
higher-degree content the analysis is trying to measure.

- `export_dtec_sample.py`: stratifies by season x solar-cycle-phase
  (solar max years 2012-2014, solar min years 2008-2009 + 2019-2020) with
  guaranteed storm (`kp_max>=5`) and true-quiet (`Kp<1` all day) buckets on
  top of ~20-per-stratum random fill. Output:
  `data/person2_dtec_sample.npz` (`dtec`, `tec_ionex`, `tec_iri`,
  `timestamps`, `kp_max`, `lats`, `lons`).
- `sakshum.py`: a differently-scoped version of the same idea — spreads
  ~100-200 samples across **deciles of daily minimum SYM-H** (a
  geomagnetic-activity index derived directly from the OMNI HRO files, not
  Kp) over 2003-2018, with an explicit flagged assumption about whether
  "low to high activity" means a continuous spread or two discrete
  storm/quiet clusters. Output: `data/person2_lmax_sample.npz` (`tec`,
  `timestamps`, `day_min_symh`, `lats`, `lons`). Also renders a two-panel
  storm-vs-quiet TEC map comparison via matplotlib as a spot-check.

### `test.py` and `yeet.py` — scratch/dev utilities

- `test.py`: a minimal manual smoke test of the CDDIS authenticated
  download path outside the main pipeline — fetches one known IONEX URL
  with a `.netrc`-backed session and prints the redirect chain and first
  200 response bytes, useful for debugging auth issues in isolation from
  the full `data_time.py` run.
- `yeet.py`: an ad hoc timing script (not meant to be imported) that
  measures how long parsing + interpolating one full year (2015) of IONEX
  data takes end-to-end, reading directly from the manifest.

---

## On-disk layout (cumulative, after all stages)

```
data/
  raw/
    ionex/{YYYY}/{DDD}/{filename}          # Stage 1, compressed, verbatim
    omni_hro/{YYYY}/omni_5min{YYYY}.asc    # Stage 1
    omni2/{YYYY}/omni2_{YYYY}.dat          # Stage 1, downloaded but currently unused downstream
    f107/f107_daily.parquet                # Stage 0
    geomag/kp_daily.parquet                # Stage 0
    geomag/kp_3hourly.parquet              # Stage 0
  manifests/
    ionex_manifest.csv
    omni_hro_manifest.csv
    omni2_manifest.csv
  interpolated_gl23x45/{year}.npz          # Stage 3, + grid.npz
  iri_gl23x45/{year}.npz                   # Stage 4 (iri-cache)
  dtec_gl23x45/{year}.npz                  # Stage 4 (dtec-cache)
  omni_aligned_gl23x45/{year}.npz          # Stage 4 (omni-cache)
  falisha_windows_gl23x45/                 # Stage 4 (windows) — final model input
    {split}_tec_input.npy   [N,6,23,45]
    {split}_omni_input.npy  [N,6,6]
    {split}_target.npy      [N,3,23,45]
    {split}_window_start_times.npy [N]
    lats.npy  lons.npy  metadata.json
  person2_dtec_sample.npz                  # one-off, export_dtec_sample.py
  person2_lmax_sample.npz                  # one-off, sakshum.py
```

## Known gaps / things worth resolving

- **OMNI2 is downloaded but dead code downstream.** `data_time.py` pulls it
  every run; nothing in `data_transformation.py` or `data_for_falisha.py`
  parses or consumes it. Kp comes from CelesTrak (`geomag_pull.py`) and
  F10.7 from CelesTrak (`pull_f107`) instead. If OMNI2 truly isn't needed,
  dropping the pull would save bandwidth and manifest bookkeeping; if Dst is
  eventually wanted as a driver, OMNI2 is still the planned source for it
  and a parser (`parse_omni2`) has not yet been written.
- **`.netrc` in this folder.** `data_pull/.netrc` exists on disk (mode
  should be 600) and holds live Earthdata Login credentials — it is not
  documented further here since credential contents shouldn't be echoed
  into project docs. Confirm it's covered by `.gitignore` (the repo's
  recent "Ignore .netrc" commit suggests this was already addressed).
