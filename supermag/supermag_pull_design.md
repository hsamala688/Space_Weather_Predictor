# SuperMAG Long-Range Data Pull — Design Doc

**Status:** spec for implementation
**Audience:** Claude Code (implementer)
**Goal:** Build a script that pulls multi-year (up to ~25-year) SuperMAG 1-minute data via the official Python web-service client, in a way that survives the constraints of a large pull: limited RAM, server limits, and inevitable mid-run failures.

---

## 1. Context

This script is the data-ingestion layer for a space-weather analysis project. Downstream, the pulled data is queried via a text-to-SQL agent (over DuckDB) and feeds a drift-monitoring component, so the priorities are: complete coverage, clean tabular output, and a format DuckDB can read out-of-core. The script itself only handles **fetching and persisting**; analysis lives elsewhere.

The data source is the SuperMAG web service (JHU/APL), accessed through the provided `supermag-api.py` client. That client exposes three core functions:

- `SuperMAGGetInventory(logon, start, extent)` — list of stations reporting in a window.
- `SuperMAGGetData(logon, start, extent, flagstring, station)` — magnetometer field data for one station.
- `SuperMAGGetIndices(logon, start, extent, flagstring)` — geomagnetic indices + solar-wind / IMF data.

Both data functions return a pandas DataFrame by default, or a list of dicts with `FORMAT='list'`.

### Key facts that shape the design

- **The web API serves 1-minute cadence data only.** (1-second data exists but is bulk-file/NetCDF only — out of scope here.)
- 1-minute cadence = **one row per minute**: 1,440 rows/day, ~526k rows/year, **~13.1M rows over 25 years** (per station for `data`; per timestamp for `indices`).
- `extent` is a **time span in seconds** (86,400 = one day). 25 years ≈ 789M seconds. The client formats it as a 12-digit field, so even multi-year extents fit syntactically — but the server will not actually serve years of minute data in one request.
- Returned columns include **nested values** that must be flattened before writing to Parquet (see §6).

---

## 2. Goals

1. Pull a configurable date range (by year) of `indices` and/or per-station `data`.
2. Chunk the pull into **monthly** requests by default.
3. Persist **each chunk to Parquet immediately** — never accumulate the full dataset in memory.
4. Be **resumable**: a re-run skips chunks already on disk and retries previously failed ones.
5. Be **polite to the server**: rate-limit between requests, retry with backoff on errors.
6. Stay within an **8 GB RAM** budget (process one chunk at a time).
7. Optionally register the output as a **DuckDB** view for out-of-core querying.

## 3. Non-goals

- 1-second data (different source path: bulk NetCDF files).
- Real-time / streaming ingestion.
- Redistributing raw SuperMAG data (see §10 — fair use).

---

## 4. Constraints & rationale

| Constraint | Why | Design response |
|---|---|---|
| 8 GB RAM | 13.1M rows × many columns will not fit as one DataFrame | Fetch + write one month at a time; nothing global held in memory |
| Server limits | No published hard `extent` cap, but large single requests time out / get rejected | Monthly chunks (~43k rows each) are safely small |
| Output format | CSV flattens nested dicts into strings and loses structured access | **Parquet** (typed, columnar, DuckDB-friendly) |
| Failures over a long run | ~300 monthly requests for 25 years — something will fail | Retry + backoff per chunk; resumable; failures logged, run continues |
| Fair use | SuperMAG data cannot be redistributed | Commit code, not data; add data dirs to `.gitignore` |

---

## 5. Inputs / configuration

| Param | Type | Notes |
|---|---|---|
| `logon` | str | SuperMAG username. **Read from env var (e.g. `SUPERMAG_LOGON`) or CLI arg — never hardcode.** |
| `start_year` | int | inclusive |
| `end_year` | int | exclusive (range is `[start_year, end_year)`) |
| `dataset` | enum | `indices`, `data`, or both |
| `stations` | list[str] | required when `dataset` includes `data` |
| `flags` | str | flagstring passed through to the client (e.g. `'all,swiall,imfall'` for indices, `'all'` for data) |
| `outdir` | path | root output directory |
| `chunk` | enum | `monthly` (default). Allow `daily` as a fallback for tighter memory. |
| `rate_limit_seconds` | float | sleep between requests (default ~1.0) |
| `max_retries` | int | per-chunk retry count (default 4) |

---

## 6. Behavior / flow

1. Resolve config; create `outdir` and subdirectories.
2. Generate the ordered list of chunk windows (month boundaries) across `[start_year, end_year)`.
3. For each chunk:
   1. Compute the chunk's output filepath.
   2. **If the file exists → skip** (resumable).
   3. Compute `extent` = seconds in the chunk window.
   4. Call the relevant SuperMAG function, wrapped in retry/backoff (§8).
   5. On success **with rows**: flatten nested fields (§7), add a `datetime` column, write Parquet.
   6. On success **with no rows**: write nothing, note in manifest (legitimate data gap).
   7. On failure after retries: record in the failures manifest, **continue** (do not abort the run).
   8. Sleep `rate_limit_seconds`.
4. Write/update the run manifest.
5. (Optional) Register the Parquet output as a DuckDB view (§9).

For `data` with multiple stations, loop stations inside each chunk (or chunks inside each station — implementer's choice, but keep one station+month per Parquet file).

---

## 7. Output layout & schema

### Directory layout

```
outdir/
  indices/
    indices_YYYYMM.parquet
  data/
    {STATION}/
      data_{STATION}_YYYYMM.parquet
  _manifest.json
```

### Flattening rules (applied before writing)

- `tval` (Unix epoch seconds) → keep as `tval` **and** add `datetime` = `pd.to_datetime(tval, unit='s', utc=True)`.
- `N` / `E` / `Z` are dicts `{nez, geo}` → split into `N_nez`, `N_geo`, `E_nez`, `E_geo`, `Z_nez`, `Z_geo`.
- Vector fields (`bgse`, `bgsm`, `vgse`, `vgsm`) are dicts `{X, Y, Z}` → split into `bgse_x`, `bgse_y`, `bgse_z`, etc.
- 24-element regional arrays (`SMLr`, `SMUr`, `SMEr`, and their `*mlat`/`*mlt`/`*glat`/`*glon`/`*stid` variants) → **v1 decision: keep as a Parquet list column, or omit if not needed downstream.** Do not silently stringify them.

### Example `data` schema (after flattening)

`tval` (int64), `datetime` (timestamp, UTC), `iaga` (str), `N_nez`, `N_geo`, `E_nez`, `E_geo`, `Z_nez`, `Z_geo` (float). Plus optional `glon`, `glat`, `mlt`, `mcolat`, `decl`, `sza` when those flags are set.

---

## 8. Error handling

- Wrap each fetch in `try/except` with **exponential backoff** (e.g. 1s, 2s, 4s, 8s) up to `max_retries`.
- The client returns `(status, payload)` — treat **`status == 0` as a failure**, not just exceptions.
- Network errors (`URLError`, timeouts) → retry.
- Persistent failure after retries → append to the failures list in the manifest and move on.
- A single bad chunk must never kill the whole run.

---

## 9. Resumability & manifest

- **Presence of a chunk's Parquet file = that chunk is done.** Skip on re-run.
- `_manifest.json` tracks: config used, run timestamps, chunks completed, chunks empty (legitimate gaps), and chunks failed (so a later run can re-attempt only those).

---

## 10. DuckDB integration (optional)

DuckDB reads Parquet out-of-core, so the full ~13.1M-row table can be queried without loading it into RAM — directly serving the 8 GB constraint and the downstream text-to-SQL layer.

```sql
CREATE VIEW indices AS
  SELECT * FROM read_parquet('outdir/indices/*.parquet');

CREATE VIEW magdata AS
  SELECT * FROM read_parquet('outdir/data/*/*.parquet');
```

---

## 11. Dependencies

- The provided `supermag-api.py` client.
- `pandas`, `pyarrow` (Parquet engine), `python-dateutil` (month arithmetic).
- `duckdb` (optional, §10).
- `certifi` (optional — required for SSL on some networks; the client already handles it if present).

---

## 12. Legal / acknowledgement

SuperMAG data are provided under **fair use and cannot be redistributed**. Include the required acknowledgement per SuperMAG's rules of the road in any publication or presentation, and **`.gitignore` the data output directories** so raw data never lands in version control. Commit the script, not the pull.

---

## 13. Open decisions for the implementer

- Handling of the 24-element regional index arrays: Parquet list column vs. omit in v1.
- Whether `indices` and `data` run in one invocation or separately.
- Whether to expose `daily` chunking as a CLI flag from the start or add later.
