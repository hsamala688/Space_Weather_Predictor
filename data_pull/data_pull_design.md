# data_pull_design.md — Stage 1: Raw Data Extraction

Target: Claude Code implementation. This document specifies **Stage 1 only** — getting authentic raw files onto disk over a multi-year span, with resumability and an honest gap record. Parsing, decompression, interpolation, NaN handling, normalization, and sample assembly are **out of scope** (Stages 2+). Stage 1's only contract is: every authentic file is on disk, or it is recorded in the failure manifest with a reason.

## Operating principles for this work

1. **Think before coding.** Confirm the open decisions below before writing the loop. Do not silently pick an analysis center or date range.
2. **Simplicity first.** Standard library plus `requests`. No framework, no abstractions for a single pull. Three near-identical loops, not a generalized "download engine."
3. **Surgical scope.** Download and record only. Do not decompress, parse, or transform. Store bytes verbatim.
4. **Goal-driven.** Success criteria and verification steps are defined at the end. Loop until they pass.

## Decisions to confirm before coding

- **Analysis center for training truth.** Recommend `COD` (CODE) for its hourly cadence, or the IGS combined product. Pick one and write it to config; changing later means re-pulling everything.
- **Date range.** The training/val/test span (e.g. one full solar cycle). Defines the loop bounds.
- **Index source.** F10.7, Kp, and Dst are coarser than 1-minute and are not in the HRO 1-minute OMNI product. Recommend pulling them from **OMNI2 (low-res hourly OMNI)**, which bundles all three in one source, rather than scattering across GFZ (Kp), WDC Kyoto (Dst), and Penticton (F10.7). Confirm.
- **Which drivers are mandatory.** Stage 1 should acquire all candidate drivers (1-min IMF/solar wind + the three indices). Exactly which become model channels is a downstream decision and does not block Stage 1.

## The three pulls

| Source | Content | Access | Iteration unit |
|---|---|---|---|
| CDDIS IONEX | TEC ground-truth maps (one file/day) | Earthdata Login (authenticated) | per (year, day-of-year) |
| SPDF OMNI HRO | 1-minute IMF + solar wind | anonymous HTTPS | per year |
| SPDF OMNI2 | hourly F10.7, Kp, Dst | anonymous HTTPS | per year |

Note the asymmetry: **only CDDIS requires authentication.** The two OMNI pulls are anonymous and much simpler. Do not over-engineer them.

## One-time setup (document in README, do not code)

1. Register a free **NASA Earthdata Login** account at `urs.earthdata.nasa.gov`.
2. Create `~/.netrc` (mode 600) with:
   ```
   machine urs.earthdata.nasa.gov
   login YOUR_USERNAME
   password YOUR_PASSWORD
   ```
3. Dependencies: `requests`. Everything else is standard library.

## Component contracts

### 1. Config

A single dataclass or dict holding: `center` (e.g. "COD"), `start_date`, `end_date`, `data_root`, and the source base URLs. No magic strings scattered through the code.

### 2. IONEX URL + path builder (the tricky one)

The IGS renamed its products effective **2023 day-of-year 219 (2023-08-07)**. Files for observation dates before the boundary use the short legacy name; on or after, the long IGS name. A single function isolates this so the rest of the code is boundary-agnostic.

```python
from datetime import date

RENAME_BOUNDARY = date(2023, 8, 7)  # DOY 219, 2023

def build_ionex_target(obs_date: date, center: str) -> tuple[str, str]:
    """Return (relative_url, local_filename) for the given observation date.
    Handles the Aug-2023 IGS naming change."""
    doy = obs_date.timetuple().tm_yday
    yyyy = obs_date.year
    center_lc = center.lower()  # e.g. "cod" -> "codg"

    if obs_date < RENAME_BOUNDARY:
        # legacy: e.g. codg0840.15i.Z   (LZW-compressed .Z)
        yy = yyyy % 100
        fname = f"{center_lc}g{doy:03d}0.{yy:02d}i.Z"
    else:
        # long IGS name: e.g. COD0OPSFIN_20232190000_01D_01H_GIM.INX.gz
        # NOTE: the exact tokens (OPS/MGX, FIN/RAP, 01H/02H) MUST be verified
        # against a live CDDIS directory listing for the chosen center.
        fname = (f"{center.upper()}0OPSFIN_{yyyy}{doy:03d}0000"
                 f"_01D_01H_GIM.INX.gz")

    rel_url = f"{yyyy}/{doy:03d}/{fname}"
    return rel_url, fname
```

**Verification step required of the implementer:** before running the full pull, fetch one directory listing on each side of the boundary for the chosen center and confirm the exact long-name tokens. Getting `01H` vs `02H` or `OPS` vs `MGX` wrong produces silent 404s across the entire post-2023 span.

Also record the compression difference (legacy `.Z` is LZW; new `.gz` is gzip). Stage 1 stores either verbatim; Stage 2 decompresses.

### 3. Authenticated download

CDDIS answers an unauthenticated request with a **302 redirect to a `proxyauth` / URS URL**. A client that does not carry the login through the redirect receives what looks like a 404 or saves an HTML login page as if it were data. Use a persistent `requests.Session` that reads `.netrc` and keeps a cookie jar so the token survives across hundreds of requests.

```python
def make_session() -> requests.Session:
    s = requests.Session()
    s.trust_env = True            # read ~/.netrc
    return s                       # cookies persist on the session

def download(session, url, dest_path, *, min_bytes=2048, retries=3) -> Result:
    """Download to dest_path.part, verify, then atomic-rename to dest_path.
    Follow redirects (URS auth). Exponential backoff on transient errors.
    Returns a Result(status, reason, n_bytes)."""
```

Required behaviors:
- **Atomic write.** Download to `dest_path + ".part"`, verify, then `os.replace` to the final name. An interrupted download must never leave a file the resumability check mistakes for complete.
- **Follow redirects** with the session so URS auth completes.
- **Retry** transient failures (timeouts, 5xx) with exponential backoff. Do not retry a clean 404 (real gap).
- **Integrity gate before rename:** size >= `min_bytes`, and the leading bytes are not `<!DOCTYPE html` or `<html` (the tell of a saved login/error page). Fail the download if these don't hold.

### 4. Manifest

A CSV (or JSONL) per source, written incrementally. It drives resumability and becomes the **gap record** Stage 6 reads so a training window never spans a missing day unknowingly.

Schema (one row per iteration unit):

| column | meaning |
|---|---|
| `source` | ionex / omni_hro / omni2 |
| `key` | `YYYY-DDD` for IONEX, `YYYY` for OMNI |
| `expected_filename` | what the builder produced |
| `status` | `present` / `downloaded` / `failed` |
| `reason` | empty, or e.g. `404`, `auth`, `timeout`, `bad_content` |
| `n_bytes` | file size on disk |
| `checked_at` | ISO timestamp |

`read_manifest(path) -> {key: row}` and `upsert_manifest_row(...)`.

### 5. Per-source orchestrators

```python
def pull_ionex(cfg, session, manifest):
    for obs_date in daterange(cfg.start_date, cfg.end_date):
        key = f"{obs_date.year}-{obs_date.timetuple().tm_yday:03d}"
        if already_present(key, manifest): continue   # resumability
        rel_url, fname = build_ionex_target(obs_date, cfg.center)
        dest = ionex_path(cfg.data_root, obs_date, fname)
        result = download(session, cfg.ionex_base + rel_url, dest)
        upsert_manifest_row(manifest, source="ionex", key=key, ...)

def pull_omni_hro(cfg, manifest): ...   # anonymous, per year
def pull_omni2(cfg, manifest):  ...     # anonymous, per year
```

`already_present` skips a unit that is `present`/`downloaded` with a valid on-disk file. A prior `failed` with reason `404` is also skipped (real gap, do not hammer it); a `failed` with `timeout`/`auth` is retried on the next run.

## On-disk layout (the handoff contract to Stage 2)

```
{data_root}/
  raw/
    ionex/{YYYY}/{DDD}/{filename}
    omni_hro/{YYYY}/{omni_min file}
    omni2/{YYYY}/{omni2 file}
  manifests/
    ionex_manifest.csv
    omni_hro_manifest.csv
    omni2_manifest.csv
```

Mirror the archive structure exactly. Files are byte-for-byte as received, still compressed. Stage 2 reads this tree plus the manifests.

## Failure handling

- A clean 404 is **data, not an error**: record `failed/404` and continue. Space-weather feeds have genuine missing days.
- Auth failure (HTML login page, 401/403, or `auth` reason) on the *first* file is a setup problem, not a per-file gap: abort the run with a clear message pointing at `.netrc`, rather than logging thousands of `auth` failures.
- Never crash the loop on a single bad day.

## Success criteria and verification

Implement these as a checkable `verify` routine; the task is done when all pass.

1. **Auth smoke test.** Before the full run, download one known-good IONEX day. Assert the file is > `min_bytes` and not HTML. (Catches the redirect/auth problem in seconds instead of after thousands of saves.)
2. **Idempotent re-run.** Run the pull twice. The second run downloads nothing new: every unit is `present` or a known real gap. Status `downloaded` count on the second run is zero.
3. **Manifest matches disk.** Count of `present`/`downloaded` rows equals the number of real files on disk for that source.
4. **Random spot-check.** Sample N downloaded files; each is nonzero, exceeds `min_bytes`, does not begin with HTML, and (manual, once) decompresses cleanly.
5. **Coverage report.** Print expected vs present day counts per source and list the gap days. This report is the artifact Stage 2/Stage 6 consume.

## Explicitly out of scope (do not implement here)

Decompression, IONEX/OMNI parsing, fill-value handling, the 71×73 to 128×256 equiangular interpolation, temporal alignment, normalization, climatology, and sample windowing. Those are later stages. Stage 1 ends at "authentic bytes on disk plus an honest manifest."
