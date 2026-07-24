# TEC Maps — Data Upload & Management Roadmap

**Status:** living document · **Last updated:** 2026-07-24
**Owner:** ionocast frontend/data
**Scope:** how observed (and, later, predicted) Total Electron Content maps get
from raw data on disk to the three.js globe, and how that stays maintained over
time on **Cloudflare R2**.

---

## 1. Purpose

The globe needs to overlay a TEC field for a chosen day/hour, for two layers:

- **Actual TEC** — observed maps derived from IONEX (2023–2025 today, growing).
- **Predicted TEC** — forecast maps from the SFNO model (future work).

This document is the plan for **serving and managing that data**: the storage
model, the upload path, the operational lifecycle, and how the predicted layer
slots in without redesigning anything.

---

## 2. Guiding principles (the "why" behind every choice)

1. **Serverless-first.** Historical maps are static files. Static files on
   object storage + a CDN need *no running server* — nothing to deploy, scale,
   patch, or page you at 3am. We only introduce compute when a requirement
   (true real-time, auth, per-request logic) forces it.
2. **Never touch the training pipeline.** The map exporters read raw
   `data/bronze/` and write to a separate `data/viz/` tree. They import the
   pipeline's IONEX parser read-only. Deleting or re-running anything here can
   never change what the model pipeline produces.
3. **Client-side coloring.** We ship *raw values*, not pre-colored images, so
   the browser shader owns the colormap, scale, and opacity. Retuning the look
   costs a shader edit, not a re-export of hundreds of files.
4. **Comparable across time.** A fixed value scale (vmin/vmax) means a color
   means the same TEC on every day — essential once we diff predicted vs actual.
5. **Incremental & automatable.** Refreshing must touch only new data and run
   unattended on a schedule.

---

## 3. Current architecture (what exists today)

Three standalone scripts under `backend/scripts/`, each a stage:

```
data/bronze/ionex/<year>/<doy>/*.gz              raw IONEX (pipeline's extract stage)
        │  export_native_tec.py   parse → keep native 71×73 grid (no GL projection)
        ▼
data/viz/tec_native_71x73/<year>.npz             full-year cubes, float32, + grid.npz
        │  build_tec_blobs.py      slice per UT day → float16 → gzip
        ▼
data/viz/tec_blobs/<year>/<date>.f16.gz          ~1096 day blobs (~146 KB each) + meta.json
        │  upload_r2.py            boto3 → R2, sets Content-Encoding: gzip
        ▼
Cloudflare R2 bucket  →  public URL  →  frontend fetch()
```

### Data contract (as built)

| Property | Value |
|---|---|
| Grid | native IONEX **71 lat × 73 lon**, uniform 2.5°×5° |
| Latitude | 87.5° → −87.5° (north→south) |
| Longitude | −180° → 180° (−180 and +180 are the same meridian) |
| Cadence | hourly (24 maps/day) |
| Per-day blob | `gzip( float16[24, 71, 73] )`, little-endian |
| Units | TECU; `NaN` = missing hour/cell |
| Sidecar | `meta.json` (grid, lats/lons, dtype, suggested scale) |

### Why these specific choices

- **Native 71×73, not the model's 23×45.** The IONEX native grid is *uniform*
  (equirectangular-friendly) and higher-resolution. The model's Gauss-Legendre
  23×45 grid is *non-uniform* — worse for texturing. So the viz path
  deliberately diverges from the training grid.
- **float16 + gzip.** Halves bytes vs float32 (directly usable as a GPU
  half-float texture, no lossy step), then gzip compresses the smooth field
  ~1.7×. Net: 533 MB → 160 MB, ~146 KB/day. Fidelity loss vs float32 is
  ≤0.03 TECU — invisible.
- **Per-day blobs.** One request per selected day (~146 KB), and it carries all
  24 hours so the client can scrub/animate a day without refetching. (A single
  Zarr cube is the alternative once file count or live appends make this awkward
  — see Phase 5.)

---

## 4. Why Cloudflare R2

| Concern | R2 | Why it matters here |
|---|---|---|
| **Egress cost** | **free** | A public map viewer = many users × many day-fetches. Egress is the bill that grows; R2 removes it. |
| **CDN** | built-in | Day blobs cache at the edge; repeat views are instant. |
| **API** | S3-compatible | `upload_r2.py` uses boto3; `aws s3`/`rclone` also work. |
| **Free tier** | 10 GB | 160 MB today; years of headroom. |

The one nuance is **public access** (see §5): the S3 endpoint is for
authenticated uploads; browser reads go through a public URL.

---

## 5. Cloudflare R2 setup & access

### Bucket

- Create bucket `tec`. Mirror the local tree as keys: `tec/2023/2023-06-21.f16.gz`,
  plus `tec/meta.json`, `tec/manifest.json` (Phase 2).

### Upload (authenticated, S3 API)

`upload_r2.py` reads credentials from env and sets per-object headers
(`Content-Encoding: gzip` on blobs, `application/json` on JSON):

```bash
export R2_ENDPOINT="https://<account_id>.r2.cloudflarestorage.com"
export R2_ACCESS_KEY_ID=...  R2_SECRET_ACCESS_KEY=...  R2_BUCKET="tec"
python backend/scripts/upload_r2.py
```

### Public read — start free, upgrade later

1. **`r2.dev` URL (no domain, start here).** Enable public access → get
   `https://pub-<hash>.r2.dev/...`. Free, but **rate-limited** and not fully
   CDN-cached — fine for dev/demo.
2. **Custom domain (production).** Point e.g. `tec.ionocast.com` (any registrar,
   on Cloudflare DNS) at the bucket for full CDN + no throttle. **The only
   frontend change is the base-URL string** — nothing about blobs or scripts
   changes.

### CORS

The bucket must allow the frontend origin (`http://localhost:5173` in dev, the
deployed origin in prod) or the browser `fetch` is blocked.

---

## 6. Roadmap (phased)

### Phase 0 — DONE: one-shot export & manual upload
Native export, float16+gzip blobs, `meta.json`, boto3 uploader. Everything is
regenerable and isolated from the pipeline.

### Phase 1 — MVP: serve Actual TEC (2023–2025) on the globe
- Upload the blob tree to R2 (`r2.dev` URL).
- Frontend: on selected date + "Actual TEC" toggle, fetch
  `…/tec/<year>/<date>.f16.gz`, decode to a half-float `DataTexture`, overlay on
  the sphere, color in-shader with the fixed scale from `meta.json`.
- **Exit criteria:** picking a date shows that day's observed TEC on the globe.

### Phase 2 — Management: make it incremental & self-describing
- **`manifest.json`** on R2: available dates, latest date, grid, and per-day
  (or global) vmin/vmax. The date picker reads this to enable exactly the days
  that exist — replaces the hardcoded 2023–2025 range.
- **Incremental blobs/upload:** skip days already built and objects already in
  the bucket (ETag/size compare or a local ledger), so a refresh touches only
  new days.
- **`refresh_maps.py`** orchestrator: export → blobs → manifest → upload, one
  command, incremental.
- **Exit criteria:** re-running after new data uploads only the new days and
  updates the manifest.

### Phase 3 — Near-live Actual TEC (still serverless)
- Schedule the pipeline's IONEX pull + `refresh_maps.py` on a **cron** (GitHub
  Actions or a cloud scheduled job). New days land on R2 automatically; the
  frontend keeps fetching static files.
- Optionally add a low-latency source (JPL GDGPS / IGS real-time) as a second
  namespace for the "now" tile.
- **Exit criteria:** yesterday's map appears without anyone running a command.

### Phase 4 — Predicted TEC integration
See §8. Adds a parallel `tec/pred/...` namespace fed by model inference, surfaced
by the "Predicted TEC" toggle. No change to the serving model — more static blobs.

### Phase 5 — Scale options (adopt only if needed)
- **Zarr cube** instead of many files (range-read time slices, natural appends).
- **Serverless function** (Cloudflare Worker) for on-the-fly coloring / the live
  tile / hiding a source key.
- **Custom domain** for production CDN.

---

## 7. Key-space / naming conventions

Namespace observed vs predicted from day one so they never collide:

```
tec/
  meta.json                         # grid + scale, shared
  manifest.json                     # what exists (both layers)
  obs/                              # Actual TEC (observed)
    2023/2023-06-21.f16.gz          # gzip(float16[24,71,73]) — hour = slice[h]
  pred/                            # Predicted TEC (model)
    <init>/<lead>.f16.gz            # see §8 for keying
```

> Note: today's blobs sit at `tec/<year>/...`. Phase 2 moves them under
> `tec/obs/<year>/...` to make room for `tec/pred/...`. That's a one-time key
> rename during the first managed upload.

---

## 8. How Predicted TEC integrates

This is the point of keeping the viz path clean: **predicted maps are just
another blob namespace behind the same shader.** The work is producing them and
reconciling the grid.

### 8.1 What the model produces

The pipeline trains an SFNO on the Gauss-Legendre **23×45** grid to forecast
**dTEC** (TEC minus the IRI climatological baseline) at
**+2/+4/+6 h** (INPUT_STEPS=6 seen, TARGET_STEPS=3, 7200 s cadence). So a single
inference, given an initialization time `t0`, yields:

```
predicted_TEC(t0 + lead) = IRI_baseline(t0 + lead) + predicted_dTEC(lead)
      for lead in {+2h, +4h, +6h}, on the 23×45 grid.
```

### 8.2 The grid reconciliation (the one real design decision)

Observed blobs are **71×73 uniform**; model output is **23×45 Gaussian**. For the
toggle to swap seamlessly — and to compute predicted−actual **difference maps** —
both layers should live on **one common "viz grid."**

**Decision:** resample predicted 23×45 → the **same 71×73 uniform grid** as
observed at export time (a `pred` counterpart to `export_native_tec.py`). Then:

- identical blob format (`gzip(float16[...])`), identical shader/overlay,
- trivial diff layer (`pred − obs` at matching valid times),
- one `meta.json` grid for everything.

(Resampling *up* to 71×73 doesn't invent resolution; it just puts both on a
shared canvas. The alternative — serve each at native res and reconcile in the
shader via a coordinate texture — is more code for no visual gain.)

### 8.3 Keying: forecasts have two time axes

Observed maps are keyed by one time (valid time). Forecasts have **init time**
and **lead time**, with `valid = init + lead`. Two viable layouts:

- **By init:** `tec/pred/<init_iso>/<lead>.f16.gz` — natural for "what did the
  model say at t0?"; good for a forecast run view.
- **By valid time:** `tec/pred/<valid_date>/<valid_hour>.f16.gz` (latest init
  wins) — natural for "best forecast for this moment"; matches the calendar UX
  (pick a date → see the forecast for it).

**Recommendation:** serve **by valid time** for the globe (matches the existing
date/hour picker), and record the originating `init`/`lead` in `manifest.json`
for provenance and a possible "forecast age" indicator.

### 8.4 Where inference runs

A batch job (same spirit as the exporters): load the trained model + recent
observed window → predict dTEC → add IRI baseline → resample to viz grid →
float16+gzip → upload to `tec/pred/...`. It reuses the **entire** upload path;
only the producer differs. Runs on the same Phase-3 schedule right after the
observed refresh (so each new obs frame triggers a fresh forecast).

### 8.5 Frontend surface

- The "Predicted TEC" toggle fetches from `tec/pred/...` instead of `tec/obs/...`
  — same decode, same overlay shader, same fixed scale.
- Natural additions once both exist:
  - **Actual vs Predicted** side-by-side or A/B toggle at the same valid time.
  - **Difference map** (`pred − obs`) with a diverging colormap to show model
    error spatially.
  - **Forecast age / lead** badge from the manifest.

### 8.6 Manifest extension

```jsonc
{
  "grid": { "nlat": 71, "nlon": 73, "lat_deg": [87.5, -87.5], "lon_deg": [-180, 180] },
  "scale": { "vmin": 0, "vmax": 100 },
  "obs":  { "dates": ["2023-01-01", "..."], "latest": "2025-12-31" },
  "pred": { "dates": ["..."], "latest": "...",
            "horizons_h": [2, 4, 6], "keyed_by": "valid_time" }
}
```

---

## 9. Operational runbook

| Task | Command / action |
|---|---|
| Re-export native maps | `python backend/scripts/export_native_tec.py --years 2023 2024 2025` |
| Rebuild day blobs | `python backend/scripts/build_tec_blobs.py` |
| Sanity-check a day | `python backend/scripts/build_tec_blobs.py --peek 2023-06-21` |
| Upload to R2 | `python backend/scripts/upload_r2.py` (env vars set) |
| Full refresh (Phase 2) | `python backend/scripts/refresh_maps.py` |
| Add a new year | pull IONEX via pipeline `extract`, then run the refresh |

**Verification after upload:** `fetch` one blob from the browser/`curl`, confirm
`Content-Encoding: gzip` on the response, decode to `Uint16Array`, and spot-check
a global-mean TEC against `--peek`.

---

## 10. Open decisions

1. **Scale:** fixed global `vmin/vmax` (comparable, simple) vs per-day
   (max local contrast). Leaning fixed 0–100 TECU, revisit against solar-max days.
2. **Grid for predicted:** confirmed as resample-to-71×73 (§8.2) — revisit only
   if native-res comparison becomes a requirement.
3. **Retention / backfill:** how far back to keep predicted forecasts (every
   init × lead is a lot of blobs) vs only latest-per-valid-time.
4. **Move to Zarr (Phase 5):** trigger = file count or live-append friction.

---

## 11. File reference

| Path | Role |
|---|---|
| `backend/scripts/export_native_tec.py` | bronze IONEX → native 71×73 npz |
| `backend/scripts/build_tec_blobs.py` | npz → per-day float16+gzip blobs + `meta.json` |
| `backend/scripts/upload_r2.py` | blob tree → Cloudflare R2 (S3 API) |
| `backend/scripts/refresh_maps.py` | *(Phase 2)* incremental orchestrator |
| `data/viz/tec_native_71x73/` | intermediate full-year cubes (not uploaded) |
| `data/viz/tec_blobs/` | upload-ready blob tree |
| `backend/docs/tec_maps_roadmap.md` | this document |
