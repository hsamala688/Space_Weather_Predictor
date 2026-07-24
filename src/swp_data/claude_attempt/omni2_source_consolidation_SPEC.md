# Source Consolidation Spec: CelesTrak to OMNI2 (Scope A)

> Handoff doc for the clean-repo rewrite of the `data_pull/` pipeline. Scopes one
> change: consolidate the driver-index sources so OMNI2 becomes the single origin
> for F10.7 and Kp, retiring CelesTrak. Everything else in the pipeline is held
> constant on purpose. All design decisions below are locked.
>
> Convention: no em dashes in this doc.

---

## 1. Goal

The current pipeline pulls indices from three places: IONEX (CDDIS), solar wind
(OMNI HRO, SPDF), and two scalar indices (F10.7 and Kp) from CelesTrak's
`SW-All.txt`. Meanwhile OMNI2 is downloaded every run by Stage 1 and never
consumed. This is source scatter and one dead download.

Consolidate to **Scope A**: OMNI2 becomes the single origin for F10.7 and Kp.
CelesTrak is deleted. OMNI HRO is kept unchanged for the five solar-wind driver
channels. The result is two live sources (CDDIS IONEX, SPDF OMNI) instead of
three, and the "OMNI2 downloaded but dead" gap closes because something finally
parses it.

This is a consolidation, not a data change. The success criterion (Section 6) is
that the rebuilt dataset is numerically identical to the pre-swap dataset.

---

## 2. Locked decisions

- **Scope A only.** OMNI2 replaces CelesTrak for F10.7 and Kp. OMNI HRO stays as
  the source for `b_magnitude, by_gsm, bz_gsm, flow_speed, proton_density`.
- **Dst is not a driver.** OMNI2 carries Dst; it is read/documented in the parser
  but is NOT added to `DRIVER_FEATURES`, NOT written as a downstream artifact, and
  NOT present in the `[N, 6]` driver array. No new dead artifact. Enabling Dst
  later is a documented one-line change, not a rebuild.
- **Driver contract unchanged.** `DRIVER_FEATURES` stays the same six channels in
  the same order (`b_magnitude, by_gsm, bz_gsm, flow_speed, proton_density,
  kp_3hour`). Only Kp's source changes, not the feature set.
- **Downstream shapes unchanged.** `tec_input [N, 6, 23, 45]`,
  `omni_input [N, 6, 6]`, `target [N, 3, 23, 45]`. Falisha's loop and
  `metadata.json` contract are untouched.
- **Stage 4 stays one file, four subcommands** (`iri`, `dtec`, `omni`, `windows`),
  as `data_for_falisha.py` runs today. Renamed for the clean repo.
- **Legacy and one-off scripts are out of scope** (`offset.py`,
  `export_dtec_sample.py`, `sakshum.py`, `test.py`, `yeet.py`). They do not appear
  in the clean repo.
- **All data is retrospective, 2000 to 2025.** OMNI2's update lag is a non-issue
  for a training set; the real-time serving question is deferred and noted in the
  README, not solved here.

---

## 3. What changes and what does not

| Stage | Change |
|---|---|
| Stage 1 extract | Delete the two CelesTrak pulls (`pull_f107`, `geomag_pull.py`). No download added: `omni2_{year}.dat` is already pulled. |
| Stage 2 parse | Add one parser: `parse_omni2`. Delete both CelesTrak parse paths and their `BEGIN/END OBSERVED` block handling. |
| Stage 3 interpolate | No change. |
| Stage 4 assemble | No code change. F10.7 still arrives at `iri` as a daily series; Kp still arrives at `omni` as a 3-hourly causal series. |
| Stage 5 dataset | No change. |

### One subtlety that is easy to get wrong

The swap is insulated in **code and shape**, but not fully insulated in **values**:

- **Kp** feeds `omni_input` channel 6 directly. A source swap changes that channel
  if and only if the Kp values differ.
- **F10.7** feeds the **IRI baseline**, and `dTEC = IONEX - IRI`. So F10.7 flows
  into `tec_input` AND `target`, and into the normalization statistics derived from
  them. F10.7 is not just a driver, it shapes the prediction target.

This is why the verification is an equality check on the two source series
(Section 6), not just a shape check. If the OMNI2 series equal the CelesTrak
series, the entire rebuilt dataset is bit-identical. If F10.7 diverges even
slightly, every TEC target moves.

---

## 4. The `parse_omni2` contract

`parse_omni2` replaces both retired CelesTrak parsers. It reads the raw hourly
OMNI2 record already on disk at `data/raw/omni2/{YYYY}/omni2_{YYYY}.dat` and emits
two artifacts matching the shape and column names the old parsers produced, so
Stage 4 consumes them unchanged.

### Outputs

- `data/raw/f107/f107_daily.parquet` : date-indexed, column `f107_obs`. (Same path
  and schema `pull_f107` wrote, so `iri` needs no change.)
- `data/raw/geomag/kp_3hourly.parquet` : one row per 3-hour UT window, column `kp`.
  (Same schema `geomag_pull.py`'s 3-hourly output had, so `omni`'s
  `align_kp_to_timestamps` needs no change.)

The old `kp_daily.parquet` (`kp_max`, `quiet_all_day`) is NOT reproduced: its only
consumers were legacy scripts, which are out of scope.

### Field mapping (must be pinned, not remembered)

The OMNI2 hourly record is fixed-column ASCII. Do NOT hardcode field positions
from memory. Pin exact word positions and `Fx.y` formats from the published OMNI2
format statement at `https://omniweb.gsfc.nasa.gov/html/ow_data.html`, and assert
the record layout on the first parsed line (same defensive first-line field-count
check `parse_omni_hro` already does).

Fields to extract:

- **F10.7** (daily solar radio flux, observed). In the hourly record this is the
  daily value repeated across that day's 24 hours. Collapse to one row per UT date
  (drop fill, take the day's value).
- **Kp** (`Kp*10` integer encoding, same as CelesTrak). In the hourly record this
  is the 3-hour value repeated across its three hours. Collapse to the 3-hour
  cadence (one row per UT window).
- **Dst** : read and document the column, but do not emit it (see Section 2).

### Fill handling

OMNI2 uses per-field sentinels (for example `999.9` for field magnitudes, `9999`,
`99` for others), not one global value. Mask each field's own sentinel to NaN with
a `>=` threshold before any scaling or collapsing, exactly as `parse_omni_hro`
does. The `pyspaceweather` `_OMNI_MISSING` dict is a clean reference for the
per-field sentinel values.

---

## 5. Three must-survive behaviors

These lived in the CelesTrak path and are correctness properties, not incidental
code. Reproduce each in `parse_omni2`. Do not let a "clean rewrite" quietly drop
them.

1. **F10.7 spike handling.** Values above 300 sfu are burst-inflated single-day
   spikes, not a valid daily EUV proxy. Set them to NaN and time-interpolate over
   them (do NOT clip to 300). Apply after masking OMNI2's own `999.9`-class fill.
   Same underlying Penticton measurement as CelesTrak, so the same spikes appear.

2. **Kp causal semantics (the future-leak trap).** OMNI2 gives Kp hourly, but a Kp
   value for a 3-hour UT window is only finalized when that window completes. Do
   NOT naively forward-fill the raw hourly column: that leaks the in-progress
   window's Kp before it is known. Reconstruct the 3-hour cadence and preserve the
   exact window-indexing and forward-fill convention `geomag_pull.py` +
   `align_kp_to_timestamps` use today. The equality gate in Section 6 confirms this
   is reproduced correctly.

3. **Observed, not adjusted, F10.7.** IRI-2016 wants observed F10.7 (varies with
   Sun-Earth distance), not adjusted-to-1AU. This is a locked convention. Assert it
   in the parser. If OMNI2's F10.7 column turns out to be adjusted, the equality
   gate will fail with a characteristic seasonal (roughly percent-level) divergence
   from CelesTrak's observed flux. That signature is the diagnostic. See Section 8.

---

## 6. Success criterion: dataset equivalence

Because both CelesTrak and OMNI2 relay the same upstream measurements (GFZ Potsdam
Kp, Penticton/Ottawa F10.7), the OMNI2-derived series should equal the retiring
CelesTrak series over any overlapping period. That gives a hard, checkable
guarantee for a swap that must not silently change the data:

> **The rebuilt dataset must be numerically identical (within float tolerance) to
> the pre-swap dataset, because the only thing that changed is where two
> numerically-identical series came from.**

This decomposes into two independent equivalence gates:

- **Gate A: Kp.** New `kp_3hourly` equals old `kp_3hourly` where both are present,
  over the full 2000 to 2025 overlap. Protects `omni_input` channel 6.
- **Gate B: F10.7.** New `f107_daily` equals old `f107_daily` where both are
  present. Protects the IRI baseline, therefore `tec_input` AND `target`.

If both gates pass, rebuilding `iri` -> `dtec` -> `omni` -> `windows` must
reproduce the prior dataset. The recorded pre-swap split sizes
(110,124 / 25,406 / 26,270 for train/val/test) are a coarse sanity magnitude:
they must not change.

---

## 7. File layout and CLI (front half of the clean repo)

```
src/swp_data/
  config.py            paths, grid contract, split years, F10.7/Kp constants
  sources/
    cddis.py           IONEX auth + download
    omni.py            OMNI HRO + OMNI2 download (both SPDF anon)
  extract.py           Stage 1 orchestration + manifests
  parse.py             parse_ionex, parse_omni_hro, parse_omni2
  interpolate.py       Stage 3
  assemble.py          Stage 4: one file, four subcommands (was data_for_falisha.py)
  dataset.py           Stage 5
  cli.py               entry point
```

```
swp-data extract
swp-data interpolate [--year Y]
swp-data assemble iri     [--year Y]
swp-data assemble dtec    [--year Y]
swp-data assemble omni    [--year Y]
swp-data assemble windows [--train-end-year Y] [--val-end-year Y]
```

No `celestrak.py`, no standalone Stage 0: F10.7/Kp derivation is now part of the
parse layer.

---

## 8. Verification plan (per boundary)

Run in order; each gate must pass before proceeding.

1. **`parse_omni2` unit correctness.**
   - First-line field-count assertion fires on a wrong OMNI product.
   - Field mapping matches the published format statement (spot-check a few known
     dates by hand against `ow_data.html`).
   - Per-field fill masking, F10.7 daily collapse, Kp 3-hourly collapse verified on
     one known year.

2. **Gate A (Kp equivalence).** Build new `kp_3hourly`, diff against the current
   CelesTrak-derived `kp_3hourly` over full overlap. Assert equality where both
   present. Enumerate and explain any divergence (expected: only coverage/fill
   edges, not value differences).

3. **Gate B (F10.7 equivalence).** Same, for `f107_daily`. A clean pass confirms
   both the spike handling and the observed-not-adjusted property. A systematic
   seasonal divergence here specifically indicates an observed-vs-adjusted
   mismatch: stop and fix the column, do not proceed.

4. **Dataset equivalence.** Rebuild `iri` -> `dtec` -> `omni` -> `windows`. Assert
   `tec_input`, `omni_input`, `target`, `metadata.json` stats, and split sizes
   match the pre-swap dataset within tolerance. This is the real acceptance test.

---

## 9. Open items to verify before building

- **Confirm F10.7 is in the standard `omni2_{YYYY}.dat`.** Expected yes (the hourly
  record carries F10.7, Kp, Dst). If it is only in the "extended" OMNI2 product
  (`low_res_omni/extended`), switch Stage 1's OMNI2 download to the extended file.
- **Confirm OMNI2's F10.7 column is observed, not adjusted.** Gate B is the
  backstop, but check the format statement directly too.
- **Confirm the Kp window-indexing convention** in the current
  `geomag_pull.py` / `align_kp_to_timestamps` before reproducing it, so Gate A is a
  true equality and not an accidental off-by-one-window match.

---

## 10. Out of scope

- Scope B (dropping OMNI HRO, pulling solar wind from OMNI2 too). Not doing this:
  it would trade away 5-minute fidelity and SYM-H for no functional gain.
- Dst as a training driver. Parsed-and-ignored now; enabling it later is a
  documented follow-up.
- Real-time / serving-time source strategy. OMNI2 lags; if the agent ever runs
  live it needs a real-time feed at inference. Note in README, do not solve here.
- All legacy/one-off analysis scripts.
