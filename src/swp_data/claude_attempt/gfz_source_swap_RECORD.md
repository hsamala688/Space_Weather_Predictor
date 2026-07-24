# Source Swap Record: CelesTrak to GFZ (revised Scope A)

> Decision and verification record for the driver-index source consolidation,
> superseding the OMNI2 plan in `omni2_source_consolidation_SPEC.md`. Written
> 2026-07-12 after the swap was executed and accepted.
>
> Convention: no em dashes in this doc.

---

## 1. What was decided and why

The original spec locked OMNI2 as the single origin for F10.7 and Kp. Both of
its equivalence gates failed, for reasons on OMNI2's side:

- **OMNI2 F10.7 is adjusted to 1 AU, not observed.** Confirmed against the
  published format statement (word 51, `ow_data.html`) and empirically: the
  OMNI2/CelesTrak ratio is 0.967 every January and 1.034 every July, the
  (r/1AU)^2 Sun-Earth distance signature the spec's Section 8 names as the
  diagnostic. IRI requires observed flux. OMNI2 has no observed column, and
  the extended OMNI2 product only adds Lyman-alpha and Proton QI. Hard fail.
- **OMNI2 Kp contradicts the definitive GFZ record for June 2006.** Exactly 50
  of 75,976 windows differ, all in 2006-06, each by one Kp third. GFZ's
  definitive record sides with CelesTrak on every checked window. OMNI2
  appears to have ingested provisional June 2006 values and never refreshed.

**Replacement source: the GFZ Potsdam combined index file**
`https://kp.gfz.de/app/files/Kp_ap_Ap_SN_F107_since_1932.txt` (one file,
1932-present, updated daily, no auth). GFZ is the Kp producer, and the file
carries F10.7 with observed and adjusted explicitly separated (`F10.7obs`,
`F10.7adj`). The parser (`swp_data.parse.parse_gfz`) asserts the
observed-vs-adjusted seasonal signature in-file on every run.

The OMNI2 yearly download is deleted from Stage 1 (nothing consumes it). OMNI
HRO is unchanged as the source of the five solar-wind driver channels.

## 2. Gate results (GFZ vs the retiring CelesTrak series, 2000-2025)

- **Gate A (Kp): PASS, bit-exact.** 75,976 / 75,976 three-hourly values equal.
  Identical coverage, no fills.
- **Gate B (F10.7): 9,465 / 9,497 days equal; 32 divergent days, all
  explained, accepted as-is (team decision, 2026-07-12).** None is a parser or
  pipeline error; all trace to genuine disagreements between the two archives'
  relays of the NRCan Penticton measurement record.

## 3. The 32 accepted F10.7 divergence days

Verified against the producer's own record
(`spaceweather.gc.ca/solar_flux_data/daily_flux_values/fluxtable.txt`).
"old" is the CelesTrak-derived value the pipeline used before the swap; "new"
is the GFZ-derived value now in `data/raw/f107/f107_daily.parquet`.

### Class 1: NRCan re-derived the measurement; the archives picked different determinations (16 days)

The producer's fluxtable contains two determinations of the same 20:00 UT
measurement on these dates. GFZ carries one, CelesTrak the other. Both are
genuine Penticton values.

| date | old (CelesTrak) | new (GFZ) |
|---|---|---|
| 2018-02-20 | 70.5 | 67.8 |
| 2018-02-22 | 69.7 | 68.4 |
| 2018-02-23 | 70.0 | 67.6 |
| 2018-02-24 | 70.0 | 68.2 |
| 2018-02-25 | 69.2 | 67.2 |
| 2018-02-26 | 72.0 | 69.8 |
| 2018-02-27 | 70.0 | 67.9 |
| 2018-02-28 | 69.9 | 68.8 |
| 2019-02-06 | 71.3 | 69.8 |
| 2019-02-08 | 71.8 | 70.0 |
| 2020-05-14 | 67.2 | 67.6 |
| 2020-05-29 | 69.6 | 69.5 |
| 2021-03-02 | 75.6 | 74.7 |
| 2022-10-23 | 105.5 | 108.4 |
| 2023-01-08 | 182.5 | 183.8 |
| 2023-02-18 | 167.2 | 167.4 |

### Class 2: no standard 20:00 UT measurement exists; GFZ substituted the nearest actual measurement, CelesTrak used an external value (2 days)

GFZ's value matches a real off-noon Penticton measurement; CelesTrak's value
matches none of the day's measurements (secondary source, likely NOAA SWPC).

| date | old (CelesTrak) | new (GFZ) | producer measurements that day |
|---|---|---|---|
| 2006-04-18 | 76.9 | 75.8 | 17h 75.4, 23h 75.8 |
| 2007-06-07 | 85.7 | 85.5 | 16h 87.9, 19h 85.5, 22h 86.5 |

### Class 3: GFZ marks the day missing; the new pipeline time-interpolates (10 days)

On these dates the producer has no usable 20:00 UT determination and GFZ
records -1. The new series carries the time-interpolated value (same rule as
burst days). CelesTrak filled these from a secondary source; where checked,
its fill matches none of the day's actual Penticton measurements.

| date | old (CelesTrak fill) | new (interpolated) |
|---|---|---|
| 2005-06-30 | 102.5 | 101.400000 |
| 2005-07-13 | 91.7 | 93.050000 |
| 2006-12-03 | 86.5 | 92.333333 |
| 2006-12-04 | 94.5 | 97.366667 |
| 2007-06-06 | 87.1 | 83.350000 |
| 2008-02-03 | 71.6 | 71.550000 |
| 2020-12-17 | 81.8 | 81.833333 |
| 2020-12-18 | 81.8 | 81.766667 |
| 2021-06-16 | 80.3 | 80.250000 |
| 2024-08-26 | 227.0 | 226.950000 |

### Class 4: pre-fluxtable dates, presumed Class 1 (3 days)

The producer's public fluxtable begins 2004-10-28, so these cannot be checked
against it; the pattern (both archives hold clean one-decimal values that
disagree by a few sfu) matches Class 1.

| date | old (CelesTrak) | new (GFZ) |
|---|---|---|
| 2002-08-30 | 174.8 | 171.9 |
| 2004-10-13 | 89.2 | 88.5 |
| 2004-10-25 | 138.5 | 141.4 |

### Class 5: burst-day interpolation artifact (1 day)

2023-02-17 is a >300 sfu burst day (343.1 in both archives). Both the old and
new pipelines mask it and time-interpolate; the 0.1 sfu difference comes from
neighbor 2023-02-18 being a Class 1 day. Old 165.2, new 165.3.

## 4. Impact statement

- Kp: zero change. `omni_input` channel 6 is bit-identical.
- F10.7: 32 of 9,497 days (0.34%) differ, by at most 5.8 sfu (2006-12-03) and
  typically 1-3 sfu. F10.7 enters only through the IRI baseline, so dTEC
  inputs/targets in windows overlapping those days shift by sub-percent
  amounts; all other windows are unchanged. Normalization statistics shift
  negligibly. The prior dataset's split sizes (110,124 / 25,406 / 26,270)
  should be reproduced exactly on rebuild, since coverage is identical.
- A downstream rebuild (`assemble iri -> dtec -> omni -> windows`) is only
  needed if bit-consistency between `f107_daily.parquet` and the existing
  window caches matters; the caches on disk still reflect the CelesTrak
  values on the 32 days above.

## 5. Artifact provenance

- `data/raw/f107/f107_daily.parquet`, `data/raw/geomag/kp_3hourly.parquet`:
  GFZ-derived as of 2026-07-12 (`swp-data extract --indices-only` +
  `swp-data parse`).
- `data/raw/f107/f107_daily.celestrak_backup.parquet`,
  `data/raw/geomag/kp_3hourly.celestrak_backup.parquet`: the retired
  CelesTrak-derived tables, kept for reference.
- `data/raw/geomag/kp_daily.parquet`: legacy CelesTrak daily summary
  (`kp_max`, `quiet_all_day`); only consumed by out-of-scope one-off scripts;
  not reproduced by the new pipeline and now frozen.
- Gate implementation: `src/swp_data/verify.py` (`swp-data verify-gates`).
