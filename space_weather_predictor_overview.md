# Space Weather Predictor

> Project overview and technology stack.
> Convention: no em dashes anywhere in this doc, to match house style.

---

## What we are building

An interactive 3D globe that renders a machine-learning forecast of the Earth's ionosphere and lets you compare that forecast against what actually happened.

The scientific core is a **Spherical Fourier Neural Operator (SFNO)** that forecasts global ionospheric **Total Electron Content (TEC)**. The surface is a **spaceweatherviz.com-class interactive globe**: you scrub across a 24-hour day, watch the ionosphere evolve on a spinning Earth, and flip a toggle between the model's predicted TEC map, the actual observed map, and the difference between them. The platform refreshes on a schedule so the globe stays current with the freshest available data product.

The deliverable is the two halves together: the model that produces the forecast, and the production-grade visualization platform that makes it legible.

---

## What the model actually does

- **Target:** global TEC, forecast on a **Gauss-Legendre 23x45 grid** (nlat=23, nlon=45, spectral truncation L_max=22).
- **Residual formulation:** the network predicts **delta-TEC = TEC_IONEX - TEC_IRI**, the correction on top of the empirical IRI-2016 baseline, rather than absolute TEC. At inference the IRI baseline is added back to reconstruct absolute TEC. This is what the globe shows.
- **Inputs:** a **6-timestep TEC history window** plus **5 OMNI HRO solar-wind driver channels** (`b_magnitude, by_gsm, bz_gsm, flow_speed, proton_density`), the exogenous conditions that drive ionospheric variability.
- **Output:** a **3-step-ahead forecast**. To fill a 24-hour day the model rolls forward +1h per hour, a leak-free convention that needs only past drivers.
- **Baseline:** IRI-2016 via PyIRI, computed from date and observed (not adjusted) F10.7.

Tensor contracts: `tec_input [batch,6,23,45]`, `omni_input [batch,6,5]`, `target [batch,3,23,45]`. Splits are chronological (train 2000-2018, val 2019-2021, test 2022-2025).

---

## What the product does

- **A 3D globe** with the TEC field draped over the Earth, spinnable, with an atmosphere limb glow.
- **Three views via one toggle:** model-predicted TEC, actual observed TEC, and the predicted-minus-actual difference field.
- **A 24-hour time scrubber** with play/pause, interpolating smoothly between hourly frames.
- **A day/night terminator** driven by the true subsolar point, which doubles as a physical sanity check: TEC is photoionization-driven, so the bright equatorial anomaly should sit under the sun.
- **Honest data semantics:** every frame carries provenance and data-age; when the observed product for a recent hour has not been published yet, the difference view says "truth not yet available" instead of showing a blank or a fabricated map.

---

## What "live" means here

**Live means scheduled refresh of the freshest available product, not true real-time.**

This is the decision that shapes the whole architecture. We stay on the exact products the model trained on: **CODE final GIMs, OMNI HRO, observed F10.7**. The consequences:

- **No train/serve product mismatch.** We never feed the model a real-time product it never saw.
- **No L1-to-bow-shock solar-wind propagation** to reconcile a real-time feed against OMNI's convention.
- **No GPU serving.** A 23x45 SFNO forward pass is milliseconds on CPU.

The tradeoff is that the freshest map carries the product's inherent latency rather than being "now." That is still a legitimately updating product; the globe reads "as of <time>" and degrades honestly. The one remaining choice inside this framing is the product-latency rung (final `codg` vs. rapid `corg`), which is gated on a rapid-vs-final equality validation before rapid is ever served.

---

## Architecture in one pass

```
live source pull  ->  raw store (verbatim)  ->  align/preprocess to GL numpy arrays
   ->  SFNO inference  ->  reconstruct + GL->uniform resample  ->  pack textures + metadata
   ->  cache  ->  FastAPI (serves cache)  ->  frontend fetch/poll
   ->  texture arrays on GPU  ->  one data-layer shader colormaps onto the globe
   ->  HUD scrubs time, flips source
```

The backend is the existing offline pipeline, factored into importable functions and run on a Prefect timer, with a thin API over a cache. Backend and frontend meet at a single seam: a **texture-array plus metadata-JSON contract**. The preprocessing that feeds the live model imports the exact same functions used at training time, so byte-identical train/serve parity is a structural property, not a discipline.

---

## Technology stack

### Backend / data / model

- **Python** — The entire pipeline, PyIRI, and the SFNO model already live here, and the scientific/ML ecosystem has no real competitor for this domain. Everything is already in it; anything else means a rewrite for zero gain.
- **NumPy** — The array format the model actually consumes (`tec_input`, `omni_input`) and the substrate for interpolation, OMNI alignment, and the delta-TEC math. It is the interchange format every other library here already speaks.
- **PyTorch** — The framework the SFNO is built and trained in; live inference reuses the exact training code. The reference SFNO implementations are PyTorch-native, so there is nothing to reimplement and no train/serve architecture drift.
- **torch-harmonics** — The differentiable spherical harmonic transforms the operator needs on the Gauss-Legendre grid at L_max=22. It is the canonical library the SFNO architecture was published with; its GL quadrature and transforms are built to match, so the grid choice and the library agree by construction.
- **PyIRI** — Computes the IRI-2016 baseline subtracted to form the delta-TEC target, in pure Python. It drops the Fortran build dependency entirely while matching IRI-2016, which the whole residual target rests on.
- **Prefect** — Orchestrates and schedules the pull -> align -> infer -> assemble flow on a cadence, with retries and idempotent writes. Existing muscle memory, and it fits the exact shape here: run a DAG on a timer, survive flaky feeds.
- **httpx (or requests)** — Pulls the CODE GIM, OMNI HRO, and F10.7 payloads. httpx if async pulls that match FastAPI's model are wanted, requests for the simplest synchronous puller; httpx is the more future-proof.
- **DuckDB + Parquet** — The cache the API serves from and the per-year columnar stores. Zero-server, fast analytical reads, and the gotchas (file locking, read_only connections) are already known.
- **FastAPI** — The REST layer serving `/latest`, `/metadata`, `/health` off the cache. Async by default, Pydantic-typed on both request and response, and auto-generates OpenAPI docs, which matters because the seam is a contract and FastAPI enforces its schema for free.
- **Uvicorn** — The ASGI server that runs FastAPI. The standard high-performance ASGI server FastAPI is designed around; no reason to deviate.
- **Pydantic** — Validates and serializes the metadata-JSON contract (color bounds, provenance, staleness flags). Ships with FastAPI and turns the seam contract into enforced types instead of hand-checked dicts.
- **pytest** — Runs the correctness gates: the byte-identical cache check after the Stage 4 refactor, and the rapid-vs-final (`corg` vs `codg`) equality gate. The existing suite already lives in it and the entire "verify before switching" discipline runs through it.

### Frontend

- **TypeScript** — Types the app shell, the Zustand store, and the metadata contract on the client so the seam is checked on both sides. Catches contract drift at compile time, which pays off most exactly at a data-heavy handoff like this one.
- **React** — The UI framework for the HUD and the structure around the canvas. It is the ecosystem react-three-fiber is built on, so the 3D scene and the HUD share one state model instead of two disconnected ones.
- **Vite** — Dev server and bundler. Near-instant hot reload is critical when iterating on shaders and scene params, and it handles TypeScript/React/GLSL with zero config; the current default over CRA or webpack.
- **three.js** — The WebGL engine rendering the globe, meshes, and materials. The mature, dominant WebGL library, and every globe-visualization in this genre (the spaceweatherviz class included) is built on it.
- **react-three-fiber** — Renders three.js as React components so the scene reacts declaratively to Zustand state. Removes the manual imperative render-loop-to-state syncing that raw three.js forces.
- **drei** — Prebuilt R3F helpers: orbit controls, loaders, shader-material utilities. Saves hand-rolling camera controls and asset loading; the standard companion to R3F.
- **GLSL** — The shader language for the TEC data-layer (texture-array sampling plus colormap LUT), the atmosphere glow, and the day/night terminator. No alternative for custom WebGL fragment shaders, and the shader work is where both the visual quality and the smooth in-shader time interpolation live.
- **@react-three/postprocessing** — The EffectComposer stack for bloom and ACES tone mapping. The R3F-native post-processing library, so effects compose into the same declarative tree as the rest of the scene.
- **Zustand** — Holds interaction state: current hour, active source, opacity, staleness. The lightweight store R3F is designed to pair with, and it can push shader-uniform updates without triggering React re-renders, which keeps scrubbing smooth.
- **Leva** — Dev-time GUI for live-tuning shader and scene parameters. The standard R3F control panel, drops in with almost no wiring, and is stripped from the production build.

### Tooling

- **Git / GitHub** — Version control and where the project and test suite already live. The existing home and the natural place the gates run.
- **GitHub Actions** — Runs the pytest gates (byte-identical cache, `corg`-vs-`codg` equality) on every change, so "verify before switching products" is enforced automatically rather than by memory. Native to the repo with no separate CI service to stand up.
