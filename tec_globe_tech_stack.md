# TEC Globe: Technology Stack

## Backend / data / model

- **Python** — The entire pipeline, PyIRI, and the SFNO model already live here, and the scientific/ML ecosystem has no real competitor for this domain. Best choice because everything you've built is already in it; anything else means a rewrite for zero gain.
- **NumPy** — The array format the model actually consumes (`tec_input`, `omni_input`) and the substrate for interpolation, OMNI alignment, and the delta-TEC math. Best choice because it's the interchange format every other library here (PyTorch, PyIRI, DuckDB) already speaks.
- **PyTorch** — The framework the SFNO is built and trained in; live inference reuses your exact training code. Best choice because the reference SFNO implementations are PyTorch-native, so there's nothing to reimplement and no train/serve architecture drift.
- **torch-harmonics (the SHT library under the SFNO)** — Provides the differentiable spherical harmonic transforms the operator needs on the Gauss-Legendre grid at L_max=22. Best choice because it's the canonical library the SFNO architecture was published with; its GL quadrature and transforms are built to match, so your grid choice and the library agree by construction.
- **PyIRI** — Computes the IRI-2016 baseline you subtract to form the delta-TEC target, in pure Python. Best choice (already locked) because it drops the Fortran build dependency entirely while matching IRI-2016, which your whole residual target rests on.
- **Prefect** — Orchestrates and schedules the pull → align → infer → assemble flow on a cadence, with retries and idempotent writes. Best choice because you already have the muscle memory from the NFL project and it fits the exact shape here: run a DAG on a timer, survive flaky feeds.
- **httpx (or requests)** — Pulls the CODE GIM, OMNI HRO, and F10.7 payloads from their providers. Best choice is httpx if you want async pulls that match FastAPI's model, or requests for the simplest synchronous puller; httpx is the more future-proof of the two.
- **DuckDB + Parquet** — The cache the API serves from and the per-year columnar stores. Best choice (already locked) because it's zero-server, gives fast analytical reads, and you already know its gotchas (file locking, read_only connections).
- **FastAPI** — The REST layer serving `/latest`, `/metadata`, `/health` off the cache. Best choice because it's async by default, Pydantic-typed on both request and response, and auto-generates OpenAPI docs, which matters because the texture-plus-metadata seam is a contract and FastAPI enforces its schema for free.
- **Uvicorn** — The ASGI server that runs FastAPI. Best choice because it's the standard high-performance ASGI server FastAPI is designed around; there's no reason to deviate.
- **Pydantic** — Validates and serializes the metadata-JSON contract (color bounds, provenance, staleness flags). Best choice because it ships with FastAPI and turns your seam contract into enforced types instead of hand-checked dicts.
- **pytest** — Runs the correctness gates: the byte-identical cache check after the Stage 4 refactor, and the rapid-vs-final (`corg` vs `codg`) equality gate. Best choice because your 21-test suite already lives in it and your entire "verify before switching" discipline runs through it.

## Frontend

- **TypeScript** — Types the app shell, the Zustand store, and the metadata contract on the client so the seam is checked on both sides. Best choice because it catches contract drift at compile time, which pays off most exactly at a data-heavy handoff like this one.
- **React** — The UI framework for the HUD and the structure around the canvas. Best choice because it's the ecosystem react-three-fiber is built on, so the 3D scene and the HUD share one state model instead of two disconnected ones.
- **Vite** — Dev server and bundler. Best choice because near-instant hot reload is critical when you're iterating on shaders and scene params, and it handles TypeScript/React/GLSL with zero config; it's the current default over CRA or webpack.
- **three.js** — The WebGL engine rendering the globe, meshes, and materials. Best choice because it's the mature, dominant WebGL library, and every globe-visualization in this genre (the spaceweatherviz class of site included) is built on it.
- **react-three-fiber** — Renders three.js as React components so the scene reacts declaratively to your Zustand state. Best choice because it removes the manual imperative render-loop-to-state syncing that raw three.js forces, which is the main reason polished apps adopt it.
- **drei** — Prebuilt R3F helpers: orbit controls, loaders, shader-material utilities. Best choice because it saves you hand-rolling camera controls and asset loading; it's the standard companion to R3F.
- **GLSL** — The shader language for the TEC data-layer (texture-array sampling plus colormap LUT), the atmosphere glow, and the day/night terminator. Best choice because there is no alternative for custom WebGL fragment shaders, and the shader work is where both the visual quality and the smooth in-shader time interpolation actually live.
- **@react-three/postprocessing** — The EffectComposer stack for bloom and ACES tone mapping. Best choice because it's the R3F-native post-processing library, so effects compose into the same declarative tree as the rest of the scene.
- **Zustand** — Holds interaction state: current hour, active source, opacity, staleness. Best choice because it's the lightweight store R3F is designed to pair with, and it can push shader-uniform updates without triggering React re-renders, which is what keeps scrubbing smooth.
- **Leva** — Dev-time GUI for live-tuning shader and scene parameters. Best choice because it's the standard R3F control panel and drops in with almost no wiring; you strip it from the production build.

## Tooling

- **Git / GitHub** — Version control and where the project and test suite already live. Best choice because it's your existing home and the natural place the gates run.
- **GitHub Actions** — Runs the pytest gates (byte-identical cache, `corg`-vs-`codg` equality) on every change, so "verify before switching products" is enforced automatically rather than by memory. Best choice because it's native to your repo with no separate CI service to stand up.
