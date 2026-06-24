# AMPERE-S2Net: Final Architecture Blueprint

**Status: Locked. This document supersedes all previous drafts.**

---

## What We Are Building

A **Spherical Fourier Neural Operator (SFNO)** that ingests AMPERE satellite measurements of Birkeland field-aligned currents as a spatial field on S², and predicts the global current density map 1–3 hours into the future — everywhere on Earth simultaneously, in magnetic coordinates.

**The scientific question:**
> Given the global distribution of field-aligned Birkeland currents right now and over the past 6 hours, what will the global current density pattern look like in 1, 2, and 3 hours?

**Why this phenomenon:** Birkeland currents are the most physically direct measurable signature of geomagnetic storm energy entering Earth's system. Predicting their global spatial evolution is more fundamental than predicting surface proxies like ground $\Delta B$ (SuperMAG) or integrated indices like Kp. AMPERE provides direct satellite measurements of these currents from the Iridium constellation.

**Why spherical:** The data is a continuous field on Earth's surface — a sphere. Birkeland currents form spatially coherent auroral ovals that expand equatorward during storms. A flat CNN distorts the geometry near the poles, which is exactly where the most important dynamics happen. An S²CNN sees the full spherical pattern at once and preserves the geometry exactly.

**Why SO(2), not SO(3):** The correct symmetry group is determined by asking: which transformations of the input leave the correct output unchanged? Shifting all magnetic longitudes by the same amount is a physical symmetry — a storm at 0° magnetic longitude and the same storm at 90° are the same physical event. But tilting Earth sideways is not a symmetry — the magnetic poles are physically special, and the auroral oval sits at fixed magnetic latitude by physics. SO(2) is the rotation group around Earth's magnetic axis. It is the complete and correct symmetry for this data — not a weakened version of SO(3).

**Coordinate system:** All spatial data is processed in **AACGM magnetic coordinates** (Altitude-Adjusted Corrected Geomagnetic coordinates), not geographic lat/lon. In geographic coordinates the magnetic pole traces a small circle as Earth rotates, breaking SO(2) symmetry. In AACGM coordinates the phenomenon is organized around the magnetic pole and SO(2) longitude-rotation equivariance holds exactly. Person 1 handles the coordinate transform in the data pipeline; Person 2 notes it in the theory section.

---

## Phenomenon and Data

### What We Are Predicting: Birkeland Field-Aligned Currents

Birkeland currents are electrical currents that flow along Earth's magnetic field lines, connecting the magnetosphere to the ionosphere at the poles. During geomagnetic storms, solar wind energy drives these currents to intensify dramatically and their auroral footprint expands equatorward. They organize into two large-scale systems:

- **Region 1 currents:** outer ring, closer to the pole, flow into the ionosphere on the dawn side and out on the dusk side
- **Region 2 currents:** inner ring, lower latitude, opposite polarity to Region 1

Predicting the global spatial pattern and intensity of $j_z$ (radial current density, $\mu A/m^2$) tells operators exactly where high-energy atmospheric corridors are forming — relevant to satellite drag, power grid induction, and aviation radiation exposure at high latitudes.

### Primary Data: AMPERE (Active Magnetosphere and Planetary Electrodynamics Response Experiment)

- **Source:** The Iridium commercial satellite constellation (~66 satellites in polar orbit) carries magnetometers that measure cross-track magnetic perturbations $\Delta B$ as satellites fly through current sheets
- **Derived product:** AMPERE processes raw $\Delta B$ measurements into global maps of radial current density $j_z$ at 10-minute cadence
- **Access:** Johns Hopkins APL, free academic access: ampere.jhuapl.edu
- **Native format:** NetCDF/HDF5 files, global maps on a fixed latitude/magnetic local time grid
- **Key challenge:** At any snapshot, Iridium orbital tracks leave longitudinal gaps. AMPERE mitigates this by accumulating data over 10-minute windows. We use these pre-processed maps rather than raw orbital tracks — this is not the same sparsity problem as raw satellite data.
- **Cadence after resampling:** 10-minute native → resample to **1-hour** to match temporal modeling depth and reduce noise

### Secondary Data: OMNI Solar Wind (Conditioning Signal Only)

- **Source:** NASA OMNI dataset, already propagated to Earth's bow shock — omniweb.gsfc.nasa.gov
- **What it provides:** IMF vector ($B_x, B_y, B_z$), solar wind velocity $v_{sw}$, proton density $n_p$, dynamic pressure
- **How it enters the model:** As a scalar time series fed into the GRU conditioning path — it does not live on the sphere and does not pass through SHT layers. It conditions the spherical layers by modulating feature maps (see architecture below).
- **Why Bz matters:** Southward IMF $B_z$ drives magnetic reconnection at the dayside magnetopause, which directly intensifies Birkeland currents. This is the primary storm driver.

### Time Range

**2010–2020.** Covers Solar Cycle 24 in full, multiple major storm events (including the March 2015 St. Patrick's Day storm), and sufficient quiet-time baseline. Train on 2010–2018, validate on 2019, test on 2020.

### Input Tensor per Sample

```
6 hours of AMPERE j_z maps in AACGM coordinates
Interpolated onto HEALPix N_side=32 grid (~12,288 pixels)
6 timesteps × 1 scalar channel = 6 channels

Shape: [batch, n_pixels, 6]

Plus OMNI solar wind time series (not on sphere):
Shape: [batch, 6, 5]   (6 timesteps × 5 variables: Bx, By, Bz, v_sw, n_p)
```

### Target Tensor per Sample

```
Predicted j_z map at t+1h, t+2h, t+3h

Shape: [batch, n_pixels, 3]   (3 forecast horizons, scalar j_z at each pixel)
```

---

## Technology Stack

**Decided. Not up for discussion.**

| Component | Choice | Reason |
|---|---|---|
| Deep learning framework | **PyTorch** | `torch-harmonics` is PyTorch-native, autograd-compatible, GPU-ready. Working toy model already exists. Switching to JAX resets implementation work with no benefit for this task. |
| SHT library | **torch-harmonics** | PyTorch-native SHT/inverse SHT with autograd support. Directly implements the spectral filtering in each SFNO layer. |
| Spherical grid | **HEALPix N_side=32** | ~12,288 equal-area pixels, standard in geophysics. `healpy` handles pixelization. |
| Data wrangling | **Polars + SciPy + healpy** | Polars for fast DataFrame manipulation, SciPy for spatial interpolation, healpy for HEALPix coordinate transforms |
| Magnetic coordinates | **AACGM via `aacgmv2`** | Python library for AACGM coordinate conversion. Applied in data pipeline before any model input. |
| Visualization | **PyVista** | 3D globe mesh rendering, interactive, handles spherical tensor output natively |
| Monitoring | **Weights & Biases** | Free for students, standard in ML research |
| Version control | **Git + DVC** | DVC for data versioning alongside code |

**JAX is not used.** The previous draft assumed JAX without team discussion. PyTorch is the correct choice given existing implementation work.

---

## Architecture

### Overview

```
OMNI solar wind time series                AMPERE j_z spherical maps
[batch, 6, 5]                              [batch, n_pixels, 6]
      │                                           │
      ▼                                           ▼
  GRU Encoder                           Spectral Window σ_l
  (temporal context)                    (Gibbs suppression)
      │                                           │
      │ context vector                            ▼
      │ [batch, 128]              SphericalConvGRU hidden state h_0
      │                           initialized from OMNI context vector
      │                                           │
      └──────────────────────────────────────────►│
                                                  │
                              ┌───────────────────▼───────────────────┐
                              │         SFNO Block × 4                │
                              │                                        │
                              │  Forward SHT                          │
                              │  f(θ,φ) → f̂_l^m                      │
                              │         │                              │
                              │  SO(2) Spectral Filter                │
                              │  f̂_l^m · ĥ_l^m  (learned weights)    │
                              │         │                              │
                              │  Inverse SHT                          │
                              │  f̂_l^m → g(θ,φ)                      │
                              │         │                              │
                              │  Pointwise Linear + GELU              │
                              │  (applied independently at each pixel) │
                              └───────────────────┬───────────────────┘
                                                  │
                                                  ▼
                                     Pointwise Linear (C → 3)
                                     (one output per forecast horizon)
                                                  │
                                                  ▼
                                     Output: predicted j_z
                                     [batch, n_pixels, 3]
                                     t+1h, t+2h, t+3h
                                     Still on HEALPix sphere.
                                     No pooling. No flattening.
```

### Component 1: OMNI GRU Encoder

A standard single-layer GRU processes the OMNI solar wind time series:

```
Input:  [batch, 6, 5]   — 6 hours × 5 solar wind variables
GRU hidden size: 128
Output: final hidden state [batch, 128]   — solar wind context vector
```

This context vector is used to initialize the SphericalConvGRU hidden state, so the spherical recurrence starts from a state informed by incoming solar wind conditions. The OMNI data never passes through SHT — it has no spatial structure on the sphere.

### Component 2: Spectral Window (Applied Once, Before All Layers)

Before the input enters any SFNO layer, a fixed spectral window $\sigma_l$ is applied in the spherical harmonic domain to suppress Gibbs ringing. AMPERE current maps are dense near the auroral oval and near-zero elsewhere — this sharp spatial boundary causes ringing when transformed via SHT.

The window is applied as:

$$\tilde{f}_l^m = \sigma_l \cdot \hat{f}_l^m$$

where $\sigma_l$ is a derived (not learned) decay sequence. **Person 2 derives the correct $\sigma_l$ analytically** (see Math Deliverables). This is implemented as a fixed pre-multiplier in the first forward pass — one line of code, but the formula is non-trivial.

### Component 3: SphericalConvGRU

Replaces channel-stacking for temporal modeling. The hidden state is itself a field on S², updated at each timestep via spherical convolution. This keeps all temporal reasoning on the sphere — nothing is flattened.

```
For each timestep t = 1, ..., 6:
  Input field:   x_t,  shape [batch, n_pixels, C_in]
  Hidden state:  h_t,  shape [batch, n_pixels, C_hidden]
  
  Reset gate:    r_t = σ(SphConv([x_t, h_{t-1}], W_r))
  Update gate:   z_t = σ(SphConv([x_t, h_{t-1}], W_z))
  Candidate:     h̃_t = tanh(SphConv([x_t, r_t ⊙ h_{t-1}], W_h))
  New hidden:    h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t

where SphConv is one SFNO layer (SHT → spectral filter → inverse SHT)
All operations are pointwise or spherical — equivariance preserved throughout
```

Initial hidden state $h_0$ is a learned linear projection of the OMNI context vector, broadcast to all pixels.

### Component 4: SFNO Block (Core Spectral Layer)

Each SFNO block is the mathematical core. It runs in three steps:

**Step 1 — Forward SHT:**

$$\hat{f}_l^m = \int_{S^2} f(\theta,\phi)\, Y_l^m(\theta,\phi)\, d\Omega$$

Implemented via `torch-harmonics` `RealSHT`. Transforms the pixel-space field into spherical harmonic coefficients indexed by degree $l \in [0, L_{max}]$ and order $m \in [-l, l]$.

**Step 2 — SO(2)-Equivariant Spectral Filter:**

$$(\hat{f} \cdot \hat{h})_l^m = \hat{f}_l^m \cdot \hat{h}_l^m$$

Learnable complex weights $\hat{h}_l^m$ depend on **both** degree $l$ and order $m$. This is SO(2)-equivariant: longitude rotation acts on coefficients by $\hat{f}_l^m \mapsto e^{im\Delta\phi}\hat{f}_l^m$, and since the filter is diagonal in the $(l,m)$ basis, equivariance is preserved exactly.

This is strictly more expressive than SO(3)-equivariant filters (which require $\hat{h}_l^m = \hat{h}_l$, independent of $m$). We have $(L_{max}+1)^2$ free parameters per filter rather than $L_{max}+1$ — a direct consequence of using the correct symmetry group.

**Step 3 — Inverse SHT:**

$$g(\theta,\phi) = \sum_{l=0}^{L_{max}} \sum_{m=-l}^{l} (\hat{f}\cdot\hat{h})_l^m\, Y_l^m(\theta,\phi)$$

Implemented via `torch-harmonics` `InverseRealSHT`. Returns a field on the same HEALPix grid.

**Step 4 — Pointwise nonlinearity:**

GELU applied independently at each pixel. Pointwise operations commute with any spatial symmetry — equivariance is preserved.

Four SFNO blocks are stacked. Channel widths: 6 → 64 → 128 → 128 → 64 → 3.

### Component 5: Autoregressive Inference

At training time, all three forecast horizons (t+1h, t+2h, t+3h) are predicted in a single forward pass from the 6-hour input window.

At inference time, predictions are fed back autoregressively for extended forecasts beyond t+3h. Training uses **scheduled teacher forcing** — early in training, true $j_z$ values are fed back; later in training, model predictions are progressively substituted. This prevents error accumulation from dominating during inference.

---

## Loss Function

$$\mathcal{L} = \underbrace{\frac{1}{\sum_t w_t} \sum_{t} w_t \cdot \|j_z(t) - \hat{j}_z(t)\|_{L^2(S^2)}^2}_{\text{importance-weighted MSE}} + \lambda \underbrace{\sum_l (1+l(l+1))\|\hat{h}_l^m\|^2}_{H^1(S^2)\text{ regularization}}$$

**Term 1:** Storm hours are rare — quiet-time $j_z \approx 0$ dominates the dataset. Standard MSE ignores storms. Weights $w_t$ are derived from the inverse empirical density of $\|j_z\|$, reweighting the loss so the model cares equally about all storm intensities. Derived analytically by Person 2.

**Term 2:** The $H^1(S^2)$ Sobolev regularization on filter weights. The $(1+l(l+1))$ factor is the Laplace-Beltrami eigenvalue at degree $l$ — penalizing filters that produce spatially rough outputs in the geometrically correct sense. Replaces the ad hoc $l^2$ penalty. Derived analytically by Person 2.

---

## Baselines

All three are required before any results are meaningful.

| Baseline | What it tests |
|---|---|
| **Persistence:** $\hat{j}_z(t+1) = j_z(t)$ | Minimum bar — does the model add anything at all? |
| **Flat CNN:** same architecture on lat/lon grid, no SHT | Does the spherical inductive bias actually help? |
| **OMNI-only GRU:** predict from solar wind alone, no spatial input | Does the spatial field matter beyond the solar wind signal? |

---

## Team Roles

### Person 1: Data & Pipeline Engineer

**Owns:** Everything from raw AMPERE files to clean `[batch, n_pixels, 6]` tensors.

- Download AMPERE NetCDF files for 2010–2020 from JHU APL
- Convert from geographic to **AACGM magnetic coordinates** using `aacgmv2`
- Resample from 10-minute to 1-hour cadence
- Interpolate onto HEALPix N_side=32 grid using SciPy spatial interpolation
- Download and align OMNI solar wind data (already in geographic coordinates, no conversion needed)
- Align all timestamps to a common 1-hour grid
- Build PyTorch Dataset/DataLoader returning `(ampere_tensor, omni_tensor, target_tensor)`
- Implement data versioning with DVC
- **Sanity check:** visualize the March 2015 St. Patrick's Day storm on the HEALPix sphere. If the auroral oval is visible expanding equatorward in AACGM coordinates, the pipeline is correct.

**Known hard part:** The AACGM coordinate transformation is non-trivial — the magnetic pole is offset and tilted relative to the geographic pole, and the conversion is altitude-dependent. Budget extra time here.

### Person 2: Math Architect (Saksham)

**Owns:** The mathematical core of the model — derivations on paper that produce formulas entering the code, and implementation of the spectral layers.

Mathematical deliverables in order of dependency:

**1. SO(2) equivariance proof and filter characterization**
Prove that diagonal spectral filters $\hat{h}_l^m$ (depending on both $l$ and $m$) are exactly SO(2)-equivariant, that SO(2) is the correct symmetry group for AACGM-coordinated geomagnetic data, and that this filter class is strictly more expressive than Cohen's SO(3)-equivariant filters. Output: one-page theorem + proof for theory section of paper.

**2. $L_{max}$ derivation from AMPERE power spectrum**
Compute empirical power spectrum $S_l = \frac{1}{T}\sum_t \sum_m |\hat{j}_l^m(t)|^2$ from AMPERE data. Fit a decay model. Estimate noise floor from quiet-time variance. Derive the minimum $L_{max}$ such that $\sum_{l > L_{max}} S_l < \epsilon$ for a specified tolerance. Output: a specific number $L_{max}$ with written justification. **This derivation gates implementation — no SHT layer can be built until $L_{max}$ is known.**

**3. Gibbs window derivation**
Derive the spectral window $\sigma_l$ that suppresses ringing from AMPERE's sharp spatial boundaries. Prove the overshoot is bounded by $\epsilon$ under the derived window. Output: a formula for $\sigma_l$ and a bound. This is one line of code in the preprocessing step but the formula is non-trivial applied analysis.

**4. Importance-weighted loss**
Derive the weights $w_t = p_0 / p(\|j_z(t)\|)$ from the inverse empirical density of current density magnitude. Prove via change-of-measure that this is equivalent to minimizing MSE under a uniform distribution over storm intensities. Output: the weight formula and proof for the methods section.

**5. Sobolev $H^1(S^2)$ regularization**
Derive the regularization term from the $H^1(S^2)$ Sobolev norm. Show that $(1+l(l+1))$ is the Laplace-Beltrami eigenvalue and that this penalty correctly measures spatial roughness on S². Output: replaces the existing $l^2$ penalty with a derived and proven formula.

**6. Equivariance error metric**
Define $\epsilon_{eq} = \mathbb{E}_{\Delta\phi}\left[\|\Phi(\rho(\Delta\phi)f) - \rho(\Delta\phi)\Phi(f)\|_{L^2} / \|\Phi(f)\|_{L^2}\right]$. Prove it is zero for exact arithmetic under the chosen architecture. Bound contributions from batch norm, floating point, and HEALPix remapping. Output: a test function + theoretical predictions for the measured values.

Implementation responsibilities (after derivations):
- Implement `SphericalConvLayer` using `torch-harmonics`
- Implement spectral window pre-multiplier
- Implement `SphericalConvGRU` building on `SphericalConvLayer`
- Implement the full loss function including importance weighting and Sobolev regularization

### Person 3: MLOps & Visualization Engineer

**Owns:** Everything after the model produces output tensors.

- Training loop with scheduled teacher forcing
- Multi-step loss across all three forecast horizons simultaneously
- GPU memory management (gradient checkpointing if needed given SFNO memory cost)
- Weights & Biases logging: loss curves, equivariance error, per-storm skill scores
- Autoregressive inference loop for extended forecasts beyond t+3h
- 3D PyVista globe visualization mapping $j_z$ output onto Earth mesh with auroral oval overlay
- Baseline model implementations and evaluation
- Evaluation metrics: RMSE, skill score vs. persistence, storm-time vs. quiet-time separated scores

---

## Known Issues and Solutions

**Issue 1: Gibbs Phenomenon**
AMPERE current maps are dense near the auroral oval and near-zero elsewhere. The sharp boundary causes spectral ringing when passed through SHT. **Solution:** Person 2 derives the correct spectral window $\sigma_l$ analytically. Applied once before the first SFNO layer.

**Issue 2: $O(L^3)$ SHT Complexity**
Evaluating associated Legendre polynomials scales cubically with $L_{max}$. High resolution will hit GPU memory limits. **Solution:** $L_{max}$ is chosen analytically by Person 2 to be the minimum sufficient value, not an arbitrary high resolution. `torch-harmonics` uses optimized routines that reduce the practical cost significantly.

**Issue 3: Autoregressive Error Accumulation**
Errors at t+1h compound when fed back as input for t+2h, t+3h predictions. **Solution:** Scheduled teacher forcing during training (Person 3). Multi-step loss that evaluates all three horizons simultaneously rather than just t+1h.

**Issue 4: AACGM Coordinate Complexity**
The magnetic pole is offset from the geographic pole. Working in geographic coordinates would break SO(2) symmetry. **Solution:** Convert all AMPERE data to AACGM coordinates in the pipeline (Person 1) before any model input. The `aacgmv2` Python library handles this.

---

## Sequential Build Steps

```
NOW — Reading (parallel, 1–2 weeks)
  Person 2:  Cohen et al. 2018 (S²CNN, SHT, spectral filtering)
  Person 1:  DeepSphere paper + aacgmv2 documentation + AMPERE data format docs
  Person 3:  Scheduled teacher forcing + autoregressive training literature
  Milestone: group meeting — everyone explains what a spherical convolution
             does and why SO(2) is the right symmetry group.

STEP 1 — Data pipeline (Person 1 leads, ~2–3 weeks)
  Download AMPERE 2010–2020, convert to AACGM, resample to 1-hour,
  interpolate onto HEALPix N_side=32, download + align OMNI.
  Milestone: Dataset object working + St. Patrick's Day 2015 storm
             visible on sphere in AACGM coordinates.

STEP 2 — L_max derivation (Person 2, ~1 week, runs parallel to Step 1)
  Compute power spectrum from early AMPERE samples, fit decay,
  derive L_max analytically.
  Milestone: A number L_max with written justification.
  This gates Step 3 — no SHT layer can be built without it.

STEP 3 — Core layer implementation (Person 2 leads, ~2–3 weeks)
  Implement SphericalConvLayer, spectral window, SphericalConvGRU,
  full SFNO architecture, loss function.
  Milestone: Forward pass runs on dummy data. Loss goes down on toy example.

STEP 4 — Training infrastructure (Person 3 leads, parallel to Step 3)
  Training loop, teacher forcing schedule, W&B logging, baseline models.
  Milestone: Persistence baseline evaluated. Training loop tested on small run.

STEP 5 — Full training run (all, ~1–2 weeks)
  Train on 2010–2018, validate on 2019.
  Milestone: Validation loss below persistence baseline.

STEP 6 — Equivariance verification (Person 2, ~1 week)
  Measure epsilon_eq on trained model. Compare to theoretical bound.
  Milestone: Equivariance error measured and documented.

STEP 7 — Evaluation and ablations (all, ~1–2 weeks)
  Test on 2020. Run all three baselines. Storm-time vs quiet-time eval.
  Ablation: spherical vs flat CNN.
  Milestone: Full results table.

STEP 8 — Writeup (all, ~2 weeks)
  NeurIPS-style paper draft. Person 2 writes theory section.
  Person 1 writes data section. Person 3 writes experiments.
  Milestone: Submittable draft. Optional: arXiv preprint.
```

---

## What Is Locked

These decisions are final. Raising them again costs more time than they could save.

| Decision | Choice |
|---|---|
| Phenomenon | Birkeland field-aligned currents (AMPERE $j_z$) |
| Coordinate system | AACGM magnetic coordinates |
| Symmetry group | SO(2) — longitude rotation around magnetic axis |
| Framework | PyTorch (not JAX) |
| SHT library | torch-harmonics |
| Grid | HEALPix N_side=32 |
| Output type | Global spherical field map — no pooling, no scalar output |
| Temporal model | SphericalConvGRU (hidden state lives on sphere) |
| Solar wind input | OMNI via GRU encoder, conditioning only — not on sphere |
| Time range | 2010–2020 |

## What Remains Open

These are tuned during implementation — do not debate them now.

| Decision | When to decide |
|---|---|
| Number of SFNO blocks (currently 4) | After first training run |
| Channel widths | After first training run |
| GRU hidden size | After first training run |
| Regularization weight $\lambda$ | Sweep after baseline results |
| Teacher forcing schedule | Person 3 tunes during training |
| Extended forecast horizon beyond t+3h | After t+1h through t+3h are working |
| Gauge equivariant upgrade (Cohen 2019) | After paper draft — future work |
