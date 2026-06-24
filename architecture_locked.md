# S2Kp-Net: Settled Architecture & Build Trajectory

**Status:** Architecture locked. No more pivots. Build from here sequentially.

---

## What We Are Building

A sphere-to-sphere neural network that takes the last 6 hours of global geomagnetic field measurements as a spatially distributed signal on S², and predicts what that field will look like 1–3 hours from now — everywhere on Earth simultaneously.

**The scientific question:**
> Given how Earth's magnetic field is disturbed right now and in the recent past, what will the global disturbance pattern look like in 1, 2, and 3 hours?

**Why spherical:** The data is genuinely a field on a sphere. A storm is spatially continuous — it propagates across the globe with structure. A flat CNN distorts the geometry near the poles. An S2CNN sees the full spherical pattern at once and respects the geometry exactly.

**Why SO(2):** The physics is invariant to longitude rotations — a storm at 0° longitude and the same storm at 90° longitude are the same physical event. North and south poles are physically special (auroral oval, field-aligned currents), so full SO(3) equivariance would be wrong — it would force the model to treat the poles identically to the equator. SO(2), the group of rotations around Earth's axis, is the correct and complete symmetry for this problem.

---

## Data

### Source: SuperMAG
- ~700 ground magnetometer stations at fixed lat/lon coordinates across Earth
- Each station reports the magnetic field disturbance vector: ΔB_north, ΔB_east, ΔB_vertical
- Native resolution: 1-minute. We resample to **1-hour**.
- Access: free academic registration at supermag.jhuapl.edu
- Time range to download: **2010–2020** (covers multiple solar cycles, enough major storms)

### Grid: HEALPix
- Stations are sparse and irregularly distributed. We interpolate onto a HEALPix grid — a standard equal-area spherical pixelization used in cosmology and climate science.
- Resolution: **N_side = 32** (~12,288 pixels). Enough to resolve storm spatial structure; cheap enough to train on.
- Library: `healpy` handles the grid. `torch-harmonics` handles the spherical transforms on it.
- Key design decision (math person): what interpolation method, and how to handle the Northern Hemisphere station bias. This is not trivial and goes in the paper.

### Input tensor per sample
```
6 timesteps × 3 field components = 18 channels
Shape: [batch, n_pixels, 18]    where n_pixels = 12,288 (HEALPix N_side=32)
```

### Target tensor per sample
```
Predicted field at t+1h, t+2h, t+3h
Shape: [batch, n_pixels, 3, 3]   (3 future steps × 3 field components)
or trained as 3 separate heads, one per forecast horizon
```

### Sanity check before any modeling
Visualize the assembled spherical field during the Halloween storms (Oct–Nov 2003). If you can see the disturbance intensify and spread at high latitudes in your data, the pipeline is correct.

---

## Architecture

```
INPUT
[batch, n_pixels, 18]
6 hours × 3 components stacked as channels
All on the HEALPix sphere — nothing flattened
         │
         ▼
┌─────────────────────────────────────┐
│  SphericalConvLayer  (18 → 64)      │
│  1. Forward SHT: pixel → spectral   │
│  2. Spectral filter ĥ_l^m           │  ← SO(2)-equivariant
│  3. Inverse SHT: spectral → pixel   │     ĥ depends on l AND m
└─────────────────────────────────────┘
         │
       ReLU  (pointwise — preserves equivariance)
         │
         ▼
┌─────────────────────────────────────┐
│  SphericalConvLayer  (64 → 128)     │
└─────────────────────────────────────┘
         │
       ReLU
         │
         ▼
┌─────────────────────────────────────┐
│  SphericalConvLayer  (128 → 64)     │
└─────────────────────────────────────┘
         │
       ReLU
         │
         ▼
┌─────────────────────────────────────┐
│  Pointwise Linear  (64 → 9)         │
│  Applied independently at every     │
│  pixel — no pooling, no flattening  │
└─────────────────────────────────────┘
         │
         ▼
OUTPUT
[batch, n_pixels, 9]
Reshape to [batch, n_pixels, 3, 3]
= 3 forecast horizons × 3 field components
Still on the sphere. Spatial structure fully preserved.
```

### What each SphericalConvLayer does

The three steps inside each layer are the core of the math:

**Step 1 — Forward Spherical Harmonic Transform (SHT):**

$$\hat{f}_l^m = \int_{S^2} f(\theta, \phi)\, Y_l^m(\theta, \phi)\, d\Omega$$

Decomposes the field into spherical harmonic coefficients, indexed by degree $l$ and order $m$. These are the Fourier modes on S² — eigenfunctions of the Laplace-Beltrami operator $\Delta_{S^2}$.

**Step 2 — Spectral filtering (the convolution):**

$$(\hat{f} \cdot \hat{h})_l^m = \hat{f}_l^m \cdot \hat{h}_l^m$$

Multiply each coefficient by a learnable weight. The weights $\hat{h}_l^m$ depend on both degree $l$ and order $m$ — this is SO(2) equivariance. (Cohen's SO(3) filters depend only on $l$, which is more constrained and less expressive for our problem.)

**Step 3 — Inverse SHT:**

$$g(\theta, \phi) = \sum_{l=0}^{L_{\max}} \sum_{m=-l}^{l} (\hat{f} \cdot \hat{h})_l^m\, Y_l^m(\theta, \phi)$$

Maps back to pixel space. Output is a field on the same HEALPix sphere.

**Library:** `torch-harmonics` for SHT/inverse SHT. PyTorch-native, autograd-compatible, GPU-ready.
**DeepSphere:** Reference for HEALPix grid setup and how to handle irregular station interpolation onto the grid. Architecture implementation follows the spectral approach above, not DeepSphere's graph convolution approach — but DeepSphere is the right paper for the data pipeline design.

### Temporal handling

Time is handled by **channel stacking** — 6 hours of data enter as 18 input channels. The S2CNN sees all timesteps simultaneously and learns temporal patterns through the channel dimension. This is the standard approach in spherical weather modeling.

**Upgrade path (implement only if channel stacking is clearly the bottleneck):** Replace with a Spherical ConvLSTM — a recurrent model whose hidden state is itself a field on S², updated at each timestep via spherical convolution. Never flattens. More complex to implement; defer until after baseline results.

---

## Loss Function

$$\mathcal{L} = \frac{1}{N} \sum_t w_t \cdot \|y_t - \hat{y}_t\|_2^2 \;+\; \lambda \sum_l (1 + l(l+1)) \|\hat{h}_l^m\|^2$$

Two terms:

**Term 1 — Importance-weighted MSE:**
$w_t$ is larger during geomagnetic storm hours, derived from the inverse empirical density of Kp. Prevents the model from ignoring storms because quiet hours dominate the dataset. The exact formula is derived by the math person (see below).

**Term 2 — Sobolev $H^1(S^2)$ spectral regularization:**
Penalizes filters that produce spatially rough outputs. The $(1 + l(l+1))$ factor is the eigenvalue of $\Delta_{S^2}$ at degree $l$ — geometrically correct, not ad hoc. Replaces the arbitrary $l^2$ penalty in the earlier draft.

---

## Math Person's Deliverables

These are not optional theory. Each one produces a specific artifact that enters the code or paper directly.

| Derivation | What it produces | Where it goes |
|---|---|---|
| SO(2) equivariance proof | Theorem that $\hat{h}_l^m$ depending on both $l,m$ is exactly SO(2)-equivariant | Theory section of paper |
| $L_{\max}$ from spectral analysis | A specific number, derived from empirical power spectrum of SuperMAG data | Model config |
| Importance-weighted loss | Formula for $w_t$ from inverse Kp density, with proof of what it minimizes | Loss function implementation |
| Sobolev $H^1(S^2)$ regularization | Replaces $l^2$ penalty with $(1+l(l+1))$ penalty, with geometric proof | Loss function implementation |
| Interpolation scheme for HEALPix | Justified choice of method + analysis of Northern Hemisphere bias | Data pipeline + paper |
| Equivariance error metric $\epsilon_{eq}$ | A test function measuring actual SO(2) equivariance of trained model | Evaluation code |

The SO(2) proof, the $L_{\max}$ derivation, and the Sobolev regularization are the three that matter most and should be done in that order. Start after finishing Cohen.

---

## Baselines (Required)

Results mean nothing without something to beat.

1. **Persistence:** predict field$(t+1)$ = field$(t)$. The standard first bar in space weather forecasting.
2. **Flat CNN:** same architecture but on a lat/lon grid without spherical transforms. Directly tests whether the spherical inductive bias helps.
3. **Station average only:** predict from global mean disturbance, no spatial structure. Tests whether spatial modeling adds anything at all.

---

## Sequential Build Trajectory

Do these in order. Do not start the next step until the current one is done.

```
STEP 1 — Reading (now, parallel)
   Math person:      Cohen et al. 2018 (spherical CNNs)
   Data friend:      RNN survey (for context; ConvLSTM if we upgrade later)
   Impl friends:     DeepSphere paper (HEALPix grid, implementation patterns)
   Deliverable: everyone can explain what a spherical convolution does
   and why SO(2) is the right symmetry. One group meeting to align.

STEP 2 — Data pipeline
   Owner: data friend, with math person advising on interpolation
   Tasks: download SuperMAG 2010–2020, resample to hourly,
          interpolate onto HEALPix N_side=32, assemble [n_pixels, 18] tensors,
          download Kp index as target, align timestamps
   Deliverable: a dataset object that returns (input_tensor, target_tensor)
                and a visualization of the Halloween 2003 storm on the sphere

STEP 3 — L_max derivation
   Owner: math person
   Tasks: compute empirical power spectrum S_l from assembled SuperMAG data,
          fit decay model, identify noise floor, set L_max analytically
   Deliverable: a number L_max with a written justification (1–2 pages)
   This must happen before model implementation — L_max sets the SHT resolution

STEP 4 — Model implementation
   Owner: impl friends, with math person advising on filter design
   Tasks: implement SphericalConvLayer using torch-harmonics,
          stack into full architecture above, verify forward pass runs,
          implement loss function with importance weighting and Sobolev regularization
   Deliverable: model that takes [batch, n_pixels, 18] and outputs [batch, n_pixels, 9]

STEP 5 — Equivariance verification
   Owner: math person (metric definition) + impl (running it)
   Tasks: define epsilon_eq, implement as a test, run on untrained and trained model
   Deliverable: equivariance error numbers before and after training

STEP 6 — Training and baselines
   Owner: impl friends
   Tasks: train on 2010–2018, validate on 2019, test on 2020,
          implement and evaluate all three baselines,
          log everything with Weights & Biases
   Deliverable: results table comparing S2CNN vs baselines

STEP 7 — Analysis and writeup
   Owner: everyone
   Tasks: ablation (spherical vs flat CNN), storm-period vs quiet-period eval,
          math person writes theory section, impl writes methods + experiments
   Deliverable: NeurIPS-style paper draft
```

---

## What Is Locked vs. Open

**Locked — do not revisit:**
- Output is a spherical field map, not a scalar Kp index
- SO(2) equivariance, not SO(3)
- `torch-harmonics` for spectral operations
- HEALPix N_side=32 grid
- Channel stacking for time (18 channels)
- SuperMAG 2010–2020 as primary data source

**Open — decide during implementation:**
- Exact number of layers and channel widths (tune after baseline)
- Whether to add ConvLSTM (only if temporal modeling is clearly the bottleneck)
- Whether to add L1 satellite data as a later extension
- Forecast horizon (start with t+1h only, add t+2h and t+3h once working)
