# Space Weather Prediction with Spherical CNNs
### Project Overview & Architecture Reference

---

## 1. Project Summary

A machine learning project to predict geomagnetic disturbances (Kp index) using a **Spherical Convolutional Neural Network** trained on globally distributed ground magnetometer data. The core motivation for a spherical architecture is that the input data is genuinely distributed across the surface of a sphere (Earth), and standard flat CNNs cannot correctly represent the geometry.

This is not a typical AI project. Spherical CNNs sit at the intersection of differential geometry, group representation theory, and deep learning — a combination that signals real depth to research internship interviewers and goes well beyond standard "I trained a CNN" projects.

---

## 2. Why This Is Impressive

- Spherical CNNs are not mainstream — most ML projects use flat architectures on standard datasets
- The math is graduate-level: spherical harmonics, SO(3) representation theory, equivariance proofs
- Space weather prediction is scientifically meaningful and operationally important (power grids, satellites, GPS)
- Using spherical geometry as an inductive bias for geophysical data is a genuinely novel angle in the space weather literature
- Potentially publishable as an arXiv preprint

---

## 3. Prediction Task

**Goal:** Predict the **spatial distribution of geomagnetic disturbance** across Earth's surface 1–3 hours ahead.

**Why spatial output (not scalar Kp):**
Predicting a single global Kp scalar collapses all spatial information at the end of the network, which wastes the spherical architecture. Instead, predict a **spherical output field** — e.g. the disturbance magnitude at every point on the globe, or the auroral electrojet intensity distribution. This makes the model truly sphere-to-sphere:

```
Input:  spherical field of magnetic disturbance at time t (and recent history)
Output: predicted spherical disturbance field at t+1h, t+2h, t+3h
```

This is what spherical CNNs are designed for. The spatial *where* matters as much as the *what*, and the geometry is preserved end to end.

**Possible spatial output targets (pick one):**
- AL/AU auroral electrojet index at each latitude band
- Auroral oval boundary location and intensity (expands equatorward during storms)
- Global magnetic disturbance field matching SuperMAG readings at future time

---

## 4. Data

### Primary: SuperMAG
- **What it is:** A network of ~700 ground-based magnetometers at fixed lat/lon positions across Earth's surface
- **What it measures:** At each timestamp, each station reports the local magnetic field disturbance vector — the deviation from quiet-time baseline — in three components: ΔB_north, ΔB_east, ΔB_vertical
- **Why it's the right data:** The stations are distributed across a sphere. At any timestamp you have a sparse spatial snapshot of how the magnetic field is disturbed across the globe. This is exactly a function on S² — the input a spherical CNN is built to process
- **Resolution:** 1-minute native; resample to hourly for this project
- **Access:** Free academic registration at supermag.jhuapl.edu
- **Key caveat:** Stations cluster in the Northern Hemisphere. The Southern Hemisphere and oceans are sparse. Handling this in the interpolation step is a real design decision

### Prediction Target: Kp Index
- **What it is:** A 0–9 scale measuring global geomagnetic disturbance, reported every 3 hours (1-hour provisional version available from NOAA)
- **Physical meaning:** Kp is itself derived from ground magnetometer readings — so there's a deep physical connection between the SuperMAG input and the Kp target
- **Source:** GFZ Potsdam (definitive) or NOAA SWPC (provisional 1-hour)

### Optional Later Addition: L1 Satellite Data (OMNI)
- **What it is:** Solar wind measurements from ACE/DSCOVR at the L1 Lagrange point, ~1.5 million km sunward
- **What it measures:** Solar wind magnetic field (especially Bz), proton density, velocity, dynamic pressure
- **Why it helps:** Solar wind takes 30–60 minutes to travel from L1 to Earth, giving advance warning of incoming disturbances. Southward Bz is the primary driver of geomagnetic storms
- **Why it's optional/secondary:** It's a single point measurement with no spatial structure, so it doesn't use the spherical architecture. It acts as a conditioning signal ("here's what's incoming") rather than a spatial input
- **Source:** NASA OMNI dataset at omniweb.gsfc.nasa.gov — already propagated to Earth's bow shock, clean and gap-filled

---

## 5. Architecture

### Core Design: Sphere-to-Sphere

```
SuperMAG spherical grid (t, t-1, t-2, ...)
         |
         | [channels: ΔB_north, ΔB_east, ΔB_vertical × N time steps]
         ↓
  Spherical CNN Layer 1     ← learns spatial filters on S²
         ↓
  Spherical CNN Layer 2     ← detects higher-order spatial patterns
         ↓
       ...
         ↓
  Spherical CNN Layer N
         ↓
  Predicted spherical disturbance field at t+1h, t+2h, t+3h

Loss: difference between predicted field and actual SuperMAG readings at future times
```

### What Spherical Convolution Actually Does

In a standard 2D CNN, a filter slides across a flat image detecting local patterns regardless of position. A spherical CNN does the same thing but on a sphere — a filter applied at the north pole behaves the same way as the same filter applied at the equator, respecting the geometry of S² rather than pretending it's flat.

The filters are defined in terms of **spherical harmonics** — the natural basis functions for signals on a sphere, analogous to Fourier modes on a line. Convolution in the spatial domain corresponds to pointwise multiplication in the spherical harmonic domain (the spectral domain), which is how the computation is actually implemented.

**Equivariance:** If you rotate the input field, the output rotates correspondingly. This is a mathematically provable property, not an approximation. For geophysical data on a globe, this is a natural and well-motivated inductive bias.

### The Sweet Spot: What You Build vs. What You Use

```
Too easy                                                  Too hard
─────────────────────────────────────────────────────────────────
SphericalCNN(data) → done   |   implement SO(3) FFT from scratch
                            ↑
                       YOUR TARGET:
           Use torch-harmonics or healpy for spherical
           harmonic transforms (like using PyTorch's conv2d
           rather than writing CUDA kernels), but build
           the architecture, training loop, loss function,
           data pipeline, and evaluation yourself
```

### Key Libraries
- `torch-harmonics` — spherical harmonic transforms in PyTorch
- `healpy` — HEALPix spherical grid (standard in cosmology/climate science)
- PyTorch — everything else

---

## 6. Data Pipeline

This is 30–40% of the total project work. Do not underestimate it.

### Steps

**Step 1 — Pick a time range**
2010–2020 recommended: covers multiple solar cycles, good station coverage, enough storms to learn from

**Step 2 — Download and align**
Download SuperMAG station files, OMNI data, and Kp index. Resample everything to a common 1-hour timestamp grid. SuperMAG, OMNI, and Kp all have different native resolutions.

**Step 3 — Build the spherical grid**
This is the critical step. At each timestamp, interpolate ~700 sparse station readings onto a HEALPix grid — a standard way of dividing a sphere into equal-area pixels. The output is a tensor of shape `[n_pixels, 3]` (three field components) at each timestamp.

Key design questions for the interpolation:
- What interpolation method? (nearest neighbor, inverse distance weighting, spherical spline)
- How do you handle the hemisphere imbalance? (more stations in the North)
- What do you do for ocean grid points with no nearby stations?

These are real scientific and mathematical decisions, not just engineering choices.

**Step 4 — Assemble dataset**
Each training sample:
- Input: spherical grid tensor for the last N hours (N=6 is a reasonable start), shape `[N, n_pixels, 3]`
- Target: spherical grid tensor at t+1h, t+2h, t+3h

**Step 5 — Sanity check**
Before any modeling, visualize the spherical field during a known major storm (e.g. Halloween storms October/November 2003). If you can see the disturbance clearly in your assembled data, the pipeline is working.

---

## 7. Training & Evaluation

### Baselines (Required — results mean nothing without them)
- **Persistence baseline:** predict field(t+1) = field(t). Surprisingly hard to beat and the standard first comparison in space weather forecasting
- **Flat CNN baseline:** same architecture but treating the spherical grid as a flat image. Directly tests whether the spherical inductive bias helps
- **Linear regression on station averages:** establishes what you get without spatial reasoning

### Loss Function
Mean squared error between predicted and actual spherical field at future times. Consider a weighted version that penalizes storm-time errors more heavily (storms are rare so MSE naturally under-weights them — the math person can derive the appropriate reweighting from the empirical Kp distribution).

### Infrastructure
- Google Colab Pro is sufficient for early experiments
- University HPC cluster for full training runs
- Log everything with Weights & Biases (free for students, looks professional)

### Evaluation Metrics
- RMSE vs. persistence baseline
- Skill score relative to persistence (standard in space weather forecasting)
- Separate evaluation during storm periods (Kp > 5) vs. quiet periods
- Ablation: spherical CNN vs. flat CNN on same data

---

## 8. Team Delegation

| Phase | Math Person | AI Friend 1 | AI Friend 2 |
|---|---|---|---|
| Foundation | Lead theory sessions, present spherical harmonics and equivariance to group | Architecture research | Data source research |
| Data pipeline | Design interpolation method formally, advise on hemisphere imbalance | Help with grid/format choices | **Lead** — download, clean, normalize, align |
| Architecture | Advise on equivariance, write theory section | **Lead** — implement model layers | Help with input tensor formatting |
| Training | Analyze loss function, derive storm reweighting | **Lead** — train, tune, run ablations | Run and document baseline models |
| Writeup | Math sections, proofs, theory derivations | Methods + architecture diagrams | Data section + experiments table |

---

## 9. Math Person's Role (Applied, Not Decorative)

The math contribution should produce specific numbers, equations, or constraints that directly change something in the model or training script — not background reading that sits in an appendix.

**Concrete contributions:**

**Spherical harmonic truncation (L_max choice)**
The spherical harmonic expansion is infinite but truncated at maximum degree L_max. This is a bias-variance tradeoff you can analyze formally. Geomagnetic fields have known spectral properties — they decay at a known rate with degree. Derive the approximation error as a function of L_max for your specific signal and use that to choose L_max analytically rather than grid-searching it.

**Weighted loss function**
Geomagnetic storms follow a heavy-tailed distribution — quiet periods dominate, storms are rare. Standard MSE under-weights storms. Formally derive a weighted loss from the empirical Kp distribution that re-weights to care more about storm prediction. This is a clean variational/statistical argument that directly produces a change to the loss function implementation.

**Spectral regularization**
Since the data has known physical smoothness properties, derive a regularization term in the spectral domain (penalizing high-degree spherical harmonic coefficients). This is mathematically better motivated than standard L2 weight decay and you can prove what it's doing.

**Equivariance error measurement**
The spherical CNN should be SO(3)-equivariant in theory, but implementation choices can break this approximately. Derive a metric for equivariance error, measure it empirically across architectural choices, and use that to guide architecture decisions.

---

## 10. Project Timeline (Part-Time During School)

| Phase | Duration | Deliverable |
|---|---|---|
| Foundation — read Cohen et al. 2018, align on data | 2–3 weeks | Everyone understands the theory; data sources identified |
| Data pipeline — download, align, interpolate, HEALPix grid | 2–3 weeks | Working dataset with sanity-checked visualizations |
| Architecture — implement spherical CNN in PyTorch | 3–4 weeks | Training model with loss going down |
| Baselines + ablations | 2 weeks | Comparison table showing spherical CNN vs. flat CNN vs. persistence |
| Writeup | 2 weeks | NeurIPS-style paper draft; optional arXiv preprint |

**Total: ~3–4 months part-time, or ~6 weeks compressed over summer**

---

## 11. Key Papers to Read

- **Cohen et al. 2018** — *Spherical CNNs* — the foundational paper; defines the architecture and equivariance properties
- **Cohen et al. 2019** — *Gauge Equivariant CNNs* — more advanced, fiber bundles and differential geometry (optional, for the math person)
- Any existing ML-for-space-weather papers using standard architectures — these are your baselines to beat and cite

---

## 12. What Makes This Brag-Worthy

When explaining to anyone — recruiter, professor, PhD interviewer:

> "We built a spherical CNN that learns to predict the global geomagnetic disturbance field on the surface of Earth, using a network of 700 ground magnetometers as spatially distributed input. The spherical architecture is motivated by SO(3) equivariance — rotating Earth's coordinate frame leaves the physics unchanged, and our model respects that symmetry by construction. Standard flat CNNs can't do this correctly."

That's a sentence that gets attention.
