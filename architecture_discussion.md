# S2Kp-Net: Architecture Decision for Role A

## What We Need to Decide

There are two valid but incompatible architectural directions. This doc lays them both out so we can align before anyone writes more code — because this decision affects what all three of us build.

---

## Option 1: S² Convolutions → Global Field Map (Current Build)

### What it outputs
A predicted magnetic field map for the entire globe, one hour from now.
`[batch, nlat, nlon, 3]` — three field components at every pixel.

### Architecture
```
Input: 6 hours of station readings interpolated onto a lat/lon grid
[batch, nlat, nlon, 18]        (6 hours × 3 components = 18)
        ↓
SphericalConvLayer(18 → 32) + ReLU
        ↓
SphericalConvLayer(32 → 64) + ReLU
        ↓
SphericalConvLayer(64 → 32) + ReLU
        ↓
Linear(32 → 3) applied at every pixel independently
        ↓
Output: predicted global field map
[batch, nlat, nlon, 3]
```

### The math in each SphericalConvLayer
Each layer does three steps:

1. **Forward SHT** — transform pixel-space signal into spherical harmonic coefficients via `RealSHT`:

$$\hat{f}_l^m = \int_{S^2} f(\theta, \phi) \, Y_l^m(\theta, \phi) \, d\Omega$$

2. **Spectral filtering** — multiply coefficients by learnable weights (this is the convolution):

$$(\hat{f} \cdot \hat{h})_l^m = \hat{f}_l^m \cdot \hat{h}_l$$

   Note: weights $\hat{h}_l$ depend only on degree $l$, not order $m$. This is what enforces rotational equivariance — the filter is the same regardless of where on the sphere you apply it.

3. **Inverse SHT** — transform back to pixel space via `InverseRealSHT`:

$$g(\theta, \phi) = \sum_{l=0}^{L_{max}} \sum_{m=-l}^{l} (\hat{f} \cdot \hat{h})_l^m \, Y_l^m(\theta, \phi)$$

### What Role A needs to derive for this option
- **L_max** — the spherical harmonic bandwidth. Controls spatial resolution vs parameter count. Should be derived from the spectral properties of the geomagnetic signal.
- **Weighted loss formula** — reweights storm hours vs quiet hours:

$$\mathcal{L} = \frac{1}{N}\sum_{t} w_t \cdot \|y_t - \hat{y}_t\|^2$$

  Where $w_t$ is larger during geomagnetic storm hours.

- **Spectral regularization term** — penalizes over-reliance on high-frequency components:

$$\mathcal{L}_{reg} = \lambda \sum_l l^2 \|\hat{h}_l\|^2$$

- **Equivariance error metric** — a test function to verify the trained model is actually equivariant.

### Library
`torch-harmonics` — PyTorch-native, GPU-compatible, autograd-compatible. Already implemented and tested in toy notebook.

---

## Option 2: S² → SO(3) Convolutions → Scalar Kp (Role A's Proposal)

### What it outputs
A single scalar Kp index (0–9) summarizing global geomagnetic activity.

### Architecture
```
Input: sparse station readings on S²
[batch, nlat, nlon, 18]
        ↓
S² Convolution Layer
(signal lives on sphere surface)
        ↓
S² → SO(3) Lifting Layer
(lifts features from sphere into the full rotation group)
[batch, SO(3) grid, channels]
        ↓
SO(3) → SO(3) Convolution Layer
(processes features in rotation space)
        ↓
Global Average Pooling
(collapses all spatial dimensions — required for SO(3) equivariance)
[batch, channels]
        ↓
Linear layer → scalar Kp value
[batch, 1]
```

### The key mathematical difference
SO(3) convolutions lift the signal from the 2D sphere surface into the 3D rotation group. The convolution theorem on SO(3) uses Wigner D-matrices instead of scalar spherical harmonics:

$$(\hat{f} * \hat{h})^l_{mn} = \sum_k \hat{f}^l_{mk} \cdot \hat{h}^l_{kn}$$

This is mathematically more powerful — full 3D rotational equivariance — but the global average pooling step required to achieve it destroys all spatial information. You get one number out, not a map.

### Library
`e2cnn` or `s2cnn` — neither is as well-maintained or PyTorch-native as `torch-harmonics`. Significantly harder to implement.

---

## The Core Tradeoff

| | Option 1 (S² → Map) | Option 2 (SO(3) → Scalar) |
|---|---|---|
| Output | Global field map `[nlat, nlon, 3]` | Scalar Kp index |
| Spatial info preserved | Yes | No — pooled away |
| Rotational equivariance | Partial (on S²) | Full (on SO(3)) |
| Library | `torch-harmonics` | `e2cnn` / `s2cnn` |
| Math complexity | Moderate | Very high |
| Current build status | Toy model working | Not started |
| Data needed | ~200-400 SuperMAG stations | 13 Intermagnet stations |

---

## Why This Decision Needs to Happen Now

Role B is currently downloading SuperMAG data for a global map model (~200-400 stations). If we switch to Option 2, they need 13 Intermagnet stations instead — completely different download. Role C (me) has a working `SphericalConvLayer` in `torch-harmonics` that would be thrown out if we go SO(3). Role A's math derivations need to target one of these two frameworks, not both.

**The scientific question is also different:**
- Option 1 answers: *what does the magnetic field look like everywhere on Earth one hour from now?*
- Option 2 answers: *how globally active is the magnetosphere in the next 3-6 hours?*

Option 1 is arguably more scientifically useful and more original. Option 2 has stronger mathematical novelty but is a harder build and a narrower output.

---

## Recommendation

Stick with Option 1. The math is solid, the implementation is already partially working, and a full global field map prediction is a stronger scientific contribution than a scalar index that existing models already predict reasonably well. Role A's derivations for L_max, weighted loss, spectral regularization, and the equivariance metric are all well-defined and non-trivial — plenty of mathematical substance for a paper.

If Role A wants to incorporate SO(3) ideas, one path forward is to use them as a theoretical motivation/framing in the paper while keeping the implementation on S² — which is common in the spherical ML literature.
