# S2Kp-Net: Architecture Decision — Role A Response

## Decision: Option 1 (S² → Global Field Map), with a precise symmetry correction

We go with Option 1. The implementation direction, the library choice (`torch-harmonics`), and the output format `[batch, nlat, nlon, 3]` are all confirmed. Role B should keep downloading SuperMAG data. Role C's `SphericalConvLayer` stays.

However the symmetry framing in the original doc needs to be corrected, because it affects what the math person derives and how we describe the model in a paper. This document replaces the characterization of Option 1 as having "partial rotational equivariance" — that framing is imprecise and undersells both the architecture and the theory.

---

## Symmetry: Why SO(2), Not SO(3)

### The domain vs. data distinction

The original doc's comparison table says Option 1 has "partial (on S²)" rotational equivariance versus Option 2's "full (on SO(3))." This makes SO(3) sound strictly better, but that framing conflates two different things:

- **Symmetry of the domain** — S² as a geometric object is acted on by SO(3). Any rotation in 3D space maps the sphere to itself.
- **Symmetry of the data distribution** — the actual geomagnetic field does *not* have SO(3) symmetry. The poles are physically special. The auroral electrojet sits at ~70° magnetic latitude by physics, not convention. Rotating Earth 90° so the poles sit at the equator would produce a completely different field distribution.

The right symmetry group to build into the architecture is the one the **data distribution** respects — and that is SO(2), the group of rotations around Earth's rotation axis (longitude shifts). Shifting all longitudes by the same amount leaves the physics unchanged. Tilting Earth sideways does not.

Enforcing full SO(3) equivariance would actively hurt prediction because it forces the model to treat the poles and the equator identically — symmetrizing over a distinction the physics requires the model to learn. Option 2 is not mathematically stronger for this problem; it imposes the wrong symmetry.

### The image CNN analogy, answered precisely

The original doc implicitly makes the analogy: just as CNNs build in translation equivariance regardless of image content, shouldn't we build in SO(3) equivariance regardless of data content?

The analogy breaks down because for natural images, the symmetry of the grid (ℤ²) and the symmetry of the data distribution (approximately translation-invariant) coincide. A cat in the corner looks like a cat in the center. For geomagnetic data on S², these do not coincide — the domain has SO(3) symmetry but the data distribution has only SO(2) symmetry. Building in SO(3) would be like building rotational equivariance into a face detector: technically valid for the domain but wrong for the data, because faces have a preferred orientation.

### What SO(2) equivariance means concretely

SO(2) acting on S² is the subgroup of SO(3) that fixes the north-south axis — it rotates the sphere around that axis, shifting φ (longitude) by a constant while leaving θ (colatitude) unchanged. A network equivariant to this satisfies:

$$\Phi(f(\theta, \phi + \Delta\phi)) = \Phi(f)(\theta, \phi + \Delta\phi) \quad \forall \Delta\phi \in [0, 2\pi)$$

Rotating the input by any longitude offset rotates the output by the same offset. This is a precise, provable, non-trivial symmetry — not a weakened version of SO(3).

---

## Why Spherical Harmonics and Cohen Still Apply

The original doc associates spherical harmonics with SO(3) and implies they are less relevant if we drop SO(3). This is wrong.

Spherical harmonics $Y_l^m(\theta, \phi)$ are eigenfunctions of the Laplace-Beltrami operator $\Delta_{S^2}$ — they are the natural basis for any square-integrable function on S², independent of which symmetry group you care about. The SHT and inverse SHT in each `SphericalConvLayer` use spherical harmonics because they are the Fourier basis on S², full stop.

Cohen 2018 is still the foundational reference. You read it to understand:
- Why spherical harmonics are the right basis (they're the irreducible representations of SO(3), which contains SO(2) as a subgroup — the basis is the same)
- What the convolution theorem on S² says and why spectral filtering is equivariant
- The relationship between SO(2) and SO(3) and why our setting is a restricted case

You then specialize Cohen's framework to SO(2) — which is itself a mathematical exercise, not just reading. The Cohen paper covers the general case; deriving the SO(2) specialization and proving it's the right restriction for geophysical data is original work that goes in our paper's theory section.

The key simplification when you specialize to SO(2): the spectral filter weights $\hat{h}_l^m$ in the full SO(3) case couple different $m$ modes through Wigner D-matrices. In the SO(2) case, different $m$ modes decouple — each longitude frequency $e^{im\phi}$ transforms independently under longitude rotation. This means:

$$(\hat{f} \cdot \hat{h})_l^m = \hat{f}_l^m \cdot \hat{h}_l^m$$

where $\hat{h}_l^m$ can now depend on both $l$ and $m$, not just $l$. This is actually a **strictly more expressive filter** than the SO(3)-equivariant one (where $\hat{h}$ depends only on $l$), because we are not over-constraining the model with a symmetry the data doesn't have. We get more parameters, more expressivity, and a provably correct symmetry — all at once.

This distinction — and the proof that it follows from the representation theory of SO(2) on L²(S²) — is Role A's first theoretical contribution to the paper.

---

## Architecture (Confirmed, With Precise Symmetry Description)

```
Input: 6 hours of SuperMAG readings interpolated onto HEALPix/lat-lon grid
[batch, nlat, nlon, 18]        (6 hours × 3 field components = 18 channels)
         |
         | Each channel is a scalar field on S²
         ↓
SphericalConvLayer(18 → 32)
  ├─ Forward SHT: f(θ,φ) → f̂_l^m
  ├─ Spectral filter: f̂_l^m · ĥ_l^m   [SO(2)-equivariant: ĥ depends on l and m]
  └─ Inverse SHT: back to pixel space
+ ReLU (pointwise, preserves equivariance)
         ↓
SphericalConvLayer(32 → 64) + ReLU
         ↓
SphericalConvLayer(64 → 32) + ReLU
         ↓
Linear(32 → 3) applied pointwise at every pixel
         ↓
Output: predicted global magnetic field map at t+1h, t+2h, t+3h
[batch, nlat, nlon, 3]
```

**Equivariance preserved throughout:** every operation (SHT, spectral filter, inverse SHT, pointwise nonlinearity, pointwise linear) commutes with SO(2) longitude rotations. No pooling. No spatial information lost. The output is a field on S², not a scalar.

**Library:** `torch-harmonics` — confirmed. PyTorch-native, autograd-compatible, GPU-compatible. No change from current build.

---

## Output Goal

**What the model answers:** *What does the global geomagnetic disturbance field look like at every point on Earth's surface 1–3 hours from now?*

The output is a tensor `[batch, nlat, nlon, 3]` — three magnetic field disturbance components (ΔB_north, ΔB_east, ΔB_vertical) at every grid point on the sphere. This is a genuine sphere-to-sphere prediction. Spatial structure is preserved end to end.

Scalar Kp can be computed from this output as a diagnostic (average disturbance magnitude), but it is not the primary target. The spatial map is more scientifically informative, more novel relative to the existing literature, and is the output that actually justifies using a spherical architecture.

---

## Role A: Mathematical Derivations

These are not supplementary — they directly determine implementation choices and go in the paper's methods section. Each produces a specific formula or number that changes something in the code.

### 1. SO(2) Equivariance Proof and Filter Characterization

**What:** Prove formally that spectral filters of the form $\hat{h}_l^m$ (depending on both degree $l$ and order $m$) give SO(2) equivariance but not SO(3) equivariance, and that this is the maximal filter class consistent with our symmetry requirement.

**Why it matters:** This is the theoretical justification for the architecture. It's the argument that separates "we used a spherical library" from "we made a principled symmetry choice." It requires understanding the decomposition of L²(S²) into irreducible representations of SO(2) — each subspace $V_m = \{f : f(\theta, \phi) = g(\theta)e^{im\phi}\}$ transforms as a one-dimensional representation of SO(2). Filters that preserve each $V_m$ independently are exactly the SO(2)-equivariant ones.

**Output:** A theorem statement and proof that goes in the paper. Also the justification for why $\hat{h}_l^m$ depends on $m$ (more expressive than Cohen's SO(3) filters, not less).

### 2. L_max Derivation from Spectral Analysis

**What:** Derive the optimal spherical harmonic bandwidth $L_{max}$ analytically from the spectral properties of the geomagnetic field.

**How:** The power spectrum of the geomagnetic field is known to decay with degree $l$. For the external (magnetospheric) field relevant to SuperMAG readings, the spectrum falls off roughly as a power law. Fit this empirically from the SuperMAG data:

$$S_l = \sum_{m=-l}^{l} |\hat{f}_l^m|^2$$

Find the degree $l^*$ at which $S_l$ drops below the noise floor (estimated from quiet-time variance). Set $L_{max} = l^*$.

This is an analytic derivation, not a hyperparameter search. It requires fitting a spectral model, estimating the noise floor, and proving that truncation at $l^*$ bounds the approximation error by a specified tolerance $\epsilon$:

$$\|f - f_{L_{max}}\|_{L^2(S^2)}^2 = \sum_{l > L_{max}} S_l < \epsilon$$

**Output:** A specific number $L_{max}$ with a proof that it's right. Goes directly into the model config.

### 3. Importance-Weighted Loss Derivation

**What:** Derive the correct storm-weighting formula from first principles, not ad hoc.

**How:** Standard MSE minimization is equivalent to maximum likelihood under a homoskedastic Gaussian. The problem is that the empirical distribution $p(\text{Kp})$ is heavily skewed — quiet hours (Kp < 3) are ~80% of all observations. This means unweighted MSE effectively ignores storms.

The fix is importance weighting. Define:

$$w_t = \frac{1}{p(\text{Kp}_t) \cdot Z}$$

where $p$ is the empirical marginal density of Kp (estimated from the training set) and $Z$ is a normalizing constant. The weighted loss:

$$\mathcal{L} = \frac{1}{N} \sum_t w_t \cdot \|y_t - \hat{y}_t\|_2^2$$

is equivalent to minimizing expected MSE under a uniform distribution over Kp values. Prove this formally: show that the minimizer of the weighted loss is the same as the minimizer of the unweighted loss under a resampled dataset with uniform Kp coverage. This is a clean result in statistical decision theory.

**Output:** The exact formula for $w_t$, a proof of what it's doing, and the implementation (a lookup table from Kp value to weight, computed once from the training set).

### 4. Sobolev Spectral Regularization

**What:** Replace the ad hoc regularizer $\mathcal{L}_{reg} = \lambda \sum_l l^2 \|\hat{h}_l\|^2$ with the geometrically correct one.

**Why the current formula is wrong:** The $l^2$ penalty is not motivated by any geometric or physical argument — it's a guess that higher-degree modes should be penalized more. The correct penalty comes from the Sobolev theory on S².

**Derivation:** The $H^s(S^2)$ Sobolev norm of order $s$ is:

$$\|f\|_{H^s}^2 = \sum_{l=0}^{\infty} (1 + l(l+1))^s \sum_{m=-l}^{l} |\hat{f}_l^m|^2$$

This measures the smoothness of $f$ on $S^2$ in a geometrically invariant way — it is the correct generalization of the Sobolev norm from flat space to the sphere. Setting $s = 1$ gives the $H^1(S^2)$ regularizer, which penalizes spatial roughness (not just magnitude of high-degree coefficients). The $l(l+1)$ factor is the eigenvalue of the Laplace-Beltrami operator $\Delta_{S^2}$ at degree $l$, so the $H^1$ norm is exactly:

$$\|f\|_{H^1}^2 = \|f\|_{L^2}^2 + \|\nabla_{S^2} f\|_{L^2}^2$$

The regularization term applied to the filter weights becomes:

$$\mathcal{L}_{reg} = \lambda \sum_l (1 + l(l+1)) \|\hat{h}_l^m\|^2$$

This is provably better motivated than $l^2$: it penalizes filters that produce physically rough outputs, where "rough" is defined by the geometry of the sphere, not by an arbitrary choice.

**Output:** Replace the existing regularization formula with this one. Requires a one-line change to the loss function. The derivation and the $H^1(S^2)$ argument go in the paper.

### 5. Equivariance Error Metric

**What:** Define a formal, measurable test of whether the trained model is actually SO(2)-equivariant, and use it to evaluate architectural choices.

**Definition:** For a longitude rotation by angle $\Delta\phi$, let $\rho(\Delta\phi)$ denote its action on fields over S². The equivariance error of model $\Phi$ is:

$$\epsilon_{eq} = \mathbb{E}_{\Delta\phi \sim \text{Uniform}[0,2\pi)} \left[ \frac{\|\Phi(\rho(\Delta\phi) \cdot f) - \rho(\Delta\phi) \cdot \Phi(f)\|_2}{\|\Phi(f)\|_2} \right]$$

A perfectly equivariant model has $\epsilon_{eq} = 0$. In practice, floating point and boundary effects introduce small errors. This metric lets you:
- Verify the model is equivariant before claiming it in the paper
- Test whether specific architectural choices (batch norm, certain nonlinearities) break equivariance
- Compare equivariance across training epochs to check that training doesn't degrade it

**Output:** A test function implemented once and run after every architectural change. The measured $\epsilon_{eq}$ values go in an appendix table.

---

## Response to the Original Doc's Tradeoff Table

The comparison table in the original doc contains one entry that needs correction:

| | Option 1 (our choice) | Option 2 |
|---|---|---|
| Output | Global field map `[nlat, nlon, 3]` | Scalar Kp |
| Spatial info preserved | Yes — fully | No — pooled away |
| Equivariance | **SO(2) — provably correct for this data** | SO(3) — wrong symmetry for geophysical data |
| Filter expressivity | **Higher** (ĥ depends on l and m) | Lower (ĥ depends only on l) |
| Library | `torch-harmonics` | `e2cnn` / `s2cnn` |
| Math for Role A | Representation theory + analysis + statistics | Wigner D-matrices only |
| Build status | Toy model working | Not started |

The original table listed Option 1's equivariance as "partial" — this was imprecise. SO(2) equivariance is not partial SO(3); it is the complete and correct equivariance for this problem. The model is fully equivariant to the symmetry group the data actually has.

---

## What Role A Is Not Doing

To be explicit: Role A is not just doing background reading or decorative theory. The five derivations above each produce a specific artifact that changes something in the model or paper:

- Equivariance proof → justifies filter design, goes in theory section
- L_max derivation → sets a hyperparameter analytically, goes in methods
- Weighted loss → changes the loss function formula and implementation
- Sobolev regularization → replaces the existing regularization term
- Equivariance metric → implemented as a test, results go in evaluation

If any of these are skipped, the model is either less correct, less well-motivated, or harder to defend in a paper. The math is load-bearing.

---

## Reading List for Role A

- **Cohen & Welling 2016** — *Group Equivariant Convolutional Networks* — equivariance from first principles, flat case
- **Cohen et al. 2018** — *Spherical CNNs* — the foundational reference; read for the spherical harmonic theory and convolution theorem, understand where SO(3) comes in, then identify how SO(2) specializes it
- **Driscoll & Healy 1994** — the original SHT algorithm paper; useful for understanding what `torch-harmonics` is actually computing
- **Taylor 1996, *Partial Differential Equations*** or any graduate analysis text covering Sobolev spaces — for the $H^s(S^2)$ norm derivation
- Any standard reference on representation theory of compact groups — for the irreducible representation argument in derivation 1
