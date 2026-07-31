# Plasmasphere Handling — Problem, Measurement, and Roadmap

Status: **decision made, implementation not started.**
This is now a main-pipeline item, not an extension. Reasoning below.

---

## 1. The problem

IONEX TEC is the electron content integrated along the whole GNSS ray
path — ground to roughly 20,200 km. That column has two physically
distinct parts:

| Region | Altitude | Storm behaviour | Modelled by IRI? |
|---|---|---|---|
| Ionosphere | ~60–1,000 km | strong, fast, structured | yes |
| Plasmasphere | ~1,000–20,000 km | slower, smoother, storm-correlated | **no** |

So when we compute

$$\Delta\text{TEC} = \text{TEC}_{\text{IONEX}} - \text{TEC}_{\text{IRI}}$$

we are not isolating "the ionospheric anomaly IRI missed." We are
getting **ionospheric anomaly + the entire plasmaspheric column**,
because IRI never accounted for the plasmasphere in the first place.

Hayden confirmed this is not fixable by raising IRI's integration
ceiling — he tested 2,000 km → 20,000 km and the gap held at ~12–13
TECU, because PyIRI models the ionosphere, not the plasmasphere. There
is simply no plasmaspheric plasma in the model to integrate.

Hayden's measured offset:

| Day | Equator | Poles |
|---|---|---|
| 2015-03-17 (storm) | ~+16 TECU | ~+4 TECU |
| quiet solar-min day | ~+3 TECU | ~+0.5 TECU |

Latitude-structured, equatorially peaked, storm-amplified.

---

## 2. Why this matters spectrally — and the measurement

The plasmasphere is organized by **L-shell** (dipole magnetic field
lines), which maps to magnetic latitude. It is approximately
*azimuthally symmetric*: it depends on latitude, barely on longitude.

In spherical harmonic terms, a latitude-only function has $m = 0$. So
a plasmaspheric offset should land almost entirely in the **zonal
modes** $Y_l^0$, at low degree.

I measured this directly on Hayden's 183-map $\Delta$TEC sample by
splitting the power at each degree into the $m=0$ part and the rest.

**Headline numbers:**

| Quantity | Value |
|---|---|
| $m=0$ share of total $\Delta$TEC power | **54.0%** |
| $m=0$ share, storm maps only | 56.4% |
| $m=0$ share, quiet maps only | 49.3% |
| Power in $l \le 4$ (any $m$) | 78.4% |
| Power in **zonal** $l \le 4$ modes | **49.9%** |

Per-degree breakdown:

| $l$ | zonal % | share of total power |
|---|---|---|
| 0 | 100.0% | **37.0%** |
| 1 | 39.4% | 17.0% |
| 2 | 38.7% | 9.8% |
| 3 | 8.4% | 10.1% |
| 4 | 34.8% | 4.5% |

**Read that $l=0$ row again.** The degree-0 mode is a single number —
the global spatial mean of the map, a constant offset applied
everywhere. It carries **37% of all $\Delta$TEC power on its own.**

That is not ionospheric dynamics. That is a DC bias from the baseline
mismatch. As things stand, a model trained on this target would spend a
large fraction of its capacity learning to reproduce a constant offset
that exists because IRI and IONEX integrate different columns.

---

## 3. Decision: main pipeline, with an ablation

Half the power in the target being a near-constant zonal offset is too
large to file under "known limitation." Two consequences if we ignore
it:

- Our headline RMSE is dominated by how well we predict a baseline
  artifact, not by forecast skill.
- The SO(2)-vs-SO(3) ablation (which is *specifically* about
  $m$-dependence) gets contaminated, since the zonal band is exactly
  where SO(3) and SO(2) filters agree.

So: **zonal/non-zonal factorization goes in the main model.** We also
run it as an on/off ablation so the paper can report exactly what it
bought.

### The honest framing (important — don't overclaim)

We are **not** claiming to have isolated and removed "the
plasmasphere." We cannot cleanly separate plasmaspheric zonal signal
from ionospheric zonal signal with the data we have.

What we claim is narrower and defensible:

> Roughly half the residual power is zonal, so we factorize the model
> into a zonal component predicted from geomagnetic indices and a
> non-zonal component predicted from the spatial field. We note
> separately that a substantial part of the zonal power is attributable
> to the plasmaspheric column not represented in the IRI baseline.

The architecture is justified by the 54% measurement alone, regardless
of the physical attribution. That makes the claim robust even if a
reviewer disputes the plasmasphere story.

---

## 4. The architecture

```
                 (Dst, F10.7, Kp)
                        │
                        ▼
              ┌──────────────────┐
              │  Zonal head      │   small MLP
              │  → ĉ_l^0,  l≤L_z │
              └──────────────────┘
                        │
                        ▼
              zonal field  z(θ) = Σ_l ĉ_l^0 Y_l^0(θ)
                        │
   ΔTEC ────────( − )───┤                     ┌──( + )──→  ΔTEC prediction
     │                  │                     │
     ▼                  ▼                     │
  non-zonal residual  ─────→  SFNO  ──────────┘
   (m ≠ 0 dominant)            (existing model, unchanged)
```

The zonal part is predicted from geomagnetic indices alone (it has no
longitude structure, so a spatial model is overkill). The SFNO handles
what it is actually good at: non-zonal spatial structure.

Nothing is thrown away — the two parts are added back together for the
final prediction. This is a **factorization**, not a removal.

---

## 5. What I (Saksham) prove

### P1 — Orthogonal decomposition, SO(2)-invariant *(easy, do first)*

**Claim.** $L^2(S^2) = V_0 \oplus V_\perp$ where
$V_0 = \overline{\text{span}}\{Y_l^0\}$ and
$V_\perp = \overline{\text{span}}\{Y_l^m : m \neq 0\}$, the sum is
orthogonal, and both summands are invariant under every longitude
rotation $\rho(\alpha)$.

**Why it matters.** This is what makes the split well-defined and
architecturally legitimate.

**Proof route.** Orthogonality is immediate from orthonormality of
$\{Y_l^m\}$. Invariance follows from my existing Lemma: rotation acts
by $\hat f_l^m \mapsto e^{-im\alpha}\hat f_l^m$, so $m=0$ coefficients
are fixed *pointwise* ($e^{0}=1$) and $m\neq0$ coefficients stay in
their own $m$-line. Three lines.

### P2 — Pythagorean error decomposition *(easy, and the one that sells it)*

**Claim.** Because $V_0 \perp V_\perp$, the squared error splits
exactly:

$$\|\Delta\text{TEC} - \widehat{\Delta\text{TEC}}\|_{L^2}^2
= \|\Pi_0(\cdot) - \hat z\|^2 + \|\Pi_\perp(\cdot) - \hat u\|^2$$

where $\Pi_0, \Pi_\perp$ are the orthogonal projections, $\hat z$ the
zonal head output, $\hat u$ the SFNO output.

**Why it matters.** Three things at once: (a) the factorization is
*lossless* — no error term is created or hidden by splitting;
(b) the two components can be trained with separate losses without
interference; (c) we can report zonal and non-zonal RMSE separately in
the paper and they provably add to the total. That last point is a
clean, reviewer-friendly result.

**Proof route.** Pythagoras in a Hilbert space, applied to the
orthogonal decomposition from P1. Short.

### P3 — Equivariance is preserved *(easy, connects to ε_eq)*

**Claim.** The combined model (zonal head + SFNO) is still exactly
SO(2)-equivariant.

**Proof route.** The head's inputs (Dst, F10.7, Kp) are scalars,
unaffected by rotating the field, so $\hat z$ is unchanged by
$\rho(\alpha)$. And $\hat z \in V_0$, which by P1 is fixed pointwise
under rotation — so $\rho(\alpha)\hat z = \hat z$. Both sides of the
equivariance identity therefore agree on the zonal branch, and the SFNO
branch is already covered by my existing Proposition. Three lines.

**This matters practically:** it means our $\epsilon_{eq}$ test still
has a predicted value of zero after adding the head. If $\epsilon_{eq}$
degrades after this change, it's a bug, not the architecture.

### P4 — Plasmaspheric projection error bound *(hard — may not survive)*

**Claim (aspirational).** Bound
$\|f_{\text{plasma}} - \Pi_0 f_{\text{plasma}}\| / \|f_{\text{plasma}}\|$
under a stated assumption about duskside-plume geometry — i.e. quantify
*how* zonal the plasmasphere actually is.

**Honest assessment.** The plasmasphere is not perfectly azimuthally
symmetric: during storm recovery it grows a duskside plume, which is
strongly non-zonal — and storm recovery is exactly the regime we care
about. I may not be able to prove a clean bound without assumptions too
strong to defend.

**Decision:** attempt it, but the paper does **not** depend on it. If it
doesn't come out cleanly, it stays as a stated limitation. Marked OPEN
in the LaTeX either way until closed.

---

## 6. What Falisha implements

### F1 — Zonal projection utilities

Two functions on SHT coefficients:

```python
def project_zonal(f_hat):
    """Keep m=0 coefficients, zero the rest."""
    # returns coefficients of Π_0 f

def project_nonzonal(f_hat):
    """Zero the m=0 coefficients, keep the rest."""
    # returns coefficients of Π_⊥ f
```

Sanity check to run once: `project_zonal(f) + project_nonzonal(f) == f`
to floating-point tolerance, and the two outputs are orthogonal in
$L^2$ (inner product ≈ 0). That's P1 verified numerically.

### F2 — Zonal head

A small MLP. Input: the geomagnetic conditioning scalars at the current
timestep — start with `(Dst, F10.7, Kp)`, plus the same UT/DOY cyclic
encodings the main model uses.

Output: zonal coefficients $\hat c_l^0$ for $l = 0, \ldots, L_z$.

```
CONFIG['L_zonal'] = 4      # start here; sweep 2–8
```

Sizing: this is a genuinely small network — input dim ~7, one or two
hidden layers of 64, output dim $L_z + 1 = 5$. It should be a rounding
error in the parameter count. If it isn't, it's too big.

Reconstruct the zonal field by evaluating
$z(\theta) = \sum_{l \le L_z} \hat c_l^0 Y_l^0(\theta)$ — note this is a
function of latitude only, so it's cheap: compute a length-71 vector and
broadcast across longitude.

### F3 — Wire it into the forward pass

```
z          = zonal_head(indices)              # (B, n_lat) → broadcast
dtec_nz    = dtec - z                         # what the SFNO sees
u_hat      = SFNO(dtec_nz, ...)               # existing model, untouched
prediction = u_hat + z_future                 # add zonal back
```

One subtlety to decide together: at forecast time $t+h$ we need the
zonal component *at the target time*, so the head should be conditioned
on the predicted or persisted indices at $t+h$, not at $t$. Simplest
defensible choice for v1: persist the indices from $t$ and note it.
Flag this to me if the timing is awkward in the data loader.

### F4 — Ablation flag

```
CONFIG['zonal_head'] = True    # or False
```

Same discipline as the SO(3) flag — a config switch, not a forked file,
so nothing else can drift between runs.

### F5 — Reporting

Report RMSE split into zonal and non-zonal components (P2 says these
add to the total, so this is a free, exact decomposition — not an
approximation). This will be one of the more informative tables in the
paper.

---

## 7. What Hayden provides

- **H1.** The plasmasphere ceiling-test numbers, written up: IRI
  integration top 2,000 km vs 20,000 km, and the residual gap in each
  case. This is the evidence that the offset is plasmaspheric rather
  than an integration-limit artifact. Goes straight into the paper.
- **H2.** Equator-vs-pole offset magnitudes for a storm day and a quiet
  day (the ~16/4 and ~3/0.5 TECU numbers) with the dates and conditions
  stated precisely.
- **H3.** Dst in the conditioning set if it isn't already — the zonal
  head wants it (Kp is a 3-hour planetary index; Dst is the
  ring-current measure and is a better predictor of the plasmaspheric
  response).
- **H4.** *Optional, only if the above goes smoothly:* assess whether an
  empirical plasmasphere model (GCPM or similar) could be added to the
  IRI baseline directly. That would be the physically cleanest fix —
  correcting the target rather than factorizing the model — but it's a
  bigger lift and we should not block on it. Worth scoping, not worth
  starting yet.

---

## 8. Order of work

1. **Saksham:** P1, P2, P3 (all short, all independent of code). Do
   these first so the architecture is justified before it's built.
2. **Hayden:** H1, H2, H3 — numbers and the Dst channel.
3. **Falisha:** F1 (+ the numerical orthogonality check), then F2, F3.
4. **Falisha:** F4 ablation flag, then train both variants.
5. **Saksham:** attempt P4; if it doesn't close cleanly, write it up as
   a limitation instead.

Nothing in step 1 blocks step 2 or 3 — the proofs and the data work can
proceed in parallel.

---

## 9. What goes in the paper

- New subsection in the model section: the zonal/non-zonal
  factorization, with P1–P3 as the justification.
- The 54% measurement and the per-degree zonal table as a figure or
  table — this is a genuinely novel empirical characterization of the
  IONEX-minus-IRI residual and I haven't seen it reported elsewhere.
  Worth foregrounding.
- The plasmasphere attribution, stated carefully (Section 3 above) with
  Hayden's ceiling test as evidence.
- The zonal-head ablation in the results table.
- P4 or, failing that, an explicit limitation paragraph about the
  duskside plume breaking the zonal approximation during storm
  recovery.
