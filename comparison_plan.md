# TEC-S2Net — Architectural Comparisons: Plan and Delegation

The paper needs comparisons that isolate our design claims. This
document says which comparisons we run, which we deliberately skip,
what the math behind each one is, and who does what.

Short version: **we build two comparison models, both cheap, and we
skip the expensive ones on purpose.** The reasoning is below so you can
push back if you disagree.

---

## The one paragraph that determines everything

Our SFNO applies a learned weight $\hat h_l^m$ to each spherical
harmonic coefficient $\hat f_l^m$. My Theorem (Section 3 of the paper)
proves two things:

1. **Any** choice of $\hat h_l^m$ is exactly SO(2)-equivariant
   (invariant under rotation about Earth's axis).
2. The **SO(3)**-equivariant filters (invariant under *all* 3D
   rotations) are exactly the subclass where
   $\hat h_l^m = \hat h_l$ — i.e. the weight does not depend on $m$.

That second point is the whole experimental plan. SO(3) is not a
different architecture we need to build from scratch. **It is our
architecture with the $m$-dependence removed.** So the comparison is a
weight-tying change, not a new codebase.

---

## What we build

### Comparison 1 — Flat CNN (already planned)

**What:** Same depth, same channel widths, same parameter budget, but
planar 2D convolutions on the raw $71\times73$ lat/lon grid instead of
SHT-based spectral convolution.

**Why:** Tests the paper's most basic claim — that spherical geometry
matters. A flat CNN treats the map as an image: it distorts area near
the poles (a pixel at $87.5^\circ$ covers a tiny fraction of the area
of one at the equator) and tears the longitude seam (column 0 and
column 72 are adjacent on Earth but maximally distant in the image).

**Your task (Falisha):**
- Keep everything identical except the conv layers.
- Use `padding_mode='circular'` on the longitude axis so we are
  comparing against a *fair* flat baseline, not a strawman. If we beat
  a flat CNN that can't even wrap longitude, a reviewer will say we
  rigged it.
- Match parameter count within ~10%. Report the exact counts for both.

**Math needed from me:** none. This one is purely empirical.

---

### Comparison 2 — SO(3)-restricted filters (the key ablation)

**What:** Our exact model, with the spectral filter weights tied so
they no longer depend on $m$.

**Why:** This is the experiment that tests our central theoretical
claim. If SO(2) filters and SO(3) filters perform identically, our
"correct symmetry class" argument is decorative. If SO(2) wins, the
theorem has teeth and the paper has a result.

**Physical intuition for why we expect SO(2) to win:** $m$ indexes
longitudinal structure. $m=0$ modes are zonal (functions of latitude
only — rings around the Earth). Large $|m|$ modes are strongly
longitude-dependent. The ionosphere has a huge $m=1$ component (the
dayside bulge, one bright region facing the Sun) that behaves
completely differently from the zonal background. Forcing
$\hat h_l^m = \hat h_l$ means the model must treat the dayside
structure and the zonal background with the *same* filter weight. That
should hurt.

**Your task (Falisha):**

The change is in how the filter parameter tensor is allocated and
indexed. Right now you presumably have something shaped like

```
h_hat : (C_out, C_in, L_max+1, 2*L_max+1)     # indexed by (l, m)
```

For the SO(3) variant, allocate

```
h_hat_so3 : (C_out, C_in, L_max+1)            # indexed by (l) only
```

and broadcast it across the $m$ axis at multiply time. Concretely, if
the current forward does something like

```python
out = f_hat * h_hat                     # (…, L+1, 2L+1) * (…, L+1, 2L+1)
```

the SO(3) version is

```python
out = f_hat * h_hat_so3.unsqueeze(-1)   # (…, L+1, 2L+1) * (…, L+1, 1)
```

That single `unsqueeze` and broadcast is the entire architectural
difference. Everything else — SHT, window, nonlinearity, GRU, loss —
stays byte-for-byte identical.

Please implement it as a **config flag**, not a forked file:

```python
CONFIG['filter_symmetry'] = 'so2'   # or 'so3'
```

so we can be certain nothing else differs between runs.

**Two things to report:**
- Parameter count for both. SO(2) has $(L_{\max}+1)^2$ weights per
  filter, SO(3) has $L_{\max}+1$. At $L_{\max}=15$ that's 256 vs 16
  — a 16× difference, so also run a **width-matched** SO(3) variant
  (wider channels to equalize total parameters). Otherwise a reviewer
  will say we only won because we had more parameters.
- Results split by geomagnetic condition. My prediction: the SO(2)
  advantage is largest during storms and smallest in quiet conditions,
  because storm structure is more strongly non-zonal. If that pattern
  shows up it's a nice result; if it doesn't, that's still worth
  reporting honestly.

**Math needed from me:** already done — this is Theorem 3.2(iii) in the
paper draft. I still owe the formal write-up of the converse direction
(that SO(3)-equivariance *forces* $m$-independence), which is currently
marked OPEN in red in the LaTeX. That's on me and doesn't block you.

---

## What we deliberately skip, and why

You may reasonably ask why we aren't comparing against every
equivariant architecture in the literature. Reasons, so we have a
consistent answer if a reviewer asks:

### Full Cohen et al. 2018 S2CNN — skipped

That architecture lifts the signal from $S^2$ to the rotation group
$SO(3)$ in the first layer, so feature maps are functions of three
Euler angles rather than two spherical coordinates. It requires
SO(3)-FFTs and Wigner-D matrices $D^l_{mn}$ as the basis, needs the
`s2cnn`/`lie_learn` stack, and costs substantially more compute
(SO(3) is 3-dimensional).

We skip it because it answers a **blurrier question**. Comparing to a
wholly different architecture confounds many variables at once. Our
Comparison 2 changes exactly one thing — the $m$-dependence — which is
precisely the variable our theorem is about. Cleaner experiment, and it
directly tests a proven statement. We cite Cohen 2018 as the
conceptual SO(3) reference in related work.

### Steerable CNN — skipped, because it's the same model

This one is worth understanding because it's a genuinely nice point for
the paper.

A "steerable" filter is one that acts as a scalar on each irreducible
representation of the symmetry group. For SO(3), irreps are
$(2l+1)$-dimensional, so steerability is a real constraint requiring
Wigner-D machinery. But **SO(2) is abelian, so all its irreps are
1-dimensional** — indexed by $m$, acting as $e^{-im\alpha}$.

A steerable filter under SO(2) is therefore just: multiply coefficient
$(l,m)$ by a scalar. Which is exactly what `h_hat[l,m] * f_hat[l,m]`
already does. **Our architecture already is the steerable
construction** — there is no separate model to build.

I'm adding a remark to the paper stating this equivalence explicitly,
since it shows the design wasn't arbitrary: for scalar fields under
SO(2), spectral-diagonal filtering and steerable filtering coincide,
and the distinction only becomes meaningful for SO(3) or for
vector-valued (gauge) settings.

### Group-equivariant CNN (Cohen & Welling 2016) — cite only

That paper is the planar ancestor — discrete symmetry groups (p4
rotations, p4m reflections) on a flat $\mathbb{Z}^2$ grid. It doesn't
operate on spherical data. Using it as a baseline would require
flattening our data, which is the exact practice our paper argues
against. Cited as intellectual lineage, not compared against.

### Gauge-equivariant CNN — future work, sketched only

Cohen et al.\ 2019, for **vector-valued** fields — sections of the
tangent bundle rather than scalar functions. TEC is scalar so it
doesn't apply here, but it's the correct generalization for the natural
next target (ground magnetic perturbation $\Delta B$, plasma drift
velocity). Building it means replacing $Y_l^m$ with spin-weighted
spherical harmonics and reformulating the whole pipeline on the bundle
— a separate paper's worth of work.

I'm writing a short problem-statement sketch for the paper's extensions
section. Nothing for you to implement.

---

## Ordering, and how this interacts with the plasmasphere issue

Hayden found that our $\Delta$TEC target carries a plasmaspheric offset:
IONEX integrates electron content along the whole GNSS path
(~20,200 km) while IRI only models the ionosphere (~2,000 km), so the
residual includes plasmaspheric content IRI never accounted for. It's
latitude-structured, storm-correlated, and approximately zonal.

**Why this matters for the comparisons:** an approximately zonal offset
lives in the $m=0$ modes. Comparison 2 is *specifically about*
$m$-dependence. So a systematic sitting in the $m=0$ band is exactly
the kind of thing that could muddy that comparison.

For the paper we're handling it by documenting it honestly rather than
correcting it (the model forecasts total-column deviation from an
ionosphere-only baseline — a defensible target). But it's worth
knowing when we read the Comparison 2 numbers: if SO(3) does
surprisingly well, one candidate explanation is that a large chunk of
the signal genuinely is zonal because of this offset, not because
$m$-dependence is useless.

**Suggested run order:**
1. Main model trains and beats persistence. Nothing else matters until
   this works.
2. Comparison 2 (SO(3) ablation) — cheapest, most theoretically
   important.
3. Comparison 1 (flat CNN) — needs a separately written model, so
   more work than #2 despite being conceptually simpler.

---

## Delegation summary

| Item | Owner | Status | Blocks? |
|---|---|---|---|
| Flat CNN baseline | Falisha | to do | no |
| SO(3) filter flag (`filter_symmetry` config) | Falisha | to do | no |
| Width-matched SO(3) variant | Falisha | to do | after flag works |
| Param counts for all variants | Falisha | to do | — |
| Results split by Kp condition | Falisha | to do | after training |
| Thm 3.2(iii) converse proof | Saksham | in progress | no |
| Steerable/SO(2) equivalence remark | Saksham | to write | no |
| Gauge-equivariant extension sketch | Saksham | to write | no |
| Plasmasphere documentation | Saksham + Hayden | in progress | no |

Nothing here blocks you — every math item I owe is paper-side, not
code-side. If any tensor shapes above don't match how you've actually
laid out the filter weights, tell me and I'll rewrite the indexing.

---

## The one-sentence version for the paper

> We compare against a flat CNN (tests whether spherical geometry
> matters) and against SO(3)-restricted filters (tests whether the
> larger SO(2)-equivariant filter class earns its parameters), and note
> that for scalar fields under SO(2), steerable and spectral-diagonal
> filtering coincide — so no separate steerable baseline exists.
