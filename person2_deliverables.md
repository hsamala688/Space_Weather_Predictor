# Person 2 (Saksham) — Deliverables Checklist

Math only — Person 3 is implementing all model code (`SphericalConvLayer`, `SphericalConvGRU`, everything) directly. Saksham's job is the formulas/proofs; Person 3 wires them into the notebook.

---

## 1. L_max derivation
**What:** The spectral truncation degree — how many spherical-harmonic frequencies the model represents, derived from the power spectrum of real TEC maps (not guessed).
**Status:** Placeholder (`l_max = 64`) in Cell 3, `H = L_max`, `W = 2 × L_max`.
**Where it lands:** Cell 3 (`CONFIG['l_max']`, `CONFIG['m_max']`, `CONFIG['H']`, `CONFIG['W']`) — one number cascades into grid size, SHT dimensions, and every filter shape in Cells 8–10.

## 2. SO(2) equivariance proof
**What:** Proof that per-(l,m) spectral filters are exactly rotation-equivariant in longitude — and the justification for *why* that symmetry needs to be deliberately broken (via Person 1's UT/day-of-year encodings), since TEC isn't actually longitude-invariant.
**Status:** Not yet delivered. Doesn't slot into a specific cell — it's the theoretical backing for a design choice already in the pipeline (Person 1's positional encodings).

## 3. Gibbs window (σ_l)
**What:** Spectral windowing formula per degree `l`, to reduce ringing artifacts from the truncated spherical transform.
**Status:** Identity placeholder. Cell 10, Step 3 of `forward()`, currently just `x = h` — **note:** the `spectral_window` buffer this should multiply against isn't actually wired into the forward pass yet, so this isn't a clean drop-in swap; the multiply itself still needs to be added when the formula arrives.

## 4. Storm weights (importance-weighted loss)
**What:** Inverse-density weighting formula based on Dst index, so rare storm-time samples aren't drowned out by abundant quiet-time samples in the loss.
**Status:** `torch.ones(B)` placeholder (uniform weighting). Cells 14 (`train_one_epoch`) and 17 (`evaluate_baselines`) — replaces `sw = torch.ones(B, device=device)`.

## 5. Sobolev H¹(S²) regularization
**What:** Proper spectral-domain Sobolev penalty, `Σ (1 + l(l+1)) |coefficient_l|²`, computed directly on SHT coefficients.
**Status:** Finite-difference proxy in place (`grad_lat`/`grad_lon` in Cell 11's `tec_loss`). Note: `sobolev_weights` (built in Cell 7) is already passed into `tec_loss()` but currently unused — the real formula needs to actually consume it against spectral coefficients, not pixel gradients.

## 6. Equivariance error metric (ε_eq)
**What:** Post-training verification — measure how far the *trained* model's behavior deviates from perfect SO(2)-equivariance, compare against a theoretical bound.
**Status:** Confirmed absent — zero occurrences anywhere in the current notebook (checked directly). Runs *after* training on a checkpoint, not during — needs its own step/cell once a real model exists, not squeezed into Cell 13's per-batch metrics.

---

### Explicitly NOT Person 2's responsibility here
The original project doc lists `SphericalConvLayer`, the spectral window pre-multiplier, `SphericalConvGRU`, and the full loss function as Saksham's implementation work. That's superseded — Person 3 owns all of Cells 8–11 end-to-end. Saksham's actual footprint is items 1, 3, 4, 5 above (as formulas) plus 2 and 6 (proof + post-hoc metric).
