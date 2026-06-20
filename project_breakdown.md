# Project Breakdown
### Space Weather Prediction with Spherical CNNs

A reference for the three-person team: who owns what, where the seams are, the order we build in, and the shared technical decisions everyone honors. Convention: no em dashes.

---

## 1. Project Summary

Predict the near-future global geomagnetic disturbance field (1 to 3 hours ahead) from a network of ground magnetometers, using a Spherical CNN. The architecture is motivated by the data living on the surface of a sphere, so the model is sphere-to-sphere: a spherical input field in, a predicted spherical field out. Team of three, part-time.

- **Primary input:** SuperMAG (~700 ground magnetometers, dB_north / dB_east / dB_vertical), resampled to hourly, interpolated onto a fixed HEALPix grid.
- **Target:** future disturbance field (and Kp as a derived scalar reference).
- **Core scientific claim:** the spherical inductive bias beats a flat CNN and persistence. The whole project hangs on that comparison being clean.

---

## 2. Team Roles and Responsibilities

The three roles are theory, data, and model, as set out. The changes below close the work that had no owner, calibrate each scope, and make the seams explicit.

### Role A: Theory + Evaluation (Math)

Evaluation is fundamentally statistical work, so it sits here. This keeps the math load-bearing rather than an appendix, and gives the role real running code.

Owns:
- SO(3) equivariance error: derive the metric, then measure it against the trained model across architectural choices.
- L_max truncation analysis: choose the spherical-harmonic bandwidth from the signal's spectral properties rather than grid search. Decided jointly with Data (couples to grid resolution).
- Storm-weighted loss: derive the reweighting from the empirical Kp distribution. Hand the equation and the computed weights to Model for implementation.
- Optional: spectral regularization (penalize high-degree coefficients) if time allows.
- The evaluation harness: skill-score-vs-persistence, storm-vs-quiet breakdowns, calibration analysis, and the statistical spherical-vs-flat comparison.
- Persistence and linear-regression baselines (they are trivial data operations that live inside the harness).

### Role B: Data / Pipeline

The job is the whole pipeline, not "the split." The split is a few lines of config; the pipeline is the single biggest chunk of the project (30 to 40% of the work). This role is rightly the heaviest on engineering volume.

Owns:
- SuperMAG, Kp, and (later) OMNI download, with raw payloads stored verbatim before transforming.
- The sparse-to-HEALPix interpolation: method choice, hemisphere imbalance, empty ocean cells. These are real scientific decisions, not just plumbing.
- Train-only normalization (fit on the training years, applied forward).
- Sample assembly: input windows and t+1 / t+2 / t+3 targets, with no sequence straddling a split boundary.
- Handling station-coverage non-stationarity (the network grew over the decades).
- The split config and the canonical sample tensor that both the model and the harness consume.

### Role C: Model / Library

Reframe "tinker with the internals" as "compose the library's layers into our architecture." Go inside the library only if a specific, concrete need forces it.

Owns:
- Library selection and integration.
- The spherical CNN architecture.
- The training loop, orchestration, and Weights & Biases logging (otherwise unowned).
- Implementation of the storm-weighted loss from Role A's derivation.
- The flat-CNN ablation baseline. It must mirror the spherical architecture exactly, so only the person who built the spherical model can guarantee the match.

---

## 3. Handoff Contracts (the seams)

Each interface is owned by two people who must agree. These are where the project breaks if left implicit.

| Seam | Between | The contract |
|---|---|---|
| Grid and bandwidth | Theory + Data | One fixed (HEALPix resolution, L_max) pair, agreed before the pipeline is built. They are not independent choices. |
| Tensor spec | Data + Model | Written spec for the sample tensor (shape, dtype, channel order, grid convention), agreed before Data finalizes output and before Model finalizes the input layer. |
| Equivariance instrumentation | Theory + Model | Model exposes the trained model as a callable with a hook; Theory supplies the rotate-input / measure-output metric that runs against it. |
| Loss handoff | Theory + Model | Theory delivers the weighted loss as an equation plus the empirical weights computed from the training distribution; Model implements it. |

---

## 4. Kickoff Sequence (Week 1)

Three decisions unblock everyone. Make them first.

1. Model picks the library.
2. Theory and Data fix the (grid, L_max) pair together.
3. Data and Model agree the tensor spec.

After that, work in parallel against stand-ins so nobody blocks:
- Data builds the pipeline on a single year before scaling to the full range.
- Model builds the architecture and training loop on synthetic tensors of the agreed shape.
- Theory builds the eval harness and equivariance metric against a toy sphere signal or the library's reference model.

Converge once real samples flow: run baselines, train, ablate, test once, report.

---

## 5. Shared Technical Decisions (everyone honors these)

- **Split:** single chronological split. Train 2000 to 2018, validate 2018 to 2020, test 2020 to 2025. Strictly time-ordered, no overlap.
- **Training data:** maximize it and put the oldest data in training. Storms are rare and heavy-tailed, so every storm example is precious, and the early years hold the largest events.
- **Test set:** the most recent stretch, touched exactly once. It is physically honest (past predicts future), has the densest station coverage, and being an active solar phase it is storm-rich enough to measure storm skill.
- **Normalization:** fit on the training years only, applied forward. Computing stats over the full range leaks future statistics into the past.
- **Grid:** a fixed HEALPix grid so input shape is constant despite the growing station network. Expect early years to be noisier (sparser stations).
- **Loss:** storm-weighted, with the weight tuned on validation against the skill score while watching the false-alarm rate. Do not over-crank it.
- **Shuffling:** shuffle whole input/target sequences for SGD, never the hours within them, and never across a split boundary.
- **Baselines:** persistence, linear regression, and the flat-CNN ablation, all run on the identical split. The spherical-vs-flat comparison is only clean if the windows match exactly.
- **Reporting:** results broken out by storm vs quiet periods with storm-event counts attached. State plainly that the hold-out is a single active-phase window and that one chronological split gives one estimate tied to that phase, with rolling-origin cross-validation as the next step if more rigor is wanted.

---

## 6. Build Workflow (with verify gates)

The order carries the discipline: define the split before touching anything, run baselines before the model, touch the test set once.

1. **Acquire raw data** -> verify: log station count and gap fraction per year, so coverage is known not assumed.
2. **Define and freeze the split** -> verify: assert the three ranges do not overlap; quarantine the test range.
3. **Build the spherical grid** -> verify: render a known storm (Halloween 2003, inside the training window) and confirm the disturbance is visible.
4. **Fit normalization on train only** -> verify: assert stats were computed from training indices only.
5. **Assemble samples** -> verify: shapes correct, no sequence straddles a boundary.
6. **Run baselines first** -> verify: skill-score pipeline runs and returns sensible numbers.
7. **Train the spherical CNN** -> verify: validation loss decreasing, beats persistence on validation, equivariance error acceptably small.
8. **Train the flat-CNN ablation** -> verify: identical split and normalization.
9. **Touch the test set once** -> verify: no config changes after this step; all four models scored together.
10. **Report honestly** -> verify: storm-vs-quiet breakdown present, single-split and active-phase caveat stated.

---

## 7. Open Item

The library choice (Role C) still shapes two things: how much of the internals Theory can reach to measure equivariance, and how thin or thick Model's integration job is. Settle it in Week 1 and the Theory/Model boundary sharpens around it.
