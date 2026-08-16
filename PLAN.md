# Research plan: interpreting the cyclic ITG heat-flux neural networks

## Purpose

The scientific goal is to determine which **geometric** patterns the trained
neural networks use to predict turbulent ion heat flux, which patterns are
shared by high-performing ensemble members, and which findings survive tests of
faithfulness, cyclic symmetry, and physical plausibility.

This is a plan for studying the existing trained networks. It is not a claim that
an attribution map is a causal explanation of gyrokinetic turbulence. The desired
end product is an evidence-weighted set of candidate physical mechanisms and
compact invariant features that can be checked against the simulation data and,
where justified, new GX calculations.

Each numbered step below is scoped for one focused Codex or Claude task.
`WORKFLOW.md` explains how to dispatch and merge those tasks.

## Read this first: five structural facts that shape the whole program

These are the reasons this plan does not look like a generic CNN-saliency
project. Each is verified in [Preliminary scoping measurements](#preliminary-scoping-measurements).

1. **The networks have a tiny invariant bottleneck.** Every member factors
   exactly as

   $$f_m(X,g_T,g_n)=\text{MLP}_m\bigl(u_m(X),\,g_T,\,g_n\bigr),\qquad
   u_{m,c}(X)=\operatorname{mean}_z\bigl[\Phi_{m,c}(X)\bigr],$$

   where $\Phi_m$ is a circular-convolution/ReLU/max-pool stack (equivariant)
   and the global average pool is a translation-invariant reduction. The
   bottleneck width is $C_m\in[7,32]$ across the 100 members (10 for the
   top-validation member), and several units are dead. **All of each network's
   geometric knowledge is 7–32 numbers.** Exhaustive analysis of the bottleneck
   is cheap and exact; input-space saliency of the scalar output is neither.

2. **The networks live inside the paper's own feature grammar.** The paper
   builds interpretable features as (equivariant operation) ∘ (invariant
   reduction), e.g.
   $f_Q=\operatorname{mean}_z([\Theta(\mathbf B\times\boldsymbol\kappa\cdot\nabla y)+0.2]|\nabla x|^3/B)$.
   Each bottleneck unit $u_{m,c}$ has *exactly that form*: the mean over $z$ of a
   learned local density. So the natural interpretive question is not "which grid
   points matter" but "**what local density does each unit average, and how does
   the head combine them?**"

3. **An exactly shift-invariant version of each network exists and is cheap.**
   The five stride-2 pools make each member exactly invariant to shifts by
   multiples of 32, so averaging the bottleneck over the 32 pooling phases,
   $\bar u_m(X)=32^{-1}\sum_{k=0}^{31}u_m(S_kX)$, is **exactly** shift-invariant
   (verified to $10^{-6}$ relative) and is **exactly** the mean over all 96
   positions of the stride-1 (à trous) pooling chain. Averaging over 96 shifts is
   redundant: 32 suffices, to $2\times10^{-6}$. The symmetrized model is also
   slightly *more* accurate than the original.

4. **A known geometric quantity is built into the target.** The HDF5 target is
   $Q_{\text{avgs}}=Q_{\text{GX}}\cdot\langle|\nabla x|\rangle$ with
   $\langle|\nabla x|\rangle=\operatorname{mean}_z(\sqrt{\texttt{gds22}}/B)\big/\operatorname{mean}_z(1/B)$,
   verified to $10^{-7}$ relative from channels 6 and 0 alone. Therefore
   $\log\langle|\nabla x|\rangle$ is an *exactly known, purely geometric,
   translation-invariant additive term* in what the network must predict. It is a
   ground-truth invariant feature with no free parameters — the single best
   available test of whether our tooling can find a feature we already know is
   there.

5. **The heavy machinery is not the first thing to reach for.** A handful of
   cheap structure-destroying counterfactuals (joint $z$-permutation,
   per-channel independent shifts, band attenuation) already bound how much of
   the network's function can possibly be spatial, spectral, or cross-channel.
   Running those first tells later steps where to spend effort.

## Facts that constrain the analysis

### Data

- The canonical local dataset is
  `/Users/mattland/20260523-01-files_for_Kosmos_interpreting_neural_networks/20250102-01_GX_stellarator_dataset.h5`.
  XAI scripts should use this as their default while retaining an optional
  `--dataset` override for portability. Agents should not require an environment
  variable or copy the file into this repository.
- 100,705 flux tubes; each appears once in `fixed_gradient_simulations` and once
  in `varied_gradient_simulations` with the *same* geometry row.
- Geometry has shape `(96, 7)`, **raw and unnormalized** — the network consumes
  `raw_feature_tensor` directly, with no per-channel standardization anywhere in
  the inference path. The authoritative HDF5 channel order and measured scales
  (3,000-tube sample) are:

  | Index | HDF5 name | Physical label | median | 1%–99% | max\|·\| |
  |---:|---|---|---:|---|---:|
  | 0 | `bmag` | $B$ | 1.05 | 0.73 – 1.57 | 2.2 |
  | 1 | `gbdrift` | $2B^{-3}\mathbf B\times\nabla B\cdot\nabla y$ | −0.11 | −0.83 – 0.70 | 4.2 |
  | 2 | `cvdrift` | $2B^{-2}\mathbf B\times\boldsymbol\kappa\cdot\nabla y$ | −0.09 | −0.79 – 0.73 | 4.2 |
  | 3 | `gbdrift0_over_shat` | $2B^{-3}\mathbf B\times\nabla B\cdot\nabla x$ | 0.00 | −0.66 – 0.66 | 1.9 |
  | 4 | `gds2` | $\|\nabla y\|^2$ | 1.71 | 0.15 – 9.88 | 443 |
  | 5 | `gds21_over_shat` | $\nabla x\cdot\nabla y$ | 0.00 | −4.10 – 4.07 | 118 |
  | 6 | `gds22_over_shat_squared` | $\|\nabla x\|^2$ | 1.42 | 0.24 – 9.24 | 97 |

  The `_over_shat` suffixes describe how GX stores the quantity; despite the
  names, channel 6 is $|\nabla x|^2$ in the same normalization used by
  $\langle|\nabla x|\rangle$ (verified by fact 4 above). Channels 4–6 have heavy
  tails spanning three orders of magnitude, so **no attribution comparison across
  channels is meaningful without an explicit robust scale**, and no perturbation
  magnitude should be specified in absolute units.
- Grid: `z` is uniform, 96 points, spanning $[-37.70, 36.91]$ in the stored
  normalization (period $96\,\Delta z$).
- The two additional inputs are $a/L_T$ and $a/L_n$. The legacy training
  convention used $a/L_T=-3$ as a marker for the fixed-gradient simulations
  (which all have the physical values $a/L_T=3$, $a/L_n=0.9$); this is not a
  physical negative temperature gradient. Fixed- and varied-gradient results must
  not be pooled without recording this distinction.
- **Target floor:** 33,891 of 100,705 varied-gradient rows (33.7%) sit at the
  clipped-log floor $\max(\log Q,-2)=-2$; 1,773 rows have $Q\le 0$ and are
  dropped by the positive-flux filter. Fixed-gradient rows have no $Q\le 0$.
  A third of the varied set therefore carries essentially no gradient signal
  about flux magnitude, only about stability. Stable/near-floor and unstable
  cases must be reported separately everywhere.
- **Grouping:** 100,705 tubes come from 23,577 distinct `equilibrium_files`
  (~4.3 tubes per equilibrium) in 5 `equilibrium_class` values (51,075 /
  12,795 / 12,791 / 8,235 / 15,809). Bootstrapping or splitting by tube rather
  than by equilibrium file will overstate precision.
- **Covariates that are not network inputs but matter for stratification and
  confounding:** `scalar_feature_matrix` columns `nfp`, `iota`, `shat`,
  `d_pressure_d_s`, `aspect`, `rho`, `aspect/rho`; `FSA_grad_xs`; `QUASR_IDs`;
  `equilibrium_class`; `tube_files`. `nfp` in particular sets the number of
  field periods sampled along the field line and therefore the dominant parallel
  Fourier content, so any "the network prefers wavenumber $k$" claim must be
  checked against `nfp`. `shat` is correlated with the magnitude of channel 5
  ($r\approx0.41$ on a 3,000-tube sample).
- **Held-out physics diagnostics that were not network inputs:**
  `Q_avgs_vs_z` (96-point parallel profile of the heat-flux contribution),
  `Q_stds` (a target-noise proxy; the paper notes $\sigma_Q$ scales roughly
  linearly with $Q$), `zonal_phi2_amplitudes`, and
  `Q_avgs_divided_by_FSA_grad_x`.

### Models

- `models/cyclic_ensemble_pre2.pt` contains 100 PyTorch ensemble members selected
  from 443 hyperparameter-search trials. Stored validation $R^2$ spans
  0.98516–0.98725 — **a spread so small it is almost certainly within
  split-resampling noise**. Consequently:
  - The top 10 by stored validation score remain the headline reporting cohort
    for continuity with the paper, but validation rank is treated as a
    *covariate*, not a hard cutoff.
  - Any analysis cheap enough to run on all 100 members must be run on all 100,
    with rank plotted as a continuous predictor.
  - S01 re-ranks members with bootstrap intervals on a held-out cohort and
    reports how much the ranking moves. Never select the primary cohort using
    the test set.
- Every member has five circular `Conv1d` layers, each followed by ReLU and
  stride-2 max pooling, then global average pooling, two scalar gradient inputs,
  two dense layers, and one scalar output. Kernel widths (all 100 tuples
  distinct), channel counts, and dense widths vary across members. Parameter
  counts range from 10,218 to 38,741.
- **Bottleneck widths** $C_m$ (last conv channel count) across the 100 members:
  7×1, 10×2, 11×1, 14×2, 15×3, 16×2, 17×3, 19×2, 20×3, 21×7, 22×9, 23×7, 24×2,
  25×4, 26×5, 27×10, 28×8, 29×10, 30×4, 31×11, 32×4. The top-validation member
  has $C=10$ with one permanently dead unit; dead-unit counts of 0–5 are common
  in the top 10.
- The native output is a prediction of $\max(\log Q,-2)$. Primary fidelity and
  attribution calculations must therefore explain this scalar, not $Q$ and not
  `exp(prediction)`.
- Five rounds of stride-2 pooling reduce 96 positions to 3. Circular convolution
  is shift-equivariant; strided max pooling is not equivariant to a
  one-grid-point shift. Invariance is exact for shifts by multiples of 32 and
  approximate otherwise — the paper reports that random cyclic-shift augmentation
  was used during training, which is why the approximate part works at all.
- Effective receptive fields wrap. For the top member (kernels 13, 5, 3, 8, 5)
  the theoretical receptive field already exceeds 96 after the fourth block, so
  the last block is not spatially local; per-member receptive fields must be
  computed, not assumed.

### Physics context

- The simulations used **periodic** parallel boundary conditions, so the
  shift-invariance of $Q$ is a property of the data-generating process, not just
  an approximation (the paper's appendix compares against twist-and-shift and
  finds $R^2=0.97$ between the two).
- The paper's leading hypotheses are flux-surface compression in regions of bad
  curvature and the magnitude of geodesic curvature. The central reference
  feature is

  $$
  f_Q=\operatorname{mean}\left(
  [\Theta(\mathbf B\times\boldsymbol\kappa\cdot\nabla y)+0.2]
  |\nabla x|^3/B\right),
  $$

  and the stability-tuned variant is
  $f_{\text{stab}}=\operatorname{mean}([\Theta(\mathbf B\times\nabla B\cdot\nabla y)+0.4]|\nabla x|/\sqrt B)$.
  These are hypotheses to test against the networks, not labels to force onto
  every discovered pattern.
- The paper's Shapley analysis of its XGBoost surrogate ranks $a/L_T$, then
  $a/L_n$, then bad-curvature-weighted compression, then compression alone
  ($\operatorname{mean}(|\nabla x|^4/B^6)$), then two geodesic-curvature features.
  That ordering is a prior, not a target.
- **Stellarator parity is a second, near-exact symmetry.** Under $z\to-z$ with
  sign reversal of channels 3 and 5 (and no sign change of 0, 1, 2, 4, 6), the
  dataset is approximately invariant (measured reversal mismatch 0.06–0.25 of
  channel variance for the correct parity versus 3.9–138 for the wrong parity).
  This gives a second physically meaningful "exact symmetry" control that the
  networks were never trained on.

## Preliminary scoping measurements

**These are unregistered scratch measurements made while writing this plan.**
They are recorded here so agents know what magnitudes to expect (a result that
disagrees by an order of magnitude signals a bug). They are *not* results, carry
no uncertainty estimate, and must be reproduced under the experiment contract in
S01–S03 before any of them is cited.

Setup: top-validation member unless stated; CPU; 256–2,000 rows;
varied-gradient; unstable rows ($\log Q>-1.9$) for the perturbation numbers.
For scale, the top member's residual standard deviation on 2,000–3,000-row slices
of the reference test cohort is **0.28–0.29** in clipped-log units (0.31 on
unstable rows only); the target standard deviation is 2.19.

| Probe | RMS change in native output | Ratio to residual std |
|---|---:|---:|
| Shift by 32 or 64 (predicted exact subgroup) | $2\times10^{-7}$ | $\sim10^{-6}$ |
| Arbitrary circular shift, all 96, top 5 members | 0.10 – 0.15 | 0.35 – 0.5 |
| Parity: $z\to-z$ with sign flip of channels 3, 5 | 0.10 – 0.11 | ~0.4 |
| Plain $z$ reversal (wrong parity, control) | 0.39 – 0.47 | ~1.5 |
| Independent circular shift per channel | 0.64 – 0.77 | ~2.3 |
| Random joint permutation of $z$ | 0.69 – 0.82 | ~2.6 |

Interpretation to be tested, not assumed: the network genuinely uses parallel
ordering (permutation is far outside the residual scale) and cross-channel
alignment carries most of that (independent shifts destroy nearly as much as
full permutation); residual arbitrary-shift error is *not* negligible — it is a
third to a half of the model's own error, so it contaminates position-resolved
explanations at a comparable level.

Symmetrization and bottleneck (top member, 2,000 reference-test rows):

- $\bar u(S_1X)=\bar u(X)$ to $10^{-6}$ relative — the 32-phase bottleneck
  average is exactly shift-invariant.
- $\bar f$ over 32 phases equals $\bar f$ over 96 phases to $2\times10^{-6}$.
- $R^2$: original 0.9826, 32-phase symmetrized 0.9851,
  $\tilde f:=\text{MLP}(\bar u,g_T,g_n)$ 0.9849. Symmetrizing costs 32 forward
  passes and *improves* accuracy by ~7% in residual std.
- Per-unit mean-replacement ablation of the 10 bottleneck units gives RMS output
  effects 0.72, 0.26, 0.25, 0.23, 0.20, 0.19, 0.18, 0.17, 0.07, 0.00 against a
  prediction std of 1.76 — one dominant unit, a broad shoulder, one dead unit.
- Linear decodability from the 10-unit bottleneck: $\log f_Q$ $R^2=0.87$;
  $\log\langle|\nabla x|\rangle$ $R^2=0.74$; $|\hat s|$ $R^2=0.08$.
- Linear fidelity to the member's own output: from $(\bar u,g_T,g_n)$
  $R^2=0.84$; from $(\log f_Q,g_T,g_n)$ $R^2=0.79$. The gap between those two is
  roughly the size of the scientific opportunity in this project.

## Questions and preregistered hypotheses

1. What local density does each bottleneck unit average, in terms of channels,
   length scales, and cross-channel alignments — and how does the head combine
   the units?
2. Do early filters detect local primitives such as curvature, compression,
   gradients, extrema, or Fourier content, and do deeper filters combine them
   into physically meaningful motifs?
3. Do high-validation members use the same motifs despite different widths and
   kernels, or do they implement different predictive strategies? Does motif
   agreement vary with validation rank at all, given how small the rank spread
   is?
4. Does the importance of geometry change with $a/L_T$, $a/L_n$, distance from
   the learned stability boundary, equilibrium family, or flux level?
5. Does the ensemble disagree most where members use different features, where
   the geometry is unusual, or where the simulations themselves are noisy?
6. Can a small set of cyclic-invariant physical features reproduce individual
   members or the ensemble mean with high fidelity — and specifically, what does
   the network capture that $\{a/L_T,a/L_n,f_Q\}$ does not?

The primary physics hypotheses are:

- High-performing members respond positively to $|\nabla x|$ concentrated in
  unfavorable-curvature regions.
- A second, distinguishable response is associated with radial magnetic drift /
  geodesic curvature, potentially connected to zonal-flow behavior.
- Relative spatial alignment between channels matters beyond each channel's
  marginal distribution or power spectrum.
- At least one bottleneck unit approximates $\log\langle|\nabla x|\rangle$ or a
  close proxy, because that quantity is an exact additive term in the target.
- These responses are common across high-validation members but their strengths
  and gradient interactions vary.

Negative or mixed results are scientifically useful and must be retained.

## Interpretation contract

Every implementation step must follow these rules.

### Explain the right function

For member $m$, denote the native output by

$$f_m(X,g_T,g_n)=\widehat{\max(\log Q,-2)}.$$

After S02 fixes the canonical object, every position-resolved explanation is
reported for the exactly shift-invariant model
$\tilde f_m(X,g_T,g_n)=\text{MLP}_m(\bar u_m(X),g_T,g_n)$ **and** for the original
$f_m$, with the difference reported. Do not silently substitute one for the
other.

Compute member-level explanations first. For attribution methods that are linear
in the explained function and use the same baseline, an explanation of
$\bar f=100^{-1}\sum_m f_m$ can be cross-checked against the mean member
explanation. Never average absolute attributions before preserving signed
member-level results. Analyze the ensemble spread
$s_f=\operatorname{std}_m f_m$ separately because it is a different, nonlinear
function.

### Respect cyclic symmetry

Let $S_k$ circularly shift all seven channels together by $k$ grid points. A
position-resolved explanation $A$ should ideally obey

$$A(S_kX)=S_kA(X),$$

while a channel- or concept-level summary should be invariant. Report prediction
invariance error and explanation equivariance error for every method. Do not
silently average over shifts. Symmetrization uses **32 phases, not 96** (they are
numerically identical; see fact 3), and must be labeled explicitly wherever used.

The parity operation $P$ ($z\to-z$ with sign flip of channels 3 and 5) is a
second, independent symmetry control. Because it is only approximate in the data,
report the data's own parity mismatch alongside the model's.

### Compare quantities with units fairly

Raw gradients are not comparable across the seven differently scaled physical
channels — channel 4 spans 0.016 to 443 while channel 0 spans 0.50 to 2.2.
Report both contribution-valued attributions (for example integrated gradients,
which include the input-minus-baseline factor) and dimensionless local
sensitivities such as $\sigma_c\,\partial f/\partial X_c$, with a **robust**
scale $\sigma_c$ (median absolute deviation or interquantile range, not standard
deviation) defined on the reference cohort. Record whether signs were retained.

For multichannel sequence models, methods that entangle "which channel" with
"which position" are known to mis-rank both; report channel-marginal and
position-marginal summaries separately, and use a temporal-saliency-rescaling
style two-stage decomposition when producing a joint map (Ismail et al., 2020).

### Do not equate predictiveness with physical causality

Changing one geometry channel alone can violate equilibrium identities and move
off the data manifold — note in particular that channels 0 and 6 are tied
together through $\langle|\nabla x|\rangle$, which is part of the target's
definition. Every perturbation must be tagged as one of:

1. exact symmetry (physically equivalent) — joint circular shift, parity;
2. observed-data comparison or matched natural experiment;
3. plausibly local but not guaranteed physical;
4. deliberately off-manifold diagnostic.

Model sensitivity under categories 3–4 explains the network, not necessarily the
plasma. A physical causal claim requires consistent observational evidence and a
valid equilibrium/GX intervention.

### Require triangulation

A candidate feature is called a **robust learned feature** only if it has:

- a faithful effect in at least one gradient/path method and one perturbation or
  hidden-intervention method;
- the expected cyclic transformation behavior;
- a stable sign or a clearly described regime-dependent sign;
- bootstrap support across equilibria and substantial agreement across members;
  and
- evidence beyond mere decodability from a hidden layer.

A candidate is called **physically supported** only after it also agrees with
held-out GX quantities or a valid new simulation intervention.

### Prevent leakage and pseudoreplication

Split or bootstrap by `equilibrium_files`, not individual flux tubes. Audit the
original random split for (a) identical geometry rows appearing through the
fixed/varied pair on opposite sides of the split and (b) multiple tubes from one
equilibrium crossing splits. Interpretation can still proceed, but the
distinction between interpolation, geometry memorization, and equilibrium-level
generalization must remain visible.

## Recommended methods and software

The state of the art here is a validated collection of complementary methods, not
a single saliency algorithm. Versions checked 2026-08.

| Need | Primary choice | Role and cautions |
|---|---|---|
| PyTorch input/layer attribution | [Captum](https://captum.ai/docs/attribution_algorithms) `0.9.0` | Integrated Gradients, GradientSHAP/Expected Gradients, grouped Feature Ablation/Occlusion, Layer Integrated Gradients / Conductance, `ShapleyValueSampling`, TCAV. Verified to import and run against this repo's torch 2.4.1 / numpy 1.26.4 / Python 3.12. The repo currently pins `0.7.0` (2023); bump it. |
| Bottleneck attribution | Exact Shapley over $C_m\le 32$ units; exact enumeration when $C_m\le 20$ | This is the highest-value method in the project and it is *cheap*. For the top member $2^{10}=1024$ coalitions give exact Shapley values with no sampling error. Use `ShapleyValueSampling` only for the widest members. |
| Attribution evaluation | Custom periodic metrics plus [Quantus](https://github.com/understandable-machine-intelligence-lab/quantus/) `0.6.0` where compatible | Faithfulness, robustness, randomization, complexity. Inspect every metric's perturbation semantics; image defaults are not automatically meaningful for geometry. |
| Sequence-model attribution caveats | Temporal Saliency Rescaling (Ismail et al., NeurIPS 2020) | Their benchmark shows standard saliency methods systematically fail on multivariate sequences by conflating feature and time importance. Implement the two-stage rescaling directly; it is ~30 lines. |
| Space masks | A periodic adaptation of [DynaMask](https://proceedings.mlr.press/v139/crabbe21a.html) / Extremal Mask | The reference implementation, [`time_interpret`](https://github.com/josephenguehard/time_interpret), has had no release since 0.3.0 (2023) — read it for reference, but implement the cyclic version in-repo rather than depending on it. The perturbation operator and wrap-around regularizer must be rewritten for the cyclic domain. Secondary method until it beats simple controls. |
| Relevance propagation | [Zennit](https://zennit.readthedocs.io/) `1.0.0` or Captum LRP | Secondary cross-check only. Both rely on module/rule behavior. S00 already provides an attribution-only `nn.ReLU` module form with verified bit-identical outputs; still document rules for circular `Conv1d`, max pool, biases, and signed/unbounded inputs before use. `zennit-crp` has had no release since 2023 — treat concept-relevance-propagation as optional. |
| Shapley attribution on inputs | Grouped Captum Shapley sampling or [SHAP](https://shap.readthedocs.io/en/latest/) `0.52.0` | Use channels, windows, spectral bands, or physical concepts as coalitions. Do not present 672 grid cells as independent players. Background choice and correlated features materially change the estimand. |
| Hidden concepts | Captum TCAV plus custom sparse probes and interventions | TCAV tests sensitivity to user-defined directions. Multiple random counterexamples, held-out concept classification, significance tests, and causal ablation along the direction are required. |
| Cross-model representations | Linear CKA, bottleneck-unit matching by functional signature, intervention validation | [CKA](https://proceedings.mlr.press/v97/kornblith19a.html) handles different widths. CKA indicates representational similarity, not identical computation; pair it with matched ablations. |
| Feature visualization | Dataset activation maximizers first; regularized input optimization second | Follow the regularization and diversity lessons in [Feature Visualization](https://distill.pub/2017/feature-visualization/). Optimize periodic Fourier coefficients or perturbations around real examples, use random circular jitter, and reject off-manifold adversarial patterns. |
| Interpretable distillation | [InterpretML EBM](https://interpret.ml/docs/) `0.7.8` and [PySR](https://ai.damtp.cam.ac.uk/pysr/v1.5.9/) `1.5.10` | Fit invariant engineered concepts to network predictions, not only to true $Q$. Report a fidelity/complexity Pareto frontier and expression stability. Symbolic regression is late-stage hypothesis compression, not the first explainer. PySR needs a Julia toolchain — budget setup time. |

**Explicitly not primary here.** Plain saliency, Guided Backprop, and a visually
attractive heat map are not sufficient. Grad-CAM is not primary because the task
is scalar regression, the final convolutional map has only three positions, and
the standard ReLU in Grad-CAM discards negative evidence. Sparse autoencoders
have recently been applied to physics surrogates, but they exist to decompose
*wide, polysemantic* representations; a 7–32 unit bottleneck can be enumerated
exactly instead, so an SAE would add estimation error and hyperparameters without
adding resolution. If an SAE is ever used here it belongs on the wider
intermediate conv layers, late in the program, with the exact bottleneck analysis
as its control.

## Standard experiment contract and artifacts

S00 established these conventions; later agents reuse them.

- Reusable code: `itg_nn/xai/`.
- Thin command-line entry points: `scripts/xai_*.py`.
- Small, committed configs: `configs/xai/`.
- Tests: `tests/xai/`, including analytic cyclic toy functions with known relevant
  features.
- Large/generated output: `output/xai/<step>/<run_id>/`, ignored by git.
- Committed step reports: `reports/xai/SNN_<name>.md` plus only the small figures
  necessary to support conclusions.
- Each run directory contains `manifest.json` with git commit, dirty status,
  command, full config, random seeds, Python/package versions, device, dataset
  path and file fingerprint, checkpoint hash, member IDs, row IDs, gradient set,
  wall time, and output hashes.
- Numeric arrays retain axes for member, sample, channel, and position. Prefer a
  self-describing HDF5/Zarr/xarray-compatible layout over anonymous flattened
  arrays.
- Every report includes methods, cohort, estimand, uncertainty, failed checks,
  negative results, interpretation limits, and exact commands to reproduce it.

### Effort budget and the minimum viable deliverable

Each step carries a **Budget** line. The intent is that one agent session
finishes one step. If a step is running past its budget, the agent must:

1. deliver the step's **minimum viable deliverable (MVD)** completely, with tests
   and a report;
2. record precisely what was dropped, in a `## Deferred` section of the step
   report; and
3. stop, rather than delivering a shallow version of everything.

Unless a step says otherwise, begin with a CPU pilot on the top member and
64–128 samples, then run the registered cohort and 512–2,000 stratified samples.
Do not launch a full-scale calculation before the pilot artifacts pass their
acceptance tests.

**Hardware.** The default target is this laptop (CPU/MPS). Steps are sized so
their production runs finish overnight on the laptop. NERSC Perlmutter is
available for genuinely heavy work; a step that wants it must first produce a
laptop-scale result plus a measured extrapolation (wall time, memory, node
hours), then use the compute decision gate. Do not port a calculation to
Perlmutter that has not run at pilot scale locally.

## Numbered research steps

### S00 — Build the reproducible XAI scaffold ✅ complete

Delivered on branch `step00_claude`: `itg_nn/xai/` (config, seeds, batching,
member selection, per-member predictor, activation capture, atomic artifacts and
manifests), `ModuleCyclicInvariantNet` with verified bit-identical outputs for all
100 members, three analytic periodic toy regressors, `scripts/xai_smoke.py`, and
`reports/xai/S00_scaffold.md`.

Two follow-ups were applied to the scaffold outside the step: the Captum pin was
raised from `0.7.0` to `0.9.0` (verified against torch 2.4.1 / numpy 1.26.4 /
Python 3.12.4), and `scripts/setup_xai_env.sh` now creates the venv in one
command. S01 additionally adds a `PeriodicPermutationToy` whose output depends
only on the multiset of per-position channel vectors, as the control for the S03
permutation ladder.

### S01 — Audit the dataset, split, ensemble ranking, and analysis cohorts

**Goal:** Freeze exactly what will be explained, and stop treating the stored
validation ranking as more informative than it is.

**Tasks**

1. Reconstruct train/validation/test membership and audit fixed/varied
   identical-geometry leakage and equilibrium-file overlap across splits. Report
   both counts and the resulting inflation of any tube-level bootstrap.
2. Compute each member's reference-test $R^2$, MSE, bias, residual quantiles, and
   performance by stable/unstable status, flux quantile, gradient bin, and
   equilibrium class.
3. **Re-rank the 100 members** with equilibrium-grouped bootstrap intervals on the
   reference cohort. Report the rank correlation between stored validation score
   and held-out score, and the bootstrap probability that the stored top-10 set
   would be reproduced. Freeze the stored-validation top 10 as the headline
   cohort regardless, but record the evidence about how arbitrary it is.
4. Freeze cohorts: top-10, ranks 11–50, ranks 51–100, all-100, ensemble.
5. Create a stratified interpretation panel including stable, near-threshold,
   low/medium/high flux, large-error, high-disagreement, all five equilibrium
   classes, and gradient regimes, sampled by equilibrium file. Include
   fixed-gradient rows to isolate geometry at constant drive. Register the exact
   row IDs.
6. Record channel robust scales, correlations, Fourier spectra, `nfp`/`iota`/
   `shat`/`aspect`/`rho`/`d_pressure_d_s`, `FSA_grad_xs`, `Q_stds`,
   `Q_avgs_vs_z`, and `zonal_phi2_amplitudes` for the panel rows.
7. Verify the $\langle|\nabla x|\rangle$ identity of fact 4 on the full dataset
   and register $\log\langle|\nabla x|\rangle$ as a ground-truth invariant
   feature for later steps.
8. Add the `PeriodicPermutationToy` control described under S00.

**Budget:** one session; minutes of compute. **MVD:** frozen cohort/panel files
plus the leakage audit.

**Deliverables:** immutable cohort/config files, tidy performance tables, audit
plots, and `reports/xai/S01_audit.md`.

**Accept when:** the varied reference cohort has 9,785 rows and reproduces the
validated ensemble $R^2$ within tolerance; every downstream sample/member can be
addressed by stable IDs; leakage is quantified rather than ignored; the ranking
uncertainty is stated numerically.

### S02 — Symmetry, the canonical invariant model, and the equivariant density

**Goal:** Establish, once, the object that every later step explains.

**Tasks**

1. For all 96 shifts, measure absolute and relative changes in each member's
   output, ensemble mean, and ensemble spread on the S01 panel. Verify the exact
   subgroup at shifts 0, 32, 64 to floating-point tolerance and confirm that
   32-phase and 96-phase averaging agree; adopt 32 phases thereafter.
2. Construct and register three functions per member: the original $f_m$; the
   shift-averaged $\bar f_m=32^{-1}\sum_k f_m(S_k\cdot)$; and
   $\tilde f_m=\text{MLP}_m(\bar u_m,g_T,g_n)$ where
   $\bar u_m=32^{-1}\sum_k u_m(S_k\cdot)$. Report accuracy, residual std, and
   cost for all three on the reference cohort, for all 100 members.
3. Implement the stride-1 (à trous) pooling chain that yields the equivariant
   density $\rho_{m,c}(z)$ at full 96-point resolution, and **verify numerically
   that $\operatorname{mean}_z\rho_{m,c}=\bar u_{m,c}$** and that
   $\rho_{m,c}(S_kX)=S_k\rho_{m,c}(X)$ exactly. This density is the primary
   position-resolved object for S05 and S07.
4. Test the parity symmetry $P$: measure the data's own parity mismatch per
   channel and each member's output change under $P$, and compare with the
   wrong-parity control (plain reversal).
5. Compute every member's theoretical and effective receptive field after each
   conv/pool block, accounting for wrap-around and even kernels; flag members
   whose late blocks are globally connected.
6. Census the bottleneck for all 100 members: width, dead units, near-dead units,
   unit activation statistics, and rank correlation of these with validation
   score.

**Budget:** one session; the 96-shift sweep over 100 members × panel is the
largest piece — pilot on 5 members first and extrapolate before running all 100.
**MVD:** items 1–3 for the top 10 members.

**Deliverables:** per-member symmetry tables, the registered
$f/\bar f/\tilde f$ API, the $\rho$ implementation with its exactness tests,
receptive-field and bottleneck-census tables, and `reports/xai/S02_symmetry.md`.

**Decision gate:** confirm which function is canonical for later steps. The
expectation from scoping is that $\tilde f_m$ becomes canonical because it is
exactly invariant, gives an exact full-resolution density, and is at least as
accurate; if the measurements contradict that, say so and choose differently.

**Accept when:** exact-subgroup and $\rho$ exactness tests pass at stated
tolerances; the arbitrary-shift error is reported relative to each member's own
residual std; no later step is left to assume an unverified invariance.

### S03 — The structure-destroying counterfactual ladder

**Goal:** Bound, cheaply and early, how much of each network's function can be
spatial, spectral, or cross-channel — before spending effort on fine-grained
attribution.

**Tasks**

1. Implement the perturbation/baseline API that later steps need: reference
   distributions (per-channel robust constant profiles; observed background
   samples matched on gradients and equilibrium class; nearest-neighbour/medoid
   backgrounds; low-pass versions of each input), wrapped spatial masks and
   window lengths tied to grid scale and member receptive fields, and the four
   validity tags. **Do not use an all-zero geometry as a default** — $B$ and the
   metric channels are positive and zero is grossly unphysical.
2. Implement a data-support score for endpoints and interpolation paths (robust
   PCA plus held-out nearest-neighbour distance). Use it as a warning, not as
   proof of physical validity, and report it at every perturbation strength.
3. Run the ladder, each with matched random controls, reporting RMS output change
   relative to each member's residual std:
   - joint circular shift (exact symmetry; should be null within S02 tolerance);
   - parity $P$ (near-exact symmetry) and wrong-parity control;
   - **random joint permutation of $z$** — the strongest single test of whether
     anything beyond the pointwise channel-vector multiset is used;
   - **block permutation** at block lengths 2, 4, 8, 16, 32 — converts the above
     into a length-scale spectrum;
   - independent circular shift per channel, and per-channel phase scrambling —
     separates cross-channel co-location from marginal structure;
   - wrapped low/mid/high Fourier band attenuation and phase-preserving amplitude
     scaling — separates marginal power from relative phase;
   - single-channel replacement by its cohort-conditional profile — a coarse
     channel-importance ranking.
4. Validate every operator on the S00/S01 toy models, including the new
   permutation toy, before running on real members.
5. Report the ladder for the top 10 members and, for the cheap entries, all 100.

**Budget:** one session. Each ladder entry is a handful of forward passes per
sample; the whole ladder for 10 members on a 1,000-row panel is minutes on CPU.
**MVD:** the perturbation API with validity tags plus ladder entries for joint
permutation, block permutation, and independent channel shifts.

**Deliverables:** reusable baseline/perturbation API, support diagnostics, the
ladder table and dose-response plots, and `reports/xai/S03_ladder.md`.

**Accept when:** all methods are deterministic under fixed seeds; wrapped windows
have no boundary artifact; toy-model relevant features outrank controls; exact
symmetries are null within S02 tolerances; every perturbation carries a validity
tag and a data-support number.

### S04 — Anatomy of the invariant bottleneck

**Goal:** Fully characterize the low-dimensional invariant summary each network
actually computes, and how its head combines it. This is the step that exploits
structural fact 1.

**Tasks**

1. For each top-10 member, compute $\bar u_m$ on the S01 panel and record it as a
   first-class artifact with stable unit IDs.
2. Compute **exact Shapley values** over the $C_m$ bottleneck units plus the two
   scalar gradients, using the cohort-conditional mean as the reference, by full
   coalition enumeration where $C_m+2\le 20$ and by `ShapleyValueSampling` with
   reported standard errors otherwise. Report both the global (variance-of-output)
   and per-sample decompositions.
3. Single-unit and pairwise interventions: zeroing, mean replacement, and
   resampling from the cohort; measure signed output change, error change, and
   pairwise interaction strength. Build the interaction graph.
4. Partial-dependence and ICE curves for every live unit, and 2-D dependence for
   each unit against $a/L_T$ and $a/L_n$ — the geometry-times-drive interaction
   question becomes a small, exact calculation at the bottleneck.
5. Fit a held-out decoder from $(\bar u_m,g_T,g_n)$ to $f_m$ to quantify how much
   of the member is linear in its own bottleneck, and how much of the residual is
   head nonlinearity.
6. Test decodability of the registered ground-truth invariants from $\bar u_m$:
   $\log\langle|\nabla x|\rangle$, $\log f_Q$, $f_{\text{stab}}$, and simple
   controls ($\hat s$, `nfp`, `aspect`). Report both linear and small-nonlinear
   decoders with equilibrium-grouped cross-validation and label-permutation
   controls.
7. Repeat the cheap parts (1, 3, 6) for all 100 members.

**Budget:** one session; exact Shapley for $C=10$ is 1,024 head evaluations per
sample — trivial. Widest members ($C=32$) need sampling. **MVD:** items 1–3 and 6
for the top 10.

**Deliverables:** bottleneck arrays with stable unit IDs, exact Shapley tables,
interaction graphs, PDP/ICE atlas, decodability matrix, and
`reports/xai/S04_bottleneck.md`.

**Accept when:** Shapley values are exact (or carry reported sampling error);
ablation and Shapley rankings are compared explicitly; "encoded" (decodable) and
"used" (changes the output) are separate columns; random-direction controls are
included.

### S05 — What each bottleneck unit measures

**Goal:** Turn the important units from S04 into named local densities.

**Tasks**

1. For each important unit, compute the equivariant density $\rho_{m,c}(z)$ from
   S02 on the panel. Report its distribution, sparsity, and typical spatial
   support.
2. Construct pointwise and windowed physics concept traces: bad curvature
   $\Theta(\mathbf B\times\boldsymbol\kappa\cdot\nabla y)$, compression
   $|\nabla x|$ and its powers, the $f_Q$ integrand family, radial drift /
   geodesic curvature, local shear
   $S=(d/dz)(\nabla x\cdot\nabla y/|\nabla x|^2)$, $B$ extrema and wells, and
   parallel Fourier scale. Use the paper's unary-operation vocabulary so results
   are directly comparable with its feature table.
3. Regress each $\rho_{m,c}(z)$ on the concept traces with circularly appropriate
   statistics: rank correlation, overlap at fixed sparsity, cross-correlation
   over lag (so a spatially shifted match is not scored as a miss), and partial
   association controlling for individual channel magnitudes.
4. Where no named concept fits, fit a small pointwise/windowed surrogate to
   $\rho_{m,c}$ directly — its input is a 7-channel window of width equal to the
   member's receptive field, which is a far smaller regression problem than the
   full network.
5. Catalog first-layer kernels and their Fourier transfer functions across all
   seven input channels; extract wrapped input receptive-field patches around
   maximal activations, aligned only by explicitly recorded circular operations,
   and display robust center/dispersion rather than cherry-picked examples.
6. Cluster natural activation exemplars to detect polysemantic units; use NMF or
   sparse dictionary learning only if single-unit exemplars are incoherent.

**Budget:** one session. **MVD:** items 1–3 for the top member's live units plus
the second-ranked member as a replication check.

**Deliverables:** density atlas, unit-to-concept alignment tables, filter/transfer
catalog, motif clusters, and `reports/xai/S05_unit_semantics.md`.

**Accept when:** every claimed unit motif has multiple natural exemplars,
bootstrap recurrence over equilibria, receptive-field coordinates, and
shift-consistency; lag is reported rather than assumed zero; no optimized
synthetic input is treated as physical evidence.

### S06 — Benchmark and scale input-space attribution (S06a benchmark, S06b scaled run)

**Goal:** Select input-attribution methods by quantitative behavior, then apply
them at scale — now against a well-defined invariant target function and a
hypothesis space already narrowed by S03–S05.

**Tasks**

1. On the top member and the pilot panel, run signed robust-scaled gradients,
   Integrated Gradients with several S03 references, GradientSHAP / Expected
   Gradients, SmoothGrad-Squared or VarGrad, cyclic grouped occlusion, a periodic
   DynaMask-style mask, and temporal-saliency-rescaling variants of the gradient
   methods. Include LRP only after documenting rule coverage on the S00 module
   form.
2. Evaluate completeness/convergence, baseline stability, cyclic explanation
   equivariance (against both $f$ and $\tilde f$), parameter-randomization
   sensitivity, infidelity/sensitivity, toy-model recovery, perturbation
   faithfulness, sparsity, and runtime.
3. For deletion/insertion tests, include random-order controls and report
   data-support drift at every deletion fraction.
4. Pre-register one primary path/gradient method and one primary perturbation
   method. Retain others as sensitivity analyses.
5. Run the selected methods for every top-10 member on the registered panel,
   retaining signed `(member, sample, channel, z)` arrays and scalar-gradient
   attributions. Aggregate only after member-level storage: median effect,
   interquartile range, sign agreement, rank agreement, and hierarchical
   bootstrap over equilibria and members.
6. Report fixed and varied datasets separately, then stratify by stability, flux,
   $a/L_T$, $a/L_n$, equilibrium class, member error, and ensemble spread. Run a
   smaller registered sensitivity sample for ranks 11–50 and 51–100.
7. Check whether attribution stability correlates with validation $R^2$ — given
   S01's ranking uncertainty, the expected answer is "barely", and a null here is
   a result.

**Budget:** one session for the benchmark (tasks 1–4) and one for the scaled run
(tasks 5–7); if both fit, good, but the benchmark is the MVD and must be complete
and honest before scaling. Split into S06a/S06b branches if needed.

**Deliverables:** method-by-metric benchmark, selected configs, consensus
maps/tables, individual-member small multiples, machine-readable bootstrap
results, and `reports/xai/S06_attribution.md`.

**Accept when:** selected methods beat random/control maps on toy recovery and
faithfulness, respond to parameter randomization, have understood baseline
sensitivity, meet the symmetry behavior permitted by S02; uncertainty includes
both model and equilibrium sampling; signed and absolute summaries are
distinguishable; no feature is called common without an explicit agreement
statistic.

### S07 — Compare learned spatial importance with physics fields and GX $Q(z)$

**Goal:** Translate reliable maps into concrete plasma-geometry hypotheses.

**Tasks**

1. Compare the S05 densities $\rho_{m,c}(z)$ and the S06 attribution maps with
   `Q_avgs_vs_z`, in signed and positive-contribution versions, using circular
   rank correlation, overlap at fixed sparsity, and cross-correlation over lag.
2. Compare sample-level summaries with `zonal_phi2_amplitudes`, which is the
   natural observable for the geodesic-curvature/zonal-flow hypothesis.
3. Use the same geometry's fixed and varied simulation pair as a natural paired
   comparison, while retaining the artificial fixed-set $a/L_T=-3$ marker as a
   potential learned interaction rather than a nuisance to be erased.
4. Identify examples that support and contradict each physics hypothesis, and
   report the contradicting cases with equal prominence.

**Budget:** one session. **MVD:** item 1 for the top-3 members.

**Deliverables:** concept-alignment tables, paired analyses, case studies, and
`reports/xai/S07_physics_alignment.md`.

**Accept when:** associations are equilibrium-bootstrap stable; spatial lag is
reported, not hidden by arbitrary alignment; the distinction between prediction
attribution and physical $Q(z)$ is explicit throughout.

### S08 — Concept probes and TCAV in the hidden layers

**Goal:** Ask whether known or discovered concepts are encoded and used in the
convolutional stack, not only at the bottleneck.

**Tasks**

1. Define continuous and high/low example sets for $f_Q$, $f_{\text{stab}}$,
   compression, bad curvature, geodesic curvature, parallel scale, cross-channel
   co-location, $\log\langle|\nabla x|\rangle$, local $Q(z)$, and zonal-flow
   magnitude. Match counterexamples on gradients, equilibrium class, and simple
   nuisance scales.
2. Fit nested-cross-validated sparse linear probes to each layer's
   representation, splitting by equilibrium. Report decodability with
   label-permutation and random-concept controls, layer by layer, so the depth at
   which a concept appears is visible.
3. Run TCAV-like directional derivatives for the scalar regression output with
   multiple random counterexample sets, member/equilibrium bootstrap intervals,
   and multiple-testing control.
4. Intervene along concept directions and their orthogonal complements; compare
   output effects with equally decodable random directions.
5. Adapt network dissection: downsample the S05 concept masks to each layer and
   quantify unit/concept IoU, mutual information, and selectivity.

**Budget:** one session. **MVD:** items 1–3 for the top-3 members.

**Deliverables:** layer-by-concept encoding/use matrix, TCAV distributions,
directional interventions, dissection tables, and `reports/xai/S08_concepts.md`.

**Accept when:** "encoded" and "used" are separate columns; concept classifiers
generalize by equilibrium; TCAV is stable across counterexample sets; direction
interventions beat matched random controls.

### S09 — Concept completeness and geometry–gradient interactions

**Goal:** Determine how much of each network's computation the candidate concept
set explains, and where it changes with drive.

**Tasks**

1. Predict each member's native output from nested sets of concept scores using a
   held-out simple decoder. Report completeness/fidelity relative to a decoder
   using the full bottleneck $\bar u_m$ — which S04 already established as a
   near-sufficient statistic, making this a clean, bounded comparison.
2. Add concepts one family at a time and report incremental fidelity with
   equilibrium-grouped cross-validation and bootstrap selection stability.
   Explicitly report the increment over the paper's $\{a/L_T,a/L_n,f_Q\}$
   baseline — this number is the headline "what did the network learn that we
   did not already know".
3. Measure geometry-concept interactions with $a/L_T$ and $a/L_n$ using
   stratified directional effects, grouped finite differences, and selected
   integrated Hessian terms rather than a full 674-by-674 Hessian.
4. Separate stability-boundary behavior from stiffness well above threshold, and
   report the floored ($\log Q=-2$) third of the varied set separately throughout.

**Budget:** one session. **MVD:** items 1–2.

**Deliverables:** completeness curves, concept residual analysis,
gradient-interaction surfaces, and `reports/xai/S09_completeness.md`.

**Accept when:** high fidelity is demonstrated on held-out equilibria; added
complexity has an uncertainty-qualified gain; interaction conclusions reproduce
across members and do not depend on the fixed-set marker alone.

### S10 — Representations and motifs common across networks

**Goal:** Find shared computations despite heterogeneous architectures.

**Tasks**

1. Match **bottleneck units across members** by functional signature: correlation
   of $\bar u_{m,c}$ across members on identical panel rows, concept selectivity,
   density shape, Fourier preference, and causal ablation signature. Use
   assignment/clustering with unmatched units allowed. This is the primary
   cross-model analysis because the bottleneck is where members are directly
   comparable despite different widths.
2. Compute linear CKA between every pair of models/layers on identical,
   standardized probe examples, for both flattened spatial activations and
   bottleneck representations. Bootstrap by equilibrium; check outlier
   sensitivity. Use CKA as supporting evidence, not proof of mechanism.
3. Cluster members by predictions, input attributions, bottleneck causal
   signatures, and concept profiles. Compare clusters with validation rank,
   bottleneck width, and architecture — in particular test whether narrow-
   bottleneck members ($C\le 11$) are qualitatively different from wide ones.
4. Define consensus motifs only when functional signatures and interventions
   agree.

**Budget:** one session. **MVD:** item 1 for the top 10 plus item 3.

**Deliverables:** matched motif catalog, cross-model CKA maps, model dendrograms,
consensus statistics, and `reports/xai/S10_cross_model.md`.

**Accept when:** correspondences survive equilibrium bootstrap and are not driven
only by flux labels; matched motifs have comparable causal effects; lower-ranked
cohorts provide a registered comparison.

### S11 — Ensemble disagreement and failure modes

**Goal:** Understand when accurate networks use different evidence or fail
together.

**Tasks**

1. Explain $s_f(X)$ and member residuals with gradients and supported
   perturbations, maintaining the distinction between variance attribution and
   mean-prediction attribution.
2. Relate disagreement and error to data-support distance, equilibrium class,
   gradients, `Q_stds` (simulation time variability), symmetry error, and
   motif/concept use.
3. Build case studies for high spread/low error, high spread/high error, low
   spread/high error, and unanimous success.
4. Test whether ensemble averaging cancels opposing but individually faithful
   feature strategies.

**Budget:** one session. **MVD:** items 1–2.

**Deliverables:** disagreement maps, calibrated diagnostic tables, failure atlas,
and `reports/xai/S11_disagreement.md`.

**Accept when:** residual analyses use held-out targets without selecting models
or features on those same residuals; model uncertainty is not called a confidence
interval; common-mode failure is reported.

### S12 — Distil the networks into invariant physical formulas

**Goal:** Compress supported learned behavior into human-readable hypotheses.

**Tasks**

1. Build a versioned table of cyclic-invariant features from the paper's
   vocabulary plus the S05/S08/S10 motifs, including gradients and a small
   registered set of interactions.
2. Fit an Explainable Boosting Regressor to (a) each top member's output, (b) the
   ensemble mean, and (c) each member's individual bottleneck units — the last is
   the easiest and most interpretable target and should be attempted first.
   Report held-out equilibrium fidelity, main effects, and a tightly limited
   number of pairwise interactions.
3. Run PySR only on the compact, physically named feature set, and separately on
   the task "express $\bar u_{m,c}$ as an invariant reduction of an equivariant
   operation" — the form the network is already known to have. Constrain
   operators and complexity to dimensionally/physically sensible forms.
4. Produce a Pareto frontier of expression complexity versus fidelity, bootstrap
   expression recurrence, and residual maps. Separately score formulas against
   true clipped $\log Q$.
5. Compare the recovered forms with $f_Q$, $f_{\text{stab}}$, the paper's
   geodesic-curvature features, and $\log\langle|\nabla x|\rangle$.

**Budget:** one session, plus PySR/Julia setup time; if the Julia toolchain
fights back, deliver the EBM half as the MVD and register PySR as a follow-up.
**MVD:** item 2 applied to bottleneck units and member outputs.

**Deliverables:** versioned feature table, EBM plots, symbolic-expression Pareto
set, fidelity report, and `reports/xai/S12_distillation.md`.

**Accept when:** formulas generalize by equilibrium, reproduce multiple top
members rather than only the mean, are stable enough to interpret, and clearly
separate model fidelity from physical predictive accuracy.

### S13 — Validate candidates with natural experiments and design GX tests

**Goal:** Move from "the network uses this" toward defensible physical evidence.

**Tasks**

1. Use fixed-gradient data for geometry-only associations, equilibrium-grouped
   matching, nearest-neighbour pairs, and doubly robust sensitivity checks where
   appropriate. Revisit actual $Q$, $Q(z)$, `Q_stds`, and zonal-flow magnitude.
2. Seek pairs that differ strongly in one candidate invariant feature while
   matching gradients and nuisance geometric summaries. Quantify remaining
   imbalance rather than calling the pair causal.
3. Test whether candidate concepts explain true-$Q$ residuals left by $f_Q$ and
   the paper's selected features.
4. Propose a minimal set of equilibrium-consistent geometry modifications and GX
   simulations that distinguish competing learned mechanisms. Include expected
   sign, controls, resolution/convergence checks, and a compute estimate in
   Perlmutter node-hours.

**Budget:** one session. **MVD:** items 1 and 3.

**Deliverables:** natural-experiment report, ranked candidate mechanisms,
contradictory cases, and a prospective GX experiment specification in
`reports/xai/S13_physical_validation.md`.

**Decision gate:** Do not generate new equilibria, retrain networks, or launch GX
simulations without the researcher's approval of the proposed intervention and
compute budget.

**Accept when:** claims are graded as model-mechanistic, observational-physical,
or intervention-ready; confounding and invalid perturbations remain visible.

### S14 — Synthesize the research program

**Goal:** Produce a decision-ready scientific account rather than a gallery of
explanations.

**Tasks**

1. Create an evidence matrix with candidate mechanisms as rows and bottleneck
   Shapley, unit semantics, input attribution, supported perturbation, hidden
   encoding, hidden intervention, cross-model consensus, distillation, $Q(z)$,
   zonal-flow, and natural-experiment evidence as columns.
2. State which hypotheses are supported, contradicted, regime-dependent, or
   unresolved, with uncertainty and negative results.
3. Separate conclusions about the original members, the ensemble mean, the
   symmetrized functions $\bar f$ and $\tilde f$, and actual gyrokinetic
   turbulence.
4. Identify the smallest next calculation that would most reduce scientific
   uncertainty.

**Budget:** one session; no new heavy computation.

**Deliverables:** `reports/xai/FINAL_REPORT.md`, a compact reproducibility index,
and a prioritized next-experiment list.

**Accept when:** every headline conclusion links to machine-readable evidence and
at least two independent method families; every causal statement identifies its
intervention; all runs can be recreated from manifests.

## Dependency and concurrency map

Use this as the default order. Parentheses indicate steps that may run
concurrently in separate worktrees after their prerequisites are merged.

```text
S00 -> S01 -> S02 -> S03 -> (S04, S06a)
S04 -> (S05, S06b)
S05 + S06 -> (S07, S08)
S08 -> S09
S04 + S05 + S08 -> S10
S06 + S10 -> S11
S05 + S09 + S10 -> S12
S07 + S11 + S12 -> S13
S10 + S11 + S12 + S13 -> S14
```

S01, S02 and S03 are deliberately sequential and cheap: each one changes what the
next should measure. S04 and the S06 method benchmark are the first safe
concurrency wave; S05 with S06's scaled run, and later S07 with S08, are the
others. Avoid running multiple full-HDF5 or all-100-member jobs on one
workstation at the same time; parallelize code development and pilots, then
serialize memory/I/O-heavy runs.

## Decision gates that require the researcher

No decision blocks S01–S12. The following choices should be presented with a
short evidence summary when reached, rather than guessed in advance:

1. **Canonical explained function** (end of S02): whether $\tilde f$, $\bar f$, or
   the original $f$ is the primary object for the rest of the program. The agent
   should present the accuracy/invariance/cost table and a recommendation.
2. **Moving a calculation to Perlmutter:** required before any run that will not
   finish overnight on the laptop. Present the measured laptop-scale timing, the
   extrapolation, and the node-hour estimate.
3. **Training-code restoration:** the researcher has training code available but
   prefers not to use it. Ask only if a step's scientific conclusion genuinely
   hinges on retraining (ROAR-style remove-and-retrain, an exactly invariant
   control model, or a concept-bottleneck model), and say what the inference-only
   substitute would be.
4. **New GX calculations:** after S13 proposes equilibrium-consistent, competing
   interventions and their compute budget.
5. **Publication scope:** whether the first paper/report should stop at robust
   network mechanisms or wait for new physical interventions.

## What not to conclude

- A smooth or visually plausible map is not evidence of faithfulness.
- A high probe score means information is available, not that the downstream
  network uses it.
- Feature ablation on an invalid geometry is not a causal plasma experiment.
- A high CKA score is not proof that two networks implement the same algorithm.
- Ensemble agreement can reflect shared training bias or leakage.
- SHAP/IG values depend on a background or baseline and do not become causal
  merely because they have an additive interpretation.
- Global average pooling removes absolute position, but strided pooling leaves
  pooling-phase artifacts of order a third of the model's own error; cyclic
  invariance is measured, not assumed.
- A difference in stored validation $R^2$ of $10^{-3}$ between ensemble members is
  not evidence that one member is better than another.
- Predicting the ensemble with a simple formula explains the ensemble only to the
  demonstrated held-out fidelity.
- Reproducing $f_Q$ is not a discovery; the interesting quantity is what the
  network adds beyond $\{a/L_T, a/L_n, f_Q\}$.

## Research basis

The plan prioritizes methods and safeguards supported by the following primary
papers and maintained project documentation:

- Sundararajan, Taly & Yan, [Axiomatic Attribution for Deep Networks](https://proceedings.mlr.press/v70/sundararajan17a.html)
  (Integrated Gradients and implementation invariance).
- Lundberg & Lee, [A Unified Approach to Interpreting Model Predictions](https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html)
  (SHAP), with the [SHAP documentation's causal warning](https://shap.readthedocs.io/en/latest/example_notebooks/overviews/Be%20careful%20when%20interpreting%20predictive%20models%20in%20search%20of%20causal%20insights.html).
- Adebayo et al., [Sanity Checks for Saliency Maps](https://proceedings.neurips.cc/paper/2018/hash/294a8ed24b1ad22ec2e7efea049b8737-Abstract.html)
  (parameter and data randomization tests).
- Ismail et al., [Benchmarking Deep Learning Interpretability in Time Series Predictions](https://proceedings.neurips.cc/paper/2020/hash/47a3893cc405396a5c30d91320572d6d-Abstract.html)
  (multivariate sequence saliency fails without temporal saliency rescaling).
- Yeh et al., [On the (In)fidelity and Sensitivity of Explanations](https://proceedings.neurips.cc/paper/2019/hash/a7471fdc77b3435276507cc8f2dc2569-Abstract.html).
- Hooker et al., [A Benchmark for Interpretability Methods in Deep Neural Networks](https://proceedings.neurips.cc/paper_files/paper/2019/hash/fe4b8556000d0f0cae99daa5c5c5a410-Abstract.html),
  and Wang & Wang, [Benchmarking Deletion Metrics with the Principled Explanations](https://proceedings.mlr.press/v235/wang24br.html)
  (distribution-shift cautions for deletion metrics).
- Crabbé & van der Schaar, [Explaining Time Series Predictions with Dynamic Masks](https://proceedings.mlr.press/v139/crabbe21a.html).
- Crabbé & van der Schaar, [Evaluating the Robustness of Interpretability Methods through Explanation Invariance and Equivariance](https://papers.neurips.cc/paper_files/paper/2023/hash/e1f418450107c4a0ddc16d008d131573-Abstract-Conference.html).
- Enguehard, [Time Interpret: a Unified Model Interpretability Library for Time Series](https://arxiv.org/abs/2306.02968)
  (reference implementations for sequence masks; unmaintained since 2023).
- Kim et al., [Interpretability Beyond Feature Attribution: TCAV](https://proceedings.mlr.press/v80/kim18d.html).
- Yeh et al., [On Completeness-aware Concept-Based Explanations](https://papers.nips.cc/paper_files/paper/2020/hash/ecb287ff763c169694f682af52c1f309-Abstract.html).
- Ghorbani et al., [Towards Automatic Concept-based Explanations](https://proceedings.neurips.cc/paper/2019/hash/77d2afcb31f6493e350fca61764efb9a-Abstract.html).
- Kornblith et al., [Similarity of Neural Network Representations Revisited](https://proceedings.mlr.press/v97/kornblith19a.html)
  (CKA), with the important functional-similarity caution from
  [Hayne, Jung & Carter](https://openreview.net/forum?id=YY2iA0hfia).
- Majdandzic et al., [Selecting deep neural networks that yield consistent attribution-based interpretations](https://proceedings.mlr.press/v200/majdandzic22a.html)
  (why validation performance alone does not ensure explanation consistency).
- Olah, Mordvintsev & Schubert, [Feature Visualization](https://distill.pub/2017/feature-visualization/).
- Cranmer, [Interpretable Machine Learning for Science with PySR and SymbolicRegression.jl](https://arxiv.org/abs/2305.01582).
- Maintained library documentation for [Captum](https://captum.ai/),
  [Quantus](https://quantus.readthedocs.io/), [Zennit](https://zennit.readthedocs.io/),
  [InterpretML](https://interpret.ml/docs/), and [PySR](https://ai.damtp.cam.ac.uk/pysr/v1.5.9/).
