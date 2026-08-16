# Research plan: interpreting the cyclic ITG heat-flux neural networks

## Purpose

The scientific goal is to determine which geometric patterns the trained neural
networks use to predict turbulent ion heat flux, which patterns are shared by the
highest-performing ensemble members, and which findings survive tests of
faithfulness, cyclic symmetry, and physical plausibility.

This is a plan for studying the existing trained networks. It is not a claim that
an attribution map is a causal explanation of gyrokinetic turbulence. The desired
end product is an evidence-weighted set of candidate physical mechanisms and
compact invariant features that can be checked against the simulation data and,
where justified, new GX calculations.

Each numbered step below is intentionally scoped for one focused Codex or Claude
task. `WORKFLOW.md` explains how to dispatch and merge those tasks.

## Facts that constrain the analysis

- The canonical local dataset is
  `/Users/mattland/20260523-01-files_for_Kosmos_interpreting_neural_networks/20250102-01_GX_stellarator_dataset.h5`.
  XAI scripts should use this as their default while retaining an optional
  `--dataset` override for portability. Agents should not require an environment
  variable or copy the file into this repository.
- `models/cyclic_ensemble_pre2.pt` contains 100 PyTorch ensemble members selected
  from 443 hyperparameter-search trials. The stored validation $R^2$ values span
  approximately 0.98516--0.98725. The primary "highest-$R^2$" cohort in this
  plan is the top 10 members by the stored validation score; ranks 11--50 and
  51--100 are comparison cohorts. Never select the primary cohort using the test
  set.
- Every member has five circular `Conv1d` layers, each followed by ReLU and
  stride-2 max pooling, then global average pooling, two scalar gradient inputs,
  two dense layers, and one scalar output. Kernel widths, channel counts, and
  dense widths vary across members.
- The native output is a prediction of $\max(\log Q,-2)$. Primary fidelity and
  attribution calculations must therefore explain this scalar, not $Q$ and not
  `exp(prediction)`. Stable/near-floor and unstable cases must be reported
  separately.
- Geometry has shape `(96, 7)`. The authoritative HDF5 channel order is:

  | Index | HDF5 name | Physical label in the file |
  |---:|---|---|
  | 0 | `bmag` | $B$ |
  | 1 | `gbdrift` | $2B^{-3}\mathbf B\times\nabla B\cdot\nabla y$ |
  | 2 | `cvdrift` | $2B^{-2}\mathbf B\times\boldsymbol\kappa\cdot\nabla y$ |
  | 3 | `gbdrift0_over_shat` | $2B^{-3}\mathbf B\times\nabla B\cdot\nabla x$ |
  | 4 | `gds2` | $|\nabla y|^2$ |
  | 5 | `gds21_over_shat` | $\nabla x\cdot\nabla y$ |
  | 6 | `gds22_over_shat_squared` | $|\nabla x|^2$ |

- The two additional inputs are $a/L_T$ and $a/L_n$. The legacy training
  convention used $a/L_T=-3$ as a marker for the fixed-gradient simulations;
  this is not a physical negative temperature gradient. Fixed- and
  varied-gradient results must not be pooled without recording this distinction.
- Five rounds of stride-2 pooling reduce 96 positions to 3. Circular convolution
  is shift-equivariant, but strided max pooling need not be equivariant to a
  one-grid-point shift. Exact invariance is expected at least for the subgroup of
  shifts by multiples of 32, while arbitrary-shift invariance is learned only
  approximately. This must be measured, not assumed.
- The HDF5 data include `Q_avgs_vs_z`, which was not a network input. It is a
  valuable independent spatial diagnostic for comparing learned saliency with
  where GX says the heat-flux contribution occurs.
- The paper's leading hypotheses are flux-surface compression in regions of bad
  curvature and the magnitude of geodesic curvature. A central reference feature
  is

  $$
  f_Q=\operatorname{mean}\left(
  [\Theta(\mathbf B\times\boldsymbol\kappa\cdot\nabla y)+0.2]
  |\nabla x|^3/B\right).
  $$

  These are hypotheses to test against the networks, not labels to force onto
  every discovered pattern.

## Questions and preregistered hypotheses

1. Which channels, positions, length scales, and cross-channel alignments change a
   member's predicted clipped log flux?
2. Do early filters detect local primitives such as curvature, compression,
   gradients, extrema, or Fourier content, and do deeper filters combine them into
   physically meaningful motifs?
3. Do the top-10 validation members use the same motifs despite different widths
   and kernels, or do they implement different predictive strategies?
4. Does the importance of geometry change with $a/L_T$, $a/L_n$, distance from the
   learned stability boundary, equilibrium family, or flux level?
5. Does the ensemble disagree most where members use different features, where
   the geometry is unusual, or where the simulations themselves are noisy?
6. Can a small set of cyclic-invariant physical features reproduce individual
   members or the ensemble mean with high fidelity?

The primary physics hypotheses are:

- High-performing members respond positively to $|\nabla x|$ concentrated in
  unfavorable-curvature regions.
- A second, distinguishable response is associated with radial magnetic drift /
  geodesic curvature, potentially connected to zonal-flow behavior.
- Relative spatial alignment between channels matters beyond each channel's
  marginal distribution or power spectrum.
- These responses are common across the top-10 members but their strengths and
  gradient interactions vary.

Negative or mixed results are scientifically useful and must be retained.

## Interpretation contract

Every implementation step must follow these rules.

### Explain the right function

For member $m$, denote the native output by

$$f_m(X,g_T,g_n)=\widehat{\max(\log Q,-2)}.$$

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
silently average over shifts. If a symmetrized model is useful, define and label it
explicitly as

$$\bar f_m(X)=96^{-1}\sum_{k=0}^{95}f_m(S_kX).$$

Its group-averaged explanations describe $\bar f_m$, not necessarily the original
$f_m$.

### Compare quantities with units fairly

Raw gradients are not comparable across the seven differently scaled physical
channels. Report both contribution-valued attributions (for example integrated
gradients, which include the input-minus-baseline factor) and dimensionless local
sensitivities such as $\sigma_c\,\partial f/\partial X_c$, with robust scale
$\sigma_c$ defined on the reference cohort. Record whether signs were retained.

### Do not equate predictiveness with physical causality

Changing one geometry channel alone can violate equilibrium identities and move
off the data manifold. Every perturbation must be tagged as one of:

1. exact symmetry (physically equivalent);
2. observed-data comparison or matched natural experiment;
3. plausibly local but not guaranteed physical;
4. deliberately off-manifold diagnostic.

Model sensitivity under categories 3--4 explains the network, not necessarily the
plasma. A physical causal claim requires consistent observational evidence and a
valid equilibrium/GX intervention.

### Require triangulation

A candidate feature is called a **robust learned feature** only if it has:

- a faithful effect in at least one gradient/path method and one perturbation or
  hidden-intervention method;
- the expected cyclic transformation behavior;
- a stable sign or a clearly described regime-dependent sign;
- bootstrap support across samples and substantial agreement across the top-10
  members; and
- evidence beyond mere decodability from a hidden layer.

A candidate is called **physically supported** only after it also agrees with
held-out GX quantities or a valid new simulation intervention.

### Prevent leakage and pseudoreplication

Where possible, split or bootstrap by `equilibrium_files`, not individual flux
tubes. Audit the original random split for (a) identical geometry rows appearing
through the fixed/varied pair on opposite sides of the split and (b) multiple
tubes from one equilibrium crossing splits. Interpretation can still proceed, but
the distinction between interpolation, geometry memorization, and equilibrium-
level generalization must remain visible.

## Recommended methods and software

The state of the art here is a validated collection of complementary methods, not
a single saliency algorithm.

| Need | Primary choice | Role and cautions |
|---|---|---|
| PyTorch input/layer attribution | [Captum](https://captum.ai/docs/attribution_algorithms) | Use Integrated Gradients, GradientSHAP/Expected Gradients, grouped Feature Ablation/Occlusion, Layer Integrated Gradients or Conductance, and TCAV. It fits the existing PyTorch code and exposes convergence or evaluation metrics. |
| Attribution evaluation | Custom periodic metrics plus [Quantus](https://github.com/understandable-machine-intelligence-lab/quantus/) where compatible | Use faithfulness, robustness, randomization, and complexity metrics. Inspect every metric's perturbation semantics; image defaults are not automatically meaningful for geometry. |
| Time/space masks | A periodic adaptation of [DynaMask](https://proceedings.mlr.press/v139/crabbe21a.html) | Useful for smooth, sparse multichannel masks. The perturbation operator and wrap-around regularizer must be rewritten for the cyclic domain. Treat this as a secondary method until it beats simple controls. |
| Relevance propagation | [Zennit](https://zennit.readthedocs.io/) or Captum LRP | A secondary cross-check only. Both rely on module/rule behavior. First make an attribution-only model with `nn.ReLU` modules and verify identical predictions; document rules for circular Conv1d, max pool, biases, and signed/unbounded inputs. |
| Shapley attribution | Grouped Captum Shapley sampling or [SHAP](https://shap.readthedocs.io/en/latest/) | Use channels, windows, spectral bands, or physical concepts as coalitions. Do not present 672 grid cells as independent players. Background choice and correlated features materially change the estimand. |
| Hidden concepts | Captum TCAV plus custom sparse probes and interventions | TCAV tests sensitivity to user-defined directions. Multiple random counterexamples, held-out concept classification, significance tests, and causal ablation along the direction are required. |
| Cross-model representations | Linear CKA, activation-signature matching, and intervention validation | [CKA](https://proceedings.mlr.press/v97/kornblith19a.html) handles different widths and initializations. CKA indicates representational similarity, not identical computation; pair it with matched ablations. |
| Feature visualization | Dataset activation maximizers first; regularized input optimization second | Follow the regularization and diversity lessons in [Feature Visualization](https://distill.pub/2017/feature-visualization/). Optimize periodic Fourier coefficients or perturbations around real examples, use random circular jitter, and reject off-manifold adversarial patterns. |
| Interpretable distillation | [InterpretML EBM](https://interpret.ml/docs/) and [PySR](https://ai.damtp.cam.ac.uk/pysr/v1.5.9/) | Fit invariant engineered concepts to network predictions, not only to true $Q$. Report a fidelity/complexity Pareto frontier and expression stability. Symbolic regression is late-stage hypothesis compression, not the first explainer. |

Plain saliency, Guided Backprop, and a visually attractive heat map are not
sufficient. Grad-CAM is not primary here because the task is scalar regression,
the final convolutional map has only three positions, and the standard ReLU in
Grad-CAM discards negative evidence. Signed layer-gradient-times-activation and
layer interventions are more informative. DeepLIFT/LRP must not be used directly
on the current functional `F.relu` implementation without verifying rule coverage.

## Standard experiment contract and artifacts

Step 00 should establish these conventions; later agents should reuse them.

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

Unless a step says otherwise, begin with a CPU pilot on the top member and 64--128
samples, then run the registered top-10 cohort and 512--2,000 stratified samples.
Use the full 100 members only for inexpensive summaries or after the top-10 result
is stable. Do not launch a full-scale calculation before the pilot artifacts pass
their acceptance tests.

## Numbered research steps

### S00 — Build the reproducible XAI scaffold

**Goal:** Make later experiments routine and comparable.

**Tasks**

1. Add an XAI optional dependency specification and create instructions for a
   project-local virtual environment. Per `AGENTS.md`, do not install new packages
   into conda environment `20240629-01-ML`.
2. Add shared configuration, manifest, hashing, deterministic seed, member
   selection, batching, activation-capture, and artifact-writing utilities.
3. Add a wrapper that returns individual member predictions while preserving the
   validated inference behavior.
4. Add an attribution-only module-form network (`nn.ReLU` rather than functional
   ReLU) if needed; prove its output is bit-for-bit or tolerance-level equivalent
   for every member on a fixture batch.
5. Add analytic periodic toy regressors whose outputs depend on known channels,
   windows, cross-channel co-location, and Fourier bands.
6. Ignore large `output/xai/` artifacts while preserving small reports.

**Deliverables:** shared modules, configs, tests, environment instructions, a
one-command smoke run, and `reports/xai/S00_scaffold.md`.

**Accept when:** existing tests pass; all 100 original/wrapped member outputs
match; the smoke artifact is reproducible from its manifest; no scientific result
is claimed yet.

### S01 — Audit the dataset, split, ensemble, and analysis cohorts

**Goal:** Freeze exactly what will be explained.

**Tasks**

1. Reconstruct train/validation/test membership and audit fixed/varied identical-
   geometry leakage and equilibrium-file overlap.
2. Compute each member's reference-test $R^2$, MSE, bias, residual quantiles, and
   performance by stable/unstable status, flux quantile, gradient bin, and
   equilibrium class. Keep stored validation rank as the primary rank.
3. Freeze top-10, ranks 11--50, ranks 51--100, and ensemble cohorts.
4. Create a stratified interpretation panel including stable, near-threshold,
   low/medium/high flux, large-error, high-disagreement, all equilibrium classes,
   and gradient regimes. Include fixed-gradient rows to isolate geometry.
5. Record channel scales, correlations, Fourier spectra, target noise proxies
   (`Q_stds`), `Q_avgs_vs_z`, and `zonal_phi2_amplitudes` for those rows.

**Deliverables:** immutable cohort/config files, tidy performance tables, audit
plots, and `reports/xai/S01_audit.md`.

**Accept when:** the varied reference cohort has 9,785 rows and reproduces the
validated ensemble $R^2$ within tolerance; every downstream sample/member can be
addressed by stable IDs; leakage is quantified rather than ignored.

### S02 — Define baselines, perturbations, and off-manifold diagnostics

**Goal:** Prevent arbitrary references from determining the answer.

**Tasks**

1. Implement and compare reference distributions: per-channel constant robust
   profiles, observed background samples matched on gradients/equilibrium class,
   nearest-neighbor or medoid backgrounds, and low-pass versions of each input.
   Do not use an all-zero geometry as the default because $B$ and metric channels
   are positive and zero is grossly unphysical.
2. Implement wrapped spatial masks and window lengths tied to grid scale and
   member receptive fields.
3. Implement a data-support score for endpoints and interpolation paths using
   robust PCA plus held-out nearest-neighbor distance. Use it as a warning, not as
   proof of physical validity.
4. Implement exact-symmetry, observed comparison, plausibly local, and
   off-manifold perturbation tags.
5. Validate baseline completeness and perturbation behavior on the analytic toy
   models from S00.

**Deliverables:** reusable baseline/perturbation API, support diagnostics,
baseline comparison report, and `reports/xai/S02_baselines.md`.

**Accept when:** all methods are deterministic under fixed seeds; wrapped windows
have no boundary artifact; toy-model relevant features outrank controls; every
perturbation carries a validity tag.

### S03 — Measure prediction invariance and layer equivariance

**Goal:** Determine what cyclic symmetry the trained networks actually possess.

**Tasks**

1. For all 96 shifts, measure absolute and relative changes in each member's
   output, ensemble mean, and ensemble spread on the interpretation panel.
2. Capture pre-ReLU, post-ReLU, post-pool, and GAP representations and measure
   the expected equivariance/invariance at every layer.
3. Verify the predicted exact subgroup at shifts 0, 32, and 64, then characterize
   arbitrary-shift phase artifacts by architecture and performance rank.
4. Compute every member's effective receptive field after each conv/pool block,
   accounting for wrap-around and even kernels.
5. Compare the original model with the explicitly shift-averaged model, including
   accuracy and cost.

**Deliverables:** per-member symmetry tables, layer heat maps, receptive-field
table, and `reports/xai/S03_symmetry.md`.

**Decision gate:** If arbitrary-shift variation is scientifically material, all
later work reports original and symmetrized-model explanations separately. A
future exactly invariant retraining experiment becomes optional; it does not
replace interpretation of the existing networks.

**Accept when:** exact-subgroup tolerances are tested; no explanation method is
run under an unverified invariance assumption.

### S04 — Benchmark candidate input-attribution methods

**Goal:** Select methods by quantitative behavior, not visual appeal.

**Tasks**

1. On the best validation member and the pilot panel, run signed standardized
   gradients, Integrated Gradients with several S02 references, GradientSHAP /
   Expected Gradients, SmoothGrad-Squared or VarGrad, cyclic grouped occlusion,
   and a periodic DynaMask-style mask. Include LRP only if S00 verified the module
   conversion and conservation rules.
2. Evaluate completeness/convergence, baseline stability, cyclic explanation
   equivariance, parameter-randomization sensitivity, infidelity/sensitivity,
   toy-model recovery, perturbation faithfulness, sparsity, and runtime.
3. For deletion/insertion tests, include random-order controls and report data-
   support drift at every deletion fraction.
4. Pre-register one primary path/gradient method and one primary perturbation
   method. Retain other methods only as sensitivity analyses.

**Deliverables:** a method-by-metric benchmark, local example plots with signed
channels, selected configs, and `reports/xai/S04_method_benchmark.md`.

**Accept when:** selected methods beat random/control maps on toy recovery and
faithfulness, respond to parameter randomization, have understood baseline
sensitivity, and meet the symmetry behavior permitted by S03.

### S05 — Scale validated input attribution across the top ensemble

**Goal:** Estimate global and regime-specific learned input importance.

**Tasks**

1. Run the selected S04 methods for every top-10 member on the registered panel,
   retaining signed `(member, sample, channel, z)` arrays and scalar-gradient
   attributions.
2. Aggregate only after member-level storage: median effect, interquartile range,
   sign agreement, rank agreement, and hierarchical bootstrap intervals over
   equilibria and members.
3. Report fixed and varied datasets separately, then stratify by stability, flux,
   $a/L_T$, $a/L_n$, equilibrium class, member error, and ensemble spread.
4. Run a smaller registered sensitivity sample for ranks 11--50 and 51--100 to
   test whether consensus is unique to the top cohort.
5. Check whether attribution stability is correlated with validation $R^2$ rather
   than assuming that accuracy guarantees interpretable maps.

**Deliverables:** consensus maps/tables, individual-member small multiples,
machine-readable bootstrap results, and `reports/xai/S05_input_consensus.md`.

**Accept when:** uncertainty includes both model and equilibrium sampling; signed
and absolute summaries are distinguishable; no feature is called common without
an explicit agreement statistic.

### S06 — Run physics-directed and spectral counterfactuals

**Goal:** Test what spatial and spectral information the network function uses.

**Tasks**

1. Apply joint circular shifts as a negative-control symmetry and independent
   channel shifts as a test of cross-channel co-location.
2. Apply wrapped low/mid/high Fourier band attenuation, phase-preserving amplitude
   changes, common phase changes, and channel-specific phase scrambling. Separate
   marginal power from relative cross-channel phase.
3. Apply smooth cyclic window replacement toward matched observed backgrounds at
   several scales; compare attribution-ranked and random windows.
4. Compute response curves for the seven channel groups and two scalar inputs
   using grouped Shapley sampling or matched finite differences.
5. Track data-support distance and explicitly distinguish model-behavior claims
   from physical interventions.

**Deliverables:** spectral sensitivity maps, alignment/co-location tests,
counterfactual dose-response plots, and `reports/xai/S06_counterfactuals.md`.

**Accept when:** valid symmetry controls are null within S03 tolerances;
attribution-ranked perturbations outperform random controls without markedly
worse support distance, or the failure is recorded.

### S07 — Compare learned spatial importance with physics fields and GX $Q(z)$

**Goal:** Translate reliable maps into concrete plasma-geometry hypotheses.

**Tasks**

1. Construct pointwise or windowed concept traces for bad curvature, surface
   compression, their product/family around $f_Q$, radial drift/geodesic
   curvature, metric shear, $B$ extrema, and parallel Fourier scale.
2. Compare signed network attributions and perturbation masks with these traces
   using circularly appropriate rank correlation, overlap at fixed sparsity,
   cross-correlation over lag, and partial association controlling individual
   channel magnitudes.
3. Compare them independently with `Q_avgs_vs_z`, including signed and positive-
   contribution versions, and with `zonal_phi2_amplitudes` at sample level.
4. Use the same geometry's fixed and varied simulation pair as a natural paired
   comparison, while retaining the artificial fixed-set $a/L_T=-3$ marker as a
   potential learned interaction.
5. Identify examples that support and contradict each physics hypothesis.

**Deliverables:** concept-alignment tables, paired analyses, case studies, and
`reports/xai/S07_physics_alignment.md`.

**Accept when:** reported associations are equilibrium-bootstrap stable; spatial
lag is not hidden by arbitrary alignment; contradictory cases and the distinction
between prediction attribution and physical $Q(z)$ are explicit.

### S08 — Build an activation atlas and learned-filter catalog

**Goal:** Inspect what convolutional channels detect before asking whether they
affect the output.

**Tasks**

1. Record pre/post-ReLU and post-pool activations for all blocks and GAP vectors
   for the top-10 members on a shared probe cohort.
2. For every filter, compute activation frequency, sparsity, moments, output-
   conditioned response, preferred Fourier content, and top natural examples.
3. Extract the wrapped input receptive-field patches around maximal activations.
   Align only by explicitly recorded circular operations and display robust
   center/dispersion, not just a cherry-picked example.
4. Visualize first-layer kernel weights and Fourier transfer functions across all
   seven input channels.
5. Cluster natural activation exemplars to detect polysemantic filters; use NMF or
   sparse dictionary learning only if single-channel exemplars are incoherent.

**Deliverables:** browsable/static atlas, filter metadata table, natural motif
clusters, and `reports/xai/S08_activation_atlas.md`.

**Accept when:** every claimed filter motif has multiple natural exemplars,
bootstrap recurrence, receptive-field coordinates, and shift-consistency; no
optimized synthetic input is treated as physical evidence.

### S09 — Test hidden units with causal activation interventions

**Goal:** Distinguish features merely encoded in a layer from features used by the
downstream network.

**Tasks**

1. Intervene after ReLU, after pooling, and at GAP by turning off one channel,
   replacing it with its cohort-conditional mean, and ablating matched channel
   groups. Measure signed output change, error change, and nonlinear interaction
   between ablations.
2. Perform wrapped spatial activation patching between matched high/low-activation
   examples and compare with random patches.
3. Compute signed layer-gradient-times-activation and layer conductance; compare
   their ranks with direct ablation.
4. Adapt network dissection: downsample S07 physical concept masks to each layer
   and quantify unit/concept IoU, mutual information, and activation selectivity.
5. Repeat key interventions across top-10 members and relevant gradient regimes.

**Deliverables:** causal channel rankings, interaction graphs, dissection tables,
and `reports/xai/S09_hidden_interventions.md`.

**Accept when:** an important unit/channel changes the actual output under a
controlled intervention; mean replacement and zeroing agree qualitatively or the
difference is explained; random-channel/patch controls are included.

### S10 — Test physics concepts with probes and TCAV

**Goal:** Ask whether known or discovered concepts are encoded and used.

**Tasks**

1. Define continuous and high/low example sets for $f_Q$, the paper's stability
   feature, compression, bad curvature, geodesic curvature, parallel scale,
   cross-channel co-location, local $Q(z)$, and zonal-flow magnitude. Match
   counterexamples on gradients, equilibrium class, and simple nuisance scales.
2. Fit nested-cross-validated sparse linear probes to each layer/GAP
   representation, splitting by equilibrium. Report decodability with label-
   permutation and random-concept controls.
3. Run TCAV-like directional derivatives for the scalar regression output with
   multiple random counterexample sets, member/equilibrium bootstrap intervals,
   and multiple-testing control.
4. Intervene along concept directions or their orthogonal complements and compare
   output effects with equally decodable random directions.

**Deliverables:** layer-by-concept encoding/use matrix, TCAV distributions,
directional interventions, and `reports/xai/S10_concepts.md`.

**Accept when:** "encoded" and "used" are separate columns; concept classifiers
generalize by equilibrium; TCAV is stable across counterexample sets; direction
interventions beat matched random controls.

### S11 — Measure concept completeness and geometry–gradient interactions

**Goal:** Determine how much of each network's computation the candidate concept
set explains and where it changes with drive.

**Tasks**

1. Predict each member's native output from nested sets of concept scores using a
   held-out simple decoder. Report completeness/fidelity relative to a decoder
   using the full GAP representation.
2. Add concepts one family at a time and report incremental fidelity with
   equilibrium-grouped cross-validation and bootstrap selection stability.
3. Measure geometry-concept interactions with $a/L_T$ and $a/L_n$ using stratified
   directional effects, grouped finite differences, and selected integrated
   Hessian terms rather than a full 674-by-674 Hessian.
4. Separate stability-boundary behavior from stiffness well above threshold.

**Deliverables:** completeness curves, concept residual analysis, gradient-
interaction surfaces, and `reports/xai/S11_completeness_interactions.md`.

**Accept when:** high fidelity is demonstrated on held-out equilibria; added
complexity has an uncertainty-qualified gain; interaction conclusions reproduce
across members and do not depend on the fixed-set marker alone.

### S12 — Identify representations and motifs common across networks

**Goal:** Find shared computations despite heterogeneous architectures.

**Tasks**

1. Compute linear CKA between every pair of models/layers on identical,
   standardized probe examples, for both flattened spatial activations and GAP
   representations. Bootstrap by equilibrium and check sensitivity to outliers.
2. Match filters across members using their activation signatures on shared
   natural patches, physics-concept selectivity, Fourier preference, and causal
   ablation signatures. Use assignment/clustering with unmatched units allowed;
   do not match raw weights alone.
3. Cluster members by predictions, input attributions, hidden causal signatures,
   TCAV/concept profiles, and representations. Compare these clusters with
   validation rank and architecture.
4. Define consensus motifs only when functional signatures and interventions
   agree; use CKA as supporting representation evidence, not proof of mechanism.

**Deliverables:** cross-model CKA maps, matched motif catalog, model dendrograms,
consensus statistics, and `reports/xai/S12_cross_model.md`.

**Accept when:** correspondences survive bootstrap samples and are not driven only
by flux labels; matched motifs have comparable causal effects; lower-ranked
cohorts provide a registered comparison.

### S13 — Explain ensemble disagreement and failure modes

**Goal:** Understand when accurate networks use different evidence or fail
together.

**Tasks**

1. Explain $s_f(X)$ and member residuals with gradients and supported
   perturbations, maintaining the distinction between variance attribution and
   mean-prediction attribution.
2. Relate disagreement/error to data-support distance, equilibrium class,
   gradients, simulation time variability, symmetry error, and motif/concept use.
3. Build case studies for high spread/low error, high spread/high error, low
   spread/high error, and unanimous success.
4. Test whether ensemble averaging cancels opposing but individually faithful
   feature strategies.

**Deliverables:** disagreement maps, calibrated diagnostic tables, failure atlas,
and `reports/xai/S13_disagreement.md`.

**Accept when:** residual analyses use held-out targets without selecting models or
features on those same residuals; model uncertainty is not called a confidence
interval; common-mode failure is reported.

### S14 — Distill the networks into invariant physical formulas

**Goal:** Compress supported learned behavior into human-readable hypotheses.

**Tasks**

1. Build a table of cyclic-invariant features from the paper plus S06--S12 motifs,
   including gradients and a small registered set of interactions.
2. Fit an Explainable Boosting Regressor to (a) each top member and (b) the
   ensemble mean. The target here is the network output. Report held-out
   equilibrium fidelity, main effects, and a tightly limited number of pairwise
   interactions.
3. Run PySR only on the compact, physically named feature set. Constrain operators
   and complexity to dimensionally/physically sensible forms where possible.
4. Produce a Pareto frontier of expression complexity versus fidelity, bootstrap
   expression recurrence, and residual maps. Separately score formulas against
   true clipped $\log Q$.
5. Compare the recovered forms with $f_Q$ and the paper's geodesic-curvature
   features.

**Deliverables:** versioned feature table, EBM plots, symbolic-expression Pareto
set, fidelity report, and `reports/xai/S14_distillation.md`.

**Accept when:** formulas generalize by equilibrium, reproduce multiple top
members rather than only the mean, are stable enough to interpret, and clearly
separate model fidelity from physical predictive accuracy.

### S15 — Validate candidates with natural experiments and design GX tests

**Goal:** Move from "the network uses this" toward defensible physical evidence.

**Tasks**

1. Use fixed-gradient data for geometry-only associations, equilibrium-grouped
   matching, nearest-neighbor pairs, and doubly robust sensitivity checks where
   appropriate. Revisit actual $Q$, $Q(z)$, `Q_stds`, and zonal-flow magnitude.
2. Seek pairs that differ strongly in one candidate invariant feature while
   matching gradients and nuisance geometric summaries. Quantify remaining
   imbalance rather than calling the pair causal.
3. Test whether candidate concepts explain true-Q residuals left by $f_Q$ and the
   paper's selected features.
4. Propose a minimal set of equilibrium-consistent geometry modifications and GX
   simulations that distinguish competing learned mechanisms. Include expected
   sign, controls, resolution/convergence checks, and compute estimate.

**Deliverables:** natural-experiment report, ranked candidate mechanisms,
contradictory cases, and a prospective GX experiment specification in
`reports/xai/S15_physical_validation.md`.

**Decision gate:** Do not generate new equilibria, retrain networks, or launch GX
simulations without the researcher's approval of the proposed intervention and
compute budget.

**Accept when:** claims are graded as model-mechanistic, observational-physical,
or intervention-ready; confounding and invalid perturbations remain visible.

### S16 — Synthesize the research program

**Goal:** Produce a decision-ready scientific account rather than a gallery of
explanations.

**Tasks**

1. Create an evidence matrix with candidate mechanisms as rows and input
   attribution, supported perturbation, hidden encoding, hidden intervention,
   cross-model consensus, distillation, $Q(z)$, zonal-flow, and natural-experiment
   evidence as columns.
2. State which hypotheses are supported, contradicted, regime-dependent, or
   unresolved, with uncertainty and negative results.
3. Separate conclusions about the original members, the ensemble mean, the
   symmetrized functions, and actual gyrokinetic turbulence.
4. Identify the smallest next calculation that would most reduce scientific
   uncertainty.

**Deliverables:** `reports/xai/FINAL_REPORT.md`, a compact reproducibility index,
and a prioritized next-experiment list.

**Accept when:** every headline conclusion links to machine-readable evidence and
at least two independent method families; every causal statement identifies its
intervention; all runs can be recreated from manifests.

## Dependency and concurrency map

Use this as the default order. Parentheses indicate tasks that may run
concurrently in separate worktrees after their prerequisites are merged.

```text
S00 -> S01 -> (S02, S03) -> S04
                         -> (S05, S06, S08)
S05 + S06 -> S07
S07 + S08 -> (S09, S10)
S09 + S10 -> S11
S08 + S09 + S10 + S11 -> S12
S05 + S06 + S12 -> S13
S07 + S11 + S12 -> S14
S07 + S13 + S14 -> S15
S12 + S13 + S14 + S15 -> S16
```

S03 can start alongside S02 because it uses only unperturbed circular shifts.
S05, S06, and S08 write to separate modules and result directories and are the
largest safe concurrency wave. Avoid running multiple full-HDF5 or all-100-member
jobs on one workstation at the same time; parallelize code development and pilots,
then serialize memory/I/O-heavy runs.

## Decision gates that require the researcher

No decision blocks S00--S14. The following choices should be presented with a
short evidence summary when reached, rather than guessed in advance:

1. **Exact invariance retraining:** only if S03 shows scientifically important
   arbitrary-shift artifacts.
2. **Large compute expansion:** before scaling expensive masks/LRP/TCAV from the
   top-10 pilot to all 100 members or substantially more rows.
3. **Training-code restoration:** before ROAR-style retraining, concept bottleneck
   training, or channel-removal retraining. The current repository is
   inference-only, and deletion without retraining has distribution-shift
   confounding.
4. **New GX calculations:** after S15 proposes equilibrium-consistent, competing
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
- Global average pooling removes absolute position, but strided pooling can leave
  pooling-phase artifacts; cyclic invariance still requires measurement.
- Predicting the ensemble with a simple formula explains the ensemble only to the
  demonstrated held-out fidelity.

## Research basis

The plan prioritizes methods and safeguards supported by the following primary
papers and maintained project documentation:

- Sundararajan, Taly & Yan, [Axiomatic Attribution for Deep Networks](https://proceedings.mlr.press/v70/sundararajan17a.html)
  (Integrated Gradients and implementation invariance).
- Lundberg & Lee, [A Unified Approach to Interpreting Model Predictions](https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html)
  (SHAP), with the [SHAP documentation's causal warning](https://shap.readthedocs.io/en/latest/example_notebooks/overviews/Be%20careful%20when%20interpreting%20predictive%20models%20in%20search%20of%20causal%20insights.html).
- Adebayo et al., [Sanity Checks for Saliency Maps](https://proceedings.neurips.cc/paper/2018/hash/294a8ed24b1ad22ec2e7efea049b8737-Abstract.html)
  (parameter and data randomization tests).
- Yeh et al., [On the (In)fidelity and Sensitivity of Explanations](https://proceedings.neurips.cc/paper/2019/hash/a7471fdc77b3435276507cc8f2dc2569-Abstract.html).
- Hooker et al., [A Benchmark for Interpretability Methods in Deep Neural Networks](https://proceedings.neurips.cc/paper_files/paper/2019/hash/fe4b8556000d0f0cae99daa5c5c5a410-Abstract.html),
  and Wang & Wang, [Benchmarking Deletion Metrics with the Principled Explanations](https://proceedings.mlr.press/v235/wang24br.html)
  (distribution-shift cautions for deletion metrics).
- Crabbé & van der Schaar, [Explaining Time Series Predictions with Dynamic Masks](https://proceedings.mlr.press/v139/crabbe21a.html).
- Crabbé & van der Schaar, [Evaluating the Robustness of Interpretability Methods through Explanation Invariance and Equivariance](https://papers.neurips.cc/paper_files/paper/2023/hash/e1f418450107c4a0ddc16d008d131573-Abstract-Conference.html).
- Kim et al., [Interpretability Beyond Feature Attribution: TCAV](https://proceedings.mlr.press/v80/kim18d.html).
- Yeh et al., [On Completeness-aware Concept-Based Explanations](https://papers.nips.cc/paper_files/paper/2020/hash/ecb287ff763c169694f682af52c1f309-Abstract.html).
- Ghorbani et al., [Towards Automatic Concept-based Explanations](https://proceedings.neurips.cc/paper/2019/hash/77d2afcb31f6493e350fca61764efb9a-Abstract.html).
- Kornblith et al., [Similarity of Neural Network Representations Revisited](https://proceedings.mlr.press/v97/kornblith19a.html)
  (CKA), with the important functional-similarity caution from
  [Hayne, Jung & Carter](https://openreview.net/forum?id=YY2iA0hfia).
- Majdandzic et al., [Selecting deep neural networks that yield consistent attribution-based interpretations](https://proceedings.mlr.press/v200/majdandzic22a.html)
  (why validation performance alone does not ensure explanation consistency).
- Olah, Mordvintsev & Schubert, [Feature Visualization](https://distill.pub/2017/feature-visualization/).
- Maintained library documentation for [Captum](https://captum.ai/),
  [Quantus](https://quantus.readthedocs.io/), [Zennit](https://zennit.readthedocs.io/),
  [InterpretML](https://interpret.ml/docs/), and [PySR](https://ai.damtp.cam.ac.uk/pysr/v1.5.9/).
