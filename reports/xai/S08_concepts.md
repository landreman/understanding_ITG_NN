# S08 — Concept probes and TCAV in the hidden layers

## Result

Known geometry concepts are widely encoded in the convolutional stack, but
encoding does not guarantee that the head uses a direction. A sparse probe (a
linear readout constrained to use few hidden features) achieved median
equilibrium-held-out $R^2=\mathbf{0.747}$ over 150 member/layer/concept cells,
versus **-0.00208** after permuting concept labels and **-0.00228** for a
separate random-concept control. Nine of ten concepts pass the preregistered
encoding gate in all 15 member/layer cells. Zonal-flow magnitude does not:
its median $R^2$ is **0.031**, and no zonal cell is called encoded or used.

TCAV-like tests (Testing with Concept Activation Vectors: measuring the native
output derivative along a hidden-layer direction associated with a concept)
are sign-stable across five independently subsampled counterexample sets in
**148/150** cells. Raw uniform-direction interventions exceed eight
norm-matched random directions in **100/150** cells. The complete claim gate is
stricter: held-out encoding, matched-example balance, counterexample sign
stability, a 500-resample equilibrium interval excluding zero, and an
intervention/random ratio above one. **72/150** cells pass all five conditions.

The most replicated concepts are $f_{\mathrm{stab}}$ and $\log f_Q$: each is
encoded and used in **13/15** member/layer cells. Compression passes in
**10/15** and $\log\langle|\nabla x|\rangle$ in **8/15**. Bad curvature,
cross-channel co-location, and geodesic-curvature magnitude each pass in
**7/15**; local $Q(z)$ concentration passes in **5/15**; parallel scale in only
**2/15**. These counts preserve signed member/layer results rather than
averaging members before testing.

Depth matters. Median TCAV derivatives for bad curvature change from **-0.457**
at layer 1 to **+0.142** at layer 5; geodesic curvature changes from **-0.562**
to **+0.141**; cross-channel co-location changes from **-0.604** to **+0.299**.
In contrast, $f_{\mathrm{stab}}$ is positive at every layer (median **+0.323**
to **+0.667**), while parallel scale is negative early (**-0.710**) and near
zero late (**+0.008**). A single unsigned or depth-averaged concept score would
hide these opposing mechanisms.

## Estimand and cohort

The explained function is each top-three member's S02 canonical exactly
shift-invariant output

$$
\tilde f_m(X,g_T,g_n)=\operatorname{MLP}_m(\bar u_m(X),g_T,g_n),
$$

in native $\max(\log Q,-2)$ units. No result exponentiates the prediction.
Each layer representation is the position mean of its full 96-position
$\grave{\mathrm a}$ trous ReLU/max-pool map, so the representation is exactly
cyclic-invariant and retains a stable feature axis. S08 does not produce a
position-resolved map, so the interpretation contract's canonical/original
map comparison is not invoked; every directional output effect is explicitly
for `invariant_tilde_f`.

The cohort is S01's frozen 1,000-row varied-gradient interpretation panel, one
tube from each of 1,000 `equilibrium_files`: **240 stable/near-floor** and
**760 unstable** rows. S02 supplies the canonical model, and S05 supplies the
concept vocabulary. Production read the external HDF5 dataset, never
`tests/data/review_slice.h5`.

The registered run is `concept-probes-top3-panel1000`. The published
[manifest](S08_artifacts/manifest.json) records the external dataset SHA-256
`9d8fa52f93f2782ad9948a38bf46943c0cd6df78cd08b94a006dad4e06c1c8ad`,
checkpoint SHA-256
`d5e092348514a5ee85b68bcdcf51dbb32eaa344beea1daa28f5aaeba9e86eefb`,
CPU execution, seed 20260823, and **2,789.14 s** of production computation. A
1.23 s hash-validated resume added the explicit balance/claim-gate columns and
republished hashes without changing any fitted direction, derivative, or
intervention value.

## Methods

### Concept scores and matched examples

The ten continuous concepts are $\log f_Q$, $f_{\mathrm{stab}}$, compression,
bad curvature, geodesic-curvature magnitude, robustly scaled parallel Fourier
scale, cross-channel co-location, $\log\langle|\nabla x|\rangle$, local $Q(z)$
concentration, and $\log_{10}$ zonal-flow magnitude. Geometry concepts are
observed comparisons, not interventions. Multichannel parallel scale divides
the seven channels by panel IQRs before combining them.

For each concept, the high/low selection first removes a linear association
with $a/L_T$, $a/L_n$, a simple geometry scale, and equilibrium class, then
selects residual extremes within class and pairs them by nearest nuisance
values. [Matched examples](S08_artifacts/matched_examples.csv) retain source
row and equilibrium IDs. The [balance table](S08_artifacts/matching_balance.csv)
reports standardized mean differences (mean separation in pooled standard-
deviation units). Nine concepts have maximum absolute imbalance below 0.25.
Zonal magnitude fails at **0.352** for $a/L_T$; it is kept but fails the use
claim gate.

### Equilibrium-grouped sparse probes

Each layer/concept/member uses five outer and three inner folds, with complete
`equilibrium_files` assigned to folds. Inner folds select among four L1
penalties using a deterministic near-best-score sparsity rule. Outer-fold
predictions give the reported $R^2$; a final all-panel fit supplies a direction
but never its own training score. Group-level label permutation and a distinct
equilibrium-level Gaussian random concept are null controls. The complete
[probe table](S08_artifacts/probe_scores.csv) contains 150 scientific and 15
random-control rows.

Stable/near-floor and unstable $R^2$ values are computed from the untouched
outer-fold predictions. Most geometry results are similar by regime. The main
exception is local $Q(z)$ concentration: median $R^2$ is **-0.087** on
stable/near-floor rows and **0.362** on unstable rows. Zonal scores are strongly
negative within either stratum because their already-small between-row variance
shrinks further.

### Directional derivatives and interventions

Five CAVs per cell come from 75% subsamples of the matched high and low sets.
The derivative sums gradients over all 96 positions because the direction is
added uniformly to the layer map. Member/sample signs are retained. Intervals
resample complete `equilibrium_files` 500 times.

The intervention shifts the real hidden representation in both directions and
continues the canonical network from that layer. It is tagged
`deliberately_off_manifold_diagnostic`: a hidden activation edit diagnoses the
network, not the plasma. Eight random directions of equal norm receive the same
edit magnitude. [TCAV/use results](S08_artifacts/tcav_use.csv) retain the raw
results even when the combined claim gate fails.

### Claim rule

A cell is called encoded and used only when all of the following hold:

1. outer-fold $R^2\ge0.1$ and at least 0.1 above its permuted control;
2. matched-example maximum absolute standardized mean difference $\le0.25$;
3. at least 80% of counterexample sets give the same derivative sign;
4. the equilibrium-bootstrap 95% interval excludes zero; and
5. the concept-direction intervention RMS exceeds the median random-direction RMS.

This rule is a reporting gate, not a tuned classifier. Every component and the
final Boolean appear in the [encoding/use matrix](S08_artifacts/encoding_use_matrix.csv).

## Negative and contradictory results

- Zonal magnitude is the clearest contradiction. It is not decodable, its
  matching misses the balance threshold, yet all 15 arbitrary zonal directions
  beat the random intervention median. This is evidence that an intervention
  ratio alone can manufacture a use story; **0/15** zonal claims are permitted.
- Direction intervention is not universal: 50/150 cells fail even the raw
  intervention/random comparison, and only 72/150 pass the complete gate.
- Parallel scale is decodable (median $R^2=0.544$) but used in only **2/15**
  cells. Information available to a probe need not drive the prediction.
- Local $Q(z)$ concentration is weakly encoded overall and fails on
  stable/near-floor rows, although five later-layer cells pass the full use gate.
- Bad-curvature, geodesic-curvature, and co-location directions reverse sign
  with depth. They cannot be summarized as monotone positive mechanisms.
- Hidden interventions are off-manifold. Even a fully gated cell explains the
  network, not a valid equilibrium or causal plasma response.

## Acceptance criteria

| PLAN criterion | Verdict and evidence |
| --- | --- |
| “encoded” and “used” are separate columns | **Pass.** All 150 rows carry separate held-out `encoded_r2`, signed derivative, intervention, component-gate, and final `use_claim_permitted` columns. Only 72/150 pass the complete rule. |
| concept classifiers generalize by equilibrium | **Qualified pass.** Median outer-fold $R^2=0.747$ versus -0.00208 permuted and -0.00228 random; nine concepts pass in all 15 cells. Zonal magnitude fails in all 15 and is retained as a negative result. Both nested levels split by `equilibrium_files`. |
| TCAV is stable across counterexample sets | **Pass with one matching failure exposed.** 148/150 cells retain their sign across five sets. Nine concept families meet the 0.25 nuisance-balance threshold; zonal magnitude fails at 0.352 and receives no use claim. |
| direction interventions beat matched random controls | **Qualified pass.** 100/150 raw interventions beat the random median; $f_{\mathrm{stab}}$ passes in 13/15 and $\log f_Q$ in 14/15. The complete gate permits 72/150; failures and ratios remain published. |

## Failed checks

- The initial test run failed all intended S08 paths with `NotImplementedError`.
- The first pilot showed a summary-only `KeyError` after adding random-concept
  rows; the artifact summary now selects only rows carrying permutation values.
- A batched random-control implementation was numerically equivalent but slower
  on CPU and was reverted before production.
- Inspection found the uniform layer derivative averaged instead of summed over
  position. A continuation/shift test and the corrected spatial sum now pin the
  intended uniform-edit derivative.
- The first production run failed its post-run nuisance audit: zonal high/low
  sets differed by 1.64 pooled standard deviations in $a/L_T$. A synthetic
  confounding test was written first and failed at 1.99; residualized matching
  made it pass. The original production run was discarded and recomputed.
- The corrected run leaves one explicit failed balance check: zonal $a/L_T$
  imbalance is 0.352 versus the 0.25 threshold. It is gated out, not hidden.

## Mutation testing

The following deliberate mutations turned the focused suite red and were
reverted:

1. assigning outer folds by row rather than `equilibrium_files` failed the
   repeated-equilibrium disjointness test;
2. removing nuisance residualization failed the strong continuous-confounder
   balance test (standardized difference 1.99); and
3. exponentiating the native scalar in the analytic direction test changed the
   known derivative away from 2.5 and failed the native-output assertion.

## Deferred

Network-dissection IoU, mutual information, and selectivity tables (item 5) are
deferred. The MVD (items 1–3) and item 4's direction interventions are complete;
adding a second mask-downsampling pipeline after a 46.5-minute corrected run
would exceed S08's one-session budget. S05 already publishes unit/concept mask
overlap at the bottleneck; S08 protects the layerwise encoding/use result.

## Reproduction

```bash
bash scripts/setup_xai_env.sh
MPLCONFIGDIR=/private/tmp/mpl-s08-pilot XDG_CACHE_HOME=/private/tmp/cache-s08-pilot \
  .venv-xai/bin/python scripts/xai_s08_concepts.py --pilot --no-publish
MPLCONFIGDIR=/private/tmp/mpl-s08-prod XDG_CACHE_HOME=/private/tmp/cache-s08-prod \
  .venv-xai/bin/python scripts/xai_s08_concepts.py
MPLCONFIGDIR=/private/tmp/mpl-s08-resume XDG_CACHE_HOME=/private/tmp/cache-s08-resume \
  .venv-xai/bin/python scripts/xai_s08_concepts.py --resume
.venv-xai/bin/python -m pytest tests/xai/test_concepts.py \
  tests/xai/test_concepts_artifacts.py -q
source .venv-xai/bin/activate && make check
```

## Reviewer reproduction

**Recomputable on the slice.** All 1,000 parent row IDs are S01 panel rows in
`tests/data/review_slice.h5`. Translate them with
`load_review_slice_index().slice_rows()` before loading. The reviewer can
recompute the ten concept scores, exact class membership, nuisance balance,
five-layer canonical representations, grouped outer predictions, permutation
and random-concept controls, TCAV signs, and intervention ratios for all three
members. The full run took 2,789 s, so the 96-row pilot is the practical proxy;
agreement on axes, grouped folds, signs, and claim gates checks the wiring.

**Checkable from committed artifacts alone.** The 150-cell matrix, 165 probe
rows, 3,940 matched-example rows, balance table, summary, figure, and manifest
are committed under [S08 artifacts](S08_artifacts/). Artifact tests recompute
the headline fractions from the CSVs and verify every manifest output hash.

**Not checkable off the researcher's machine, and why.** No headline number
depends on an off-panel row or a git-ignored scientific array. Exact bytewise
reproduction of the external-data run requires the 678 MB source HDF5 and about
47 CPU minutes. The review slice contains the same 1,000 panel rows and all
required held-out diagnostics, so a full slice rerun is numerically equivalent;
the pilot is the cheaper nearest proxy.
