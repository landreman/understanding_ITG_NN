# S06a open researcher decision: control-aware selection and the mask background

## Decision

**Approved by the researcher on 2026-08-22.** Option 2 from the
[S06a report](S06a_attribution_benchmark.md)'s `## Acceptance criteria` section
— rerun a control-aware selection across all five path candidates — plus a
robust-constant (z-median) background variant of the periodic extremal mask to
resolve its fixed-background symmetry failure by construction. The two
outstanding should-fix review items on
[PR #9](https://github.com/landreman/understanding_ITG_NN/pull/9) are fixed in
the same iteration. Estimated cost: one code iteration and under 10 minutes of
production compute. The researcher accepts that this may change the registered
baseline family; a benchmark step is the right place for that to happen.

## The question, as posed before the gate was answered

Two S06a acceptance criteria ended not fully met. First, the network-free
control map $|X-B|$ — rank cells by distance from the baseline, no trained
network involved — matches or beats the selected low-pass IG map on the
stable/near-floor stratum: paired per-row-oriented method-minus-control gaps
are -0.00043 (-0.00376, 0.00209) deletion and -0.00202 (-0.00911, 0.00241)
insertion, while both unstable directions favour the method (0.00964 and
0.01105, both intervals excluding zero). The automated review further computed
that this control *clears the registered per-stratum faithfulness gate* in all
four low-pass cells (stable 1.7404/0.7917, wider than the selected method's
1.5026/0.2113; unstable 0.1684/0.1526) and passes the toy floor (channel top-1
1.000, position AP 1.000); the registered rule rejects it only through the
parameter-randomization clause, where its correlation with the randomized-model
map is exactly 1.000 because the map is model-independent by construction.
Because the control was measured only for the selected pair, whether
robust-constant, medoid, matched-observed, or Expected Gradients would rank
differently under a control-aware rule was unknown. Second, the registered
periodic mask uses a fixed matched-observed background and fails cyclic
explanation equivariance under that registered convention (relative RMS error
1.009), passing only when the background is co-shifted with the input
($2.70\times10^{-7}$).

The three options were: (1) proceed with the selected pair as qualified S06b
sensitivities at no additional compute; (2) rerun control-aware selection
across all five path candidates; (3) demote the mask and select a new
perturbation primary. The recommendation was option 2 plus the median-background
mask variant. It was accepted without change.

## Why this option

- The cost is one iteration and minutes of compute, against anchoring the
  scaled S06b run — and the S07 physics comparison downstream of it — on a
  method whose selection rule is known to be passable by a map containing no
  network information.
- Robust-constant IG was already fully eligible and has a stronger
  parameter-randomization response than low-pass (0.235 versus 0.406); low-pass
  won only on the infidelity tie-break. A control-aware rerun either replaces
  it defensibly or makes its registration genuinely defensible.
- A z-constant background is shift-invariant by construction — there is no
  background z-origin to get wrong — so the mask's fixed-background symmetry
  failure vanishes without registering a co-shift convention. The
  `robust_constant_profile` helper (per-channel median over the 512-row S03
  reference cohort and over z) already exists and is already an IG baseline
  family in this benchmark.
- The stable-stratum control tie is expected to persist under any baseline:
  those rows sit at the clipped-log floor and their endpoint differences are
  ~0.003 native units, so the tie is at least partly mechanical. It is handled
  as a standing caveat, not gated on.

## Instructions to the implementer

Work stays on `codex/xai-s06a-attribution-benchmark` under the existing S06a
step; tests first, mutations named in the PR body, standard definition of done.
The registered 128-row panel, the top member, the explained functions, and the
512-row S03 reference cohort are unchanged. No review-slice regeneration is
expected because no panel row changes.

1. **Fix the two outstanding should-fix review items first.**
   (a) In the report's control-map discussion and next to the open-decision
   text, quote both halves of the reviewer's negative result: the $|X-B|$
   control clears the toy floor and the per-stratum faithfulness gate in all
   four low-pass cells (numbers above, already present in
   `benchmark_metrics.csv`), and is excluded by the registered rule only
   through the randomization clause at exactly 1.000.
   (b) Refresh the PR body's acceptance-criteria list to match the report's
   current verdicts (the pooled 5.247/5.293 and the pre-correction "PASS"
   entries must go) and state this decision and its resolution in the body.

2. **Address the three note-severity findings; they are cheap.** Add one
   assertion on the committed CSV's paired control-gap signs so the production
   control wiring is pinned, not only the statistic; extend the multiplicity
   sentence to cover the control-map paired intervals; and either label the
   `*_margin_vs_control_map` columns as normalized ratios in the report's
   artifact description or drop their pooled `all` rows.

3. **Register a control-aware selection clause and rerun selection over all
   five path candidates** (robust-constant, matched-observed, medoid, low-pass,
   Expected Gradients). For each candidate, compute its own $|X-B|$ control map
   from its own baseline, score it through the same trained-network
   deletion/insertion curves, and publish paired per-row-oriented native-unit
   method-minus-control gaps with 500-draw whole-`equilibrium_files` intervals,
   per stratum and direction. The added registered clause: an eligible path
   method must have paired method-minus-control intervals excluding zero in
   favour of the method in **both unstable directions**. The stable stratum is
   published but not gated, with the near-floor mechanical-tie rationale stated
   in the report. All prior clauses (toy floor, per-stratum positive margins in
   both directions, randomization maximum 0.95, symmetry) and the infidelity
   tie-break are retained unchanged. Record the clause in the config, pilot and
   production selection records, and the manifest, as with the existing rule.
   If the winner changes, that is an accepted outcome of this decision; keep
   the displaced baseline as a published sensitivity analysis either way.

4. **Add a robust-constant-background variant of the periodic extremal mask**
   using the existing z-constant per-channel-median profile as the replacement
   content, leaving the mask optimizer and its registered hyperparameters
   unchanged. Evaluate it through the full gate: toy recovery, per-stratum
   faithfulness in both directions, randomization, its own $|X-B|$ control
   comparison, and fixed-background cyclic equivariance — which should now be
   near machine precision *without* co-shifting; verify that, do not assume it.
   If it passes, register it as the perturbation primary and demote the
   matched-observed mask to a published secondary sensitivity carrying its
   existing fixed-background caveat. If it fails any clause, keep the
   matched-observed mask as the perturbation method, explicitly secondary to
   the path primary and labelled with the fixed-background symmetry failure,
   and report the median-background failure as a negative result. Commit the
   seven per-channel median values as a small artifact so the reviewer can
   reconstruct this background exactly and recompute the median-mask numbers on
   the slice; that moves the new mask variant out of the "not checkable"
   reproduction list that the matched-observed background is confined to.

5. **Carry one standing caveat into the report and forward to S06b:**
   attribution maps on the stable/near-floor stratum are near-uninformative for
   any method, because the clipped output barely moves under edits there
   (median low-pass endpoint difference 0.0014 native units). S06b reports that
   stratum but must not base feature-level claims on it.

6. **Regenerate pilot and production, and update every dependent artifact**:
   `benchmark_metrics.csv`, `grouped_uncertainty.csv`, `faithfulness_curves.csv`,
   `selected_methods.json`, `selected_review_maps.h5` if the selected pair
   changes, the committed manifest copy, the step report, and the executive
   summary. New tests must include an analytic fixture for the control-aware
   clause that fails when the clause is dropped or the control is degenerate,
   and an equivariance test for the median-background mask; show at least two
   step-relevant mutations red and name them in the PR body.

The ~10-minute production rerun is deliberately **not** started in the session
that records this memo; the implementing agent executes it.

## What this does not decide

No feature-level or physics claim. Whether the control-aware winner is low-pass
or another baseline family is an empirical outcome, not part of this decision.
The S06a/S06b split, the registered panel, and the pending member-sampling
uncertainty criterion are unchanged.
