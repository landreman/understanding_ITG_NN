# S14 executive summary — the decision-ready story

## What this final step did

The earlier steps deliberately asked different questions: whether the networks
respect the right symmetry, which inputs and hidden units matter to them,
whether ensemble members agree, whether their behavior can be rewritten using
named geometry quantities, and whether those quantities also track real GX
simulations. S14 joins those answers without running a new model or simulation.

It builds a machine-readable evidence matrix with 11 candidate explanations and
all 11 evidence types required by the plan. Every headline conclusion has at
least two independent analysis families behind it. Every source number can be
traced to an exact row in a committed CSV, and every production run from S00 to
S13 has a hash-checked manifest in the reproducibility index.

Throughout, the quantity being explained is the native network target
`max(log Q,-2)`: logarithmic heat flux with a floor at -2. Individual network
members, their ensemble mean, the symmetry-averaged member functions, and true
GX output are kept separate. That distinction is the guardrail that prevents a
finding about a neural network from being presented as a finding about plasma
physics.

## The central scientific picture

The ensemble has learned a real, compact geometry story, but not one simple
formula or one common spatial mechanism.

The strongest repeated ingredients are the paper's $f_Q$ and $f_{\rm stab}$
features, a localized peak in bad-curvature-weighted compression,
bad-curvature and geodesic-curvature combinations, alignment among geometry
channels, and interactions between geometry and the imposed gradients. Several
quite different tests find this same family: credit assigned to hidden units,
edits to inputs and hidden states, comparisons across members, and readable
regression models fitted to network outputs.

The networks care about arrangement along the field line. Independently shifting
the seven geometry channels changes member outputs by a median 2.41 residual
standard deviations. Removing the low-frequency spatial structure changes them
by 3.85, while the high-frequency control—applied at a roughly seven-times
smaller robust input dose—changes them by only 0.10. The middle band gives an
intermediate 1.22 response, and the exact common-shift control is only
0.00000081. Those results show that alignment and broad spatial structure matter. They
do not yet tell us whether parallel order itself matters after broad structure
is held fixed; that decomposition remains unresolved.

The members agree most strongly at a coarse level. Their channel rankings have
median pairwise agreement 0.964, but signed position-by-position agreement is
only 0.749. A 17-feature readable model reproduces three separate members at
85.6–86.4% of held-out output variance. That is useful compression, not an exact
replacement: its residual errors remain much larger than the networks' errors
against GX, and only 13 of 64 tested internal units are reproduced at 80% or
better.

## Two appealing explanations were rejected

First, the networks do not share a signed spatial mechanism that focuses where
GX transports heat. For three leading members, signed attribution-to-$Q(z)$
rank correlations are -0.021, -0.013, and -0.012—essentially zero—and their
best spatial offsets disagree. A positive-only picture looks more similar only
because it throws away negative contributions.

Second, the tested evidence contradicts a specific learned zonal-flow mechanism.
GX zonal-flow magnitude is not predictably encoded in the selected hidden
layers, and none of the 15 hidden-state use tests passes. Zonal flow can still be
associated with geometry in the physical simulations; that is different from
showing that the feed-forward network directly represents and uses it.

## What has physical support

The leading network features are not arbitrary: all four physical candidates
are associated with true GX heat flux at fixed gradient drive. But the existing
equilibria change several geometry properties together. Even after matching and
statistical adjustment, the high- and low-candidate rows remain too different
to isolate one feature.

The best next target is geodesic-curvature/compression. Its adjusted association
with true clipped-log heat flux is +0.559, with a 95% range [0.318, 0.764], and
it supplies a resolved improvement beyond the paper's selected quantities in
all seven alternative fold assignments. Yet the remaining nuisance mismatch is
1.068 against an allowed 0.5, and usable overlap is 0.478 against a required
0.8. It is therefore the most informative candidate to test, not a proven
physical cause.

That distinction is important. Input and hidden-state edits performed earlier
were deliberately unrealistic changes used to diagnose the network. The
physical comparisons used real force-balanced equilibria but did not intervene.
No step has yet changed only one candidate in a recomputed equilibrium and then
measured the GX response. There is consequently no physical causal conclusion
in the final claim register.

## How to use ensemble spread

Member disagreement is a useful alarm. Across all 1,000 varied-gradient rows,
ensemble spread and absolute prediction error have rank correlation 0.761, with
a resampling range [0.730, 0.794]. But spread is not an error bar. The absolute
error is larger than the spread on 44.9% of rows, and eight registered cases
show all members agreeing closely while being collectively wrong. Shared
training data and architecture make this unsurprising.

The practical rule is: use spread to prioritize cases for checking, not to put
a confidence interval around a prediction.

## The next decision

The smallest calculation that reduces the largest scientific uncertainty is a
VMEC-only feasibility test, not the proposed full GX campaign.

At three typical unstable equilibria, the test would search boundary-coefficient
directions that move geodesic/compression or the competing
bad-curvature/compression quantity up and down. Every move would recompute a
force-balanced equilibrium. Success requires at least half a panel interquartile
range of candidate movement while every protected constraint stays within one
tenth of its panel interquartile range, in both signs at all three anchors.

The planning allowance is two Perlmutter node-hours and no GX runs. If it fails,
the proposed physical intervention does not exist in its current form and the
expensive campaign should be redesigned. If it passes, a separate small GX
timing and convergence pilot comes next; only after that would the 32.5-node-hour
competing intervention test be considered.

The recommendation is to approve the VMEC feasibility test when the researcher
is ready to authorize external computation. S14 itself does not run it.

## What is now reproducible

The committed S14 package contains:

- an [11-candidate evidence matrix](S14_artifacts/evidence_matrix.csv), including
  uncertainty and negative results;
- a [64-record evidence ledger](S14_artifacts/evidence_ledger.csv) with exact
  source selectors and values;
- a [nine-headline claim register](S14_artifacts/claim_register.csv), with at
  least two method families per claim;
- an [18-run/19-record provenance index](S14_artifacts/reproducibility_index.csv),
  with manifest paths, the S03 publication verification, SHA-256 hashes, and an
  explicit content-hash pin for every evidence artifact; and
- a [five-item next-experiment list](S14_artifacts/next_experiments.csv).

The full interpretation, limitations, commands, mutation tests, acceptance
criteria, and reviewer instructions are in [FINAL_REPORT.md](FINAL_REPORT.md).

## Deferred

Nothing from S14. Earlier unfinished work remains visible: the symbolic PySR
search and true mixed network derivatives were not retroactively claimed. VMEC
and GX calculations are future experiments that require explicit authorization.

## Reviewer reproduction

**Recomputable on the slice.** S14 adds no row-level calculation. The registered
1,000 varied and 1,000 fixed panel rows can be mapped through
`load_review_slice_index().slice_rows()` for the upstream proxy checks; the
synthesis itself is reproduced from committed CSVs.

**Checkable from committed artifacts alone.** All candidate counts, statuses,
headline claims, exact source values, family counts, manifest hashes, and next
experiment gates are committed and covered by tests.

**Not checkable off the researcher's machine, and why.** Full S01–S13 reruns
need the external HDF5 dataset and some ignored large arrays. The review slice
checks the same code path and registered panel rows but not claims outside it.
The proposed VMEC/GX experiments cannot be reproduced because they have not run.
