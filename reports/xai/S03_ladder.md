# S03 — The structure-destroying counterfactual ladder

## Status and headline result

Complete. Every result explains the trained native output
`max(log Q, -2)`, never `Q` or `exp(prediction)`. The primary estimand is S02's
exactly shift-invariant canonical member function
`tilde_f_m = MLP_m(mean_z rho_m, a/L_T, a/L_n)`; the original `f_m` is retained
and compared. Member-level signed predictions and changes are preserved before
aggregation.

The networks use substantially more than the multiset of pointwise
seven-channel vectors. On the 1,000-row varied panel, random joint permutation
changes canonical top-10-member outputs by a median **4.54 member residual
standard deviations** (10th–90th member range 4.21–4.78). Independent channel
shifts give 3.35 (3.11–3.63), so cross-channel co-location matters, but it does
not account for the entire ordering effect.

The length and spectral ladders narrow the useful search space. Wrapped block
permutation falls monotonically from 4.10 residual SD at block length 2 to 1.25
at length 32. Full attenuation of Fourier bins 1–4 changes the output by 5.30,
versus 1.66 for bins 5–16 and only 0.140 for bins 17–48. Later attribution work
should therefore prioritize low-frequency and short-to-intermediate ordering
structure rather than fine-grid high-frequency detail.

The ignored production run is
`output/xai/S03/ladder-top10-all100-panel2000-wrappedblocks/`. Its manifest
hashes the full signed prediction tensor, the full 5.9 MB member/function/stratum
table, support paths, figures, protected checkpoint, and external dataset. The
committed compact table is
[`S03_artifacts/ladder_summary.csv`](S03_artifacts/ladder_summary.csv), with
machine-readable checks in [`S03_artifacts/summary.json`](S03_artifacts/summary.json).

## Registered cohort, functions, and uncertainty

The production cohort is the frozen S01 panel: 1,000 varied-gradient rows from
1,000 distinct `equilibrium_files` and their 1,000 fixed-gradient twins. No row
or member was selected after seeing S03 results. The complete ladder was run for
the stored-validation top 10; 13 cheap entries were run for all 100 registered
members. Random operators have three fixed-seed replicates for the top 10, with
replicate zero covering all 100.

For varied rows, RMS output change is divided by each member's S02 full-reference
residual standard deviation for the same function and all/stable/unstable
stratum. Fixed rows are reported in native clipped-log units without borrowing a
varied-row denominator. Top-10 member intervals use 200 resamples of whole
`equilibrium_files`; because the panel has one row per equilibrium in each
gradient set, this is also the row count, but the registered sampling unit
remains the equilibrium. All ten joint-permutation and independent-shift
intervals exclude zero; their median interval widths are 0.582 and 0.595
residual SD, respectively.

## Perturbation and baseline API

`itg_nn.xai.perturbations` provides validity-tagged, deterministic operators and
the reusable baseline family required downstream:

- per-channel robust constant profiles using the pooled reference median;
- observed backgrounds matched on `(a/L_T, a/L_n)` within equilibrium class,
  excluding the source row;
- nearest-neighbour and observed medoid backgrounds;
- input-specific periodic low-pass backgrounds;
- equilibrium-class/gradient-conditional channel profiles;
- hard wrapped windows with no truncation at index 0, and window registries tied
  to both grid scales and every member's S02 receptive fields; and
- linear endpoint paths with support diagnostics at doses 0, 0.25, 0.5, 0.75,
  and 1.

An all-zero geometry is explicitly forbidden as a default. Every ladder row is
tagged `exact_symmetry`, `observed_comparison`,
`plausibly_local_not_guaranteed_physical`, or
`deliberately_off_manifold_diagnostic`. The structure-destroying edits and
single-channel replacements are deliberately off manifold: they explain the
network, not the plasma.

Block permutations use a seeded random cyclic origin per sample before
permuting contiguous blocks, then roll back, so index 0 is never a privileged
boundary. Shared phase scrambling applies the same random phase at each
frequency to all channels and therefore preserves cross-channel phase
differences; per-channel scrambling destroys them. Both preserve every
channel's Fourier amplitude exactly.

## Exact and near-exact controls

All exact controls pass S02's `atol=rtol=2e-5` standard. Across production,
maximum absolute errors are 9.54e-6 for original `f` under shift 32, 7.63e-6
for `tilde_f` under shift 32, and at most 7.63e-6 for `tilde_f` under arbitrary
per-sample joint shifts. Normalized top-10 canonical shift RMS is about 1.0e-6.

The near-exact stellarator parity changes canonical outputs by median 0.193
residual SD, whereas the matched wrong-parity reversal changes them by 1.75.
The all-100 medians are 0.178 and 1.80, so the result is not a top-10 accident.
Parity is not called a numerical null: S02 showed that channels 3 and 5 obey it
only approximately in observed data.

The original model behaves as expected: arbitrary joint shifts change top-10
outputs by median 0.536 residual SD, while canonical `tilde_f` is null. The
larger structure tests agree across functions: original `f` gives medians 4.35
for joint permutation and 3.22 for independent shifts, versus 4.54 and 3.35 for
`tilde_f`.

## Ordering, co-location, and length scale

For canonical `tilde_f` on varied rows, replicate-zero top-10 and all-100
member distributions agree:

| Operator | Top-10 median (10%–90%) | All-100 median (10%–90%) |
| --- | ---: | ---: |
| Joint permutation | 4.54 (4.21–4.78) | 4.45 (4.07–4.93) |
| Independent channel shifts | 3.35 (3.11–3.63) | 3.15 (2.93–3.45) |
| Shared phase scramble | 4.47 (3.70–5.57) | 4.18 (3.30–5.80) |
| Per-channel phase scramble | 4.05 (3.22–6.97) | 3.84 (3.18–5.17) |

Joint permutation preserves every pointwise channel vector but destroys their
parallel order, so its large effect is a direct rejection of a multiset-only
description. The median signed changes are negative: -0.682 clipped-log units
for joint permutation, -0.201 for independent shifts, -0.436 for shared phase
scrambling, and -0.228 for per-channel scrambling. Signed member values remain
in the full table; the signs are not silently absolutized.

The preliminary expectation that cross-channel alignment carries *most* of the
ordering signal is not supported. Shared phase scrambling, which preserves
relative phase between channels, is at least as disruptive as per-channel
scrambling. That negative/contradictory result means marginal spatial ordering
or waveform organization is itself important, and these off-manifold probes do
not cleanly isolate a unique cross-channel mechanism.

Wrapped block-permutation medians are 4.10, 3.35, 2.53, 1.73, and 1.25 for
lengths 2, 4, 8, 16, and 32. Across three random replicates, the range of the
member median is at most 0.181 residual SD. The monotonic curve indicates a
broad length spectrum: preserving progressively longer local runs reduces the
damage, but reordering three length-32 blocks still exceeds the model's own
residual scale.

## Spectral and channel ladders

![Phase-preserving Fourier dose response](S03_artifacts/dose_response.png)

Low-band and all-non-DC attenuation are strong and monotone at every registered
dose. At doses 0.25, 0.5, 0.75, and 1, the low-band top-10 medians are 2.59,
4.22, 5.00, and 5.30 residual SD. Mid-band values are 0.582, 1.04, 1.40, and
1.66; high-band values are only 0.040, 0.082, 0.114, and 0.140. Removing all
non-DC amplitude gives 2.65, 4.34, 5.28, and 5.55. These are phase-preserving
amplitude interventions; they do not establish that a particular Fourier mode
is physically causal.

Conditional single-channel replacement gives this coarse top-10 ranking:

| Channel | Median RMS / residual SD |
| --- | ---: |
| `gbdrift` | 3.06 |
| `gds22_over_shat_squared` | 2.81 |
| `gds2` | 2.73 |
| `gbdrift0_over_shat` | 2.52 |
| `gds21_over_shat` | 1.77 |
| `cvdrift` | 1.59 |
| `bmag` | 0.880 |

This is not a causal or independent channel attribution. Channel correlations,
the channel-0/channel-6 target identity, and the off-manifold replacement path
all prevent that interpretation. It is a prioritization result for later
joint/concept studies.

![Canonical structure-destroying ladder](S03_artifacts/ladder_overview.png)

## Drive dependence and support warnings

Stable/near-floor and unstable varied rows are never pooled. Top-10 canonical
medians for joint permutation are 1.76 residual SD on stable/near-floor rows and
4.55 on unstable rows; independent shifts are 2.10 and 3.35; shared phase
scrambling is 1.85 and 4.44; per-channel phase scrambling is 2.99 and 4.01.
Even after each stratum uses its own residual scale, the driven branch is more
sensitive to most structure destruction.

Fixed-gradient twins are much less sensitive in absolute units. Their top-10
canonical median RMS changes are 0.0212 for joint permutation, 0.0286 for
independent shifts, 0.0470 for shared phase scrambling, and 0.0590 for
per-channel phase scrambling. No fixed/varied ratio is formed.

The data-support warning uses per-channel median/IQR scaling, a 24-component PCA
fit on 1,536 equilibrium-unique reference rows, and nearest-neighbour calibration
on 512 held-out equilibria. Cyclic phase is canonicalized before fitting and
scoring. Both unusually large and unusually small reconstruction/distance
values warn. This diagnostic is not a physical-validity test.

Exact-shift endpoints retain essentially the source support distribution: about
10.2%–10.8% of rows exceed the held-out 95th-percentile warning threshold,
versus 10.6% at dose zero. Their linear interpolation midpoints are less
supported (20.4%–22.4%), illustrating why an exact endpoint does not validate a
linear attribution path. Complete non-DC removal is strongly warned (median
score 1.0; 73.4% above threshold), as are full low-band attenuation (median
0.863; 31.5%) and mid-band attenuation (0.824; 22.8%).

An important negative check is that PCA support does **not** flag every known
off-manifold edit: joint permutation has only 5.7% above threshold and
independent shifts 5.2%. The warning captures coarse reference variance, not
equilibrium constraints or fine ordering. Low warning values must never be
reported as evidence of realizability.

## Toy controls and failed checks

All registered operator tests pass. The S01 permutation toy is invariant to
joint and wrapped-block permutations to numerical tolerance but changes under
independent shifts. The colocation toy is invariant to joint permutation and
changes under independent shifts. The Fourier toy's relevant-band effect is
24.0 versus 5.72e-6 for the matched high-band control. Phase scrambling preserves
the amplitude-spectrum control. Wrapped windows keep identical support when
crossing index 0. All seeded operators reproduce bit for bit.

The first production implementation used block partitions anchored at index 0
and a raw-coordinate support PCA. Artifact inspection caught both weaknesses.
The final registered run randomizes and wraps block origins and uses cyclic
support canonicalization. The superseded ignored run is retained rather than
deleted, but no number from it is cited. Its non-block results were numerically
identical; the block-family median changed from 2.58 to 2.53.

Negative results retained here are: shared phase scrambling is not smaller than
per-channel scrambling; support PCA misses several deliberately off-manifold
operators; sensitivity is strongly drive dependent; and the conditional
replacement ranking cannot isolate independent physical channel effects.

## Reproduction and artifacts

Pilot, production, and resume commands:

```bash
MPLCONFIGDIR=/private/tmp/mpl-s03 .venv-xai/bin/python \
  scripts/xai_s03_ladder.py --config configs/xai/S03_ladder.json \
  --pilot --no-publish

MPLCONFIGDIR=/private/tmp/mpl-s03 .venv-xai/bin/python \
  scripts/xai_s03_ladder.py --config configs/xai/S03_ladder.json

MPLCONFIGDIR=/private/tmp/mpl-s03 .venv-xai/bin/python \
  scripts/xai_s03_ladder.py --config configs/xai/S03_ladder.json --resume
```

The final run took 3,093 s (51.6 minutes) on CPU. The pilot used one member,
64 varied rows plus fixed twins, two random replicates, and passed before
production. The production manifest records Python 3.12.4, torch 2.4.1, numpy
1.26.4, h5py 3.11.0, and Captum 0.9.0. The external 678,040,404-byte dataset
SHA-256 is
`9d8fa52f93f2782ad9948a38bf46943c0cd6df78cd08b94a006dad4e06c1c8ad`;
the protected checkpoint SHA-256 is
`d5e092348514a5ee85b68bcdcf51dbb32eaa344beea1daa28f5aaeba9e86eefb`.
Both were read only.

`predictions.h5` retains `(member, function, perturbation, sample)` axes,
including signed native outputs. Resume validates dataset, checkpoint, row IDs,
member IDs, perturbation registry, resolved configuration, and hashes of the CLI
and perturbation implementation before accepting any cache. The full ladder,
support model, and arrays remain ignored; committed summaries and figures are
copies of manifest-hashed outputs.

Verification commands:

```bash
conda run -n 20240629-01-ML python -m pytest
.venv-xai/bin/python -m pytest
git diff --check
```

## Acceptance criteria

| Criterion | Evidence | Status |
| --- | --- | --- |
| Deterministic fixed-seed methods | Unit tests, toy checks, experiment fingerprint | Pass |
| Wrapped windows have no boundary artifact | Shifted wraparound support equality | Pass |
| Toy relevant features outrank controls | Permutation, colocation, Fourier, and window checks | Pass |
| Exact symmetries null within S02 tolerance | Maximum absolute error 9.54e-6 < 2e-5 | Pass |
| Every perturbation has a validity tag | Machine-readable tag on all 56 entries | Pass |
| Every strength has a support number | 56 entries x five path doses | Pass |
| Top-10 full ladder | Both `f` and `tilde_f`, fixed/varied and flux strata | Pass |
| Cheap entries cover all 100 members | 13 entries, signed member predictions retained | Pass |
| Baseline API avoids all-zero default | Six registered baseline families | Pass |
| Member-level grouped uncertainty | 200 equilibrium-file resamples for top 10 | Pass |

## Deferred

Nothing. All five S03 tasks, the full registered panel, the top-10 ladder, and
the all-100 cheap entries were completed.
