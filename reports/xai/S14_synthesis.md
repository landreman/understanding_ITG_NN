# S14 — synthesis handoff

The canonical detailed step report and PLAN deliverable is
[FINAL_REPORT.md](FINAL_REPORT.md). The researcher-oriented account is
[S14_executive_summary.md](S14_executive_summary.md). Machine-readable outputs
are under [S14_artifacts](S14_artifacts/).

The registered run is `synthesis-registered-evidence-s01-s13`. It produced an
11-candidate matrix, 64 exact evidence records, nine headline claims, an 18-run
plus one-publication-record provenance index, and five prioritized next
experiments without recomputing model
or GX outputs.

## Acceptance criteria

- Every headline conclusion links to machine-readable evidence and at least two
  independent method families: pass, verified for all nine rows in
  [claim_register.csv](S14_artifacts/claim_register.csv).
- Every causal statement identifies its intervention: pass. There are zero
  physical causal statements; every executed model diagnostic and the
  prospective VMEC intervention are explicitly named in
  [evidence_ledger.csv](S14_artifacts/evidence_ledger.csv) and
  [next_experiments.csv](S14_artifacts/next_experiments.csv).
- All runs can be recreated from manifests: qualified historical failure.
  Seventeen of 18 indexed run manifests are independently recreatable; the
  `S03_PHASE` correction has no recorded Git commit and is honestly marked
  false. The 19th provenance row still pins its exact corrected publication,
  and the S14 [manifest](S14_artifacts/manifest.json) recreates this synthesis.

## Deferred

Nothing from S14. Upstream symbolic-regression and mixed-derivative deferrals
remain visible in [FINAL_REPORT.md](FINAL_REPORT.md).

## Reviewer reproduction

**Recomputable on the slice.** S14 adds no new row calculation. Upstream proxy
checks use the registered S01 panel rows mapped with
`load_review_slice_index().slice_rows()`.

**Checkable from committed artifacts alone.** All synthesis counts, statuses,
source values, hashes, and acceptance checks.

**Not checkable off the researcher's machine, and why.** Full upstream reruns
need the external dataset; prospective VMEC/GX interventions have not run. See
the detailed three-part audit in [FINAL_REPORT.md](FINAL_REPORT.md).
