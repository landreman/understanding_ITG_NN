# Reference validation

The inference-only implementation was validated against
`pred_vs_actual_plot_pre2.pdf` from the legacy neural-network directory using
the external `20250102-01_GX_stellarator_dataset.h5` file.

## Data equivalence

Direct array comparisons established the following exact mappings:

- pickle `tensor` = HDF5 `raw_feature_tensor`
- pickle `tprims` = HDF5 `*/a_over_LT`
- pickle `fprims` = HDF5 `*/a_over_Ln`
- pickle `Q_avgs_without_FSA_grad_x` = HDF5 `*/Q_avgs`

The reference split reconstruction produced 9,785 varied-gradient test rows,
matching the legacy run. It uses the original seed (42), concatenation order
(fixed followed by varied), positive-heat-flux filter, and 80/10/remainder
split.

## Fixed-gradient input convention

The legacy training script negates `tprims` for the fixed-gradient half of the
data (`Cyclic_net.py:152`), describing it as a trick so the reported test score
uses varied-gradient rows only. That negation postdates the ensemble's training
and does **not** describe the inputs the members learned from. Established
2026-08-20 from three sources:

- The serialized `train_val_test_dataset_5_pre_2.pth` holds 199,637 rows with no
  negative `a/L_T`; its fixed half is exactly `+3`. Its saved `test_dataset` has
  19,965 rows — the unfiltered split — so it was written before the `< 0` filter
  had anything to drop.
- `out-35793251.log`, the run behind `pred_vs_actual_plot_pre2.pdf`, records
  `New test_dataset: 19965 9785`, so the negation *was* active by the time the
  reference figure was made.
- The checkpoint itself: on 1,000 fixed rows, all 100 members reach R²
  0.973–0.987 at `+3`, while at `-3` every member flattens against the
  clipped-log floor — member prediction spreads fall to 0.0036–0.218 from a
  minimum of 1.122 at `+3`, and the ensemble mean spans only [-2.101, -2.004] —
  and no member exceeds R² of −8. A network trained
  with `-3` on half its rows would fit them there.

This repository therefore supplies fixed rows at `+3`. Nothing above changes:
the reference figure and its R² use varied-gradient test rows only, so the
comparison below is unaffected by the convention either way. Full evidence and
the artifacts it changed are in
`reports/xai/S03_fixed_gradient_decision.md`.

## Model equivalence

All 100 selected models use preprocessing method 2 and omit batch
normalization. The pruned model produced bit-for-bit identical outputs to the
legacy class for a 64-sample comparison using the same state dictionary.

The consolidated inference run produced:

- New batched CPU R2 in clipped-log space: `0.989310659379`
- Legacy logged R2: `0.989310562611`
- Absolute difference: `0.000000096768`

The tiny difference is attributable to batched floating-point evaluation; it
does not affect the displayed three-decimal score (`0.989`).

## Figure equivalence

Both one-page, 6-by-6-inch PDFs were rendered at 160 DPI to 960-by-960 RGB
images. Comparing the renders gave:

- Exact-pixel fraction: `0.9976421441`
- Mean absolute channel error: `0.00652778` on a 0-255 scale
- Root-mean-square channel error: `0.253992` on a 0-255 scale

Visual inspection confirmed matching axes, labels, title, score, point cloud,
uncertainty bars, reference line, and layout without clipping or overlap.
