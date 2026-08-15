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
