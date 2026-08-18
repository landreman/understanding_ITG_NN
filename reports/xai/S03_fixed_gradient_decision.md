# S03 fixed-gradient premise conflict

## Decision

All S03 fixed-gradient model-sensitivity results are withdrawn. They must not be
used as constant-drive evidence in S07 or S13. The varied-gradient S03 ladder is
unaffected.

Correcting the shared fixed-gradient loader inside S03 would silently change a
registered premise inherited by S00–S02. Before any later step uses fixed rows,
change the loader to the checkpoint's training convention (`a/L_T = +3`) and
refresh the fixed-row validation artifacts that depend on it.

## Evidence

The trusted serialized training `TensorDataset` at
`neural_networks/cyclic_invariant_models/train_val_test_dataset_5_pre_2.pth`
contains no negative `a/L_T` values in its train, validation, or test tensors;
the full stored `a/L_T` tensor has minimum 0.000129. The checkpoint was therefore
trained with the positive convention. A legacy source comment describing the
negative fixed marker does not override the serialized training input.

As an independent inference check, on the first 500 rows of the registered S01
panel, actual fixed targets have mean 1.853 and standard deviation 1.206. For the
stored-validation top three members:

| Fixed input | Prediction means | Prediction standard deviations | R² against fixed target |
| --- | --- | --- | --- |
| Loader convention, `a/L_T = -3` | -1.971 / -2.045 / -2.242 | 0.022 / 0.011 / 0.112 | -10.07 / -10.47 / -11.43 |
| Training convention, `a/L_T = +3` | 1.860 / 1.842 / 1.855 | 1.198 / 1.195 / 1.204 | 0.982 / 0.985 / 0.979 |

The negative marker drives the members to a nearly constant clipped-log floor.
Consequently, the earlier small fixed-twin changes measured saturation wobble,
not geometry sensitivity at constant drive. The earlier classification of fixed
rows as stable or unstable from their *target* while predictions remained at the
floor was also scientifically incoherent.

## Scope and cost

S03's registered scientific question can still be answered from its 1,000
varied-gradient rows. No new inference is needed for those rows: random twin
pairing changes only how the withdrawn fixed half is generated, and all other
review corrections are analysis or reporting corrections.

A shared loader correction will require refreshing the fixed portions of S00–S02
before S07/S13. That work is intentionally not folded into this S03 review fix,
because it changes a cross-step baseline convention rather than an S03-local
implementation detail.
