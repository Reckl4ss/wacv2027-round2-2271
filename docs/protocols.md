# Protocol definitions

## Orientation

The 29-feature zoo uses either per-space accuracy-informed signs, no orientation,
or one fixed sign map applied to both spaces. Column sign reflection preserves
PCA eigenvalues. The fixed rule is inherited from the documented prior map; it
is not a newly established theoretical orientation for every proxy. Exploratory
partial-correlation intervals are marginal and not multiple-testing-certified.

## Sampling

Finite-mask filtering precedes TSS subsampling. Each of 1,000 deterministic
seeds selects 138 of the 317 finite records without replacement. These draws
summarize algorithmic subsampling sensitivity, not 1,000 independent experiments.
Selection randomizations compare subsets with their containing populations;
within-axis Holm correction is retained.

## Search

Two protocols have different resampling units:

1. **Position-resampled split**: resample target positions, then split into
   calibration/evaluation. Duplicate canonical identities can cross the split.
2. **Identity-disjoint split**: partition canonical identities first, then
   resample within each side. Identity overlap must remain zero.

Both use 40 splits, 1,000 outer draws, K=1/5/10/50, and all fixed baselines.
The estimand is the mean cached accuracy of the selected K architectures,
averaged across splits; it is not best-of-K accuracy. The sign-stability screen
reads target-CALIB labels, and evaluation labels are used only to score selections.
Percentile intervals are protocol-specific. Positive/zero/negative sign mass
is a resampling diagnostic, not an inferential p-value.

Earlier single-split reporting and inferential treatment of resampling draws
were superseded. They are not part of the default released results. Current
results use the corrected schema-v4 report. This methodological correction is
retained here so the two protocols and inference boundaries remain explicit.

## Frozen record identifiers

Some identifiers are preserved for hash/schema compatibility:

| Identifier | Scientific meaning |
|---|---|
| `per_space_r1` | Per-space accuracy-informed orientation |
| `round1_compatible` | Position-resampled split protocol above |
| `--round1-results` | Compatibility alias for `--frozen-results`, the fixed measurement cache directory |
| `e1_` / `e2_` / `e3_` in filenames | Orientation / dimension sensitivity / sampling output families |

These identifiers do not require different submission versions to reproduce the
analyses. The reader-facing entry points use descriptive experiment names.

## Provenance

Numerical engines were exported from the verified research snapshot. Historical
module narratives/comments were removed with an AST-equivalence assertion;
data, seeds, mathematical operations, and frozen output records were retained.
Path adapters and the new command runner provide portable input/output locations.
Legacy core report prose is not an additional scientific claim: interpretation
is defined by the paper and `limitations.md`, particularly for exploratory
partial correlations and the NTK target versus trained accuracy.

Readable CLI aliases, input-path defaults, and output descriptions were adapted
without changing measured data or mathematical formulas. A separate full trace
verification wrapper vectorizes tied ranking only after checking exact equality
against the reference ranking implementation; its checks are recorded.
