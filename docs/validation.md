# Local validation — 2026-10-30

Executed in a fresh Python 3.13.13 Windows environment using the pinned CPU
requirements. This is an actual local validation record, separate from the
historical server reports kept with frozen outputs. Logs and machine-readable
reports are in `validation/`. Commands are shown without personal machine paths.

| Check | Fresh result | Scope |
|---|---|---|
| Capacity | PASS: 3,162 values; maximum error 2.70e-13 | Complete cached full-zoo and standard-proxy analyses |
| Orientation | PASS: 12 PCA rows and 58 partial rows | Full 2,000-draw runs; max CSV error 8.89e-16 |
| Dimension sensitivities | PASS: 24 CSV rows | Full 2,000-draw runs; max CSV error 3.34e-16 |
| Sampling | PASS: all 1,000 matched draws and all reference CSVs | Full 2,000-randomization selection audit; max CSV error 8.89e-16 |
| Sampling reconstruction | PASS | Independent selected-ID and statistic reconstruction |
| Search point estimates | PASS: 192 means; max error 1.43e-14 | 40 splits; cached source summaries and fixed target measurements |
| Public search artifact | PASS: archive SHA-256 and 179/179 files | Fresh public download, safe extraction, per-file hashes |
| Full search trace verification | PASS | Counts and bounds below; no inferential p-values |
| External coverage | PASS: 49,674 joined records | Saved joins/coverage rechecked; upstream rejoin not repeated |
| Recent proxy statistics | PASS: 52,725 raw and 2,850 architecture rows | 8 correlations and 10,000-draw intervals reconstructed; networks not rerun |
| Frozen NTK analysis | PASS: 670 values; max error 2.00e-15 | From stored Gram features/target |
| Early-window / width partials | PASS: 8 / 12 numeric report rows | Logged rounded values and interval endpoints compared |
| Portability and overview figures | PASS | Regenerated and visually inspected; original numeric cells retained |
| LaTeX main paper | PASS: 9 pages | All page text and 72 dpi raster pixels identical to submitted PDF |
| LaTeX appendix | PASS: 13 pages | All page text and 72 dpi raster pixels identical to submitted PDF |
| Renamed paper entry points | PASS | `main.tex` / `appendix.tex`; stable cross-document labels and resolved static references |
| Clean-directory portability | PASS | Hash, search-point and dimension commands run from a separate path with spaces and Unicode |

The main paper's 9 pages include references. These are the preserved submitted
documents, not a new camera-ready version. LaTeX logs contain benign draft-mode,
unused-subcaption, and float-placement warnings, with no undefined references
or overfull boxes in the final logs. Tectonic 0.17.0 required network access to
its official TeX bundle before compilation.

The latest compilation uses `scripts/build_paper.py`, the renamed entry points,
and `paper/crossrefs/` snapshots. See `validation/paper_build.json` and
`validation/paper_cross_references.json`. This supersedes the old entry-point
build records; the paper's scientific content and reference PDFs are unchanged.

## Full search verifier coverage

- Trace chunks: 80.
- Deterministic selected indices and stored scores: 105,705,600 each.
- Screen decisions recomputed: 320,320.
- Seed-reconstructed random indices in sampled chunks: 2,376,000.
- Point-trace random indices: 1,056,000.
- Paired comparisons recomputed: 160.
- Identity-disjoint maximum overlap: 0.

Every chunk and every deterministic selection is checked. Seed reconstruction
of random choices uses the recorded offsets/splits in the verifier, rather than
claiming every random index was independently seed-reconstructed. Aggregation
uses the complete stored arrays. The faster wrapper's rank equivalence was
checked exactly on 660 arrays, including tied and nonfinite cases.

## Boundaries

PASS is scoped to the operation above. This package does not claim a fresh GPU
rerun of SWAP/AZ-NAS, a fresh upstream join of all external benchmarks, recovery
of the original 500-cohort sampling algorithm, or regeneration of every proxy
measurement and every original figure from scratch. True dimension matching
and the full external decomposition remain scientifically blocked, not passed.

The distribution content scan checks common credential patterns and personal paths
in text, decompressed gzip records, and trusted pickle strings. It does not scan
or erase the original public Git history, and is not proof against every
possible form of sensitive data. Original source hashes were rechecked unchanged.
