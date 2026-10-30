# Paper-to-result map

The included PDFs are the submitted paper and appendix, with their wording and
numbers preserved. This map names scientific topics and exact source labels;
it does not require familiarity with the submission workflow.

| Paper topic / source label | Authoritative records under `results/reference/` | Command after `python scripts/` |
|---|---|---|
| Capacity / trainability, `fig:results` | `core/zoo_analysis_results.pkl` | `reproduce.py capacity` |
| Orientation, `tab:r2-orientation` | `orientation_controls/e1_pca_summary.csv`, `e1_global_trainability_partials.csv` | `reproduce.py orientation` |
| Sampling, `tab:r2-sampling` | `sampling_robustness/e3_matched_n_runs.csv`, `e3_matched_n_summary.csv`, inclusion/test tables | `reproduce.py sampling`, `reproduce.py sampling-verify` |
| Cached-length / effective-rank sensitivity, `tab:spectral-body`, `tab:r2-dimension` | `dimension_sensitivity/e2_score_correlations.csv`, `e2_erank_common_range.csv`, strata | `reproduce.py dimension` |
| Search cost, `tab:searchtransfer`, `tab:r2-search` | `search_transfer/paired_comparisons.csv`, `method_summary.csv`, separate full traces | `reproduce.py search-points`; trace verifier in `reproduction.md` |
| External coverage, `tab:r2-external` | `external_coverage/field_coverage.csv`, `join_summary.csv`, `join_records/` | `reproduce.py external-verify` |
| Recent methods, `tab:r2-recent` | `recent_proxies/correlations.csv`, raw/architecture scores | `reproduce.py recent-verify` |
| Raw portability, `fig:portability` | `core/crossspace_out.txt`, `crossspace_results.pkl` | `supporting_analysis.py figures` renders logged numbers |
| Frozen at-init NTK target | `core/ntk_experiment_results.pkl`; input in `data/frozen/ntk_gram_features.pkl` | `supporting_analysis.py ntk-analysis` |
| Early target windows / within-sample partials | `core/t_window_out.txt`, `width_partial_out.txt` | `supporting_analysis.py early-windows`, `supporting_analysis.py width-partials` |

Source labels containing `r2` are retained cross-reference identifiers, not
experiment names. The appendix controls section covers, in order: orientation,
sampling, dimension, search, external coverage, and recent proxies. Use these
names when discussing evidence, not file prefixes as appendix numbering.

The full portability figure is regenerated with all 13 x 7 numeric cells and
the distinction between 9 descriptive reversals and 4 CI-separated cases.
`src/figures/pipeline.py` regenerates the analysis overview. The original plotting
source for the supplied headline sign-flip figure and composite capacity image
is not included in this snapshot; those final assets are preserved. Their
underlying measurements are available, but exact recreation of every original
visual is not claimed.

Cross-dataset decomposition and broader portability caches are provided. The
complete upstream benchmark environments for those analyses have not been
freshly reconstructed here. A frozen cache, a log-driven rendering, and a full
upstream rerun are distinct evidence levels.
