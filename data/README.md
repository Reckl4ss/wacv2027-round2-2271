# Inputs and provenance

`frozen/` contains fixed per-architecture measurement caches used by the analyses.
These are trusted project pickle files; verify their hashes before loading them.
They are not the complete upstream benchmarks or a replacement for measuring
all proxies from initialized networks.

- `base_tss_cifar10.pkl`, `base_sss_cifar10.pkl`: 1,000 topology and 425 size records.
- `exp_alpha_components.pkl`: spectral summaries on 500 architectures per axis.
- `premise_data.pkl`: capacity and early-training measurements.
- `trainability_probes.pkl`: recorded initialization-time probes.
- `_tss_archstr.pkl`, `_tss_intersection_idx.pkl`: architecture identity maps.
- `base_T_*.pkl`: recorded early-training targets for base populations.
- `ntk_gram_features.pkl`: recorded at-initialization NTK features and target.

The full-zoo intersection contains 324 / 138 records; finite 29-feature PCA has
317 / 138. Cached `n_layers_used` is a count, not a certified representation
dimension. The upstream sampling algorithm for the 500-record cohorts cannot
be reconstructed from these caches.

`provenance/*/official_sources.json` records upstream URLs, pinned commits,
source-file IDs, and checksums. `export_manifest.json` records byte-identical
artifacts and adapted source files. `python scripts/verify_bundle.py` validates
unchanged inputs and results. Run `python scripts/verify_distribution.py`
to validate every file in the final candidate against `MANIFEST.sha256`.

Full external benchmark inputs and large search traces are not in this compact
candidate. See `docs/external_runs.md` and `docs/release_status.md`.
Repository code licenses do not by themselves establish redistribution rights
for every associated benchmark dataset; see `docs/licensing.md`.
