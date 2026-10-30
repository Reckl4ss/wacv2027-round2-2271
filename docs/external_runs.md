# External inputs and full execution

The README CPU commands work from bundled measurements. Commands below require
external datasets or GPU dependencies. They are full-run entry points, not claims
that those runs were repeated during packaging.

## Public search artifact

```bash
python scripts/fetch_search_traces.py --extract
python scripts/verify_search_traces.py --output artifacts/search_traces/payload/outputs/E4_search_nested --no-write
```

Allow at least 4 GB free for the 1.7 GB archive and extraction. Hashes are pinned
in the downloader and `data/provenance/search_transfer/`. The fixed Release name
is an archival identifier; the scientific task is search transfer.

## Regenerate search traces

Provision the NASLib zero-cost tables from the pinned source recorded in
`data/provenance/external_coverage/official_sources.json`. The `--official-zc`
directory must contain the original zero-cost JSON files, including NB101 and
TransNAS scene. The engine's `--help` lists supported arguments.

```bash
python src/engines/search_transfer/run.py prepare --frozen-results data/frozen --official-zc data/external/zc --output runs/search-full
python src/engines/search_transfer/run.py run-point --frozen-results data/frozen --official-zc data/external/zc --output runs/search-full --protocol round1_compatible --pair nb201_tss_to_sss
python src/engines/search_transfer/run.py run-chunk --frozen-results data/frozen --official-zc data/external/zc --output runs/search-full --protocol round1_compatible --pair nb201_tss_to_sss --start 0 --stop 100
```

For a complete run, execute `run-point` and ten consecutive non-overlapping
chunks `[0,100), ..., [900,1000)` for every pair and protocol:

- pairs: `nb201_tss_to_sss`, `nb201_sss_to_tss`, `transnas_scene_to_nb101`, `transnas_scene_to_nb201`;
- protocols: `round1_compatible` (position-resampled), `identity_disjoint`.

Then call `aggregate`, `validate` with the same three directory arguments, and
the independent verifier on `runs/search-full`. Do not overwrite references.
K values, 40 splits, 50 random selections per split, 1,000 outer draws, and seeds
are fixed in the engine. Both protocols use calibration labels.

## External field-coverage audit

`data/provenance/external_coverage/official_sources.json` records pinned NASLib,
NB101, NB301, and TransNAS repositories, file IDs, and known hashes. Upstream
software licenses and dataset redistribution terms must be checked separately.
Do not substitute a different version of a similarly named file.

```bash
python src/engines/external_coverage/run.py prepare --zc-dir data/external/zc --nb101 data/external/nasbench101.pkl --nb301 data/external/nasbench301.pkl --transnas data/external/transnas-bench_v10141024.pth --output runs/external-full
python src/engines/external_coverage/run.py audit --zc-dir data/external/zc --nb101 data/external/nasbench101.pkl --nb301 data/external/nasbench301.pkl --transnas data/external/transnas-bench_v10141024.pth --output runs/external-full
python src/engines/external_coverage/run.py validate --output runs/external-full
python src/engines/external_coverage/verify.py --output runs/external-full --report runs/external-full/independent_verification.json
```

Loading the TransNAS `.pth` requires compatible PyTorch. This audit is not a full
decomposition: eligible row count is zero because the required capacity, proxy,
and target fields do not coexist on any population.

## SWAP and AZ-NAS from initialized networks

Use a separate Python 3.12 GPU environment. The recorded scoring environment
was PyTorch 2.5.1+cu124/CUDA 12.4 on RTX 4090. The Python 3.13 CPU requirements
are not a GPU lock. Pin official code to:

- SWAP: `0853fc866051dca2b3b99d068502549de3686bd1` at `pym1024/SWAP`.
- AZ-NAS: `5e6683a2cfa5c6d0dc34a1317a842497ba7eae47` at `cvlab-yonsei/AZ-NAS`.

URLs and licenses are recorded in
`data/provenance/recent_proxies/official_sources.json`. Fetch the official
CIFAR-10 Python archive and prepare the exact batch using its hash gate:

```bash
python src/engines/recent_proxies/prepare_inputs.py --archive data/external/cifar-10-python.tar.gz --output data/external/cifar_first16.npy
python src/engines/recent_proxies/run.py --mode identity --frozen-results data/frozen --swap-repo data/external/SWAP --az-repo data/external/AZ-NAS --output-dir runs/recent-full
python src/engines/recent_proxies/run.py --mode run --method both --frozen-results data/frozen --swap-repo data/external/SWAP --az-repo data/external/AZ-NAS --cifar-first16 data/external/cifar_first16.npy --output-dir runs/recent-full
python src/engines/recent_proxies/run.py --mode aggregate --frozen-results data/frozen --output-dir runs/recent-full
python src/engines/recent_proxies/run.py --mode validate --frozen-results data/frozen --output-dir runs/recent-full
```

Do not set `--limit-per-space` for a formal run. Retain 32 SWAP repeats, AZ-NAS
seeds 1-5, degenerate-path sentinels, and the 1,425-identity gate. GPU dependencies
and scoring have not been freshly tested on this local machine.

## Advanced core engines

`crossspace.py`, `scaleup.py`, `fdr_recount.py`, and the GPU phase of
`ntk_experiment.py` in `src/analysis/` require external benchmark/source layouts.
They are included for inspection, not advertised as turnkey CPU tasks. Path
overrides include `NASLIB_DATA_DIR`, `NASLIB_BASELINES_DIR`, `NAS_PROJECT_ROOT`,
`NAS_RESULTS_DIR`, `NB201_CORRELATION_DIR`, and `NB201_ZEROCOST_DIR` where applicable.
Full upstream remeasurement remains outside this distribution's tested scope.
