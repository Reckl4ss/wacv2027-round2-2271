# Capacity Meters That Change Sign

Reproducibility materials for **Capacity Meters That Change Sign: Diagnosing
Zero-Cost NAS Proxy Transport Across Search Spaces** (WACV 2027).

The study examines how zero-cost neural architecture search scores behave when
transported between search spaces. It reports capacity-associated structure,
orientation and sampling sensitivity, and controlled search consequences. These
are observational diagnostics. Reversal is not universal, and the experiments
do not establish a causal mechanism.

## Quick start

The CPU reproduction environment uses Python 3.13. A fresh virtual environment
is recommended. Run commands from the repository root:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/verify_distribution.py
python scripts/verify_bundle.py
python scripts/reproduce.py orientation
python scripts/reproduce.py dimension
python scripts/reproduce.py sampling
```

Each task writes to `runs/<task>/`, leaving reference data intact. Full analysis
parameters are used: 2,000 bootstrap draws for orientation/dimension and 1,000
matched-sample resamples plus 2,000 selection randomizations for sampling. The
runner compares regenerated CSVs to frozen references with absolute tolerance
`1e-10`; original references remain byte-identical.

Additional checks:

```bash
python scripts/reproduce.py sampling-verify
python scripts/reproduce.py external-verify
python scripts/reproduce.py recent-verify
python scripts/reproduce.py search-points
python scripts/reproduce.py capacity
python scripts/reproduce.py all-cpu
python scripts/supporting_analysis.py figures
```

`recent-verify` reconstructs aggregation and 10,000-draw architecture-bootstrap
intervals from the recorded raw SWAP/AZ-NAS scores. It does not rerun networks.
`search-points` recomputes all 192 method/K/protocol point estimates from frozen
target manifests and cached source summaries. It does not regenerate the full
search-bootstrap traces. The full public trace artifact can be downloaded and
checked separately. See [reproduction scope](docs/reproduction.md).

## Contents

- [`experiments/`](experiments/README.md): named experiments and their entry points.
- `src/engines/`: control, search, coverage, and recent-proxy implementations.
- `src/analysis/`: capacity and supporting analysis kernels.
- [`data/`](data/README.md): fixed measurement caches and upstream provenance.
- `results/reference/`: frozen numerical outputs and reference plots.
- [`paper/`](paper/README.md): paper and appendix source (`main.tex`, `appendix.tex`).
- [`docs/paper_result_map.md`](docs/paper_result_map.md): paper-to-result mapping.
- [`docs/protocols.md`](docs/protocols.md): definitions and protocol differences.
- [`docs/limitations.md`](docs/limitations.md): scientific and reproducibility limits.

## Principal evidence

- Rank-PCA first-component variance is approximately 52.9% / 49.4% on the
  topology / size axes. Its capacity association survives orientation changes,
  but loading cosine falls from about 0.87 to 0.46 under global orientation.
- Matched sample count preserves the three headline reversals in 1,000 draws;
  cohort representativeness remains a separate limitation.
- Named search traps have `K=10` costs of +5.401575 and +1.972606 accuracy points;
  two null directions have zero cost. These are not average gains or universal
  superiority claims. Calibration uses cached target labels.
- SWAP and AZ-NAS are sign-stable on these two populations.
- External coverage is audited for 49,674 records, but none qualifies for the
  full decomposition under the required field contract.

The detailed numerical records, including unfavorable baselines, are retained.
The [validation report](docs/validation.md) distinguishes fresh reruns from
checks of saved outputs and from work that remains unverified in this package.
