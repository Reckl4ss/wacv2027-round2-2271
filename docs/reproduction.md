# Reproduction levels

## Recompute from fixed measurements

`orientation`, `dimension`, and `sampling` use the original draw counts/seeds,
write fresh outputs, and compare every reference CSV field (absolute tolerance
1e-10). Differences smaller than tolerance are reported explicitly. Plot bytes
are not expected to match across plotting-library/font versions.

`capacity` executes the full cached-population capacity/decomposition engine in
an isolated working directory. It loads the supplied early-target caches for
the standard-proxy population; no extraction from the full NATS benchmark is
needed. All 3,162 nested numerical values are compared to the frozen output.
Do not use the legacy `--no-base-t` flag: it also skips loading those caches.

## Reconstruct statistics from saved raw outputs

`sampling-verify` independently reconstructs all 1,000 selected architecture-ID
sets and analysis summaries. It writes its generated IDs to `runs/`.

`external-verify` checks all 49,674 saved join rows, uniqueness, field coverage,
target definitions, integer parameter agreement, and eligibility. It does not
redownload/rejoin the complete upstream benchmark files.

`recent-verify` checks 52,725 raw score rows, 2,850 architecture-level rows,
initialization seeds/repeat counts, cached model-identity fields, and reconstructs
correlations and 10,000-draw intervals for both targets. It does not instantiate
or rescore the networks; that requires the pinned official implementations and GPU
environment documented in `external_runs.md`.

`search-points` replays all 40 point-estimate splits for four directions, four K
values, six methods, and two protocol labels. Source selection uses cached source
correlations/coverage; upstream source proxies are not recomputed. The point
protocols coincide before outer resampling. All 192 means must match reference
results. The 1,000-draw percentile intervals require the separate public traces:

```bash
python scripts/fetch_search_traces.py --extract
python scripts/verify_search_traces.py --output artifacts/search_traces/payload/outputs/E4_search_nested --no-write
```

The downloader checks the pinned archive size/SHA-256 and all 179 payload files,
rejects archive links/unsafe paths, and never executes downloaded code. The
verifier recomputes statistics and selections from saved traces; this is distinct
from generating the traces again from upstream inputs.

The wrapper uses vectorized tie ranking after an exact comparison with the
unchanged independent verifier on all target columns, deterministic resamples,
and nonfinite edge cases. All selections, stored scores, seeds, protocol gates,
and aggregate comparisons remain checked. The reference verifier can also be
invoked directly at `src/engines/search_transfer/verify.py`; it is slower.

## Full external execution

Low-level engines and independent verifiers are included. Follow
`external_runs.md` to provision external inputs and run them. A command or a
historical PASS report is not evidence of a fresh full rerun of this distribution.

## Environment

`requirements.txt` pins the independently tested Python 3.13 CPU environment.
The original server analyses used Python 3.12.3, NumPy 2.1.3, Pandas 2.2.3 and
Matplotlib 3.9.2; GPU scoring used PyTorch 2.5.1+cu124, CUDA 12.4 and RTX 4090.
Do not install that old GPU stack into Python 3.13 and assume compatibility.

Every execution records its actual environment and result comparison. See
`validation.md` for measured coverage and outstanding checks.
