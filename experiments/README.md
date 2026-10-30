# Experiment index

| Experiment | CPU command | Reference results |
|---|---|---|
| Capacity decomposition | `python scripts/reproduce.py capacity` | `results/reference/core/` |
| Orientation controls | `python scripts/reproduce.py orientation` | `results/reference/orientation_controls/` |
| Cached-length/effective-rank sensitivity | `python scripts/reproduce.py dimension` | `results/reference/dimension_sensitivity/` |
| Sampling and matched sample count | `python scripts/reproduce.py sampling` | `results/reference/sampling_robustness/` |
| Search transfer | `python scripts/reproduce.py search-points` | `results/reference/search_transfer/` |
| External field coverage | `python scripts/reproduce.py external-verify` | `results/reference/external_coverage/` |
| Recent proxies | `python scripts/reproduce.py recent-verify` | `results/reference/recent_proxies/` |

The command names describe different verification levels; consult
`docs/reproduction.md` before interpreting a PASS. Each command records its
arguments, stdout/stderr, runtime, and installed package versions under `runs/`.

Low-level engines retain frozen output filenames and schema identifiers for
compatibility with published numerical records. These identifiers are explained
in `docs/protocols.md`; users need not know any submission workflow to run the
commands above.
