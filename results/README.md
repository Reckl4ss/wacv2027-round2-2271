# Reference results

`reference/` is immutable evidence. Its measured values, including unfavorable
baselines and incomplete coverage, are retained. New runs belong under `runs/`.

Output family prefixes and historical schema keys are documented in
`docs/protocols.md`. They are retained to avoid silently changing data hashes.
`core/*_out.txt` are archived reports; their narrative headings can reflect early
analysis terminology. They supply numerical tables, not additional claims
beyond the paper and documented limitations. Machine-readable CSV/JSON/pickle
results should be preferred for analysis.

`reference/search_transfer/independent_verification_nested.json` records the
original verification. Fresh local verification is separately recorded under
`validation/`; do not treat an archived PASS as a newly executed test.
