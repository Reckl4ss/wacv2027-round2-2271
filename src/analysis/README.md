# Supporting analysis kernels

Use `scripts/reproduce.py capacity` and `scripts/supporting_analysis.py` from the
repository root. Those entry points create isolated working directories and
load the correct frozen caches. Running a kernel directly assumes sibling
`code/` and `results/` directories; it is not the public quick-start interface.

The capacity and frozen-NTK outputs are numerically compared with the saved
objects. Early-window and width-partial reports are checked against their logged
tables. These are analyses of cached measurements, not new network runs.

`crossspace.py`, `scaleup.py`, and `fdr_recount.py` additionally need external
NASLib benchmark files. Their result caches are provided, but full upstream runs
have not been verified for this candidate. See `docs/external_runs.md`.

Legacy schema keys `R1`, `R2`, `R3` denote dimensionality, decomposition, and
residual-access analyses within a numerical output. They are not three rounds
of conference submission. Interpret printed exploratory summaries using
`docs/limitations.md`; failure to exclude zero does not establish a mechanism.
