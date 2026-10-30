# Export adaptations

The research archive is retained separately. This distribution is a selected export,
not a copy of the old Git history.

- Use topic-based source/result directories and public command entry points.
- Omit reviewer reports, rebuttal drafts, coauthor assignments, cross-review
  discussions, submission checklists, intermediate manuscript builds, and the
  superseded single-split search demonstration.
- Preserve fixed measurement caches and authoritative numerical outputs, including
  adverse/null results. Record original and exported hashes in the manifest.
- Remove historical source narratives/comments using an AST-equivalence gate;
  replace machine-specific paths with explicit inputs or isolated staging.
- Add descriptive command aliases, environment pins, argument/log recording,
  value comparisons, and separate public artifact download/integrity checks.
- Ensure the capacity runner loads cached full-population early-training targets.
  The legacy `--no-base-t` option skips their loading as well as extraction and
  is not used by the public runner.
- Replace causal or universal-inaccessibility language in auxiliary reports with
  association/interval descriptions. Measurements and formulas are unchanged.
- Add the exact-equivalence vectorized ranking wrapper for full trace checking.
  All independent verifier checks are retained; no trace subsampling limits are
  used in its formal validation.
- Preserve the submitted paper/appendix sources and PDFs used in the validation
  record. Separate camera-ready publication files are not substituted into those
  frozen compilation checks.
- Rename the paper entry points to `main.tex` and `appendix.tex`, and place the
  cross-document label snapshots in `paper/crossrefs/`. An isolated alternating
  build refreshes both label sets to a stable state. Remove the temporary author
  review notice from the project README; distribution scope is recorded in the
  dedicated release-status document.

Historical result identifiers are documented in `protocols.md`, rather than
renaming frozen fields and breaking their checksums. This export does not erase
historical material already present in the original public repository.
