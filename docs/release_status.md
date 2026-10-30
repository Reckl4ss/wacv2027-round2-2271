# Public distribution

This repository publishes the author-reviewed reproducibility snapshot as one
initial commit. Its source, measurement caches, reference results, and validation
records are independent of the development repository's Git history. Subsequent
public changes can be tracked normally.

The frozen paper sources and comparison PDFs are the anonymous submitted
documents used by the recorded compilation checks. They are not the separate
camera-ready publication files. The entry points are `paper/main.tex` and
`paper/appendix.tex`; the main paper has 9 pages including references and the
appendix has 13 pages. See `paper/README.md` and `docs/validation.md`.

The large trace artifact is available in this repository's `e4-traces-v1`
Release. This tag points to the public snapshot. The provenance bundled with
the artifact records the original experiment revision, which is a separate
identifier from the public release tag's commit.

The large search artifact has 179 manifest entries and archive SHA-256
`8325148fe975c9eaf24fa6fca912c8ef5a2a8df96e9d3e7176742e2417ce344a`
(1,744,138,240 bytes). The compact package contains 11 of those manifest targets;
the other 168 are downloaded separately. All 179 files were downloaded and
verified during local validation. The payload is excluded from the compact ZIP.

The preserved measurements and numerical reference files are unchanged from
the reviewed package. GPU network scoring and the upstream external-benchmark
rejoin were not repeated locally; their validation scope and unresolved data
limitations remain documented. No project-wide software license has been
selected; see `docs/licensing.md` for the distinction between public access and
reuse permissions.
