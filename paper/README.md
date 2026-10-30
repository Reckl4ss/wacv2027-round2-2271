# Paper source

`main.tex` builds the paper; `appendix.tex` builds the supplementary appendix.
Figures, bibliography/style files, and local fonts are included. The submitted
comparison PDFs are `reference/paper.pdf` and `reference/appendix.pdf`.

The source preserves the anonymous submitted documents used in the validation
record. These are reproducibility reference documents, not the separate
camera-ready publication files. Their content and formatting are retained so
that the recorded text and raster comparisons remain reproducible.

The tested compiler is Tectonic 0.17.0 (XeTeX). From the repository root, run:

```bash
python scripts/build_paper.py
```

If Tectonic is not on PATH, pass `--tectonic /path/to/tectonic`. The compiler may
need internet access to retrieve its official TeX bundle. The numerical Python
dependencies are not needed for this build script.

The script stages the paper under `runs/paper-build-renamed/source/`, alternates
the two builds, and refreshes `crossrefs/main.aux` and `crossrefs/appendix.aux`
until both label files stabilize. It checks final logs for unresolved citations,
references, and overfull boxes. The packaged source and comparison PDFs are not
overwritten. Final outputs are:

- `runs/paper-build-renamed/main.pdf`
- `runs/paper-build-renamed/appendix.pdf`
- `runs/paper-build-renamed/build_report.json`

For manual builds from this directory, use `build/main/` and `build/appendix/`
as output directories, copy their newly generated `.aux` files into `crossrefs/`,
then rebuild both documents. The shipped cross-reference snapshots allow initial
compilation; refreshing both avoids silently retaining obsolete page numbers.

The validation report records the executed commands and PDF comparisons.
