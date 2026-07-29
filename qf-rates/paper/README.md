# Paper and reproducibility guide

This directory contains the submission-ready manuscript and all committed inputs
required for arXiv compilation. The paper is an implementation and validation
study; it does not claim a new stochastic model.

## Requirements

- C++20 compiler and CMake 3.20+
- Python 3.9+
- `pybind11`, `QuantLib`, and `matplotlib`
- either `latexmk` with a standard TeX Live installation or Tectonic

## Full reproduction

From the repository root:

```bash
python3 -m pip install pybind11 QuantLib matplotlib
./scripts/reproduce_paper.sh
./scripts/package_arxiv.sh
```

The first script builds the library, runs all tests, regenerates the validation
CSVs, rebuilds the Python bindings, runs the QuantLib checks, recreates the
figures and LaTeX tables, and compiles the manuscript. The second script creates
the arXiv source archive.

Expected outputs:

- `output/pdf/qf-rates-paper.pdf`
- `output/pdf/qf-rates-arxiv-source.tar.gz`

## Directory map

- `main.tex`: manuscript source
- `references.bib`: bibliography
- `data/`: frozen machine-readable results and provenance
- `figures/`: generated PDF figures for LaTeX and PNG inspection copies
- `generated/`: generated LaTeX macros and complete diagnostic tables
- `arxiv-metadata.md`: suggested submission metadata and checklist

## Interpretation

All stochastic prices expose standard errors and fixed seeds. Reproducibility
does not imply bitwise-identical normal variates across every C++ standard
library. Acceptance thresholds are uncertainty-aware. The paper's limitations
section defines what the results do and do not validate.

arXiv posting is a public preprint, not peer review or guaranteed acceptance.
After submission, inspect arXiv's compiled PDF before confirming the upload.
