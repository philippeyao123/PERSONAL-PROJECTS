# Paper build and arXiv package

The manuscript is generated from frozen CSV evidence. Do not edit numerical
values directly in `main.tex`; regenerate the macros and tables instead.

```bash
python scripts/generate_paper_artifacts.py
TECTONIC_BIN=tectonic bash scripts/build_paper.sh
bash scripts/package_arxiv.sh
```

The final files are:

- `output/pdf/european-power-fair-value-paper.pdf`
- `output/pdf/european-power-fair-value-arxiv-source.tar.gz`

The archive contains `main.tex`, the bibliography, generated tables, the
compiled bibliography, and all fourteen vector figures. It intentionally
excludes code, CSV files, credentials, raw API caches, and LLM logs.

For a full numerical reproduction from the committed dataset:

```bash
PYTHON_BIN=python3 TECTONIC_BIN=tectonic bash scripts/reproduce_paper.sh
```
