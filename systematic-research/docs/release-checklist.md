# Release checklist

1. Update `CHANGELOG.md` and the version in `pyproject.toml`.
2. Run `scripts/run_all.sh` in a clean environment.
3. Run the flagship experiment twice and compare `metrics.json` and `metadata.json`.
4. Build the package with `python -m build`.
5. Install the wheel in a new environment and run `sysresearch --help`.
6. Inspect the sdist to exclude data, reports, caches, and secrets.
7. Create a signed `vX.Y.Z` tag.
8. Publish to TestPyPI first, then install and retest.
9. Publish to PyPI after explicit approval from the owner.
10. Create the remote release with notes and distribution checksums.

External publication, tagging, and remote releases require an explicit destination and
authorization.

