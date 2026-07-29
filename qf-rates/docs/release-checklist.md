# v0.1.0 release checklist

1. Confirm the README badge targets `philippeyao123/qf-rates`.
2. Configure and build Release with tests and warnings-as-errors.
3. Run the sanitizer configuration and `scripts/python_reference.py --require-quantlib
   --require-bindings`.
4. Record benchmark hardware/compiler and output in the release notes.
5. Confirm `git status` contains no build products, credentials or notebooks.
6. Commit with `Release v0.1.0`, split the `qf-rates/` subtree, create tag `v0.1.0`, and push the
   autonomous branch and tag.
7. Create a public GitHub release from the tag; GitHub supplies the source `.zip` and `.tar.gz`
   archives.
8. Verify the clean-room commands from the README on macOS and Linux.

The root workflow `.github/workflows/qf-rates-ci.yml` validates the integrated monorepo path. The
nested workflow becomes `.github/workflows/ci.yml` in the autonomous subtree and validates the
independent repository.
