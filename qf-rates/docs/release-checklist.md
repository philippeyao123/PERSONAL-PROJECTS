# v0.1.0 release checklist

1. Replace `OWNER` in the README badge with the GitHub account or organization.
2. Configure and build Release with tests and warnings-as-errors.
3. Run the sanitizer configuration and `scripts/python_reference.py`.
4. Record benchmark hardware/compiler and output in the release notes.
5. Confirm `git status` contains no build products, credentials or notebooks.
6. Commit with `Release v0.1.0`, create signed tag `v0.1.0`, and push branch and tag.
7. Create a public GitHub release from the tag and attach the source archives.
8. Verify the clean-room commands from the README on macOS and Linux.

