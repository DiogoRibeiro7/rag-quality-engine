# Release Process

This project uses semantic versioning and publishes GitHub releases from tags on
`main`. Zenodo archives are created from GitHub releases, so release metadata
must be correct before a tag is pushed.

## Version Metadata

Keep these files aligned for every release:

- `pyproject.toml`
- `src/ragops_lab/__init__.py`
- `.zenodo.json`
- `CITATION.cff`
- `CHANGELOG.md`

The automated release metadata tests verify that these files agree on the
current version and Zenodo concept DOI.

## Pre-Release Checklist

1. Update version fields and the top `CHANGELOG.md` entry.
2. Keep the Zenodo all-versions DOI unchanged unless Zenodo changes the concept
   DOI:

   `10.5281/zenodo.21805398`

3. Run:

   ```bash
   make release-check
   ```

4. Commit the release preparation:

   ```bash
   git add pyproject.toml src/ragops_lab/__init__.py .zenodo.json CITATION.cff CHANGELOG.md
   git commit -m "Bump version to X.Y.Z"
   git push origin main
   ```

5. Confirm GitHub CI passes on `main`.

Dependency updates are managed by Dependabot through `.github/dependabot.yml`.
Before merging dependency update pull requests, confirm the CI matrix passes and
run `make release-check` locally for release-affecting changes.

## Create the Release

Create an annotated tag and publish the GitHub release:

```bash
git tag -a vX.Y.Z -m "rag-quality-engine X.Y.Z"
git push origin vX.Y.Z
gh release create vX.Y.Z --title "RAG Quality Engine X.Y.Z" --notes-file RELEASE_NOTES.md
```

Use the release notes to summarize user-facing changes, metadata changes,
validation, and DOI guidance.

## Post-Release Checklist

1. Confirm the GitHub release is published and not marked as a prerelease.
2. Confirm CI passed for the release commit.
3. Confirm Dependabot open alerts are zero:

   ```bash
   gh api repos/DiogoRibeiro7/rag-quality-engine/dependabot/alerts --paginate \
     --jq '[.[] | select(.state == "open")] | length'
   ```

4. Confirm Zenodo archived the release and inspect the displayed title,
   description, creators, ORCID, license, and DOI metadata.
