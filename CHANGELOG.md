# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Added refusal benchmark fixtures and refusal-accuracy reporting for
  unanswerable and weak-context evaluation cases.
- Added a dedicated API usage guide with endpoint summaries, request payloads,
  response examples, trace inspection examples, runtime settings, and error
  formats.

### Changed

- Improved the deterministic heuristic generator to parse multi-line contexts
  and refuse answers when retrieved evidence has insufficient query overlap.

## [0.2.0] - 2026-08-06

### Added

- Added a deterministic RAG evaluation regression gate with JSON, CSV, and
  Markdown artifacts.
- Added a dedicated GitHub Actions job that uploads evaluation artifacts from
  CI.
- Added notebook execution checks with `nbval` in CI.
- Added environment-backed runtime settings for project paths and API limits.
- Added API request-size, query-length, text-length, and `top_k` guards.
- Added structured API errors and clearer CLI exits for missing local resources
  and invalid runtime inputs.
- Added trace summaries, filtered trace listing, and a richer dashboard table.
- Added a persistent local vector index with CLI/API indexing and reload support.
- Added named retrieval profiles with request-level overrides for mode, `top_k`,
  and hybrid weights.
- Added configurable LLM and embedding provider settings for CLI/API generation
  and vector indexing, with deterministic offline defaults preserved.
- Added claim-level faithfulness scoring with cited-evidence matching,
  unsupported-claim details, and stricter numeric mismatch detection.
- Added a reusable dataset benchmark runner, repeated-run benchmark artifacts,
  and a `rag-quality-engine benchmark` CLI command.

### Changed

- Bumped package, module, Zenodo, and citation metadata versions to `0.2.0`.

## [0.1.4] - 2026-08-05

### Changed

- Updated GitHub Actions workflow dependencies used by CI.
- Updated Streamlit and development tooling dependencies managed by Dependabot.
- Adjusted Ruff configuration and imports for the newer linting toolchain.
- Bumped package, module, Zenodo, and citation metadata versions to `0.1.4`.

## [0.1.3] - 2026-08-05

### Changed

- Improved the Zenodo and citation title to better describe the software as a
  RAG operations platform.
- Expanded the Zenodo and citation description with the project scope and core
  capabilities.
- Bumped package, module, Zenodo, and citation metadata versions to `0.1.3`.

## [0.1.2] - 2026-08-05

### Added

- Added `CHANGELOG.md` to the tagged release archive.

### Changed

- Bumped package, module, Zenodo, and citation metadata versions to `0.1.2`.

## [0.1.1] - 2026-08-05

### Added

- Added `CITATION.cff` for GitHub citation support.
- Added the Zenodo all-versions DOI badge and citation reference to `README.md`.
- Added the Zenodo all-versions DOI as a related identifier in `.zenodo.json`.

### Changed

- Bumped package, module, Zenodo, and citation metadata versions to `0.1.1`.

## [0.1.0] - 2026-08-05

### Added

- Added the first public release of `rag-quality-engine`.
- Added a reusable Python package with typed domain models for documents, chunks,
  retrieval results, generated answers, evaluations, and traces.
- Added ingestion for text, Markdown, CSV, and optional PDF extraction boundaries.
- Added lexical, vector, and hybrid retrieval flows with retrieval evaluation
  metrics.
- Added evidence-grounded generation with citation validation and refusal
  handling.
- Added evaluation metrics for context quality, faithfulness, citation support,
  relevance, and refusal correctness.
- Added FastAPI service, Typer CLI, JSONL trace persistence, and a minimal trace
  dashboard.
- Added analytical notebooks covering retrieval baselines, retrieval strategy
  comparison, grounded generation, and RAG evaluation.
- Added CI, pre-commit hooks, issue templates, PR template, MIT license,
  contribution guide, security policy, Docker hygiene, and Zenodo metadata.

### Fixed

- Fixed default vector retrieval by fitting the deterministic fake embedding
  client vocabulary on indexed documents before query embedding.
- Added explicit validation for unsupported chunking strategies.
- Updated `GitPython` to a patched version to clear Dependabot advisories.
- Fixed GitHub Actions and branch protection check naming for the Python matrix.

[Unreleased]: https://github.com/DiogoRibeiro7/rag-quality-engine/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/DiogoRibeiro7/rag-quality-engine/compare/v0.1.4...v0.2.0
[0.1.4]: https://github.com/DiogoRibeiro7/rag-quality-engine/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/DiogoRibeiro7/rag-quality-engine/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/DiogoRibeiro7/rag-quality-engine/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/DiogoRibeiro7/rag-quality-engine/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/DiogoRibeiro7/rag-quality-engine/releases/tag/v0.1.0
