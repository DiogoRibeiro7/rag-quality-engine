# Roadmap

This roadmap tracks the product and engineering direction for `rag-quality-engine` as an
evaluation-first RAG and LLMOps platform. It is organised as: current state,
known issues and bugs, milestones, planned features, and near-term priorities.

## Current state

The repository includes a mature evaluation-first MVP with:

- typed domain models for documents, chunks, retrieval results, generated answers, evaluations, and traces
- reusable ingestion for text, markdown, csv, and optional pdf adapters
- lexical, vector, and hybrid retrieval flows
- grounded answer generation with citation validation and refusal handling
- evaluation metrics (context precision/recall, claim-level faithfulness, citation support, answer relevance, refusal correctness) and report export
- CLI commands for ingestion, question answering, indexing, and dataset benchmarks
- FastAPI endpoints for ingestion, retrieval, answering, evaluation, and trace lookup
- local trace persistence and a minimal dashboard
- an analytical notebook suite (retrieval tuning, strategy comparison, grounded generation, end-to-end evaluation) built on package code, committed with executed outputs and figures
- automated linting, type-checking, and test coverage (~90%)
- CI coverage across Python 3.11, 3.12, 3.13, and 3.14
- CI-backed deterministic RAG evaluation regression checks with persisted
  artifacts
- notebook execution checks in CI with `nbval`
- configurable runtime paths and API request safety limits
- trace summary, filtering, and dashboard inspection views
- persistent local vector indexing and reload support
- configurable retrieval profiles for lexical, vector, and hybrid search
- configurable LLM and embedding providers wired into CLI/API execution paths

## Recently completed

- Fixed default `FakeEmbeddingClient` behavior so document and query vectors
  share the same fitted vocabulary.
- Added validation for unsupported chunking strategies instead of silently
  ignoring `ChunkingConfig.strategy`.
- Made character chunking boundary-aware so ordinary words are not split at
  fixed-width seams, while preserving exact source offsets and safe progress on
  long unbroken tokens.
- Added professional repository hygiene: license, contribution guide, security
  policy, pre-commit hooks, Docker build hygiene, and improved CI.
- Added a deterministic RAG evaluation regression gate with faithfulness and
  citation-support thresholds, JSON/CSV/Markdown artifacts, and CI artifact
  upload.
- Added notebook execution checks in CI with `nbval`.
- Added environment-backed runtime settings for data, artifact, chunk, trace,
  and model paths.
- Added API request-size, query-length, text-length, and `top_k` guards.
- Added structured API errors and clearer CLI exits for missing files,
  unsupported inputs, and invalid evaluation requests.
- Added trace summaries, a filtered `GET /traces` endpoint, and a richer
  dashboard table for trace inspection.
- Added a persistent local vector index with CLI/API build and reload flows.
- Added named retrieval profiles with request-level overrides for mode, `top_k`,
  and hybrid weights.
- Wired configurable LLM and embedding providers into the CLI and API while
  keeping deterministic offline defaults.
- Added deterministic claim-level faithfulness scoring with cited-evidence
  matching, numeric mismatch checks, and unsupported-claim details.
- Added a dataset benchmark CLI command with repeated-run summaries and
  per-run artifacts.
- Added refusal-set benchmark coverage for unanswerable and weak-context
  questions.
- Added richer API usage documentation with concrete request and response
  examples.
- Added benchmark provenance fingerprints covering corpus bytes, evaluation
  fixtures, and chunking configuration.
- Added benchmark comparison tooling with provenance checks and regression
  classification.
- Added per-query benchmark comparison to identify the exact golden questions
  responsible for regressions.
- Added an optional CI benchmark-baseline gate with deliberate, reviewed
  baseline promotion.
- Added baseline manifest metadata so approved benchmarks record project version,
  Git commit, promotion time, and benchmark fingerprint.
- Added baseline integrity validation before CI regression comparison.
- Added optional p95 latency and token budget gates to benchmark regression
  checks.
- Added golden-set rebuild tooling based on stable evidence phrases.
- Added sentence-aware chunking with exact source offsets and safe fallback for
  oversized sentences.
- Added token-aware chunking with token-count overlap and exact source offsets.
- Added opt-in embedding-backed semantic judges for answer relevance and
  claim-level support.
- Added opt-in provider-backed LLM judges for relevance and claim-level support.
- Added configurable lexical, embedding, and LLM evaluation strategies across
  normal CLI/API execution paths.
- Added evaluation strategy to benchmark provenance so judge changes cannot be
  mistaken for like-for-like regressions.
- Added evaluation provider and model identity to benchmark provenance so
  provider/model changes also invalidate like-for-like comparisons.
- Added configurable embedding-backed semantic reranking while retaining the
  lexical reranker as the default.
- Added trace-dashboard KPIs, quality-state columns, and attention highlighting
  for weak traces.
- Modernized the API container, added a health endpoint, and documented local
  and public-demo deployment workflows.

## Known issues and design constraints

There are no confirmed open defects tracked at the time of this roadmap refresh.
The main remaining constraints are architectural rather than correctness bugs:

- **Lexical evaluation remains the deterministic default.** Embedding-backed and
  provider-backed LLM judges are available, but the default remains lexical so
  the repository works offline and CI remains deterministic.
- **Storage is local-first.** Chunks, vector indexes, traces, benchmark artifacts,
  and approved baselines are filesystem-backed. This is appropriate for a
  portfolio/demo service but not for multi-replica production deployment.
- **Public deployment controls are intentionally minimal.** The API has request
  limits and health checks, but authentication, rate limiting, and production
  telemetry are still deployment-level extensions.

## Milestone 1 — Complete the evaluation pipeline

- [x] Add a dedicated CI job for regression-style RAG evaluation (the notebook
  04 gate logic, lifted into a runnable script).
- [x] Enforce failure thresholds for faithfulness and citation support.
- [x] Persist evaluation outputs to `artifacts/evaluation`.
- [x] Upload evaluation artifacts from CI for inspection.
- [x] Execute the notebooks in CI with `nbval` (already a dev dependency) so they
  cannot silently rot.
- [x] Add clearer reporting for retrieval metrics and answer-quality metrics.

## Milestone 2 — Strengthen runtime and API behavior

- [x] Add request-size and payload-size guards to the API surface.
- [x] Make runtime paths and storage locations configurable through environment settings.
- [x] Improve error responses and validation messages across API and CLI entrypoints.
- [x] Add better trace summaries, filtering, and inspection views in the dashboard.

## Milestone 3 — Move beyond in-memory retrieval

- [x] Add a persistent local vector store for embeddings.
- [x] Add indexing and reload flows for processed documents and chunks.
- [x] Separate retrieval configuration from runtime execution paths.
- [x] Support configurable retrieval profiles for lexical, vector, and hybrid search.

## Milestone 4 — Improve generation and evaluation depth

- [x] Add provider-backed LLM and embedding integrations behind the existing
  abstractions (the `SentenceTransformerEmbeddingClient` adapter exists but is
  now wired into the CLI and API defaults).
- [x] Add stronger claim extraction and evidence matching for faithfulness checks.
- [x] Add dataset-oriented evaluation commands for repeated benchmark runs.
- [x] Expand refusal evaluation for unanswerable and weak-context cases.

## Milestone 5 — Product and deployment polish

- [x] Improve the dashboard into a more useful inspection surface for traces and metrics.
- [x] Add richer API usage documentation and example payloads.
- [x] Add release notes and a first tagged release from `main`.
- [x] Add deployment guidance for local demos and portfolio presentation.

## Planned features

The originally planned MVP capabilities are now complete: reranking, RRF,
sentence/token-aware chunking, semantic and LLM evaluation judges, cost/latency
budgets, golden-set rebuilding, dataset/version provenance, benchmark comparison,
baseline promotion/validation, dashboard inspection, persistent local retrieval,
and deployment guidance.

The next phase is post-MVP hardening and longitudinal evaluation:

- **Approved benchmark baseline activation.** Promote the first reviewed
  baseline so the optional CI comparison gate becomes active on every pull
  request.
- **Benchmark history and trend reporting.** Persist comparable benchmark
  summaries across releases and render metric trends instead of only pairwise
  diffs.
- **Pluggable storage.** Introduce storage abstractions for traces, chunks,
  indexes, and benchmark metadata so local JSON/JSONL remains the default but
  production backends can be substituted.
- **Operational observability.** Add structured logging, request/trace correlation,
  and exportable metrics for latency, token usage, refusals, retrieval quality,
  and evaluation failures.
- **Public API controls.** Add optional authentication and rate limiting for
  internet-facing deployments while preserving the zero-config local demo path.
- **Release 0.3.0.** Consolidate the post-MVP architecture, changelog, baseline,
  and deployment contract into the next tagged release.

## Near-term priorities

1. Promote and commit the first reviewed benchmark baseline so CI regression
   comparison becomes active rather than optional.
2. Add benchmark-history storage and release-to-release trend reporting.
3. Add structured observability and trace/request correlation for public demos.
4. Introduce pluggable storage interfaces while keeping filesystem-backed local
   implementations as the default.
5. Add optional authentication and rate limiting for public API deployments.
6. Prepare the 0.3.0 release with an updated changelog, baseline snapshot, and
   deployment contract.
