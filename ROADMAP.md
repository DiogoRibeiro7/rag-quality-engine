# Roadmap

This roadmap tracks the product and engineering direction for `ragops-lab` as an
evaluation-first RAG and LLMOps platform. It is organised as: current state,
known issues and bugs, milestones, planned features, and near-term priorities.

## Current state

The repository includes a working local MVP with:

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
- CI coverage across Python 3.11 and 3.12
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

## Known issues and bugs

These are confirmed defects with concrete reproduction paths. They should be
fixed before the behaviours they affect are relied on.

- **Sentence-aware chunking is not implemented yet.** The default character
  chunker now snaps ordinary boundaries to whitespace and preserves exact source
  offsets, so it no longer introduces artificial mid-word fragments. A future
  sentence-aware strategy can improve semantic coherence further.
- **`OverlapJudge` answer-relevance is a weak proxy.** It scores how many query
  terms reappear in the answer, which yields structurally low values and is not
  safe to use as a quality gate (only as a trend signal). *Fix:* add a
  model-based or embedding-based relevance judge behind the existing
  `RelevanceJudge` protocol.
- **Faithfulness still uses lexical matching.** Claim support is now scored at
  claim level against cited evidence with stopword filtering, light stemming,
  and strict number matching. It is stronger than subset matching, but it is
  still not semantic entailment. *Fix:* add an embedding or LLM judge behind the
  existing `ClaimSupportJudge` protocol.

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

- [ ] Improve the dashboard into a more useful inspection surface for traces and metrics.
- [x] Add richer API usage documentation and example payloads.
- [x] Add release notes and a first tagged release from `main`.
- [ ] Add deployment guidance for local demos and portfolio presentation.

## Planned features

- **Reranking stage.** The `Reranker` protocol and retrieve-then-rerank path are
  now implemented with a deterministic offline lexical reranker. A provider-backed
  cross-encoder remains a future extension for semantic second-stage scoring.
- **Reciprocal Rank Fusion.** Completed: `HybridRetriever` now supports RRF as
  an alternative to weighted score fusion, removing the need to calibrate raw
  lexical and vector score scales.
- **Token- and sentence-aware chunking.** Implement the chunking strategies the
  config already advertises, with boundary snapping.
- **Model-as-judge evaluation.** Pluggable LLM judges for relevance and
  faithfulness, with the deterministic lexical judges kept as the offline path.
- **Cost and latency budgets.** Track token and latency distributions per
  request and fail evaluation when p95 budgets are breached, not just on quality.
- **Golden-set tooling.** A small command to (re)build the golden set by
  resolving answer phrases to current chunk ids, keeping fixtures in sync with
  the chunker.
- **Dataset versioning.** Record the chunking config and corpus hash alongside
  evaluation artifacts so results are reproducible and comparable across runs.

## Near-term priorities

1. Finish CI-backed evaluation regression checks and artifact publishing, and
   run the notebooks under `nbval` in CI.
2. Harden API safety limits and runtime configuration.
3. Add persistent retrieval storage instead of relying on in-memory indexing.
4. Add stronger faithfulness checks with claim extraction and evidence matching.
