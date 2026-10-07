# Architecture

RAG Quality Engine is organized as a reusable Python package with thin CLI and API
entrypoints. The project is intentionally offline-first: default retrieval,
generation, and evaluation paths are deterministic so CI and notebooks can run
without external credentials.

```text
Raw documents
  -> Ingestion
  -> Chunk store
  -> Retriever
  -> Generator
  -> Evaluator
  -> Trace store
  -> API / dashboard
```

The central design principle is evaluation-first development. Every generated answer should be linked to retrieved context and every failure should be measurable.

## Package Boundaries

| Package | Responsibility |
| --- | --- |
| `ragops_lab.domain` | Pydantic models for documents, chunks, retrieval results, generated answers, evaluation reports, and traces. |
| `ragops_lab.ingestion` | File discovery, document loading, chunk validation, and JSONL chunk persistence. |
| `ragops_lab.retrieval` | Tokenization, BM25 retrieval, deterministic vector retrieval, hybrid search, and retrieval metrics. |
| `ragops_lab.generation` | LLM client abstraction and evidence-grounded answer generation. |
| `ragops_lab.evaluation` | Context, citation, faithfulness, relevance, and refusal scoring. |
| `ragops_lab.tracing` | JSONL trace storage for request inspection and dashboard views. |
| `ragops_lab.api` | FastAPI service exposing ingestion, search, ask, evaluation, trace lookup, and dashboard routes. |

## Runtime Flow

1. Documents are loaded from a local directory and normalized into `Document`
   models.
2. Documents are split into `DocumentChunk` records with stable IDs and offsets.
3. Chunks are persisted as JSONL so CLI, API, tests, and notebooks share the
   same data contract.
4. Retrieval ranks chunks using lexical, vector, or hybrid search.
5. Generation receives the question and retrieved evidence, then produces a
   cited answer or a refusal.
6. Evaluation scores the answer against retrieved and reference evidence.
7. The API path stores a `RagTrace` with latency, token estimates, retrieved
   chunks, answer, and evaluation output.

## Quality Strategy

- Unit tests cover domain validation, ingestion, retrieval, generation,
  evaluation, tracing, CLI behavior, and API behavior.
- Prompt and retrieval fixtures live under `data/golden`.
- CI runs Ruff, mypy strict mode, and pytest with coverage on Python 3.11 and
  3.12.
- Notebooks are demos over package code, not an alternate implementation path.

## Extension Points

- `PdfExtractor` allows optional PDF support without making PDF parsing a hard
  dependency.
- `EmbeddingClient` supports deterministic fake embeddings and optional
  provider-backed or local embedding adapters.
- Generation and evaluation services are structured so model-backed clients and
  judges can be added without replacing the offline defaults.
