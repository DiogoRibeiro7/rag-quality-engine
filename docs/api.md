# API Usage Guide

RAG Quality Engine exposes the local retrieval, answering, evaluation, and trace
inspection workflow through FastAPI. The service uses deterministic offline
defaults, so the examples below work without external model credentials after
the sample corpus has been ingested.

## Start the Service

Install dependencies and start the API:

```bash
poetry install --with dev
poetry run uvicorn ragops_lab.api.app:app --host 0.0.0.0 --port 8000
```

Interactive OpenAPI documentation is available at:

- `http://localhost:8000/docs`
- `http://localhost:8000/redoc`

## Endpoint Summary

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/ingest` | Load local documents and write chunk JSONL. |
| `POST` | `/index` | Build a persistent local vector index from chunks. |
| `POST` | `/search` | Retrieve relevant chunks for a query. |
| `POST` | `/ask` | Retrieve evidence, generate an answer, evaluate it, and persist a trace. |
| `POST` | `/evaluate` | Evaluate a supplied answer against retrieved evidence. |
| `GET` | `/traces` | List trace summaries with optional filters. |
| `GET` | `/traces/{trace_id}` | Fetch one full trace. |
| `GET` | `/dashboard` | Inspect traces in a browser. |

## Ingest Documents

`POST /ingest` reads files from a local directory and writes chunk records to a
JSONL file.

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "input_dir": "data/sample_documents",
    "out_path": "data/processed/chunks.jsonl",
    "chunk_size": 500,
    "overlap": 50
  }'
```

Example response:

```json
{
  "chunks_written": 12,
  "out_path": "data/processed/chunks.jsonl"
}
```

Supported input formats are `.txt`, `.md`, `.csv`, and `.pdf` when the optional
PDF dependency is installed.

## Build a Vector Index

`POST /index` builds a local vector index that can be reused by vector and
hybrid retrieval requests.

```bash
curl -X POST http://localhost:8000/index \
  -H "Content-Type: application/json" \
  -d '{
    "chunks_path": "data/processed/chunks.jsonl",
    "index_path": "artifacts/index/vector_index.json"
  }'
```

Example response:

```json
{
  "chunks_indexed": 12,
  "index_path": "artifacts/index/vector_index.json"
}
```

The default embedding provider is deterministic and local. Set
`RAGOPS_EMBEDDING_PROVIDER=sentence-transformers` to build an index with a local
sentence-transformer model.

## Search Chunks

`POST /search` returns ranked chunks. Use `profile` for the built-in retrieval
profiles, or override individual retrieval settings per request.

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What does citation support measure?",
    "profile": "hybrid",
    "chunks_path": "data/processed/chunks.jsonl",
    "index_path": "artifacts/index/vector_index.json",
    "top_k": 3
  }'
```

Example response:

```json
[
  {
    "chunk": {
      "chunk_id": "rag-quality-engine-doc-001-chunk-000",
      "document_id": "rag-quality-engine-doc-001",
      "text": "Citation support measures whether answer citations point to retrieved evidence.",
      "start_offset": 0,
      "end_offset": 82,
      "token_count": 11,
      "source_path": "data/sample_documents/rag_quality_engine.md",
      "metadata": {}
    },
    "score": 0.91,
    "rank": 1,
    "retrieval_method": "hybrid",
    "matched_terms": ["citation", "support", "measure"]
  }
]
```

Built-in profiles are:

| Profile | Default behavior |
| --- | --- |
| `lexical` | BM25-style lexical retrieval. |
| `vector` | Embedding similarity retrieval. |
| `hybrid` | Weighted lexical and vector retrieval. |

Request overrides:

| Field | Notes |
| --- | --- |
| `mode` | `lexical`, `vector`, or `hybrid`. |
| `top_k` | Defaults to the profile value and is capped by `RAGOPS_API_MAX_TOP_K`. |
| `lexical_weight` | Hybrid lexical score weight, from `0.0` to `1.0`. |
| `vector_weight` | Hybrid vector score weight, from `0.0` to `1.0`. |

## Ask a Question

`POST /ask` runs retrieval, answer generation, automatic evaluation, and trace
persistence in one request.

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Which Apollo mission first landed on the Moon?",
    "profile": "hybrid",
    "chunks_path": "data/processed/chunks.jsonl",
    "index_path": "artifacts/index/vector_index.json",
    "top_k": 3
  }'
```

Example response:

```json
{
  "trace_id": "trace-7f3c2a1b9d42",
  "answer": {
    "question": "Which Apollo mission first landed on the Moon?",
    "answer_text": "Apollo 11 first landed humans on the Moon. [apollo-11-chunk-000]",
    "citations": ["apollo-11-chunk-000"],
    "model_name": "heuristic-grounded",
    "refusal": false,
    "grounded": true,
    "prompt": "..."
  },
  "evaluation": {
    "context_precision": 1.0,
    "context_recall": 1.0,
    "answer_relevance": 0.83,
    "faithfulness": 1.0,
    "citation_support": 1.0,
    "unsupported_claim_count": 0,
    "claim_count": 1,
    "supported_claim_count": 1,
    "unsupported_claims": [],
    "claim_support": [],
    "refusal_correct": null,
    "notes": [],
    "created_at": "2026-08-06T12:00:00Z"
  }
}
```

When retrieved evidence is too weak, the deterministic default generator returns
a refusal-style answer and the evaluation can score refusal correctness when the
expected case is marked unanswerable.

## Evaluate a Supplied Answer

`POST /evaluate` is useful for scoring a candidate answer generated outside the
API. Provide retrieved chunk IDs from the same chunk store.

```bash
curl -X POST http://localhost:8000/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What does context recall measure?",
    "answer_text": "Context recall measures how much expected evidence was retrieved.",
    "citations": ["rag-quality-engine-doc-001-chunk-002"],
    "chunks_path": "data/processed/chunks.jsonl",
    "retrieved_chunk_ids": ["rag-quality-engine-doc-001-chunk-002"],
    "reference_chunk_ids": ["rag-quality-engine-doc-001-chunk-002"],
    "expected_answer": "Context recall measures whether expected evidence appears in retrieval results.",
    "expected_unanswerable": false
  }'
```

Example response:

```json
{
  "context_precision": 1.0,
  "context_recall": 1.0,
  "answer_relevance": 0.75,
  "faithfulness": 1.0,
  "citation_support": 1.0,
  "unsupported_claim_count": 0,
  "claim_count": 1,
  "supported_claim_count": 1,
  "unsupported_claims": [],
  "claim_support": [
    {
      "claim": "Context recall measures how much expected evidence was retrieved",
      "supported": true,
      "score": 1.0,
      "evidence_chunk_id": "rag-quality-engine-doc-001-chunk-002",
      "matched_terms": ["context", "recall", "evidence"],
      "missing_terms": []
    }
  ],
  "refusal_correct": null,
  "notes": [],
  "created_at": "2026-08-06T12:00:00Z"
}
```

## Inspect Traces

List recent traces:

```bash
curl "http://localhost:8000/traces?limit=10"
```

Filter traces by question text and minimum faithfulness:

```bash
curl "http://localhost:8000/traces?q=citation&min_faithfulness=0.8&limit=10"
```

Example trace summary:

```json
[
  {
    "trace_id": "trace-7f3c2a1b9d42",
    "question": "Which Apollo mission first landed on the Moon?",
    "model_name": "heuristic-grounded",
    "created_at": "2026-08-06T12:00:01Z",
    "latency_ms": 18.4,
    "token_estimate": 213,
    "retrieved_chunk_count": 3,
    "faithfulness": 1.0,
    "citation_support": 1.0,
    "answer_relevance": 0.83,
    "grounded": true,
    "refusal": false
  }
]
```

Fetch a full trace:

```bash
curl http://localhost:8000/traces/trace-7f3c2a1b9d42
```

Open the local dashboard:

```text
http://localhost:8000/dashboard
```

The dashboard accepts the same `q`, `min_faithfulness`, and `limit` filters as
`GET /traces`.

## Error Format

Runtime and validation errors use a stable JSON envelope:

```json
{
  "error": {
    "code": "resource_not_found",
    "message": "Input directory not found: data/raw"
  }
}
```

Common API error codes:

| Code | Status | Meaning |
| --- | --- | --- |
| `invalid_request` | `400` | Local path, payload, retrieval, or evaluation input is invalid. |
| `invalid_header` | `400` | `Content-Length` is malformed. |
| `resource_not_found` | `404` | A requested local file, directory, or trace is missing. |
| `request_too_large` | `413` | The request body exceeds `RAGOPS_API_MAX_REQUEST_BYTES`. |
| `provider_error` | `502` | The configured model provider failed at runtime. |

FastAPI's native validation errors may still return its standard `422`
response shape when required fields or field types are invalid.

## Runtime Settings

The most relevant API environment variables are:

| Variable | Default |
| --- | --- |
| `RAGOPS_CHUNK_PATH` | `data/processed/chunks.jsonl` |
| `RAGOPS_VECTOR_INDEX_PATH` | `artifacts/index/vector_index.json` |
| `RAGOPS_TRACE_PATH` | `artifacts/traces/traces.jsonl` |
| `RAGOPS_API_MAX_REQUEST_BYTES` | `1000000` |
| `RAGOPS_API_MAX_TOP_K` | `20` |
| `RAGOPS_API_MAX_QUERY_CHARS` | `1000` |
| `RAGOPS_API_MAX_TEXT_CHARS` | `20000` |
| `RAGOPS_LLM_PROVIDER` | `heuristic` |
| `RAGOPS_LLM_ENDPOINT` | unset |
| `RAGOPS_LLM_API_KEY_ENV` | `OPENAI_API_KEY` |
| `RAGOPS_EMBEDDING_PROVIDER` | `fake` |
| `RAGOPS_EMBEDDING_MODEL` | `fake-bow` |

Use `RAGOPS_LLM_PROVIDER=openai-compatible` with `RAGOPS_LLM_ENDPOINT` and an API
key environment variable for provider-backed chat-completions calls. Keep the
default `heuristic` provider for offline tests, deterministic demos, and CI.
