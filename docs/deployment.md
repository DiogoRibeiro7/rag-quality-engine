# Deployment Guide

RAG Quality Engine is designed to run locally by default, with deterministic
offline providers and filesystem-backed data, indexes, artifacts, and traces.
The simplest deployment target is the FastAPI service in the repository
container.

## Local Docker Compose

Build and start the API:

```bash
docker compose up --build
```

The service is available at:

- API: `http://localhost:8000`
- OpenAPI docs: `http://localhost:8000/docs`
- Dashboard: `http://localhost:8000/dashboard`
- Health check: `http://localhost:8000/health`

Compose mounts the repository's `data/` and `artifacts/` directories into the
container so processed chunks, indexes, evaluation outputs, and traces survive
container replacement.

Stop the service with:

```bash
docker compose down
```

## Direct Docker Run

Build the image:

```bash
docker build -t rag-quality-engine .
```

Run it with persistent local directories:

```bash
docker run --rm \
  -p 8000:8000 \
  -v "${PWD}/data:/app/data" \
  -v "${PWD}/artifacts:/app/artifacts" \
  rag-quality-engine
```

The image starts Uvicorn on `0.0.0.0:8000` and exposes a Docker health check
against `GET /health`.

Inspect container health with:

```bash
docker inspect --format='{{json .State.Health}}' <container-id>
```

## Runtime Configuration

The container uses the same `RAGOPS_*` environment variables as local CLI/API
execution. For example:

```bash
docker run --rm \
  -p 8000:8000 \
  -v "${PWD}/data:/app/data" \
  -v "${PWD}/artifacts:/app/artifacts" \
  -e RAGOPS_CHUNK_PATH=/app/data/processed/chunks.jsonl \
  -e RAGOPS_VECTOR_INDEX_PATH=/app/artifacts/index/vector_index.json \
  -e RAGOPS_TRACE_PATH=/app/artifacts/traces/traces.jsonl \
  rag-quality-engine
```

Provider-backed generation can be enabled explicitly:

```bash
docker run --rm \
  -p 8000:8000 \
  -v "${PWD}/data:/app/data" \
  -v "${PWD}/artifacts:/app/artifacts" \
  -e RAGOPS_LLM_PROVIDER=openai-compatible \
  -e RAGOPS_LLM_ENDPOINT=https://example.invalid/v1/chat/completions \
  -e RAGOPS_LLM_MODEL=your-model \
  -e OPENAI_API_KEY \
  rag-quality-engine
```

Keep secrets in the runtime environment or a secret manager. Do not bake API
keys into the image, Compose file, or repository.

## Preparing Data

The API container does not automatically ingest or index documents at startup.
Create the required assets before serving queries, either through the API or the
CLI.

Example local preparation:

```bash
poetry run rag-quality-engine ingest data/sample_documents \
  --out data/processed/chunks.jsonl

poetry run rag-quality-engine index \
  --chunks data/processed/chunks.jsonl \
  --out artifacts/index/vector_index.json
```

Because `data/` and `artifacts/` are mounted into the container, the service
can use those files immediately after startup.

## Public Deployment

For a public demo or portfolio deployment, run the container behind a platform
or reverse proxy that provides:

- HTTPS/TLS termination;
- request and connection limits;
- access logging and monitoring;
- secret injection;
- persistent storage for `data/` and `artifacts/`;
- restart and health-check handling.

The built-in Uvicorn command is suitable for the repository demo and a single
container. For higher availability, let the hosting platform manage multiple
container replicas and routing rather than adding process management inside the
image.

Because traces and indexes are currently filesystem-backed, replicas must share
the same persistent storage or use separate isolated datasets. The current
storage model is intentionally simple and is not a substitute for a distributed
database or managed vector store.

## Health Check

`GET /health` is intentionally lightweight and does not require data files,
model credentials, or provider connectivity.

Example:

```bash
curl http://localhost:8000/health
```

Expected response:

```json
{
  "status": "ok",
  "version": "0.2.0"
}
```

A successful health response means the API process is running. It does not
guarantee that optional external LLM or embedding providers are reachable.

## Deployment Checklist

Before exposing the service publicly:

- run `make check`;
- run `make rag-eval`;
- build the container from the intended release commit;
- verify `GET /health`;
- verify the required chunks/index files exist;
- configure persistent storage;
- inject provider credentials only at runtime;
- place HTTPS and request controls in front of the service;
- inspect `GET /dashboard` and trace persistence after a test request.
