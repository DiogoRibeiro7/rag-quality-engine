from __future__ import annotations

import importlib
from pathlib import Path

from fastapi.testclient import TestClient
from pytest import MonkeyPatch

from ragops_lab import __version__
from ragops_lab.api.app import SETTINGS, TRACE_STORE, app
from ragops_lab.ingestion import ChunkingConfig, ingest_directory, load_chunks_jsonl


def test_api_end_to_end(tmp_path: Path) -> None:
    client = TestClient(app)
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "apollo.txt").write_text(
        "Apollo 11 was the first mission to land on the Moon in July 1969.",
        encoding="utf-8",
    )
    chunks_path = tmp_path / "chunks.jsonl"
    ingest_directory(raw_dir, chunks_path, ChunkingConfig(chunk_size=120, overlap=10))
    TRACE_STORE.path = tmp_path / "traces.jsonl"

    search_response = client.post(
        "/search", json={"query": "moon landing mission", "chunks_path": str(chunks_path)}
    )
    ask_response = client.post(
        "/ask", json={"query": "Which mission landed on the Moon?", "chunks_path": str(chunks_path)}
    )

    assert search_response.status_code == 200
    assert ask_response.status_code == 200
    trace_id = ask_response.json()["trace_id"]
    trace_response = client.get(f"/traces/{trace_id}")
    traces_response = client.get("/traces", params={"q": "moon", "min_faithfulness": 0.0})
    dashboard_response = client.get("/dashboard", params={"q": "moon"})

    assert trace_response.status_code == 200
    assert trace_response.json()["trace_id"] == trace_id
    assert traces_response.status_code == 200
    assert traces_response.json()[0]["trace_id"] == trace_id
    assert dashboard_response.status_code == 200
    assert "RAGOps Traces" in dashboard_response.text
    assert trace_id in dashboard_response.text



def test_dashboard_renders_quality_kpis_and_attention_rows(tmp_path: Path) -> None:
    client = TestClient(app)
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "apollo.txt").write_text(
        "Apollo 11 was the first mission to land humans on the Moon.",
        encoding="utf-8",
    )
    chunks_path = tmp_path / "chunks.jsonl"
    ingest_directory(raw_dir, chunks_path, ChunkingConfig(chunk_size=120, overlap=10))
    TRACE_STORE.path = tmp_path / "traces.jsonl"

    first = client.post(
        "/ask",
        json={
            "query": "Which mission landed humans on the Moon?",
            "chunks_path": str(chunks_path),
        },
    )
    assert first.status_code == 200

    trace = TRACE_STORE.list()[0]
    degraded = trace.model_copy(
        update={
            "trace_id": "trace-degraded",
            "latency_ms": 40.0,
            "token_estimate": 12,
            "evaluation": trace.evaluation.model_copy(
                update={
                    "faithfulness": 0.5,
                    "citation_support": 0.5,
                    "answer_relevance": 0.25,
                }
            )
            if trace.evaluation
            else None,
        }
    )
    TRACE_STORE.save(degraded)

    response = client.get("/dashboard")

    assert response.status_code == 200
    assert "RAG Quality Engine Traces" in response.text
    assert "Avg faithfulness" in response.text
    assert "Avg citation support" in response.text
    assert "Avg latency ms" in response.text
    assert "Total tokens" in response.text
    assert "Relevance" in response.text
    assert "Grounded" in response.text
    assert "Refusal" in response.text
    assert 'class="attention"' in response.text

def test_api_ingest_and_evaluate_endpoints(tmp_path: Path) -> None:
    client = TestClient(app)
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "apollo.txt").write_text(
        "Apollo 11 landed humans on the Moon.",
        encoding="utf-8",
    )
    chunks_path = tmp_path / "chunks.jsonl"

    ingest_response = client.post(
        "/ingest",
        json={
            "input_dir": str(raw_dir),
            "out_path": str(chunks_path),
            "chunk_size": 120,
            "overlap": 10,
        },
    )
    evaluate_response = client.post(
        "/evaluate",
        json={
            "question": "Which mission landed humans on the Moon?",
            "answer_text": "Apollo 11 landed humans on the Moon.",
            "citations": ["apollo:0"],
            "chunks_path": str(chunks_path),
            "retrieved_chunk_ids": ["apollo:0"],
            "reference_chunk_ids": ["apollo:0"],
        },
    )

    assert ingest_response.status_code == 200
    assert ingest_response.json()["chunks_written"] == 1
    assert evaluate_response.status_code == 200
    assert evaluate_response.json()["citation_support"] == 1.0


def test_api_builds_and_uses_persistent_vector_index(tmp_path: Path) -> None:
    client = TestClient(app)
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "metrics.txt").write_text(
        "Faithfulness and citation support are critical RAG metrics.",
        encoding="utf-8",
    )
    chunks_path = tmp_path / "chunks.jsonl"
    index_path = tmp_path / "vector_index.json"
    ingest_directory(raw_dir, chunks_path, ChunkingConfig(chunk_size=120, overlap=10))

    index_response = client.post(
        "/index",
        json={"chunks_path": str(chunks_path), "index_path": str(index_path)},
    )
    search_response = client.post(
        "/search",
        json={
            "query": "citation support",
            "index_path": str(index_path),
            "mode": "vector",
            "top_k": 1,
        },
    )

    assert index_response.status_code == 200
    assert index_response.json()["chunks_indexed"] == 1
    assert index_path.exists()
    assert search_response.status_code == 200
    assert search_response.json()[0]["chunk"]["chunk_id"] == "metrics:0"


def test_api_search_uses_named_retrieval_profile_with_overrides(tmp_path: Path) -> None:
    client = TestClient(app)
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "apollo.txt").write_text(
        "Apollo 11 landed on the Moon in 1969.",
        encoding="utf-8",
    )
    chunks_path = tmp_path / "chunks.jsonl"
    index_path = tmp_path / "vector_index.json"
    ingest_directory(raw_dir, chunks_path, ChunkingConfig(chunk_size=120, overlap=10))
    client.post("/index", json={"chunks_path": str(chunks_path), "index_path": str(index_path)})

    response = client.post(
        "/search",
        json={
            "query": "moon landing",
            "profile": "hybrid",
            "chunks_path": str(chunks_path),
            "index_path": str(index_path),
            "top_k": 1,
            "lexical_weight": 0.8,
            "vector_weight": 0.2,
        },
    )

    assert response.status_code == 200
    assert response.json()[0]["retrieval_method"] == "hybrid"
    assert response.json()[0]["chunk"]["chunk_id"] == "apollo:0"



def test_api_search_can_enable_reranking(tmp_path: Path) -> None:
    client = TestClient(app)
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "first.txt").write_text(
        "Apollo 11 landed on the Moon in 1969.",
        encoding="utf-8",
    )
    (raw_dir / "second.txt").write_text(
        "Citation support measures whether answer citations point to retrieved evidence.",
        encoding="utf-8",
    )
    chunks_path = tmp_path / "chunks.jsonl"
    ingest_directory(raw_dir, chunks_path, ChunkingConfig(chunk_size=120, overlap=10))

    response = client.post(
        "/search",
        json={
            "query": "citation support",
            "chunks_path": str(chunks_path),
            "profile": "lexical",
            "top_k": 1,
            "rerank": True,
            "rerank_candidate_multiplier": 3,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload[0]["chunk"]["chunk_id"] == "second:0"
    assert payload[0]["retrieval_method"] == "reranked-lexical"


def test_api_uses_configured_default_chunk_path() -> None:
    assert app.version == __version__
    assert SETTINGS.paths.chunk_path.as_posix() == "data/processed/chunks.jsonl"


def test_api_rejects_top_k_above_runtime_limit(tmp_path: Path) -> None:
    client = TestClient(app)
    chunks_path = tmp_path / "chunks.jsonl"
    chunks_path.write_text("", encoding="utf-8")

    response = client.post(
        "/search",
        json={
            "query": "moon landing",
            "chunks_path": str(chunks_path),
            "top_k": SETTINGS.api_max_top_k + 1,
        },
    )

    assert response.status_code == 422
    assert "top_k" in response.text


def test_api_rejects_overlong_query() -> None:
    client = TestClient(app)
    response = client.post(
        "/search",
        json={
            "query": "x" * (SETTINGS.api_max_query_chars + 1),
            "chunks_path": "missing.jsonl",
        },
    )

    assert response.status_code == 422
    assert "query" in response.text


def test_api_rejects_unsupported_retrieval_mode(tmp_path: Path) -> None:
    client = TestClient(app)
    chunks_path = tmp_path / "chunks.jsonl"
    chunks_path.write_text("", encoding="utf-8")

    response = client.post(
        "/search",
        json={"query": "moon landing", "chunks_path": str(chunks_path), "mode": "unknown"},
    )

    assert response.status_code == 400
    assert "Unsupported retrieval mode" in response.text


def test_api_rejects_unknown_retrieval_profile(tmp_path: Path) -> None:
    client = TestClient(app)
    chunks_path = tmp_path / "chunks.jsonl"
    chunks_path.write_text("", encoding="utf-8")

    response = client.post(
        "/search",
        json={"query": "moon landing", "chunks_path": str(chunks_path), "profile": "missing"},
    )

    assert response.status_code == 400
    assert "Unknown retrieval profile" in response.text


def test_api_reports_missing_chunks_file() -> None:
    client = TestClient(app)
    response = client.post(
        "/search",
        json={"query": "moon landing", "chunks_path": "missing/chunks.jsonl"},
    )

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "resource_not_found"
    assert "missing" in response.json()["error"]["message"]


def test_api_reports_missing_ingest_directory(tmp_path: Path) -> None:
    client = TestClient(app)
    missing_dir = tmp_path / "missing"
    response = client.post(
        "/ingest",
        json={"input_dir": str(missing_dir), "out_path": str(tmp_path / "chunks.jsonl")},
    )

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "resource_not_found"
    assert "Input directory not found" in response.json()["error"]["message"]


def test_api_reports_missing_manual_evaluation_chunk_ids(tmp_path: Path) -> None:
    client = TestClient(app)
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "apollo.txt").write_text("Apollo 11 landed on the Moon.", encoding="utf-8")
    chunks_path = tmp_path / "chunks.jsonl"
    ingest_directory(raw_dir, chunks_path, ChunkingConfig(chunk_size=120, overlap=10))

    response = client.post(
        "/evaluate",
        json={
            "question": "Which mission landed on the Moon?",
            "answer_text": "Apollo 11 landed on the Moon.",
            "chunks_path": str(chunks_path),
            "retrieved_chunk_ids": ["missing:0"],
        },
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_request"
    assert "Retrieved chunk ids not found" in response.json()["error"]["message"]


def test_api_reports_embedding_provider_runtime_errors(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    client = TestClient(app, raise_server_exceptions=False)
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "metrics.txt").write_text(
        "Faithfulness and citation support are critical RAG metrics.",
        encoding="utf-8",
    )
    chunks_path = tmp_path / "chunks.jsonl"
    ingest_directory(raw_dir, chunks_path, ChunkingConfig(chunk_size=120, overlap=10))
    api_app_module = importlib.import_module("ragops_lab.api.app")

    def fail_embedding_provider(_: object) -> object:
        raise RuntimeError("embedding provider is unavailable")

    monkeypatch.setattr(api_app_module, "build_embedding_client", fail_embedding_provider)

    response = client.post(
        "/search",
        json={
            "query": "citation support",
            "chunks_path": str(chunks_path),
            "mode": "vector",
        },
    )

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "provider_error"
    assert "embedding provider is unavailable" in response.json()["error"]["message"]


def test_api_rejects_oversized_request_body() -> None:
    client = TestClient(app)
    response = client.post(
        "/search",
        content=b"x" * (SETTINGS.api_max_request_bytes + 1),
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 413
    assert "Request body is too large" in response.text


def test_api_ingest_supports_sentence_strategy(tmp_path: Path) -> None:
    client = TestClient(app)
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "sentences.txt").write_text(
        "First sentence. Second sentence. Third sentence.",
        encoding="utf-8",
    )
    chunks_path = tmp_path / "chunks.jsonl"

    response = client.post(
        "/ingest",
        json={
            "input_dir": str(raw_dir),
            "out_path": str(chunks_path),
            "chunk_size": 35,
            "overlap": 10,
            "strategy": "sentence",
        },
    )

    assert response.status_code == 200
    chunks = load_chunks_jsonl(chunks_path)
    assert len(chunks) >= 2
    assert all(chunk.text.endswith((".", "!", "?")) for chunk in chunks)
