from __future__ import annotations

import json
import logging
from pathlib import Path

from fastapi.testclient import TestClient
from pytest import LogCaptureFixture, MonkeyPatch

from ragops_lab.api.app import TRACE_STORE, app
from ragops_lab.ingestion import ChunkingConfig, ingest_directory


def _prepare_chunks(tmp_path: Path) -> Path:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "apollo.txt").write_text(
        "Apollo 11 landed humans on the Moon in 1969.",
        encoding="utf-8",
    )
    chunks_path = tmp_path / "chunks.jsonl"
    ingest_directory(
        raw_dir,
        chunks_path,
        ChunkingConfig(chunk_size=120, overlap=10),
    )
    return chunks_path


def test_api_generates_request_id_header() -> None:
    response = TestClient(app).get("/health")

    request_id = response.headers["x-request-id"]
    assert request_id.startswith("req-")
    assert len(request_id) > len("req-")


def test_api_preserves_request_id_across_response_and_trace(tmp_path: Path) -> None:
    client = TestClient(app)
    chunks_path = _prepare_chunks(tmp_path)
    TRACE_STORE.path = tmp_path / "traces.jsonl"
    request_id = "demo-request-123"

    response = client.post(
        "/ask",
        headers={"X-Request-ID": request_id},
        json={
            "query": "Which mission landed humans on the Moon?",
            "chunks_path": str(chunks_path),
        },
    )

    assert response.status_code == 200
    assert response.headers["x-request-id"] == request_id
    assert response.json()["request_id"] == request_id
    trace = TRACE_STORE.get(response.json()["trace_id"])
    assert trace is not None
    assert trace.request_id == request_id

    summaries = client.get("/traces", params={"q": request_id})
    assert summaries.status_code == 200
    assert summaries.json()[0]["request_id"] == request_id


def test_api_emits_correlated_structured_events(
    tmp_path: Path,
    caplog: LogCaptureFixture,
) -> None:
    client = TestClient(app)
    chunks_path = _prepare_chunks(tmp_path)
    TRACE_STORE.path = tmp_path / "traces.jsonl"
    request_id = "structured-event-123"
    caplog.set_level(logging.INFO, logger="rag_quality_engine.observability")

    response = client.post(
        "/ask",
        headers={"X-Request-ID": request_id},
        json={
            "query": "Which mission landed humans on the Moon?",
            "chunks_path": str(chunks_path),
        },
    )

    assert response.status_code == 200
    events = [
        json.loads(record.getMessage())
        for record in caplog.records
        if record.name == "rag_quality_engine.observability"
    ]
    event_names = {event["event"] for event in events}
    assert {"request.started", "trace.created", "request.completed"} <= event_names
    correlated = [event for event in events if event.get("request_id") == request_id]
    assert len(correlated) >= 3
    assert all(event.get("path") == "/ask" for event in correlated)
    assert all(event.get("method") == "POST" for event in correlated)


def test_api_emits_structured_error_event_for_handled_failure(
    tmp_path: Path,
    caplog: LogCaptureFixture,
) -> None:
    client = TestClient(app)
    request_id = "handled-error-123"
    caplog.set_level(logging.INFO, logger="rag_quality_engine.observability")

    response = client.post(
        "/search",
        headers={"X-Request-ID": request_id},
        json={
            "query": "moon landing",
            "chunks_path": str(tmp_path / "missing.jsonl"),
        },
    )

    assert response.status_code == 404
    assert response.headers["x-request-id"] == request_id
    events = [
        json.loads(record.getMessage())
        for record in caplog.records
        if record.name == "rag_quality_engine.observability"
    ]
    error_events = [
        event
        for event in events
        if event["event"] == "request.error"
        and event.get("request_id") == request_id
    ]
    assert len(error_events) == 1
    assert error_events[0]["status_code"] == 404


def test_unexpected_failure_keeps_request_correlation(
    monkeypatch: MonkeyPatch,
) -> None:
    api_module = __import__("ragops_lab.api.app", fromlist=["_search"])
    request_id = "unexpected-error-123"

    def fail_search(_: object) -> object:
        raise KeyError("boom")

    monkeypatch.setattr(api_module, "_search", fail_search)
    response = TestClient(app, raise_server_exceptions=False).post(
        "/search",
        headers={"X-Request-ID": request_id},
        json={"query": "moon", "chunks_path": "unused.jsonl"},
    )

    assert response.status_code == 500
    assert response.headers["x-request-id"] == request_id
    assert response.json()["error"]["code"] == "internal_error"
