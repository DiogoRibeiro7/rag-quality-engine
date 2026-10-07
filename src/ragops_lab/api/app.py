"""FastAPI application for RAG Quality Engine."""

from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable
from html import escape
from pathlib import Path
from time import perf_counter

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel, Field, field_validator

from ragops_lab import __version__
from ragops_lab.config import RuntimeSettings
from ragops_lab.domain import RagTrace, RetrievalResult
from ragops_lab.evaluation import evaluate_answer
from ragops_lab.generation import GenerationService, build_llm_client
from ragops_lab.ingestion import ChunkingConfig, ingest_directory, load_chunks_jsonl
from ragops_lab.retrieval import (
    BM25Retriever,
    HybridRetriever,
    LocalVectorIndex,
    VectorRetriever,
    build_embedding_client,
    search_with_profile,
)
from ragops_lab.tracing import JsonlTraceStore

SETTINGS = RuntimeSettings.from_env()

app = FastAPI(title="RAG Quality Engine API", version=__version__)


def _error_response(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "message": message}},
    )


@app.exception_handler(FileNotFoundError)
async def handle_file_not_found(_: Request, exc: FileNotFoundError) -> JSONResponse:
    """Return a clear API error when a configured local file is missing."""
    return _error_response(404, "resource_not_found", str(exc))


@app.exception_handler(ValueError)
async def handle_value_error(_: Request, exc: ValueError) -> JSONResponse:
    """Return a clear API error for invalid local runtime inputs."""
    return _error_response(400, "invalid_request", str(exc))


@app.exception_handler(RuntimeError)
async def handle_runtime_error(_: Request, exc: RuntimeError) -> JSONResponse:
    """Return a clear API error for provider runtime failures."""
    return _error_response(502, "provider_error", str(exc))


@app.middleware("http")
async def enforce_request_size(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    """Reject request bodies above the configured API limit."""
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            declared_size = int(content_length)
        except ValueError:
            return _error_response(400, "invalid_header", "Invalid Content-Length header.")
        if declared_size > SETTINGS.api_max_request_bytes:
            return _error_response(
                413,
                "request_too_large",
                (
                    "Request body is too large. "
                    f"Maximum size is {SETTINGS.api_max_request_bytes} bytes."
                ),
            )
    body = await request.body()
    if len(body) > SETTINGS.api_max_request_bytes:
        return _error_response(
            413,
            "request_too_large",
            (f"Request body is too large. Maximum size is {SETTINGS.api_max_request_bytes} bytes."),
        )
    return await call_next(request)


class IngestRequest(BaseModel):
    """Request body for ingestion."""

    input_dir: str = Field(default_factory=lambda: str(SETTINGS.paths.data_dir / "raw"))
    out_path: str = Field(default_factory=lambda: str(SETTINGS.paths.chunk_path))
    chunk_size: int = Field(default=500, gt=0)
    overlap: int = Field(default=50, ge=0)

    @field_validator("input_dir", "out_path")
    @classmethod
    def validate_path_text(cls, value: str) -> str:
        return _validate_text_limit(value, field_name="path")


class IndexRequest(BaseModel):
    """Request body for local vector indexing."""

    chunks_path: str = Field(default_factory=lambda: str(SETTINGS.paths.chunk_path))
    index_path: str = Field(default_factory=lambda: str(SETTINGS.paths.vector_index_path))

    @field_validator("chunks_path", "index_path")
    @classmethod
    def validate_path_text(cls, value: str) -> str:
        return _validate_text_limit(value, field_name="path")


class SearchRequest(BaseModel):
    """Search request."""

    query: str = Field(min_length=1)
    profile: str = Field(default="lexical")
    chunks_path: str = Field(default_factory=lambda: str(SETTINGS.paths.chunk_path))
    index_path: str | None = Field(default=None)
    top_k: int | None = Field(default=None, ge=1)
    mode: str | None = Field(default=None)
    lexical_weight: float | None = Field(default=None, ge=0.0, le=1.0)
    vector_weight: float | None = Field(default=None, ge=0.0, le=1.0)
    fusion_strategy: str | None = Field(default=None)
    rrf_k: int | None = Field(default=None, ge=1)
    rerank: bool | None = Field(default=None)
    rerank_candidate_multiplier: int | None = Field(default=None, ge=1)

    @field_validator("query")
    @classmethod
    def validate_query_length(cls, value: str) -> str:
        return _validate_text_limit(
            value,
            field_name="query",
            max_chars=SETTINGS.api_max_query_chars,
        )

    @field_validator("profile", "chunks_path", "index_path", "mode", "fusion_strategy")
    @classmethod
    def validate_short_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _validate_text_limit(value, field_name="request field")

    @field_validator("top_k")
    @classmethod
    def validate_top_k(cls, value: int | None) -> int | None:
        if value is not None and value > SETTINGS.api_max_top_k:
            raise ValueError(f"top_k must be less than or equal to {SETTINGS.api_max_top_k}.")
        return value


class AskRequest(SearchRequest):
    """Ask request."""


class EvaluateRequest(BaseModel):
    """Evaluation request."""

    question: str = Field(min_length=1)
    answer_text: str = Field(min_length=1)
    citations: list[str] = Field(default_factory=list)
    chunks_path: str = Field(default_factory=lambda: str(SETTINGS.paths.chunk_path))
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    reference_chunk_ids: list[str] = Field(default_factory=list)
    expected_answer: str = Field(default="")
    expected_unanswerable: bool | None = Field(default=None)

    @field_validator("question")
    @classmethod
    def validate_question_length(cls, value: str) -> str:
        return _validate_text_limit(
            value,
            field_name="question",
            max_chars=SETTINGS.api_max_query_chars,
        )

    @field_validator("answer_text", "expected_answer")
    @classmethod
    def validate_payload_text_length(cls, value: str) -> str:
        return _validate_text_limit(
            value,
            field_name="evaluation text",
            max_chars=SETTINGS.api_max_text_chars,
        )

    @field_validator("chunks_path")
    @classmethod
    def validate_chunks_path(cls, value: str) -> str:
        return _validate_text_limit(value, field_name="path")

    @field_validator("citations", "retrieved_chunk_ids", "reference_chunk_ids")
    @classmethod
    def validate_identifier_list_length(cls, value: list[str]) -> list[str]:
        if len(value) > SETTINGS.api_max_top_k:
            raise ValueError(
                f"Identifier lists must contain at most {SETTINGS.api_max_top_k} entries."
            )
        return value


TRACE_STORE = JsonlTraceStore(SETTINGS.paths.trace_path)


def _validate_text_limit(
    value: str,
    *,
    field_name: str,
    max_chars: int | None = None,
) -> str:
    limit = max_chars or SETTINGS.api_max_text_chars
    if len(value) > limit:
        raise ValueError(f"{field_name} must contain at most {limit} characters.")
    return value


def _search(request: SearchRequest) -> list[RetrievalResult]:
    profile = SETTINGS.resolve_retrieval_profile(
        request.profile,
        mode=request.mode,
        top_k=request.top_k,
        lexical_weight=request.lexical_weight,
        vector_weight=request.vector_weight,
        fusion_strategy=request.fusion_strategy,
        rrf_k=request.rrf_k,
        rerank=request.rerank,
        rerank_candidate_multiplier=request.rerank_candidate_multiplier,
    )
    if profile.top_k > SETTINGS.api_max_top_k:
        raise ValueError(f"top_k must be less than or equal to {SETTINGS.api_max_top_k}.")
    vector_index = LocalVectorIndex.load(Path(request.index_path)) if request.index_path else None
    chunks = vector_index.chunks if vector_index else load_chunks_jsonl(Path(request.chunks_path))
    lexical = BM25Retriever(chunks)
    vector = (
        vector_index.as_retriever()
        if vector_index
        else VectorRetriever(
            chunks,
            build_embedding_client(SETTINGS.embeddings),
        )
    )
    if profile.mode == "lexical":
        retriever = lexical
    elif profile.mode == "vector":
        retriever = vector
    elif profile.mode == "hybrid":
        retriever = HybridRetriever(
            lexical,
            vector,
            lexical_weight=profile.lexical_weight,
            vector_weight=profile.vector_weight,
            fusion_strategy=profile.fusion_strategy,
            rrf_k=profile.rrf_k,
        )
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported retrieval mode: {profile.mode}")
    return search_with_profile(retriever, profile, request.query)


@app.post("/ingest")
def ingest(request: IngestRequest) -> dict[str, int | str]:
    """Ingest local files into chunk JSONL."""
    input_dir = Path(request.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not input_dir.is_dir():
        raise ValueError(f"Input path is not a directory: {input_dir}")
    chunks = ingest_directory(
        input_dir,
        Path(request.out_path),
        config=ChunkingConfig(chunk_size=request.chunk_size, overlap=request.overlap),
    )
    return {"chunks_written": len(chunks), "out_path": request.out_path}


@app.post("/index")
def index(request: IndexRequest) -> dict[str, int | str]:
    """Build a persistent local vector index from chunk JSONL."""
    chunks = load_chunks_jsonl(Path(request.chunks_path))
    embedding_client = build_embedding_client(SETTINGS.embeddings)
    LocalVectorIndex.build(
        chunks,
        embedding_client,
        embedding_provider=SETTINGS.embeddings.provider,
        embedding_model=SETTINGS.embeddings.model,
    ).save(Path(request.index_path))
    return {"chunks_indexed": len(chunks), "index_path": request.index_path}


@app.post("/search")
def search(request: SearchRequest) -> list[dict[str, object]]:
    """Search indexed chunks."""
    return [result.model_dump(mode="json") for result in _search(request)]


@app.post("/ask")
def ask(request: AskRequest) -> dict[str, object]:
    """Answer a question using retrieved evidence."""
    started = perf_counter()
    results = _search(request)
    llm_client, model_name = build_llm_client(SETTINGS.llm)
    service = GenerationService(llm_client)
    answer = service.answer(request.query, results, model_name=model_name)
    evaluation = evaluate_answer(request.query, answer, results)
    trace = RagTrace(
        trace_id=f"trace-{uuid.uuid4().hex[:12]}",
        question=request.query,
        retrieved_chunks=results,
        answer=answer,
        evaluation=evaluation,
        model_name=answer.model_name,
        latency_ms=(perf_counter() - started) * 1000,
        token_estimate=sum(result.chunk.token_count for result in results),
    )
    TRACE_STORE.save(trace)
    return {
        "trace_id": trace.trace_id,
        "answer": answer.model_dump(mode="json"),
        "evaluation": evaluation.model_dump(mode="json"),
    }


@app.post("/evaluate")
def evaluate(request: EvaluateRequest) -> dict[str, object]:
    """Evaluate an answer against retrieved evidence."""
    chunks = {chunk.chunk_id: chunk for chunk in load_chunks_jsonl(Path(request.chunks_path))}
    missing_chunk_ids = sorted(set(request.retrieved_chunk_ids) - set(chunks))
    if missing_chunk_ids:
        raise ValueError(f"Retrieved chunk ids not found: {missing_chunk_ids}")
    results = [
        RetrievalResult(chunk=chunks[chunk_id], score=1.0, rank=index, retrieval_method="manual")
        for index, chunk_id in enumerate(request.retrieved_chunk_ids, start=1)
        if chunk_id in chunks
    ]
    from ragops_lab.domain import GeneratedAnswer

    answer = GeneratedAnswer(
        question=request.question,
        answer_text=request.answer_text,
        citations=request.citations,
        model_name="manual-eval",
        refusal=False,
        grounded=bool(request.citations),
    )
    report = evaluate_answer(
        request.question,
        answer,
        results,
        reference_chunk_ids=request.reference_chunk_ids,
        expected_answer=request.expected_answer,
        expected_unanswerable=request.expected_unanswerable,
    )
    return report.model_dump(mode="json")


@app.get("/traces")
def list_traces(
    q: str | None = Query(default=None, max_length=SETTINGS.api_max_query_chars),
    min_faithfulness: float | None = Query(default=None, ge=0.0, le=1.0),
    limit: int = Query(default=50, ge=1, le=SETTINGS.api_max_top_k * 5),
) -> list[dict[str, object]]:
    """List trace summaries with optional filters."""
    summaries = TRACE_STORE.list_summaries(
        query=q,
        min_faithfulness=min_faithfulness,
        limit=limit,
    )
    return [summary.model_dump(mode="json") for summary in summaries]


@app.get("/traces/{trace_id}")
def get_trace(trace_id: str) -> dict[str, object]:
    """Fetch a single trace."""
    trace = TRACE_STORE.get(trace_id)
    if trace is None:
        raise HTTPException(status_code=404, detail="Trace not found.")
    return trace.model_dump(mode="json")


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(
    q: str | None = Query(default=None, max_length=SETTINGS.api_max_query_chars),
    min_faithfulness: float | None = Query(default=None, ge=0.0, le=1.0),
    limit: int = Query(default=25, ge=1, le=SETTINGS.api_max_top_k * 5),
) -> str:
    """Render a trace dashboard with filtering and summary metrics."""
    summaries = TRACE_STORE.list_summaries(
        query=q,
        min_faithfulness=min_faithfulness,
        limit=limit,
    )
    rows = "\n".join(
        "<tr>"
        f'<td><a href="/traces/{escape(summary.trace_id)}">{escape(summary.trace_id)}</a></td>'
        f"<td>{escape(summary.question)}</td>"
        f"<td>{escape(summary.model_name)}</td>"
        f"<td>{summary.retrieved_chunk_count}</td>"
        f"<td>{_format_optional_score(summary.faithfulness)}</td>"
        f"<td>{_format_optional_score(summary.citation_support)}</td>"
        f"<td>{summary.latency_ms:.1f}</td>"
        f"<td>{summary.token_estimate}</td>"
        f"<td>{escape(summary.created_at.isoformat())}</td>"
        "</tr>"
        for summary in summaries
    )
    query_value = escape(q or "")
    faithfulness_value = "" if min_faithfulness is None else f"{min_faithfulness:.2f}"
    empty_row = '<tr><td colspan="9">No traces found.</td></tr>'
    return (
        "<html><head><title>RAGOps Traces</title>"
        "<style>"
        "body{font-family:Arial,sans-serif;margin:2rem;color:#1f2933;}"
        "form{display:flex;gap:.75rem;align-items:end;margin-bottom:1rem;flex-wrap:wrap;}"
        "label{display:flex;flex-direction:column;font-size:.85rem;font-weight:600;}"
        "input{padding:.45rem;border:1px solid #b8c2cc;border-radius:4px;}"
        "button{padding:.5rem .75rem;border:1px solid #1f2933;background:#1f2933;color:white;"
        "border-radius:4px;}"
        "table{border-collapse:collapse;width:100%;font-size:.9rem;}"
        "th,td{border-bottom:1px solid #d9e2ec;padding:.55rem;text-align:left;vertical-align:top;}"
        "th{background:#f0f4f8;}"
        "</style></head><body><h1>RAGOps Traces</h1>"
        '<form method="get">'
        f'<label>Search<input name="q" value="{query_value}" /></label>'
        "<label>Min faithfulness"
        f'<input name="min_faithfulness" value="{faithfulness_value}" /></label>'
        f'<label>Limit<input name="limit" value="{limit}" /></label>'
        '<button type="submit">Apply</button>'
        "</form>"
        "<table><thead><tr><th>Trace</th><th>Question</th><th>Model</th>"
        "<th>Chunks</th><th>Faithfulness</th><th>Citation</th><th>Latency ms</th>"
        "<th>Tokens</th><th>Created</th></tr></thead><tbody>"
        f"{rows or empty_row}"
        "</tbody></table></body></html>"
    )


def _format_optional_score(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}"
