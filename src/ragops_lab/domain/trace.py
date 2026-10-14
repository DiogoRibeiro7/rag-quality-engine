"""Trace domain model."""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, Field, field_validator

from .answer import GeneratedAnswer
from .document import ensure_aware_timestamp, validate_identifier
from .evaluation import EvaluationResult
from .retrieval import RetrievalResult


class RagTrace(BaseModel):
    """Persisted trace for a retrieval and answer generation request."""

    trace_id: str = Field(description="Stable trace identifier.")
    request_id: str = Field(default="unknown", min_length=1)
    question: str = Field(min_length=1)
    retrieved_chunks: list[RetrievalResult] = Field(default_factory=list)
    answer: GeneratedAnswer
    evaluation: EvaluationResult | None = Field(default=None)
    model_name: str = Field(min_length=1)
    latency_ms: float = Field(ge=0.0)
    token_estimate: int = Field(ge=0, default=0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("trace_id")
    @classmethod
    def validate_trace_id(cls, value: str) -> str:
        return validate_identifier(value, "trace_id")

    @field_validator("request_id")
    @classmethod
    def validate_request_id(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("request_id must not be empty.")
        return stripped

    @field_validator("question", "model_name")
    @classmethod
    def normalize_text_fields(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("value must not be empty.")
        return stripped

    @field_validator("created_at")
    @classmethod
    def validate_created_at(cls, value: datetime) -> datetime:
        return ensure_aware_timestamp(value)


class RagTraceSummary(BaseModel):
    """Compact trace representation for list views and dashboards."""

    trace_id: str
    request_id: str
    question: str
    model_name: str
    created_at: datetime
    latency_ms: float = Field(ge=0.0)
    token_estimate: int = Field(ge=0)
    retrieved_chunk_count: int = Field(ge=0)
    faithfulness: float | None = Field(default=None, ge=0.0, le=1.0)
    citation_support: float | None = Field(default=None, ge=0.0, le=1.0)
    answer_relevance: float | None = Field(default=None, ge=0.0, le=1.0)
    grounded: bool
    refusal: bool

    @classmethod
    def from_trace(cls, trace: RagTrace) -> RagTraceSummary:
        """Build a compact summary from a full trace."""
        return cls(
            trace_id=trace.trace_id,
            request_id=trace.request_id,
            question=trace.question,
            model_name=trace.model_name,
            created_at=trace.created_at,
            latency_ms=trace.latency_ms,
            token_estimate=trace.token_estimate,
            retrieved_chunk_count=len(trace.retrieved_chunks),
            faithfulness=trace.evaluation.faithfulness if trace.evaluation else None,
            citation_support=trace.evaluation.citation_support if trace.evaluation else None,
            answer_relevance=trace.evaluation.answer_relevance if trace.evaluation else None,
            grounded=trace.answer.grounded,
            refusal=trace.answer.refusal,
        )
