"""Trace storage."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from ragops_lab.domain import RagTrace, RagTraceSummary


class JsonlTraceStore:
    """Simple append-only trace store."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def save(self, trace: RagTrace) -> None:
        """Append a trace to storage."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(trace.model_dump_json())
            handle.write("\n")

    def list(self) -> list[RagTrace]:
        """List all traces."""
        if not self.path.exists():
            return []
        with self.path.open("r", encoding="utf-8") as handle:
            return [RagTrace.model_validate(json.loads(line)) for line in handle if line.strip()]

    def list_summaries(
        self,
        *,
        query: str | None = None,
        min_faithfulness: float | None = None,
        limit: int = 50,
    ) -> Sequence[RagTraceSummary]:
        """List filtered trace summaries, newest first."""
        normalized_query = query.lower().strip() if query else None
        summaries = [RagTraceSummary.from_trace(trace) for trace in self.list()]
        summaries.sort(key=lambda summary: summary.created_at, reverse=True)
        if normalized_query:
            summaries = [
                summary
                for summary in summaries
                if normalized_query in summary.question.lower()
                or normalized_query in summary.trace_id.lower()
                or normalized_query in summary.request_id.lower()
                or normalized_query in summary.model_name.lower()
            ]
        if min_faithfulness is not None:
            summaries = [
                summary
                for summary in summaries
                if summary.faithfulness is not None and summary.faithfulness >= min_faithfulness
            ]
        return summaries[:limit]

    def get(self, trace_id: str) -> RagTrace | None:
        """Retrieve a trace by id."""
        for trace in self.list():
            if trace.trace_id == trace_id:
                return trace
        return None
