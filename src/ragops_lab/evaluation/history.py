"""Benchmark history storage and longitudinal trend reporting."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field

from .benchmark import BenchmarkSummary, load_benchmark_summary


class BenchmarkHistoryEntry(BaseModel):
    """One persisted benchmark-history record."""

    recorded_at: datetime
    label: str = Field(min_length=1)
    git_commit: str = Field(min_length=1)
    summary: BenchmarkSummary

    @property
    def identity(self) -> tuple[str, str, str]:
        """Return the stable duplicate-detection identity."""
        return (
            self.label,
            self.git_commit,
            self.summary.provenance.fingerprint,
        )


class BenchmarkTrend(BaseModel):
    """Longitudinal benchmark changes across comparable history entries."""

    entry_count: int = Field(ge=1)
    fingerprint: str = Field(min_length=64, max_length=64)
    first_label: str
    latest_label: str
    first_recorded_at: datetime
    latest_recorded_at: datetime
    recall_at_k_delta: float
    mean_reciprocal_rank_delta: float
    faithfulness_delta: float
    citation_support_delta: float
    refusal_accuracy_delta: float
    p95_latency_ms_delta: float
    p95_token_estimate_delta: int


def load_benchmark_history(path: Path) -> list[BenchmarkHistoryEntry]:
    """Load benchmark history from an append-only JSONL file."""
    if not path.exists():
        return []

    entries: list[BenchmarkHistoryEntry] = []
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw_line.strip()
        if not line:
            continue
        try:
            entries.append(BenchmarkHistoryEntry.model_validate_json(line))
        except ValueError as exc:
            raise ValueError(
                f"Invalid benchmark history record at line {line_number}."
            ) from exc
    return sorted(entries, key=lambda entry: entry.recorded_at)


def append_benchmark_history(
    *,
    history_path: Path,
    summary_path: Path,
    label: str,
    git_commit: str,
    recorded_at: datetime | None = None,
) -> BenchmarkHistoryEntry:
    """Append one benchmark summary to the local history file."""
    summary = load_benchmark_summary(summary_path)
    entry = BenchmarkHistoryEntry(
        recorded_at=recorded_at or datetime.now(UTC),
        label=label.strip(),
        git_commit=git_commit.strip(),
        summary=summary,
    )
    existing = load_benchmark_history(history_path)
    if any(item.identity == entry.identity for item in existing):
        raise ValueError(
            "Benchmark history already contains this label/commit/fingerprint."
        )

    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("a", encoding="utf-8") as handle:
        handle.write(entry.model_dump_json() + "\n")
    return entry


def build_benchmark_trend(
    entries: list[BenchmarkHistoryEntry],
    *,
    fingerprint: str | None = None,
) -> BenchmarkTrend:
    """Build a trend from chronologically ordered, comparable history entries."""
    if not entries:
        raise ValueError("Benchmark history is empty.")

    ordered = sorted(entries, key=lambda entry: entry.recorded_at)
    selected_fingerprint = fingerprint or ordered[-1].summary.provenance.fingerprint
    comparable = [
        entry
        for entry in ordered
        if entry.summary.provenance.fingerprint == selected_fingerprint
    ]
    if not comparable:
        raise ValueError(
            f"No benchmark history entries match fingerprint {selected_fingerprint}."
        )

    first = comparable[0]
    latest = comparable[-1]
    first_summary = first.summary
    latest_summary = latest.summary
    return BenchmarkTrend(
        entry_count=len(comparable),
        fingerprint=selected_fingerprint,
        first_label=first.label,
        latest_label=latest.label,
        first_recorded_at=first.recorded_at,
        latest_recorded_at=latest.recorded_at,
        recall_at_k_delta=(
            latest_summary.average_recall_at_k - first_summary.average_recall_at_k
        ),
        mean_reciprocal_rank_delta=(
            latest_summary.mean_reciprocal_rank
            - first_summary.mean_reciprocal_rank
        ),
        faithfulness_delta=(
            latest_summary.average_faithfulness
            - first_summary.average_faithfulness
        ),
        citation_support_delta=(
            latest_summary.average_citation_support
            - first_summary.average_citation_support
        ),
        refusal_accuracy_delta=(
            latest_summary.average_refusal_accuracy
            - first_summary.average_refusal_accuracy
        ),
        p95_latency_ms_delta=(
            latest_summary.worst_run_p95_latency_ms
            - first_summary.worst_run_p95_latency_ms
        ),
        p95_token_estimate_delta=(
            latest_summary.worst_run_p95_token_estimate
            - first_summary.worst_run_p95_token_estimate
        ),
    )


def write_benchmark_trend_markdown(trend: BenchmarkTrend, path: Path) -> None:
    """Write a compact Markdown trend report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Benchmark Trend",
        "",
        f"- Entries: {trend.entry_count}",
        f"- Fingerprint: {trend.fingerprint}",
        f"- First label: {trend.first_label}",
        f"- Latest label: {trend.latest_label}",
        f"- Recall@k delta: {trend.recall_at_k_delta:+.4f}",
        f"- MRR delta: {trend.mean_reciprocal_rank_delta:+.4f}",
        f"- Faithfulness delta: {trend.faithfulness_delta:+.4f}",
        f"- Citation support delta: {trend.citation_support_delta:+.4f}",
        f"- Refusal accuracy delta: {trend.refusal_accuracy_delta:+.4f}",
        f"- p95 latency delta (ms): {trend.p95_latency_ms_delta:+.4f}",
        f"- p95 token estimate delta: {trend.p95_token_estimate_delta:+d}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
