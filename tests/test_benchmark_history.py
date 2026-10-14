from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from ragops_lab.evaluation import (
    append_benchmark_history,
    build_benchmark_trend,
    load_benchmark_history,
    write_benchmark_trend_markdown,
)
from ragops_lab.evaluation.benchmark import BenchmarkSummary


def _write_summary(
    path: Path,
    *,
    fingerprint: str,
    recall: float = 0.8,
    mrr: float = 0.7,
    faithfulness: float = 1.0,
    citation_support: float = 1.0,
    refusal_accuracy: float = 1.0,
    latency_ms: float = 10.0,
    tokens: int = 100,
) -> None:
    payload = {
        "run_count": 1,
        "case_count": 1,
        "answerable_case_count": 1,
        "unanswerable_case_count": 0,
        "top_k": 2,
        "average_recall_at_k": recall,
        "mean_reciprocal_rank": mrr,
        "average_faithfulness": faithfulness,
        "lowest_run_faithfulness": faithfulness,
        "average_citation_support": citation_support,
        "lowest_run_citation_support": citation_support,
        "average_refusal_accuracy": refusal_accuracy,
        "lowest_run_refusal_accuracy": refusal_accuracy,
        "min_faithfulness": 0.8,
        "min_citation_support": 1.0,
        "min_refusal_accuracy": 1.0,
        "worst_run_p95_latency_ms": latency_ms,
        "worst_run_p95_token_estimate": tokens,
        "max_p95_latency_ms": None,
        "max_p95_token_estimate": None,
        "provenance": {
            "corpus_sha256": "a" * 64,
            "golden_sha256": "b" * 64,
            "refusal_sha256": "c" * 64,
            "chunking_strategy": "chars",
            "chunk_size": 400,
            "overlap": 60,
            "evaluation_strategy": "lexical",
            "evaluation_provider": "builtin",
            "evaluation_model": "lexical-default",
            "fingerprint": fingerprint,
        },
        "passed": True,
    }
    summary = BenchmarkSummary.model_validate(payload)
    path.write_text(summary.model_dump_json(indent=2), encoding="utf-8")


def test_benchmark_history_appends_and_loads_chronologically(tmp_path: Path) -> None:
    history_path = tmp_path / "history.jsonl"
    summary_path = tmp_path / "summary.json"
    _write_summary(summary_path, fingerprint="d" * 64)
    later = datetime(2026, 1, 2, tzinfo=UTC)
    earlier = later - timedelta(days=1)

    append_benchmark_history(
        history_path=history_path,
        summary_path=summary_path,
        label="later",
        git_commit="bbb",
        recorded_at=later,
    )
    append_benchmark_history(
        history_path=history_path,
        summary_path=summary_path,
        label="earlier",
        git_commit="aaa",
        recorded_at=earlier,
    )

    entries = load_benchmark_history(history_path)

    assert [entry.label for entry in entries] == ["earlier", "later"]


def test_benchmark_history_rejects_duplicate_identity(tmp_path: Path) -> None:
    history_path = tmp_path / "history.jsonl"
    summary_path = tmp_path / "summary.json"
    _write_summary(summary_path, fingerprint="d" * 64)

    append_benchmark_history(
        history_path=history_path,
        summary_path=summary_path,
        label="v0.2.0",
        git_commit="abc123",
    )

    with pytest.raises(ValueError, match="already contains"):
        append_benchmark_history(
            history_path=history_path,
            summary_path=summary_path,
            label="v0.2.0",
            git_commit="abc123",
        )


def test_benchmark_trend_uses_latest_fingerprint_by_default(tmp_path: Path) -> None:
    history_path = tmp_path / "history.jsonl"
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    incompatible_path = tmp_path / "incompatible.json"

    _write_summary(first_path, fingerprint="d" * 64, recall=0.7, latency_ms=10.0)
    _write_summary(second_path, fingerprint="d" * 64, recall=0.9, latency_ms=11.0)
    _write_summary(incompatible_path, fingerprint="e" * 64, recall=0.1, latency_ms=50.0)

    base_time = datetime(2026, 1, 1, tzinfo=UTC)
    append_benchmark_history(
        history_path=history_path,
        summary_path=first_path,
        label="first",
        git_commit="aaa",
        recorded_at=base_time,
    )
    append_benchmark_history(
        history_path=history_path,
        summary_path=second_path,
        label="second",
        git_commit="bbb",
        recorded_at=base_time + timedelta(days=1),
    )
    append_benchmark_history(
        history_path=history_path,
        summary_path=incompatible_path,
        label="other",
        git_commit="ccc",
        recorded_at=base_time - timedelta(days=1),
    )

    trend = build_benchmark_trend(load_benchmark_history(history_path))

    assert trend.entry_count == 2
    assert trend.fingerprint == "d" * 64
    assert trend.first_label == "first"
    assert trend.latest_label == "second"
    assert trend.recall_at_k_delta == pytest.approx(0.2)
    assert trend.p95_latency_ms_delta == pytest.approx(1.0)


def test_benchmark_trend_writes_markdown(tmp_path: Path) -> None:
    summary_path = tmp_path / "summary.json"
    history_path = tmp_path / "history.jsonl"
    report_path = tmp_path / "trend.md"
    _write_summary(summary_path, fingerprint="d" * 64)

    append_benchmark_history(
        history_path=history_path,
        summary_path=summary_path,
        label="baseline",
        git_commit="abc123",
        recorded_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    trend = build_benchmark_trend(load_benchmark_history(history_path))
    write_benchmark_trend_markdown(trend, report_path)

    text = report_path.read_text(encoding="utf-8")
    assert "# Benchmark Trend" in text
    assert "baseline" in text
    assert "Recall@k delta" in text
