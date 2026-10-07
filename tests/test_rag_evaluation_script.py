from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = Path("scripts/evaluate_rag.py")

spec = importlib.util.spec_from_file_location("evaluate_rag", SCRIPT_PATH)
assert spec is not None
assert spec.loader is not None
evaluate_rag = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = evaluate_rag
spec.loader.exec_module(evaluate_rag)


def test_rag_evaluation_regression_gate_writes_artifacts(tmp_path: Path) -> None:
    summary, cases = evaluate_rag.run_evaluation(
        source_dir=Path("data/sample_documents"),
        golden_path=Path("data/golden/qa.json"),
        refusal_path=Path("data/golden/refusal.json"),
        chunks_path=tmp_path / "chunks.jsonl",
        top_k=2,
        chunk_size=400,
        overlap=60,
        min_faithfulness=0.80,
        min_citation_support=1.00,
    )
    output_dir = tmp_path / "evaluation"
    evaluate_rag.write_artifacts(summary, cases, output_dir)

    persisted_summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))

    assert summary.passed
    assert persisted_summary["passed"] is True
    assert persisted_summary["case_count"] == len(cases)
    assert persisted_summary["unanswerable_case_count"] == 3
    assert persisted_summary["refusal_accuracy"] == 1.0
    assert (output_dir / "summary.md").exists()
    assert (output_dir / "cases.csv").exists()
    assert (output_dir / "cases.json").exists()


def test_rag_benchmark_writes_repeated_run_artifacts(tmp_path: Path) -> None:
    output_dir = tmp_path / "benchmark"
    summary, runs = evaluate_rag.run_benchmark(
        source_dir=Path("data/sample_documents"),
        golden_path=Path("data/golden/qa.json"),
        refusal_path=Path("data/golden/refusal.json"),
        output_dir=output_dir,
        runs=2,
        top_k=2,
        chunk_size=400,
        overlap=60,
        min_faithfulness=0.80,
        min_citation_support=1.00,
    )
    evaluate_rag.write_benchmark_artifacts(summary, runs, output_dir)

    persisted_summary = json.loads(
        (output_dir / "benchmark-summary.json").read_text(encoding="utf-8")
    )

    assert summary.passed
    assert persisted_summary["run_count"] == 2
    assert persisted_summary["passed"] is True
    assert persisted_summary["unanswerable_case_count"] == 3
    assert persisted_summary["average_refusal_accuracy"] == 1.0
    assert (output_dir / "benchmark-runs.csv").exists()
    assert (output_dir / "run-001" / "cases.csv").exists()
    assert (output_dir / "run-002" / "summary.json").exists()


def test_rag_benchmark_rejects_missing_source_dir(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Source directory not found"):
        evaluate_rag.run_benchmark(
            source_dir=tmp_path / "missing",
            golden_path=Path("data/golden/qa.json"),
            output_dir=tmp_path / "benchmark",
            runs=1,
            top_k=2,
            chunk_size=400,
            overlap=60,
            min_faithfulness=0.80,
            min_citation_support=1.00,
            min_refusal_accuracy=1.00,
        )


def test_benchmark_provenance_is_deterministic() -> None:
    first = evaluate_rag.build_benchmark_provenance(
        source_dir=Path("data/sample_documents"),
        golden_path=Path("data/golden/qa.json"),
        refusal_path=Path("data/golden/refusal.json"),
        chunk_size=400,
        overlap=60,
    )
    second = evaluate_rag.build_benchmark_provenance(
        source_dir=Path("data/sample_documents"),
        golden_path=Path("data/golden/qa.json"),
        refusal_path=Path("data/golden/refusal.json"),
        chunk_size=400,
        overlap=60,
    )

    assert first == second
    assert len(first.fingerprint) == 64


def test_benchmark_provenance_changes_with_chunking() -> None:
    baseline = evaluate_rag.build_benchmark_provenance(
        source_dir=Path("data/sample_documents"),
        golden_path=Path("data/golden/qa.json"),
        refusal_path=Path("data/golden/refusal.json"),
        chunk_size=400,
        overlap=60,
    )
    changed = evaluate_rag.build_benchmark_provenance(
        source_dir=Path("data/sample_documents"),
        golden_path=Path("data/golden/qa.json"),
        refusal_path=Path("data/golden/refusal.json"),
        chunk_size=300,
        overlap=60,
    )

    assert baseline.fingerprint != changed.fingerprint


def test_benchmark_artifacts_persist_provenance(tmp_path: Path) -> None:
    output_dir = tmp_path / "benchmark"
    summary, runs = evaluate_rag.run_benchmark(
        source_dir=Path("data/sample_documents"),
        golden_path=Path("data/golden/qa.json"),
        refusal_path=Path("data/golden/refusal.json"),
        output_dir=output_dir,
        runs=1,
        top_k=2,
        chunk_size=400,
        overlap=60,
        min_faithfulness=0.80,
        min_citation_support=1.00,
    )
    evaluate_rag.write_benchmark_artifacts(summary, runs, output_dir)

    payload = json.loads(
        (output_dir / "benchmark-summary.json").read_text(encoding="utf-8")
    )
    report = (output_dir / "benchmark-summary.md").read_text(encoding="utf-8")

    assert payload["provenance"]["fingerprint"] == summary.provenance.fingerprint
    assert payload["provenance"]["chunk_size"] == 400
    assert payload["provenance"]["overlap"] == 60
    assert summary.provenance.fingerprint in report


def test_benchmark_records_p95_performance_metrics(tmp_path: Path) -> None:
    summary, runs = evaluate_rag.run_benchmark(
        source_dir=Path("data/sample_documents"),
        golden_path=Path("data/golden/qa.json"),
        refusal_path=Path("data/golden/refusal.json"),
        output_dir=tmp_path / "benchmark",
        runs=1,
        top_k=2,
        chunk_size=400,
        overlap=60,
        min_faithfulness=0.80,
        min_citation_support=1.00,
    )

    assert summary.worst_run_p95_latency_ms >= 0.0
    assert summary.worst_run_p95_token_estimate > 0
    assert runs[0].summary.p95_latency_ms >= 0.0
    assert runs[0].summary.p95_token_estimate > 0
    assert all(case.latency_ms >= 0.0 for case in runs[0].cases)
    assert all(case.token_estimate >= 0 for case in runs[0].cases)


def test_token_budget_can_fail_benchmark(tmp_path: Path) -> None:
    summary, _ = evaluate_rag.run_benchmark(
        source_dir=Path("data/sample_documents"),
        golden_path=Path("data/golden/qa.json"),
        refusal_path=Path("data/golden/refusal.json"),
        output_dir=tmp_path / "benchmark",
        runs=1,
        top_k=2,
        chunk_size=400,
        overlap=60,
        min_faithfulness=0.80,
        min_citation_support=1.00,
        max_p95_token_estimate=1,
    )

    assert summary.worst_run_p95_token_estimate > 1
    assert summary.max_p95_token_estimate == 1
    assert summary.passed is False


def test_benchmark_artifacts_persist_performance_metrics(tmp_path: Path) -> None:
    output_dir = tmp_path / "benchmark"
    summary, runs = evaluate_rag.run_benchmark(
        source_dir=Path("data/sample_documents"),
        golden_path=Path("data/golden/qa.json"),
        refusal_path=Path("data/golden/refusal.json"),
        output_dir=output_dir,
        runs=1,
        top_k=2,
        chunk_size=400,
        overlap=60,
        min_faithfulness=0.80,
        min_citation_support=1.00,
        max_p95_token_estimate=10000,
    )
    evaluate_rag.write_benchmark_artifacts(summary, runs, output_dir)

    payload = json.loads(
        (output_dir / "benchmark-summary.json").read_text(encoding="utf-8")
    )
    report = (output_dir / "benchmark-summary.md").read_text(encoding="utf-8")

    assert payload["worst_run_p95_latency_ms"] >= 0.0
    assert payload["worst_run_p95_token_estimate"] > 0
    assert payload["max_p95_token_estimate"] == 10000
    assert "Worst-run p95 latency" in report
    assert "Worst-run p95 token estimate" in report


def test_benchmark_rejects_invalid_performance_budget(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="max_p95_token_estimate"):
        evaluate_rag.run_benchmark(
            source_dir=Path("data/sample_documents"),
            golden_path=Path("data/golden/qa.json"),
            refusal_path=Path("data/golden/refusal.json"),
            output_dir=tmp_path / "benchmark",
            runs=1,
            top_k=2,
            chunk_size=400,
            overlap=60,
            min_faithfulness=0.80,
            min_citation_support=1.00,
            max_p95_token_estimate=0,
        )
