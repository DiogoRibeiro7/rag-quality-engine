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


def _benchmark_summary_payload(*, fingerprint: str, faithfulness: float = 1.0) -> dict[str, object]:
    return {
        "run_count": 1,
        "case_count": 1,
        "answerable_case_count": 1,
        "unanswerable_case_count": 0,
        "top_k": 2,
        "average_recall_at_k": 1.0,
        "mean_reciprocal_rank": 1.0,
        "average_faithfulness": faithfulness,
        "lowest_run_faithfulness": faithfulness,
        "average_citation_support": 1.0,
        "lowest_run_citation_support": 1.0,
        "average_refusal_accuracy": 1.0,
        "lowest_run_refusal_accuracy": 1.0,
        "min_faithfulness": 0.8,
        "min_citation_support": 1.0,
        "min_refusal_accuracy": 1.0,
        "worst_run_p95_latency_ms": 10.0,
        "worst_run_p95_token_estimate": 100,
        "max_p95_latency_ms": None,
        "max_p95_token_estimate": None,
        "provenance": {
            "corpus_sha256": "a" * 64,
            "golden_sha256": "b" * 64,
            "refusal_sha256": "c" * 64,
            "chunking_strategy": "chars",
            "chunk_size": 400,
            "overlap": 60,
            "fingerprint": fingerprint,
        },
        "passed": faithfulness >= 0.8,
    }


def test_compare_benchmark_summaries_reports_regressions(tmp_path: Path) -> None:
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    baseline_path.write_text(
        json.dumps(_benchmark_summary_payload(fingerprint="d" * 64)),
        encoding="utf-8",
    )
    candidate_payload = _benchmark_summary_payload(
        fingerprint="d" * 64,
        faithfulness=0.7,
    )
    candidate_payload["worst_run_p95_latency_ms"] = 12.0
    candidate_payload["worst_run_p95_token_estimate"] = 120
    candidate_path.write_text(json.dumps(candidate_payload), encoding="utf-8")

    baseline = evaluate_rag.load_benchmark_summary(baseline_path)
    candidate = evaluate_rag.load_benchmark_summary(candidate_path)
    comparison = evaluate_rag.compare_benchmark_summaries(
        baseline=baseline,
        candidate=candidate,
        baseline_path=baseline_path,
        candidate_path=candidate_path,
    )

    assert comparison.fingerprint_match is True
    assert comparison.faithfulness_delta == pytest.approx(-0.3)
    assert comparison.p95_latency_ms_delta == 2.0
    assert comparison.p95_token_estimate_delta == 20
    assert "faithfulness" in comparison.regressions
    assert "p95_latency_ms" in comparison.regressions
    assert "p95_token_estimate" in comparison.regressions
    assert comparison.has_regressions is True


def test_compare_benchmark_summaries_rejects_fingerprint_mismatch(tmp_path: Path) -> None:
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    baseline_path.write_text(
        json.dumps(_benchmark_summary_payload(fingerprint="d" * 64)),
        encoding="utf-8",
    )
    candidate_path.write_text(
        json.dumps(_benchmark_summary_payload(fingerprint="e" * 64)),
        encoding="utf-8",
    )

    baseline = evaluate_rag.load_benchmark_summary(baseline_path)
    candidate = evaluate_rag.load_benchmark_summary(candidate_path)

    with pytest.raises(ValueError, match="fingerprints differ"):
        evaluate_rag.compare_benchmark_summaries(
            baseline=baseline,
            candidate=candidate,
            baseline_path=baseline_path,
            candidate_path=candidate_path,
        )


def test_compare_benchmark_summaries_allows_explicit_mismatch(tmp_path: Path) -> None:
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    baseline_path.write_text(
        json.dumps(_benchmark_summary_payload(fingerprint="d" * 64)),
        encoding="utf-8",
    )
    candidate_path.write_text(
        json.dumps(_benchmark_summary_payload(fingerprint="e" * 64)),
        encoding="utf-8",
    )

    comparison = evaluate_rag.compare_benchmark_summaries(
        baseline=evaluate_rag.load_benchmark_summary(baseline_path),
        candidate=evaluate_rag.load_benchmark_summary(candidate_path),
        baseline_path=baseline_path,
        candidate_path=candidate_path,
        allow_mismatched_fingerprints=True,
    )

    assert comparison.fingerprint_match is False


def test_compare_evaluation_cases_identifies_regressed_query(tmp_path: Path) -> None:
    _, cases = evaluate_rag.run_evaluation(
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
    baseline_case = cases[0]
    candidate_case = baseline_case.model_copy(
        update={
            "recall_at_k": max(0.0, baseline_case.recall_at_k - 0.5),
            "latency_ms": baseline_case.latency_ms + 10.0,
            "token_estimate": baseline_case.token_estimate + 5,
            "evaluation": baseline_case.evaluation.model_copy(
                update={
                    "faithfulness": max(
                        0.0,
                        baseline_case.evaluation.faithfulness - 0.5,
                    )
                }
            ),
        }
    )

    comparisons = evaluate_rag.compare_evaluation_cases(
        [baseline_case],
        [candidate_case],
    )

    assert len(comparisons) == 1
    comparison = comparisons[0]
    assert comparison.query == baseline_case.query
    assert comparison.status == "matched"
    assert "recall_at_k" in comparison.regressions
    assert "faithfulness" in comparison.regressions
    assert "latency_ms" in comparison.regressions
    assert "token_estimate" in comparison.regressions


def test_compare_evaluation_cases_reports_added_and_missing_queries(
    tmp_path: Path,
) -> None:
    _, cases = evaluate_rag.run_evaluation(
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
    baseline_case = cases[0]
    added_case = cases[1].model_copy(update={"query": "A newly added query?"})

    comparisons = evaluate_rag.compare_evaluation_cases(
        [baseline_case],
        [added_case],
    )
    by_status = {comparison.status: comparison for comparison in comparisons}

    assert by_status["missing"].query == baseline_case.query
    assert by_status["missing"].regressions == ["missing_query"]
    assert by_status["added"].query == "A newly added query?"


def test_summary_comparison_can_include_case_regressions(tmp_path: Path) -> None:
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
    baseline_case = runs[0].cases[0]
    candidate_case = baseline_case.model_copy(
        update={"recall_at_k": max(0.0, baseline_case.recall_at_k - 0.5)}
    )

    comparison = evaluate_rag.compare_benchmark_summaries(
        baseline=summary,
        candidate=summary,
        baseline_path=tmp_path / "baseline.json",
        candidate_path=tmp_path / "candidate.json",
        baseline_cases=[baseline_case],
        candidate_cases=[candidate_case],
    )

    assert comparison.case_comparisons
    assert any(item.startswith("query:") for item in comparison.regressions)


def test_promote_benchmark_baseline_writes_validated_artifacts(tmp_path: Path) -> None:
    output_dir = tmp_path / "evaluation"
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

    baseline_dir = tmp_path / "baseline"
    summary_out, cases_out, manifest_out = evaluate_rag.promote_benchmark_baseline(
        summary_path=output_dir / "benchmark-summary.json",
        cases_path=output_dir / "cases.json",
        output_dir=baseline_dir,
        git_commit="abc123",
    )

    assert summary_out == baseline_dir / "benchmark-summary.json"
    assert cases_out == baseline_dir / "cases.json"
    assert manifest_out == baseline_dir / "manifest.json"
    assert evaluate_rag.load_benchmark_summary(summary_out).passed is True
    assert len(evaluate_rag.load_evaluation_cases(cases_out)) == summary.case_count
    manifest = evaluate_rag.load_benchmark_baseline_manifest(manifest_out)
    assert manifest.project_version == "0.2.0"
    assert manifest.git_commit == "abc123"
    assert manifest.benchmark_fingerprint == summary.provenance.fingerprint


def test_promote_benchmark_baseline_rejects_failed_summary(tmp_path: Path) -> None:
    output_dir = tmp_path / "evaluation"
    summary, runs = evaluate_rag.run_benchmark(
        source_dir=Path("data/sample_documents"),
        golden_path=Path("data/golden/qa.json"),
        refusal_path=Path("data/golden/refusal.json"),
        output_dir=output_dir,
        runs=1,
        top_k=2,
        chunk_size=400,
        overlap=60,
        min_faithfulness=1.0,
        min_citation_support=1.0,
        min_refusal_accuracy=1.0,
        max_p95_token_estimate=1,
    )
    evaluate_rag.write_benchmark_artifacts(summary, runs, output_dir)

    with pytest.raises(ValueError, match="did not pass"):
        evaluate_rag.promote_benchmark_baseline(
            summary_path=output_dir / "benchmark-summary.json",
            cases_path=output_dir / "cases.json",
            output_dir=tmp_path / "baseline",
        )


def test_promote_benchmark_baseline_rejects_case_count_mismatch(tmp_path: Path) -> None:
    output_dir = tmp_path / "evaluation"
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

    cases = evaluate_rag.load_evaluation_cases(output_dir / "cases.json")
    (output_dir / "cases.json").write_text(
        json.dumps([case.model_dump(mode="json") for case in cases[:-1]]),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Case count does not match"):
        evaluate_rag.promote_benchmark_baseline(
            summary_path=output_dir / "benchmark-summary.json",
            cases_path=output_dir / "cases.json",
            output_dir=tmp_path / "baseline",
        )


def test_validate_benchmark_baseline_accepts_consistent_artifacts(tmp_path: Path) -> None:
    output_dir = tmp_path / "evaluation"
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
    baseline_dir = tmp_path / "baseline"
    evaluate_rag.promote_benchmark_baseline(
        summary_path=output_dir / "benchmark-summary.json",
        cases_path=output_dir / "cases.json",
        output_dir=baseline_dir,
        git_commit="abc123",
    )

    validation = evaluate_rag.validate_benchmark_baseline(
        summary_path=baseline_dir / "benchmark-summary.json",
        cases_path=baseline_dir / "cases.json",
        manifest_path=baseline_dir / "manifest.json",
    )

    assert validation.valid is True
    assert validation.benchmark_fingerprint == summary.provenance.fingerprint
    assert validation.case_count == summary.case_count


def test_validate_benchmark_baseline_rejects_manifest_fingerprint_mismatch(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "evaluation"
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
    baseline_dir = tmp_path / "baseline"
    _, _, manifest_path = evaluate_rag.promote_benchmark_baseline(
        summary_path=output_dir / "benchmark-summary.json",
        cases_path=output_dir / "cases.json",
        output_dir=baseline_dir,
        git_commit="abc123",
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["benchmark_fingerprint"] = "f" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="fingerprint does not match"):
        evaluate_rag.validate_benchmark_baseline(
            summary_path=baseline_dir / "benchmark-summary.json",
            cases_path=baseline_dir / "cases.json",
            manifest_path=manifest_path,
        )


def test_validate_benchmark_baseline_rejects_duplicate_queries(tmp_path: Path) -> None:
    output_dir = tmp_path / "evaluation"
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
    baseline_dir = tmp_path / "baseline"
    evaluate_rag.promote_benchmark_baseline(
        summary_path=output_dir / "benchmark-summary.json",
        cases_path=output_dir / "cases.json",
        output_dir=baseline_dir,
        git_commit="abc123",
    )

    cases = json.loads((baseline_dir / "cases.json").read_text(encoding="utf-8"))
    cases[1]["query"] = cases[0]["query"]
    (baseline_dir / "cases.json").write_text(json.dumps(cases), encoding="utf-8")

    with pytest.raises(ValueError, match="unique queries"):
        evaluate_rag.validate_benchmark_baseline(
            summary_path=baseline_dir / "benchmark-summary.json",
            cases_path=baseline_dir / "cases.json",
            manifest_path=baseline_dir / "manifest.json",
        )
