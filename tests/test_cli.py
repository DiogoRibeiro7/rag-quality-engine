from __future__ import annotations

import json
from pathlib import Path

from pytest import MonkeyPatch
from typer.testing import CliRunner

from ragops_lab import cli
from ragops_lab.cli import app
from ragops_lab.ingestion import ChunkingConfig, ingest_directory


def test_cli_ask_reports_missing_chunks_file(tmp_path: Path) -> None:
    runner = CliRunner()
    missing_path = tmp_path / "missing.jsonl"

    result = runner.invoke(app, ["ask", "What happened?", "--chunks", str(missing_path)])

    assert result.exit_code == 1
    assert "Chunks file not found" in result.output


def test_cli_ingest_reports_missing_input_directory(tmp_path: Path) -> None:
    runner = CliRunner()
    missing_dir = tmp_path / "missing"
    out_path = tmp_path / "chunks.jsonl"

    result = runner.invoke(app, ["ingest", str(missing_dir), "--out", str(out_path)])

    assert result.exit_code == 1
    assert "Input directory not found" in result.output


def test_cli_builds_index_and_asks_with_vector_mode(tmp_path: Path) -> None:
    runner = CliRunner()
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "metrics.txt").write_text(
        "Faithfulness and citation support are critical RAG metrics.",
        encoding="utf-8",
    )
    chunks_path = tmp_path / "chunks.jsonl"
    index_path = tmp_path / "vector_index.json"
    ingest_directory(raw_dir, chunks_path, ChunkingConfig(chunk_size=120, overlap=10))

    index_result = runner.invoke(
        app,
        ["index", "--chunks", str(chunks_path), "--out", str(index_path)],
    )
    ask_result = runner.invoke(
        app,
        [
            "ask",
            "What metrics matter?",
            "--chunks",
            str(chunks_path),
            "--index-path",
            str(index_path),
            "--profile",
            "vector",
            "--mode",
            "vector",
        ],
    )

    assert index_result.exit_code == 0
    assert index_path.exists()
    assert ask_result.exit_code == 0
    assert "Faithfulness" in ask_result.output


def test_cli_reports_unknown_retrieval_profile(tmp_path: Path) -> None:
    runner = CliRunner()
    chunks_path = tmp_path / "chunks.jsonl"
    chunks_path.write_text("", encoding="utf-8")

    result = runner.invoke(
        app,
        ["ask", "What happened?", "--chunks", str(chunks_path), "--profile", "missing"],
    )

    assert result.exit_code == 1
    assert "Unknown retrieval profile" in result.output


def test_cli_reports_missing_llm_provider_configuration(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runner = CliRunner()
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "apollo.txt").write_text("Apollo 11 landed on the Moon.", encoding="utf-8")
    chunks_path = tmp_path / "chunks.jsonl"
    ingest_directory(raw_dir, chunks_path, ChunkingConfig(chunk_size=120, overlap=10))
    monkeypatch.setenv("RAGOPS_LLM_PROVIDER", "openai-compatible")
    monkeypatch.delenv("RAGOPS_LLM_ENDPOINT", raising=False)

    result = runner.invoke(app, ["ask", "Which mission landed?", "--chunks", str(chunks_path)])

    assert result.exit_code == 1
    assert "RAGOPS_LLM_ENDPOINT" in result.output


def test_cli_index_reports_embedding_provider_errors(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    runner = CliRunner()
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "metrics.txt").write_text(
        "Faithfulness and citation support are critical RAG metrics.",
        encoding="utf-8",
    )
    chunks_path = tmp_path / "chunks.jsonl"
    index_path = tmp_path / "vector_index.json"
    ingest_directory(raw_dir, chunks_path, ChunkingConfig(chunk_size=120, overlap=10))

    def fail_embedding_provider(_: object) -> object:
        raise RuntimeError("embedding provider is unavailable")

    monkeypatch.setattr(cli, "build_embedding_client", fail_embedding_provider)

    result = runner.invoke(
        app,
        ["index", "--chunks", str(chunks_path), "--out", str(index_path)],
    )

    assert result.exit_code == 1
    assert "embedding provider is unavailable" in result.output


def test_cli_runs_dataset_benchmark(tmp_path: Path) -> None:
    runner = CliRunner()
    output_dir = tmp_path / "benchmark"

    result = runner.invoke(
        app,
        [
            "benchmark",
            "--source-dir",
            "data/sample_documents",
            "--golden-path",
            "data/golden/qa.json",
            "--out",
            str(output_dir),
            "--runs",
            "2",
        ],
    )

    summary = json.loads((output_dir / "benchmark-summary.json").read_text(encoding="utf-8"))

    assert result.exit_code == 0
    assert "Benchmark Summary" in result.output
    assert summary["run_count"] == 2
    assert summary["passed"] is True
    assert summary["unanswerable_case_count"] == 3
    assert summary["average_refusal_accuracy"] == 1.0
    assert (output_dir / "benchmark-runs.csv").exists()
    assert (output_dir / "run-001" / "cases.json").exists()
    assert (output_dir / "run-002" / "summary.md").exists()


def test_cli_benchmark_compare_exits_on_regression(tmp_path: Path) -> None:
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    payload = {
        "run_count": 1,
        "case_count": 1,
        "answerable_case_count": 1,
        "unanswerable_case_count": 0,
        "top_k": 2,
        "average_recall_at_k": 1.0,
        "mean_reciprocal_rank": 1.0,
        "average_faithfulness": 1.0,
        "lowest_run_faithfulness": 1.0,
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
            "fingerprint": "d" * 64,
        },
        "passed": True,
    }
    baseline_path.write_text(json.dumps(payload), encoding="utf-8")
    candidate_payload = payload | {
        "average_faithfulness": 0.9,
        "lowest_run_faithfulness": 0.9,
    }
    candidate_path.write_text(json.dumps(candidate_payload), encoding="utf-8")

    result = CliRunner().invoke(
        app,
        ["benchmark-compare", str(baseline_path), str(candidate_path)],
    )

    assert result.exit_code == 1
    assert "faithfulness" in result.output


def test_cli_benchmark_compare_shows_per_query_regressions(tmp_path: Path) -> None:
    output_dir = tmp_path / "benchmark"
    summary, runs = cli.run_benchmark(
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

    baseline_summary = tmp_path / "baseline-summary.json"
    candidate_summary = tmp_path / "candidate-summary.json"
    baseline_cases = tmp_path / "baseline-cases.json"
    candidate_cases = tmp_path / "candidate-cases.json"

    baseline_summary.write_text(summary.model_dump_json(indent=2), encoding="utf-8")
    candidate_summary.write_text(summary.model_dump_json(indent=2), encoding="utf-8")

    case = runs[0].cases[0]
    regressed = case.model_copy(
        update={"recall_at_k": max(0.0, case.recall_at_k - 0.5)}
    )
    baseline_cases.write_text(
        json.dumps([case.model_dump(mode="json")]),
        encoding="utf-8",
    )
    candidate_cases.write_text(
        json.dumps([regressed.model_dump(mode="json")]),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "benchmark-compare",
            str(baseline_summary),
            str(candidate_summary),
            "--baseline-cases",
            str(baseline_cases),
            "--candidate-cases",
            str(candidate_cases),
        ],
    )

    assert result.exit_code == 1
    assert "Per-query Comparison" in result.output
    assert case.query in result.output


def test_cli_benchmark_promote_writes_baseline(tmp_path: Path) -> None:
    output_dir = tmp_path / "evaluation"
    summary, runs = cli.run_benchmark(
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
    cli.write_benchmark_artifacts(summary, runs, output_dir)
    baseline_dir = tmp_path / "baseline"

    result = CliRunner().invoke(
        app,
        [
            "benchmark-promote",
            "--summary",
            str(output_dir / "benchmark-summary.json"),
            "--cases",
            str(output_dir / "cases.json"),
            "--out",
            str(baseline_dir),
        ],
    )

    assert result.exit_code == 0
    assert (baseline_dir / "benchmark-summary.json").exists()
    assert (baseline_dir / "cases.json").exists()
    assert (baseline_dir / "manifest.json").exists()


def test_cli_benchmark_baseline_validate(tmp_path: Path) -> None:
    output_dir = tmp_path / "evaluation"
    summary, runs = cli.run_benchmark(
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
    cli.write_benchmark_artifacts(summary, runs, output_dir)
    baseline_dir = tmp_path / "baseline"
    cli.promote_benchmark_baseline(
        summary_path=output_dir / "benchmark-summary.json",
        cases_path=output_dir / "cases.json",
        output_dir=baseline_dir,
        git_commit="abc123",
    )

    result = CliRunner().invoke(
        app,
        [
            "benchmark-baseline-validate",
            "--summary",
            str(baseline_dir / "benchmark-summary.json"),
            "--cases",
            str(baseline_dir / "cases.json"),
            "--manifest",
            str(baseline_dir / "manifest.json"),
        ],
    )

    assert result.exit_code == 0
    assert "benchmark_fingerprint" in result.output
