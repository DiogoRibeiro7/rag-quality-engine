"""Run the deterministic RAG evaluation regression gate."""

from __future__ import annotations

import argparse
from pathlib import Path

from ragops_lab.config import RuntimeSettings
from ragops_lab.evaluation.benchmark import (
    BenchmarkBaselineManifest,
    BenchmarkBaselineValidation,
    BenchmarkCaseComparison,
    BenchmarkComparison,
    BenchmarkProvenance,
    BenchmarkRegressionTolerances,
    BenchmarkRun,
    BenchmarkSummary,
    EvaluationCase,
    EvaluationSummary,
    build_benchmark_provenance,
    compare_benchmark_summaries,
    compare_evaluation_cases,
    load_benchmark_baseline_manifest,
    load_benchmark_summary,
    load_evaluation_cases,
    load_golden_examples,
    promote_benchmark_baseline,
    run_benchmark,
    run_evaluation,
    validate_benchmark_baseline,
    write_artifacts,
    write_benchmark_artifacts,
)
from ragops_lab.generation import LLMClient, build_llm_client
from ragops_lab.retrieval import EmbeddingClient, build_embedding_client

__all__ = [
    "BenchmarkBaselineManifest",
    "BenchmarkBaselineValidation",
    "BenchmarkCaseComparison",
    "BenchmarkComparison",
    "BenchmarkProvenance",
    "BenchmarkRegressionTolerances",
    "BenchmarkRun",
    "BenchmarkSummary",
    "EvaluationCase",
    "EvaluationSummary",
    "build_benchmark_provenance",
    "compare_benchmark_summaries",
    "compare_evaluation_cases",
    "load_benchmark_baseline_manifest",
    "load_benchmark_summary",
    "load_evaluation_cases",
    "load_golden_examples",
    "promote_benchmark_baseline",
    "validate_benchmark_baseline",
    "parse_args",
    "run_benchmark",
    "run_evaluation",
    "write_artifacts",
    "write_benchmark_artifacts",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path("data/sample_documents"))
    parser.add_argument("--golden-path", type=Path, default=Path("data/golden/qa.json"))
    parser.add_argument("--refusal-path", type=Path, default=Path("data/golden/refusal.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/evaluation"))
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=400)
    parser.add_argument("--overlap", type=int, default=60)
    parser.add_argument("--min-faithfulness", type=float, default=0.80)
    parser.add_argument("--min-citation-support", type=float, default=1.00)
    parser.add_argument("--min-refusal-accuracy", type=float, default=1.00)
    parser.add_argument("--max-p95-latency-ms", type=float, default=None)
    parser.add_argument("--max-p95-token-estimate", type=int, default=None)
    parser.add_argument("--evaluation-strategy", type=str, default=None)
    parser.add_argument("--runs", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = RuntimeSettings.from_env()
    strategy = args.evaluation_strategy or settings.evaluation.strategy
    embedding_client: EmbeddingClient | None = None
    llm_client: LLMClient | None = None
    evaluation_provider = "builtin"
    evaluation_model = "lexical-default"
    if strategy == "embedding":
        embedding_client = build_embedding_client(settings.embeddings)
        evaluation_provider = settings.embeddings.provider
        evaluation_model = settings.embeddings.model
    elif strategy == "llm":
        llm_client, _ = build_llm_client(settings.llm)
        evaluation_provider = settings.llm.provider
        evaluation_model = settings.llm.model
    summary, runs = run_benchmark(
        source_dir=args.source_dir,
        golden_path=args.golden_path,
        refusal_path=args.refusal_path,
        output_dir=args.output_dir,
        runs=args.runs,
        top_k=args.top_k,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        min_faithfulness=args.min_faithfulness,
        min_citation_support=args.min_citation_support,
        min_refusal_accuracy=args.min_refusal_accuracy,
        max_p95_latency_ms=args.max_p95_latency_ms,
        max_p95_token_estimate=args.max_p95_token_estimate,
        evaluation_strategy=strategy,
        evaluation_provider=evaluation_provider,
        evaluation_model=evaluation_model,
        embedding_client=embedding_client,
        llm_client=llm_client,
    )
    write_benchmark_artifacts(summary, runs, args.output_dir)
    print(summary.model_dump_json(indent=2))
    return 0 if summary.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
