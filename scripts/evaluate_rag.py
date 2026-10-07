"""Run the deterministic RAG evaluation regression gate."""

from __future__ import annotations

import argparse
from pathlib import Path

from ragops_lab.evaluation.benchmark import (
    BenchmarkProvenance,
    BenchmarkRun,
    BenchmarkSummary,
    EvaluationCase,
    EvaluationSummary,
    build_benchmark_provenance,
    load_golden_examples,
    run_benchmark,
    run_evaluation,
    write_artifacts,
    write_benchmark_artifacts,
)

__all__ = [
    "BenchmarkProvenance",
    "BenchmarkRun",
    "BenchmarkSummary",
    "EvaluationCase",
    "EvaluationSummary",
    "build_benchmark_provenance",
    "load_golden_examples",
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
    parser.add_argument("--runs", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
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
    )
    write_benchmark_artifacts(summary, runs, args.output_dir)
    print(summary.model_dump_json(indent=2))
    return 0 if summary.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
