"""Evaluation package."""

from .benchmark import (
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
from .golden import GoldenSetExample, rebuild_golden_set
from .service import (
    ClaimSupportJudge,
    EmbeddingClaimSupportJudge,
    EmbeddingRelevanceJudge,
    LexicalClaimSupportJudge,
    OverlapJudge,
    RelevanceJudge,
    evaluate_answer,
    export_evaluation_report_csv,
    export_evaluation_report_markdown,
)

__all__ = [
    "RelevanceJudge",
    "OverlapJudge",
    "EmbeddingRelevanceJudge",
    "ClaimSupportJudge",
    "LexicalClaimSupportJudge",
    "EmbeddingClaimSupportJudge",
    "GoldenSetExample",
    "rebuild_golden_set",
    "evaluate_answer",
    "export_evaluation_report_csv",
    "export_evaluation_report_markdown",
    "EvaluationCase",
    "EvaluationSummary",
    "BenchmarkProvenance",
    "BenchmarkRun",
    "BenchmarkSummary",
    "build_benchmark_provenance",
    "load_golden_examples",
    "run_evaluation",
    "run_benchmark",
    "write_artifacts",
    "write_benchmark_artifacts",
]
