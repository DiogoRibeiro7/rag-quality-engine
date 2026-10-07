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
from .service import (
    ClaimSupportJudge,
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
    "ClaimSupportJudge",
    "LexicalClaimSupportJudge",
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
