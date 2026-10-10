"""Dataset-oriented RAG benchmark runner."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

from pydantic import BaseModel, Field

from ragops_lab import __version__
from ragops_lab.domain import EvaluationResult, GeneratedAnswer
from ragops_lab.evaluation.service import evaluate_answer
from ragops_lab.generation import GenerationService, HeuristicLLMClient
from ragops_lab.ingestion import ChunkingConfig, ingest_directory
from ragops_lab.retrieval import BM25Retriever
from ragops_lab.retrieval.evaluation import recall_at_k, reciprocal_rank


class BenchmarkGoldenExample(BaseModel):
    """Golden benchmark example for answerable or unanswerable questions."""

    query: str = Field(min_length=1)
    relevant_chunk_ids: list[str] = Field(default_factory=list)
    expected_unanswerable: bool = Field(default=False)


class EvaluationCase(BaseModel):
    """Serializable result for one golden-set example."""

    query: str
    retrieved_chunk_ids: list[str]
    relevant_chunk_ids: list[str]
    expected_unanswerable: bool = Field(default=False)
    recall_at_k: float = Field(ge=0.0, le=1.0)
    reciprocal_rank: float = Field(ge=0.0, le=1.0)
    latency_ms: float = Field(ge=0.0)
    token_estimate: int = Field(ge=0)
    answer: GeneratedAnswer
    evaluation: EvaluationResult


class BenchmarkProvenance(BaseModel):
    """Stable identity for benchmark data and chunking inputs."""

    corpus_sha256: str = Field(min_length=64, max_length=64)
    golden_sha256: str = Field(min_length=64, max_length=64)
    refusal_sha256: str | None = Field(default=None, min_length=64, max_length=64)
    chunking_strategy: str = Field(min_length=1)
    chunk_size: int = Field(ge=1)
    overlap: int = Field(ge=0)
    fingerprint: str = Field(min_length=64, max_length=64)


class EvaluationSummary(BaseModel):
    """Aggregate regression metrics and threshold status for one run."""

    case_count: int = Field(ge=0)
    answerable_case_count: int = Field(ge=0)
    unanswerable_case_count: int = Field(ge=0)
    top_k: int = Field(ge=1)
    average_recall_at_k: float = Field(ge=0.0, le=1.0)
    mean_reciprocal_rank: float = Field(ge=0.0, le=1.0)
    average_faithfulness: float = Field(ge=0.0, le=1.0)
    average_citation_support: float = Field(ge=0.0, le=1.0)
    refusal_accuracy: float = Field(ge=0.0, le=1.0)
    min_faithfulness: float = Field(ge=0.0, le=1.0)
    min_citation_support: float = Field(ge=0.0, le=1.0)
    min_refusal_accuracy: float = Field(ge=0.0, le=1.0)
    p95_latency_ms: float = Field(ge=0.0)
    p95_token_estimate: int = Field(ge=0)
    max_p95_latency_ms: float | None = Field(default=None, gt=0.0)
    max_p95_token_estimate: int | None = Field(default=None, ge=1)
    provenance: BenchmarkProvenance
    passed: bool


class BenchmarkRun(BaseModel):
    """One benchmark repeat with its case-level results."""

    run_id: int = Field(ge=1)
    summary: EvaluationSummary
    cases: list[EvaluationCase]


class BenchmarkSummary(BaseModel):
    """Aggregate metrics across repeated benchmark runs."""

    run_count: int = Field(ge=1)
    case_count: int = Field(ge=0)
    answerable_case_count: int = Field(ge=0)
    unanswerable_case_count: int = Field(ge=0)
    top_k: int = Field(ge=1)
    average_recall_at_k: float = Field(ge=0.0, le=1.0)
    mean_reciprocal_rank: float = Field(ge=0.0, le=1.0)
    average_faithfulness: float = Field(ge=0.0, le=1.0)
    lowest_run_faithfulness: float = Field(ge=0.0, le=1.0)
    average_citation_support: float = Field(ge=0.0, le=1.0)
    lowest_run_citation_support: float = Field(ge=0.0, le=1.0)
    average_refusal_accuracy: float = Field(ge=0.0, le=1.0)
    lowest_run_refusal_accuracy: float = Field(ge=0.0, le=1.0)
    min_faithfulness: float = Field(ge=0.0, le=1.0)
    min_citation_support: float = Field(ge=0.0, le=1.0)
    min_refusal_accuracy: float = Field(ge=0.0, le=1.0)
    worst_run_p95_latency_ms: float = Field(ge=0.0)
    worst_run_p95_token_estimate: int = Field(ge=0)
    max_p95_latency_ms: float | None = Field(default=None, gt=0.0)
    max_p95_token_estimate: int | None = Field(default=None, ge=1)
    provenance: BenchmarkProvenance
    passed: bool



class BenchmarkBaselineManifest(BaseModel):
    """Identity metadata for an approved benchmark baseline."""

    project_version: str = Field(min_length=1)
    git_commit: str = Field(min_length=1)
    promoted_at: datetime
    benchmark_fingerprint: str = Field(min_length=64, max_length=64)


class BenchmarkCaseComparison(BaseModel):
    """Per-query benchmark deltas between baseline and candidate cases."""

    query: str
    status: str
    recall_at_k_delta: float = 0.0
    reciprocal_rank_delta: float = 0.0
    faithfulness_delta: float = 0.0
    citation_support_delta: float = 0.0
    latency_ms_delta: float = 0.0
    token_estimate_delta: int = 0
    regressions: list[str] = Field(default_factory=list)

    @property
    def has_regressions(self) -> bool:
        """Return whether this query has any metric regressions."""
        return bool(self.regressions)


class BenchmarkComparison(BaseModel):
    """Metric deltas between two benchmark summaries."""

    baseline_path: str
    candidate_path: str
    fingerprint_match: bool
    baseline_passed: bool
    candidate_passed: bool
    recall_at_k_delta: float
    mean_reciprocal_rank_delta: float
    faithfulness_delta: float
    citation_support_delta: float
    refusal_accuracy_delta: float
    p95_latency_ms_delta: float
    p95_token_estimate_delta: int
    regressions: list[str] = Field(default_factory=list)
    case_comparisons: list[BenchmarkCaseComparison] = Field(default_factory=list)

    @property
    def has_regressions(self) -> bool:
        """Return whether any monitored metric regressed."""
        return bool(self.regressions)


def load_evaluation_cases(path: Path) -> list[EvaluationCase]:
    """Load persisted benchmark case results."""
    if not path.exists():
        raise ValueError(f"Benchmark cases not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [EvaluationCase.model_validate(item) for item in payload]


def compare_evaluation_cases(
    baseline_cases: list[EvaluationCase],
    candidate_cases: list[EvaluationCase],
) -> list[BenchmarkCaseComparison]:
    """Compare benchmark cases aligned by query."""
    baseline_by_query = {case.query: case for case in baseline_cases}
    candidate_by_query = {case.query: case for case in candidate_cases}
    queries = sorted(set(baseline_by_query) | set(candidate_by_query))
    comparisons: list[BenchmarkCaseComparison] = []

    for query in queries:
        baseline_case = baseline_by_query.get(query)
        candidate_case = candidate_by_query.get(query)
        if baseline_case is None:
            comparisons.append(BenchmarkCaseComparison(query=query, status="added"))
            continue
        if candidate_case is None:
            comparisons.append(
                BenchmarkCaseComparison(
                    query=query,
                    status="missing",
                    regressions=["missing_query"],
                )
            )
            continue

        recall_delta = candidate_case.recall_at_k - baseline_case.recall_at_k
        rank_delta = candidate_case.reciprocal_rank - baseline_case.reciprocal_rank
        faithfulness_delta = (
            candidate_case.evaluation.faithfulness
            - baseline_case.evaluation.faithfulness
        )
        citation_delta = (
            candidate_case.evaluation.citation_support
            - baseline_case.evaluation.citation_support
        )
        latency_delta = candidate_case.latency_ms - baseline_case.latency_ms
        token_delta = candidate_case.token_estimate - baseline_case.token_estimate

        regressions: list[str] = []
        monitored = {
            "recall_at_k": recall_delta,
            "reciprocal_rank": rank_delta,
            "faithfulness": faithfulness_delta,
            "citation_support": citation_delta,
        }
        regressions.extend(name for name, delta in monitored.items() if delta < 0.0)
        if latency_delta > 0.0:
            regressions.append("latency_ms")
        if token_delta > 0:
            regressions.append("token_estimate")

        comparisons.append(
            BenchmarkCaseComparison(
                query=query,
                status="matched",
                recall_at_k_delta=recall_delta,
                reciprocal_rank_delta=rank_delta,
                faithfulness_delta=faithfulness_delta,
                citation_support_delta=citation_delta,
                latency_ms_delta=latency_delta,
                token_estimate_delta=token_delta,
                regressions=regressions,
            )
        )
    return comparisons


def _resolve_git_commit(repository_root: Path = Path(".")) -> str:
    """Resolve the current commit without invoking a subprocess."""
    github_sha = os.getenv("GITHUB_SHA")
    if github_sha:
        return github_sha.strip()

    git_dir = repository_root / ".git"
    head_path = git_dir / "HEAD"
    if not head_path.exists():
        return "unknown"

    head = head_path.read_text(encoding="utf-8").strip()
    if not head.startswith("ref: "):
        return head or "unknown"

    ref_path = git_dir / head.removeprefix("ref: ")
    if ref_path.exists():
        return ref_path.read_text(encoding="utf-8").strip() or "unknown"
    return "unknown"


def load_benchmark_baseline_manifest(path: Path) -> BenchmarkBaselineManifest:
    """Load and validate a promoted baseline manifest."""
    if not path.exists():
        raise ValueError(f"Benchmark baseline manifest not found: {path}")
    return BenchmarkBaselineManifest.model_validate_json(
        path.read_text(encoding="utf-8")
    )


def promote_benchmark_baseline(
    *,
    summary_path: Path,
    cases_path: Path,
    output_dir: Path,
    git_commit: str | None = None,
    promoted_at: datetime | None = None,
) -> tuple[Path, Path, Path]:
    """Validate and promote benchmark artifacts into a CI baseline directory."""
    summary = load_benchmark_summary(summary_path)
    cases = load_evaluation_cases(cases_path)

    if not summary.passed:
        raise ValueError("Cannot promote a benchmark that did not pass.")
    if len(cases) != summary.case_count:
        raise ValueError(
            "Case count does not match benchmark summary: "
            f"{len(cases)} != {summary.case_count}."
        )

    queries = [case.query for case in cases]
    if len(set(queries)) != len(queries):
        raise ValueError("Benchmark cases must contain unique queries.")

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_out = output_dir / "benchmark-summary.json"
    cases_out = output_dir / "cases.json"
    manifest_out = output_dir / "manifest.json"
    summary_out.write_text(summary.model_dump_json(indent=2) + "\n", encoding="utf-8")
    cases_out.write_text(
        json.dumps([case.model_dump(mode="json") for case in cases], indent=2) + "\n",
        encoding="utf-8",
    )
    manifest = BenchmarkBaselineManifest(
        project_version=__version__,
        git_commit=(git_commit or _resolve_git_commit()).strip() or "unknown",
        promoted_at=promoted_at or datetime.now(UTC),
        benchmark_fingerprint=summary.provenance.fingerprint,
    )
    manifest_out.write_text(
        manifest.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    return summary_out, cases_out, manifest_out


def load_benchmark_summary(path: Path) -> BenchmarkSummary:
    """Load one persisted benchmark summary JSON file."""
    if not path.exists():
        raise ValueError(f"Benchmark summary not found: {path}")
    return BenchmarkSummary.model_validate_json(path.read_text(encoding="utf-8"))


def compare_benchmark_summaries(
    *,
    baseline: BenchmarkSummary,
    candidate: BenchmarkSummary,
    baseline_path: Path,
    candidate_path: Path,
    allow_mismatched_fingerprints: bool = False,
    baseline_cases: list[EvaluationCase] | None = None,
    candidate_cases: list[EvaluationCase] | None = None,
) -> BenchmarkComparison:
    """Compare persisted benchmark summaries and classify regressions."""
    fingerprint_match = baseline.provenance.fingerprint == candidate.provenance.fingerprint
    if not fingerprint_match and not allow_mismatched_fingerprints:
        raise ValueError(
            "Benchmark fingerprints differ. Compare only like-for-like runs or "
            "set allow_mismatched_fingerprints=True explicitly."
        )

    recall_delta = candidate.average_recall_at_k - baseline.average_recall_at_k
    mrr_delta = candidate.mean_reciprocal_rank - baseline.mean_reciprocal_rank
    faithfulness_delta = candidate.average_faithfulness - baseline.average_faithfulness
    citation_delta = (
        candidate.average_citation_support - baseline.average_citation_support
    )
    refusal_delta = candidate.average_refusal_accuracy - baseline.average_refusal_accuracy
    latency_delta = (
        candidate.worst_run_p95_latency_ms - baseline.worst_run_p95_latency_ms
    )
    token_delta = (
        candidate.worst_run_p95_token_estimate
        - baseline.worst_run_p95_token_estimate
    )

    regressions: list[str] = []
    monitored = {
        "recall_at_k": recall_delta,
        "mean_reciprocal_rank": mrr_delta,
        "faithfulness": faithfulness_delta,
        "citation_support": citation_delta,
        "refusal_accuracy": refusal_delta,
    }
    regressions.extend(name for name, delta in monitored.items() if delta < 0.0)
    if latency_delta > 0.0:
        regressions.append("p95_latency_ms")
    if token_delta > 0:
        regressions.append("p95_token_estimate")
    if baseline.passed and not candidate.passed:
        regressions.append("passed")

    case_comparisons: list[BenchmarkCaseComparison] = []
    if baseline_cases is not None or candidate_cases is not None:
        if baseline_cases is None or candidate_cases is None:
            raise ValueError(
                "Both baseline_cases and candidate_cases are required for case comparison."
            )
        case_comparisons = compare_evaluation_cases(baseline_cases, candidate_cases)
        regressions.extend(
            f"query:{comparison.query}"
            for comparison in case_comparisons
            if comparison.has_regressions
        )

    return BenchmarkComparison(
        baseline_path=str(baseline_path),
        candidate_path=str(candidate_path),
        fingerprint_match=fingerprint_match,
        baseline_passed=baseline.passed,
        candidate_passed=candidate.passed,
        recall_at_k_delta=recall_delta,
        mean_reciprocal_rank_delta=mrr_delta,
        faithfulness_delta=faithfulness_delta,
        citation_support_delta=citation_delta,
        refusal_accuracy_delta=refusal_delta,
        p95_latency_ms_delta=latency_delta,
        p95_token_estimate_delta=token_delta,
        regressions=regressions,
        case_comparisons=case_comparisons,
    )


def _sha256_file(path: Path) -> str:
    """Hash a file's bytes with SHA-256."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_directory(path: Path) -> str:
    """Hash relative paths and bytes for all files in a directory tree."""
    digest = hashlib.sha256()
    for file_path in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        relative_path = file_path.relative_to(path).as_posix()
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        with file_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def build_benchmark_provenance(
    *,
    source_dir: Path,
    golden_path: Path,
    refusal_path: Path | None,
    chunk_size: int,
    overlap: int,
    chunking_strategy: str = "chars",
) -> BenchmarkProvenance:
    """Build deterministic provenance for a benchmark configuration."""
    corpus_sha256 = _sha256_directory(source_dir)
    golden_sha256 = _sha256_file(golden_path)
    refusal_sha256 = _sha256_file(refusal_path) if refusal_path is not None else None
    payload = {
        "corpus_sha256": corpus_sha256,
        "golden_sha256": golden_sha256,
        "refusal_sha256": refusal_sha256,
        "chunking_strategy": chunking_strategy,
        "chunk_size": chunk_size,
        "overlap": overlap,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    fingerprint = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return BenchmarkProvenance(
        corpus_sha256=corpus_sha256,
        golden_sha256=golden_sha256,
        refusal_sha256=refusal_sha256,
        chunking_strategy=chunking_strategy,
        chunk_size=chunk_size,
        overlap=overlap,
        fingerprint=fingerprint,
    )


def load_golden_examples(path: Path) -> list[BenchmarkGoldenExample]:
    """Load golden benchmark examples from JSON."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    examples = [BenchmarkGoldenExample.model_validate(example) for example in payload]
    invalid_answerable = [
        example.query
        for example in examples
        if not example.expected_unanswerable and not example.relevant_chunk_ids
    ]
    if invalid_answerable:
        raise ValueError(
            f"Answerable golden examples must include relevant chunk ids: {invalid_answerable}"
        )
    return examples


def _percentile_nearest_rank(values: list[float], percentile: float) -> float:
    """Return a deterministic nearest-rank percentile."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, math.ceil(percentile * len(ordered)))
    return ordered[rank - 1]


def run_evaluation(
    *,
    source_dir: Path,
    golden_path: Path,
    chunks_path: Path,
    top_k: int,
    chunk_size: int,
    overlap: int,
    min_faithfulness: float,
    min_citation_support: float,
    refusal_path: Path | None = Path("data/golden/refusal.json"),
    min_refusal_accuracy: float = 1.0,
    max_p95_latency_ms: float | None = None,
    max_p95_token_estimate: int | None = None,
) -> tuple[EvaluationSummary, list[EvaluationCase]]:
    """Run one deterministic RAG evaluation pass over a golden dataset."""
    _validate_benchmark_inputs(
        source_dir=source_dir,
        golden_path=golden_path,
        refusal_path=refusal_path,
        top_k=top_k,
        chunk_size=chunk_size,
        overlap=overlap,
        min_faithfulness=min_faithfulness,
        min_citation_support=min_citation_support,
        min_refusal_accuracy=min_refusal_accuracy,
        max_p95_latency_ms=max_p95_latency_ms,
        max_p95_token_estimate=max_p95_token_estimate,
    )
    chunking = ChunkingConfig(chunk_size=chunk_size, overlap=overlap)
    provenance = build_benchmark_provenance(
        source_dir=source_dir,
        golden_path=golden_path,
        refusal_path=refusal_path,
        chunk_size=chunking.chunk_size,
        overlap=chunking.overlap,
        chunking_strategy=chunking.strategy,
    )
    chunks = ingest_directory(
        source_dir,
        chunks_path,
        chunking,
    )
    retriever = BM25Retriever(chunks)
    generation = GenerationService(HeuristicLLMClient())
    golden_examples = load_golden_examples(golden_path)
    if refusal_path is not None:
        golden_examples.extend(load_golden_examples(refusal_path))
    available_chunk_ids = {chunk.chunk_id for chunk in chunks}

    cases: list[EvaluationCase] = []
    for example in golden_examples:
        missing_ids = set(example.relevant_chunk_ids) - available_chunk_ids
        if missing_ids:
            raise ValueError(f"Golden example references missing chunk ids: {sorted(missing_ids)}")

        started = perf_counter()
        results = retriever.search(example.query, top_k=top_k)
        retrieved_ids = [result.chunk.chunk_id for result in results]
        relevant_ids = set(example.relevant_chunk_ids)
        answer = generation.answer(
            example.query,
            results,
            model_name="heuristic-grounded",
        )
        report = evaluate_answer(
            example.query,
            answer,
            results,
            reference_chunk_ids=example.relevant_chunk_ids,
            expected_unanswerable=example.expected_unanswerable,
        )
        latency_ms = (perf_counter() - started) * 1000.0
        token_estimate = sum(result.chunk.token_count for result in results)
        cases.append(
            EvaluationCase(
                query=example.query,
                retrieved_chunk_ids=retrieved_ids,
                relevant_chunk_ids=example.relevant_chunk_ids,
                expected_unanswerable=example.expected_unanswerable,
                recall_at_k=recall_at_k(retrieved_ids, relevant_ids),
                reciprocal_rank=reciprocal_rank(retrieved_ids, relevant_ids),
                latency_ms=latency_ms,
                token_estimate=token_estimate,
                answer=answer,
                evaluation=report,
            )
        )

    case_count = len(cases)
    divisor = max(case_count, 1)
    answerable_cases = [case for case in cases if not case.expected_unanswerable]
    retrieval_divisor = max(len(answerable_cases), 1)
    average_faithfulness = sum(case.evaluation.faithfulness for case in cases) / divisor
    average_citation_support = sum(case.evaluation.citation_support for case in cases) / divisor
    refusal_accuracy = (
        sum(1.0 for case in cases if case.evaluation.refusal_correct is True) / divisor
    )
    p95_latency_ms = _percentile_nearest_rank([case.latency_ms for case in cases], 0.95)
    p95_token_estimate = int(
        _percentile_nearest_rank([float(case.token_estimate) for case in cases], 0.95)
    )
    latency_within_budget = (
        max_p95_latency_ms is None or p95_latency_ms <= max_p95_latency_ms
    )
    tokens_within_budget = (
        max_p95_token_estimate is None or p95_token_estimate <= max_p95_token_estimate
    )
    summary = EvaluationSummary(
        case_count=case_count,
        answerable_case_count=len(answerable_cases),
        unanswerable_case_count=case_count - len(answerable_cases),
        top_k=top_k,
        average_recall_at_k=sum(case.recall_at_k for case in answerable_cases) / retrieval_divisor,
        mean_reciprocal_rank=sum(case.reciprocal_rank for case in answerable_cases)
        / retrieval_divisor,
        average_faithfulness=average_faithfulness,
        average_citation_support=average_citation_support,
        refusal_accuracy=refusal_accuracy,
        min_faithfulness=min_faithfulness,
        min_citation_support=min_citation_support,
        min_refusal_accuracy=min_refusal_accuracy,
        p95_latency_ms=p95_latency_ms,
        p95_token_estimate=p95_token_estimate,
        max_p95_latency_ms=max_p95_latency_ms,
        max_p95_token_estimate=max_p95_token_estimate,
        provenance=provenance,
        passed=(
            average_faithfulness >= min_faithfulness
            and average_citation_support >= min_citation_support
            and refusal_accuracy >= min_refusal_accuracy
            and latency_within_budget
            and tokens_within_budget
        ),
    )
    return summary, cases


def run_benchmark(
    *,
    source_dir: Path,
    golden_path: Path,
    output_dir: Path,
    runs: int,
    top_k: int,
    chunk_size: int,
    overlap: int,
    min_faithfulness: float,
    min_citation_support: float,
    refusal_path: Path | None = Path("data/golden/refusal.json"),
    min_refusal_accuracy: float = 1.0,
    max_p95_latency_ms: float | None = None,
    max_p95_token_estimate: int | None = None,
) -> tuple[BenchmarkSummary, list[BenchmarkRun]]:
    """Run repeated benchmark passes and aggregate their metrics."""
    if runs < 1:
        raise ValueError("runs must be greater than or equal to 1.")
    _validate_benchmark_inputs(
        source_dir=source_dir,
        golden_path=golden_path,
        refusal_path=refusal_path,
        top_k=top_k,
        chunk_size=chunk_size,
        overlap=overlap,
        min_faithfulness=min_faithfulness,
        min_citation_support=min_citation_support,
        min_refusal_accuracy=min_refusal_accuracy,
        max_p95_latency_ms=max_p95_latency_ms,
        max_p95_token_estimate=max_p95_token_estimate,
    )

    benchmark_runs: list[BenchmarkRun] = []
    for run_id in range(1, runs + 1):
        run_dir = output_dir / f"run-{run_id:03d}"
        summary, cases = run_evaluation(
            source_dir=source_dir,
            golden_path=golden_path,
            refusal_path=refusal_path,
            chunks_path=run_dir / "chunks.jsonl",
            top_k=top_k,
            chunk_size=chunk_size,
            overlap=overlap,
            min_faithfulness=min_faithfulness,
            min_citation_support=min_citation_support,
            min_refusal_accuracy=min_refusal_accuracy,
            max_p95_latency_ms=max_p95_latency_ms,
            max_p95_token_estimate=max_p95_token_estimate,
        )
        benchmark_runs.append(BenchmarkRun(run_id=run_id, summary=summary, cases=cases))

    divisor = len(benchmark_runs)
    benchmark_summary = BenchmarkSummary(
        run_count=divisor,
        case_count=benchmark_runs[0].summary.case_count if benchmark_runs else 0,
        answerable_case_count=(
            benchmark_runs[0].summary.answerable_case_count if benchmark_runs else 0
        ),
        unanswerable_case_count=(
            benchmark_runs[0].summary.unanswerable_case_count if benchmark_runs else 0
        ),
        top_k=top_k,
        average_recall_at_k=sum(run.summary.average_recall_at_k for run in benchmark_runs)
        / divisor,
        mean_reciprocal_rank=sum(run.summary.mean_reciprocal_rank for run in benchmark_runs)
        / divisor,
        average_faithfulness=sum(run.summary.average_faithfulness for run in benchmark_runs)
        / divisor,
        lowest_run_faithfulness=min(run.summary.average_faithfulness for run in benchmark_runs),
        average_citation_support=sum(run.summary.average_citation_support for run in benchmark_runs)
        / divisor,
        lowest_run_citation_support=min(
            run.summary.average_citation_support for run in benchmark_runs
        ),
        average_refusal_accuracy=sum(run.summary.refusal_accuracy for run in benchmark_runs)
        / divisor,
        lowest_run_refusal_accuracy=min(run.summary.refusal_accuracy for run in benchmark_runs),
        min_faithfulness=min_faithfulness,
        min_citation_support=min_citation_support,
        min_refusal_accuracy=min_refusal_accuracy,
        worst_run_p95_latency_ms=max(run.summary.p95_latency_ms for run in benchmark_runs),
        worst_run_p95_token_estimate=max(
            run.summary.p95_token_estimate for run in benchmark_runs
        ),
        max_p95_latency_ms=max_p95_latency_ms,
        max_p95_token_estimate=max_p95_token_estimate,
        provenance=benchmark_runs[0].summary.provenance,
        passed=all(run.summary.passed for run in benchmark_runs),
    )
    return benchmark_summary, benchmark_runs


def write_artifacts(
    summary: EvaluationSummary,
    cases: list[EvaluationCase],
    output_dir: Path,
) -> None:
    """Write one evaluation run's JSON, CSV, and Markdown artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        summary.model_dump_json(indent=2),
        encoding="utf-8",
    )
    (output_dir / "cases.json").write_text(
        json.dumps([case.model_dump(mode="json") for case in cases], indent=2),
        encoding="utf-8",
    )

    with (output_dir / "cases.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "query",
                "retrieved_chunk_ids",
                "relevant_chunk_ids",
                "expected_unanswerable",
                "recall_at_k",
                "reciprocal_rank",
                "latency_ms",
                "token_estimate",
                "refusal",
                "refusal_correct",
                "faithfulness",
                "citation_support",
                "unsupported_claim_count",
            ],
        )
        writer.writeheader()
        for case in cases:
            writer.writerow(
                {
                    "query": case.query,
                    "retrieved_chunk_ids": " ".join(case.retrieved_chunk_ids),
                    "relevant_chunk_ids": " ".join(case.relevant_chunk_ids),
                    "expected_unanswerable": case.expected_unanswerable,
                    "recall_at_k": f"{case.recall_at_k:.4f}",
                    "reciprocal_rank": f"{case.reciprocal_rank:.4f}",
                    "latency_ms": f"{case.latency_ms:.4f}",
                    "token_estimate": case.token_estimate,
                    "refusal": case.answer.refusal,
                    "refusal_correct": case.evaluation.refusal_correct,
                    "faithfulness": f"{case.evaluation.faithfulness:.4f}",
                    "citation_support": f"{case.evaluation.citation_support:.4f}",
                    "unsupported_claim_count": case.evaluation.unsupported_claim_count,
                }
            )

    lines = [
        "# RAG Evaluation Regression Report",
        "",
        f"- Cases: {summary.case_count}",
        f"- Answerable cases: {summary.answerable_case_count}",
        f"- Unanswerable cases: {summary.unanswerable_case_count}",
        f"- Top k: {summary.top_k}",
        f"- Recall@k: {summary.average_recall_at_k:.2f}",
        f"- MRR: {summary.mean_reciprocal_rank:.2f}",
        f"- Faithfulness: {summary.average_faithfulness:.2f}",
        f"- Citation support: {summary.average_citation_support:.2f}",
        f"- Refusal accuracy: {summary.refusal_accuracy:.2f}",
        f"- Required faithfulness: {summary.min_faithfulness:.2f}",
        f"- Required citation support: {summary.min_citation_support:.2f}",
        f"- Required refusal accuracy: {summary.min_refusal_accuracy:.2f}",
        f"- p95 latency (ms): {summary.p95_latency_ms:.2f}",
        f"- p95 token estimate: {summary.p95_token_estimate}",
        f"- Max p95 latency budget: {summary.max_p95_latency_ms}",
        f"- Max p95 token budget: {summary.max_p95_token_estimate}",
        f"- Benchmark fingerprint: {summary.provenance.fingerprint}",
        f"- Corpus SHA-256: {summary.provenance.corpus_sha256}",
        (
            "- Chunking: "
            f"{summary.provenance.chunking_strategy} "
            f"(size={summary.provenance.chunk_size}, overlap={summary.provenance.overlap})"
        ),
        f"- Status: {'passed' if summary.passed else 'failed'}",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_benchmark_artifacts(
    summary: BenchmarkSummary,
    runs: list[BenchmarkRun],
    output_dir: Path,
) -> None:
    """Write aggregate and per-run benchmark artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for run in runs:
        write_artifacts(run.summary, run.cases, output_dir / f"run-{run.run_id:03d}")
    if len(runs) == 1:
        write_artifacts(runs[0].summary, runs[0].cases, output_dir)

    (output_dir / "benchmark-summary.json").write_text(
        summary.model_dump_json(indent=2),
        encoding="utf-8",
    )
    (output_dir / "benchmark-runs.json").write_text(
        json.dumps(
            [
                {"run_id": run.run_id, "summary": run.summary.model_dump(mode="json")}
                for run in runs
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    with (output_dir / "benchmark-runs.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "run_id",
                "case_count",
                "recall_at_k",
                "mrr",
                "faithfulness",
                "citation_support",
                "refusal_accuracy",
                "p95_latency_ms",
                "p95_token_estimate",
                "passed",
            ],
        )
        writer.writeheader()
        for run in runs:
            writer.writerow(
                {
                    "run_id": run.run_id,
                    "case_count": run.summary.case_count,
                    "recall_at_k": f"{run.summary.average_recall_at_k:.4f}",
                    "mrr": f"{run.summary.mean_reciprocal_rank:.4f}",
                    "faithfulness": f"{run.summary.average_faithfulness:.4f}",
                    "citation_support": f"{run.summary.average_citation_support:.4f}",
                    "refusal_accuracy": f"{run.summary.refusal_accuracy:.4f}",
                    "p95_latency_ms": f"{run.summary.p95_latency_ms:.4f}",
                    "p95_token_estimate": run.summary.p95_token_estimate,
                    "passed": run.summary.passed,
                }
            )

    lines = [
        "# RAG Benchmark Summary",
        "",
        f"- Runs: {summary.run_count}",
        f"- Cases per run: {summary.case_count}",
        f"- Answerable cases per run: {summary.answerable_case_count}",
        f"- Unanswerable cases per run: {summary.unanswerable_case_count}",
        f"- Top k: {summary.top_k}",
        f"- Average recall@k: {summary.average_recall_at_k:.2f}",
        f"- Mean reciprocal rank: {summary.mean_reciprocal_rank:.2f}",
        f"- Average faithfulness: {summary.average_faithfulness:.2f}",
        f"- Lowest run faithfulness: {summary.lowest_run_faithfulness:.2f}",
        f"- Average citation support: {summary.average_citation_support:.2f}",
        f"- Lowest run citation support: {summary.lowest_run_citation_support:.2f}",
        f"- Average refusal accuracy: {summary.average_refusal_accuracy:.2f}",
        f"- Lowest run refusal accuracy: {summary.lowest_run_refusal_accuracy:.2f}",
        f"- Required faithfulness: {summary.min_faithfulness:.2f}",
        f"- Required citation support: {summary.min_citation_support:.2f}",
        f"- Required refusal accuracy: {summary.min_refusal_accuracy:.2f}",
        f"- Worst-run p95 latency (ms): {summary.worst_run_p95_latency_ms:.2f}",
        f"- Worst-run p95 token estimate: {summary.worst_run_p95_token_estimate}",
        f"- Max p95 latency budget: {summary.max_p95_latency_ms}",
        f"- Max p95 token budget: {summary.max_p95_token_estimate}",
        f"- Benchmark fingerprint: {summary.provenance.fingerprint}",
        f"- Corpus SHA-256: {summary.provenance.corpus_sha256}",
        (
            "- Chunking: "
            f"{summary.provenance.chunking_strategy} "
            f"(size={summary.provenance.chunk_size}, overlap={summary.provenance.overlap})"
        ),
        f"- Status: {'passed' if summary.passed else 'failed'}",
    ]
    (output_dir / "benchmark-summary.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def _validate_benchmark_inputs(
    *,
    source_dir: Path,
    golden_path: Path,
    top_k: int,
    chunk_size: int,
    overlap: int,
    min_faithfulness: float,
    min_citation_support: float,
    refusal_path: Path | None,
    min_refusal_accuracy: float,
    max_p95_latency_ms: float | None,
    max_p95_token_estimate: int | None,
) -> None:
    if not source_dir.exists():
        raise ValueError(f"Source directory not found: {source_dir}")
    if not source_dir.is_dir():
        raise ValueError(f"Source path is not a directory: {source_dir}")
    if not golden_path.exists():
        raise ValueError(f"Golden dataset not found: {golden_path}")
    if refusal_path is not None and not refusal_path.exists():
        raise ValueError(f"Refusal dataset not found: {refusal_path}")
    if top_k < 1:
        raise ValueError("top_k must be greater than or equal to 1.")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size.")
    if not 0.0 <= min_faithfulness <= 1.0:
        raise ValueError("min_faithfulness must be between 0 and 1.")
    if not 0.0 <= min_citation_support <= 1.0:
        raise ValueError("min_citation_support must be between 0 and 1.")
    if not 0.0 <= min_refusal_accuracy <= 1.0:
        raise ValueError("min_refusal_accuracy must be between 0 and 1.")
    if max_p95_latency_ms is not None and max_p95_latency_ms <= 0.0:
        raise ValueError("max_p95_latency_ms must be positive when provided.")
    if max_p95_token_estimate is not None and max_p95_token_estimate < 1:
        raise ValueError("max_p95_token_estimate must be at least 1 when provided.")
