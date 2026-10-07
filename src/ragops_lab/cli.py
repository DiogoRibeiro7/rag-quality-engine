"""Command line interface for the project."""

from __future__ import annotations

from pathlib import Path
from typing import NoReturn

import typer
from rich.console import Console
from rich.table import Table

from .config import RuntimeSettings
from .evaluation import evaluate_answer, run_benchmark, write_benchmark_artifacts
from .generation import GenerationService, build_llm_client
from .ingestion import ChunkingConfig, ingest_directory, load_chunks_jsonl
from .retrieval import (
    BM25Retriever,
    HybridRetriever,
    LocalVectorIndex,
    build_embedding_client,
    Retriever,
    search_with_profile,
)

app = typer.Typer(help="Portfolio project command line interface.")
console = Console()


@app.command()
def info() -> None:
    """Print validated runtime settings."""
    settings = RuntimeSettings.from_env()
    console.print(settings.model_dump())


def _fail(message: str) -> NoReturn:
    console.print(f"Error: {message}", style="red")
    raise typer.Exit(1)


@app.command()
def ingest(
    input_dir: Path,
    out: Path | None = None,
    chunk_size: int = 500,
    overlap: int = 50,
) -> None:
    """Ingest documents into chunk JSONL."""
    settings = RuntimeSettings.from_env()
    output_path = out or settings.paths.chunk_path
    if not input_dir.exists():
        _fail(f"Input directory not found: {input_dir}")
    if not input_dir.is_dir():
        _fail(f"Input path is not a directory: {input_dir}")
    try:
        chunks = ingest_directory(
            input_dir,
            output_path,
            config=ChunkingConfig(chunk_size=chunk_size, overlap=overlap),
        )
    except ValueError as exc:
        _fail(str(exc))
    console.print({"chunks_written": len(chunks), "out": str(output_path)})


@app.command()
def index(
    chunks: Path | None = None,
    out: Path | None = None,
) -> None:
    """Build a persistent local vector index from chunk JSONL."""
    settings = RuntimeSettings.from_env()
    chunks_path = chunks or settings.paths.chunk_path
    output_path = out or settings.paths.vector_index_path
    if not chunks_path.exists():
        _fail(f"Chunks file not found: {chunks_path}")
    chunk_list = load_chunks_jsonl(chunks_path)
    try:
        embedding_client = build_embedding_client(settings.embeddings)
    except (RuntimeError, ValueError) as exc:
        _fail(str(exc))
    LocalVectorIndex.build(
        chunk_list,
        embedding_client,
        embedding_provider=settings.embeddings.provider,
        embedding_model=settings.embeddings.model,
    ).save(output_path)
    console.print({"chunks_indexed": len(chunk_list), "out": str(output_path)})


@app.command()
def ask(
    question: str,
    chunks: Path | None = None,
    index_path: Path | None = None,
    profile: str = "lexical",
    mode: str | None = None,
    top_k: int | None = None,
    lexical_weight: float | None = None,
    vector_weight: float | None = None,
    fusion_strategy: str | None = None,
    rrf_k: int | None = None,
    rerank: bool | None = None,
    rerank_candidate_multiplier: int | None = None,
) -> None:
    """Ask a grounded question over ingested chunks."""
    settings = RuntimeSettings.from_env()
    try:
        retrieval = settings.resolve_retrieval_profile(
            profile,
            mode=mode,
            top_k=top_k,
            lexical_weight=lexical_weight,
            vector_weight=vector_weight,
            fusion_strategy=fusion_strategy,
            rrf_k=rrf_k,
            rerank=rerank,
            rerank_candidate_multiplier=rerank_candidate_multiplier,
        )
    except ValueError as exc:
        _fail(str(exc))
    chunks_path = chunks or settings.paths.chunk_path
    vector_index_path = index_path or settings.paths.vector_index_path
    if retrieval.mode in {"lexical", "hybrid"} and not chunks_path.exists():
        _fail(f"Chunks file not found: {chunks_path}")
    if retrieval.mode in {"vector", "hybrid"} and not vector_index_path.exists():
        _fail(f"Vector index not found: {vector_index_path}")
    retriever: Retriever
    if retrieval.mode == "lexical":
        chunk_list = load_chunks_jsonl(chunks_path)
        retriever = BM25Retriever(chunk_list)
    elif retrieval.mode == "vector":
        retriever = LocalVectorIndex.load(vector_index_path).as_retriever()
    else:
        chunk_list = load_chunks_jsonl(chunks_path)
        lexical = BM25Retriever(chunk_list)
        vector = LocalVectorIndex.load(vector_index_path).as_retriever()
        retriever = HybridRetriever(
            lexical,
            vector,
            lexical_weight=retrieval.lexical_weight,
            vector_weight=retrieval.vector_weight,
            fusion_strategy=retrieval.fusion_strategy,
            rrf_k=retrieval.rrf_k,
        )
    results = search_with_profile(retriever, retrieval, question)
    try:
        llm_client, model_name = build_llm_client(settings.llm)
    except (RuntimeError, ValueError) as exc:
        _fail(str(exc))
    try:
        answer = GenerationService(llm_client).answer(question, results, model_name=model_name)
    except (RuntimeError, ValueError) as exc:
        _fail(str(exc))
    evaluation = evaluate_answer(question, answer, results)
    table = Table(title="RAG Answer")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("Question", question)
    table.add_row("Answer", answer.answer_text)
    table.add_row("Citations", ", ".join(answer.citations) or "none")
    table.add_row("Faithfulness", f"{evaluation.faithfulness:.2f}")
    table.add_row("Citation support", f"{evaluation.citation_support:.2f}")
    console.print(table)


@app.command()
def benchmark(
    source_dir: Path = Path("data/sample_documents"),
    golden_path: Path = Path("data/golden/qa.json"),
    refusal_path: Path = Path("data/golden/refusal.json"),
    out: Path = Path("artifacts/evaluation"),
    runs: int = 1,
    top_k: int = 2,
    chunk_size: int = 400,
    overlap: int = 60,
    min_faithfulness: float = 0.80,
    min_citation_support: float = 1.00,
    min_refusal_accuracy: float = 1.00,
) -> None:
    """Run a dataset benchmark over a golden QA set."""
    if not source_dir.exists():
        _fail(f"Source directory not found: {source_dir}")
    if not source_dir.is_dir():
        _fail(f"Source path is not a directory: {source_dir}")
    if not golden_path.exists():
        _fail(f"Golden dataset not found: {golden_path}")
    if not refusal_path.exists():
        _fail(f"Refusal dataset not found: {refusal_path}")
    if runs < 1:
        _fail("runs must be greater than or equal to 1.")
    if top_k < 1:
        _fail("top_k must be greater than or equal to 1.")
    try:
        summary, benchmark_runs = run_benchmark(
            source_dir=source_dir,
            golden_path=golden_path,
            refusal_path=refusal_path,
            output_dir=out,
            runs=runs,
            top_k=top_k,
            chunk_size=chunk_size,
            overlap=overlap,
            min_faithfulness=min_faithfulness,
            min_citation_support=min_citation_support,
            min_refusal_accuracy=min_refusal_accuracy,
        )
    except ValueError as exc:
        _fail(str(exc))
    write_benchmark_artifacts(summary, benchmark_runs, out)

    table = Table(title="Benchmark Summary")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Runs", str(summary.run_count))
    table.add_row("Cases per run", str(summary.case_count))
    table.add_row("Recall@k", f"{summary.average_recall_at_k:.2f}")
    table.add_row("MRR", f"{summary.mean_reciprocal_rank:.2f}")
    table.add_row("Faithfulness", f"{summary.average_faithfulness:.2f}")
    table.add_row("Citation support", f"{summary.average_citation_support:.2f}")
    table.add_row("Refusal accuracy", f"{summary.average_refusal_accuracy:.2f}")
    table.add_row("Status", "passed" if summary.passed else "failed")
    table.add_row("Artifacts", str(out))
    console.print(table)
    if not summary.passed:
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
