from __future__ import annotations

from pathlib import Path

from ragops_lab.domain import DocumentChunk, GeneratedAnswer, RetrievalResult
from ragops_lab.evaluation import (
    EmbeddingClaimSupportJudge,
    EmbeddingRelevanceJudge,
    evaluate_answer,
    export_evaluation_report_csv,
    export_evaluation_report_markdown,
)


def test_evaluation_metrics_and_exports(tmp_path: Path) -> None:
    chunk = DocumentChunk(
        chunk_id="apollo:0",
        document_id="apollo",
        text="Apollo 11 was the first mission to land humans on the Moon.",
        start_offset=0,
        end_offset=60,
        token_count=12,
    )
    result = RetrievalResult(chunk=chunk, score=1.0, rank=1, retrieval_method="lexical")
    answer = GeneratedAnswer(
        question="Which mission first landed humans on the Moon?",
        answer_text="Apollo 11 was the first mission to land humans on the Moon.",
        citations=["apollo:0"],
        model_name="heuristic",
        grounded=True,
    )

    report = evaluate_answer(
        answer.question,
        answer,
        [result],
        reference_chunk_ids=["apollo:0"],
        expected_answer=chunk.text,
        expected_unanswerable=False,
    )
    csv_path = tmp_path / "report.csv"
    md_path = tmp_path / "report.md"
    export_evaluation_report_csv(report, csv_path)
    export_evaluation_report_markdown(report, md_path)

    assert report.faithfulness == 1.0
    assert report.citation_support == 1.0
    assert report.claim_count == 1
    assert report.supported_claim_count == 1
    assert csv_path.exists()
    assert md_path.exists()


def test_claim_support_tolerates_light_paraphrase() -> None:
    chunk = DocumentChunk(
        chunk_id="metrics:0",
        document_id="metrics",
        text="Context precision measures the share of retrieved chunks that are relevant.",
        start_offset=0,
        end_offset=72,
        token_count=10,
    )
    answer = GeneratedAnswer(
        question="What does context precision measure?",
        answer_text="Context precision is the proportion of relevant retrieved chunks.",
        citations=["metrics:0"],
        model_name="heuristic",
        grounded=True,
    )

    report = evaluate_answer(
        answer.question,
        answer,
        [RetrievalResult(chunk=chunk, score=1.0, rank=1, retrieval_method="lexical")],
    )

    assert report.faithfulness == 1.0
    assert report.unsupported_claim_count == 0
    assert report.claim_support[0].supported is True


def test_claim_support_flags_hallucinated_claims() -> None:
    chunk = DocumentChunk(
        chunk_id="apollo:0",
        document_id="apollo",
        text="Apollo 11 landed humans on the Moon in 1969.",
        start_offset=0,
        end_offset=46,
        token_count=9,
    )
    answer = GeneratedAnswer(
        question="What happened on Apollo 11?",
        answer_text=(
            "Apollo 11 landed humans on the Moon in 1969. Buzz Aldrin commanded the mission."
        ),
        citations=["apollo:0"],
        model_name="heuristic",
        grounded=True,
    )

    report = evaluate_answer(
        answer.question,
        answer,
        [RetrievalResult(chunk=chunk, score=1.0, rank=1, retrieval_method="lexical")],
    )

    assert report.faithfulness == 0.5
    assert report.unsupported_claim_count == 1
    assert report.unsupported_claims == ["Buzz Aldrin commanded the mission."]


def test_claim_support_requires_matching_numbers() -> None:
    chunk = DocumentChunk(
        chunk_id="apollo:0",
        document_id="apollo",
        text="Apollo 11 was the first mission to land humans on the Moon.",
        start_offset=0,
        end_offset=60,
        token_count=12,
    )
    answer = GeneratedAnswer(
        question="Which mission landed humans on the Moon?",
        answer_text="Apollo 12 was the first mission to land humans on the Moon.",
        citations=["apollo:0"],
        model_name="heuristic",
        grounded=True,
    )

    report = evaluate_answer(
        answer.question,
        answer,
        [RetrievalResult(chunk=chunk, score=1.0, rank=1, retrieval_method="lexical")],
    )

    assert report.faithfulness == 0.0
    assert report.claim_support[0].missing_terms == ["12"]


def test_claim_support_prefers_cited_evidence() -> None:
    cited_chunk = DocumentChunk(
        chunk_id="apollo:0",
        document_id="apollo",
        text="Apollo 11 landed humans on the Moon.",
        start_offset=0,
        end_offset=36,
        token_count=7,
    )
    uncited_chunk = DocumentChunk(
        chunk_id="apollo:1",
        document_id="apollo",
        text="Neil Armstrong commanded Gemini 8.",
        start_offset=37,
        end_offset=70,
        token_count=5,
    )
    answer = GeneratedAnswer(
        question="What did Neil Armstrong command?",
        answer_text="Neil Armstrong commanded Gemini 8.",
        citations=["apollo:0"],
        model_name="heuristic",
        grounded=True,
    )

    report = evaluate_answer(
        answer.question,
        answer,
        [
            RetrievalResult(chunk=cited_chunk, score=1.0, rank=1, retrieval_method="lexical"),
            RetrievalResult(chunk=uncited_chunk, score=0.8, rank=2, retrieval_method="lexical"),
        ],
    )

    assert report.faithfulness == 0.0
    assert report.claim_support[0].evidence_chunk_id == "apollo:0"


class _StubEmbeddingClient:
    """Small deterministic embedding stub for semantic judge tests."""

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            normalized = text.lower()
            if "moon" in normalized or "lunar" in normalized:
                vectors.append([1.0, 0.0, 0.0])
            elif "citation" in normalized or "source" in normalized:
                vectors.append([0.0, 1.0, 0.0])
            else:
                vectors.append([0.0, 0.0, 1.0])
        return vectors

    def embed_query(self, query: str) -> list[float]:
        return self.embed_texts([query])[0]


def test_embedding_relevance_judge_scores_semantic_alignment() -> None:
    judge = EmbeddingRelevanceJudge(_StubEmbeddingClient())

    score = judge.score(
        "What happened on the Moon?",
        "The lunar landing was successful.",
        "Apollo 11 completed the Moon landing.",
    )

    assert score == 1.0


def test_embedding_claim_support_judge_selects_best_evidence() -> None:
    moon_chunk = DocumentChunk(
        chunk_id="moon:0",
        document_id="moon",
        text="Apollo 11 completed the lunar landing.",
        start_offset=0,
        end_offset=38,
        token_count=6,
    )
    citation_chunk = DocumentChunk(
        chunk_id="citation:0",
        document_id="citation",
        text="Citation support checks whether sources back an answer.",
        start_offset=0,
        end_offset=54,
        token_count=8,
    )
    judge = EmbeddingClaimSupportJudge(
        _StubEmbeddingClient(),
        support_threshold=0.8,
    )

    result = judge.score_claim(
        "Apollo 11 completed the Moon landing.",
        [
            RetrievalResult(
                chunk=citation_chunk,
                score=1.0,
                rank=1,
                retrieval_method="lexical",
            ),
            RetrievalResult(
                chunk=moon_chunk,
                score=0.8,
                rank=2,
                retrieval_method="lexical",
            ),
        ],
    )

    assert result.supported is True
    assert result.evidence_chunk_id == "moon:0"
    assert result.score == 1.0


def test_embedding_claim_support_rejects_numeric_mismatch() -> None:
    chunk = DocumentChunk(
        chunk_id="apollo:0",
        document_id="apollo",
        text="Apollo 11 landed humans on the Moon.",
        start_offset=0,
        end_offset=36,
        token_count=7,
    )
    judge = EmbeddingClaimSupportJudge(
        _StubEmbeddingClient(),
        support_threshold=0.8,
    )

    result = judge.score_claim(
        "Apollo 12 landed humans on the Moon.",
        [RetrievalResult(chunk=chunk, score=1.0, rank=1, retrieval_method="lexical")],
    )

    assert result.supported is False
    assert result.score == 0.0
    assert "12" in result.missing_terms


def test_evaluate_answer_accepts_semantic_judges() -> None:
    chunk = DocumentChunk(
        chunk_id="apollo:0",
        document_id="apollo",
        text="Apollo 11 completed the lunar landing.",
        start_offset=0,
        end_offset=38,
        token_count=6,
    )
    result = RetrievalResult(chunk=chunk, score=1.0, rank=1, retrieval_method="lexical")
    answer = GeneratedAnswer(
        question="What happened on the Moon?",
        answer_text="Apollo 11 completed the Moon landing.",
        citations=["apollo:0"],
        model_name="heuristic",
        grounded=True,
    )

    report = evaluate_answer(
        answer.question,
        answer,
        [result],
        judge=EmbeddingRelevanceJudge(_StubEmbeddingClient()),
        claim_judge=EmbeddingClaimSupportJudge(
            _StubEmbeddingClient(),
            support_threshold=0.8,
        ),
    )

    assert report.answer_relevance == 1.0
    assert report.faithfulness == 1.0
