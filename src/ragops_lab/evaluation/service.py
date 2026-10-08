"""Evaluation services and report exporters."""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Protocol

from ragops_lab.domain import ClaimSupportResult, EvaluationResult, GeneratedAnswer, RetrievalResult

from ..retrieval.tokenizer import tokenize
from ..retrieval.vector import EmbeddingClient, cosine_similarity

CLAIM_SUPPORT_THRESHOLD = 0.65
STOPWORDS = frozenset(
    {
        "a",
        "about",
        "above",
        "after",
        "again",
        "against",
        "all",
        "also",
        "am",
        "an",
        "and",
        "any",
        "are",
        "as",
        "at",
        "be",
        "because",
        "been",
        "before",
        "being",
        "between",
        "both",
        "but",
        "by",
        "can",
        "could",
        "did",
        "do",
        "does",
        "doing",
        "during",
        "each",
        "few",
        "for",
        "from",
        "had",
        "has",
        "have",
        "having",
        "he",
        "her",
        "here",
        "hers",
        "him",
        "his",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "itself",
        "me",
        "more",
        "most",
        "my",
        "no",
        "not",
        "of",
        "on",
        "once",
        "only",
        "or",
        "other",
        "our",
        "out",
        "over",
        "own",
        "same",
        "she",
        "should",
        "so",
        "some",
        "such",
        "than",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "through",
        "to",
        "too",
        "under",
        "until",
        "up",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "while",
        "who",
        "whom",
        "why",
        "will",
        "with",
        "you",
        "your",
    }
)


class RelevanceJudge(Protocol):
    """Judge abstraction used for answer relevance scoring."""

    def score(self, question: str, answer_text: str, references: str) -> float:
        """Return a normalized score."""


class OverlapJudge:
    """Simple lexical overlap judge for local evaluation."""

    def score(self, question: str, answer_text: str, references: str) -> float:
        question_terms = set(tokenize(question))
        answer_terms = set(tokenize(answer_text))
        reference_terms = set(tokenize(references))
        if not question_terms or not answer_terms:
            return 0.0
        aligned = len((question_terms & answer_terms) & reference_terms)
        return min(1.0, aligned / len(question_terms))


class EmbeddingRelevanceJudge:
    """Semantic answer relevance judge backed by embeddings."""

    def __init__(self, embedding_client: EmbeddingClient) -> None:
        self.embedding_client = embedding_client

    def score(self, question: str, answer_text: str, references: str) -> float:
        """Score answer relevance from semantic similarity to question and references."""
        if not question.strip() or not answer_text.strip():
            return 0.0
        vectors = self.embedding_client.embed_texts([question, answer_text, references])
        question_score = cosine_similarity(vectors[0], vectors[1])
        if not references.strip():
            return question_score
        reference_score = cosine_similarity(vectors[1], vectors[2])
        return min(1.0, (question_score + reference_score) / 2.0)


class ClaimSupportJudge(Protocol):
    """Judge abstraction used for claim-level faithfulness scoring."""

    def score_claim(
        self,
        claim: str,
        evidence: list[RetrievalResult],
    ) -> ClaimSupportResult:
        """Return a claim-level support result."""


class LexicalClaimSupportJudge:
    """Deterministic lexical claim support judge for local evaluation.

    The judge compares each claim with individual evidence chunks, ignores
    low-signal function words, applies light stemming, and treats missing
    numeric tokens as unsupported even when the rest of the overlap is high.
    """

    def __init__(self, *, support_threshold: float = CLAIM_SUPPORT_THRESHOLD) -> None:
        self.support_threshold = support_threshold

    def score_claim(
        self,
        claim: str,
        evidence: list[RetrievalResult],
    ) -> ClaimSupportResult:
        claim_terms = _content_terms(claim)
        if not claim_terms:
            return ClaimSupportResult(
                claim=claim,
                supported=True,
                score=1.0,
                matched_terms=[],
                missing_terms=[],
            )
        if not evidence:
            return ClaimSupportResult(
                claim=claim,
                supported=False,
                score=0.0,
                matched_terms=[],
                missing_terms=claim_terms,
            )

        best_score = -1.0
        best_chunk_id: str | None = None
        best_matched: list[str] = []
        best_missing: list[str] = claim_terms
        claim_numbers = {term for term in claim_terms if term.isdigit()}

        for result in evidence:
            evidence_terms = set(_content_terms(result.chunk.text))
            matched = sorted(set(claim_terms) & evidence_terms)
            missing = sorted(set(claim_terms) - evidence_terms)
            score = len(matched) / len(set(claim_terms))
            missing_numbers = claim_numbers - evidence_terms
            if missing_numbers:
                score = 0.0
            if score > best_score:
                best_score = score
                best_chunk_id = result.chunk.chunk_id
                best_matched = matched
                best_missing = missing

        supported = best_score >= self.support_threshold
        return ClaimSupportResult(
            claim=claim,
            supported=supported,
            score=max(0.0, best_score),
            evidence_chunk_id=best_chunk_id,
            matched_terms=best_matched,
            missing_terms=best_missing if not supported else [],
        )


class EmbeddingClaimSupportJudge:
    """Semantic claim-support judge backed by embeddings."""

    def __init__(
        self,
        embedding_client: EmbeddingClient,
        *,
        support_threshold: float = CLAIM_SUPPORT_THRESHOLD,
    ) -> None:
        if not 0.0 <= support_threshold <= 1.0:
            raise ValueError("support_threshold must be between 0 and 1.")
        self.embedding_client = embedding_client
        self.support_threshold = support_threshold

    def score_claim(
        self,
        claim: str,
        evidence: list[RetrievalResult],
    ) -> ClaimSupportResult:
        """Score a claim against the most semantically similar evidence chunk."""
        claim_terms = _content_terms(claim)
        if not claim_terms:
            return ClaimSupportResult(
                claim=claim,
                supported=True,
                score=1.0,
                matched_terms=[],
                missing_terms=[],
            )
        if not evidence:
            return ClaimSupportResult(
                claim=claim,
                supported=False,
                score=0.0,
                matched_terms=[],
                missing_terms=claim_terms,
            )

        texts = [claim, *[result.chunk.text for result in evidence]]
        vectors = self.embedding_client.embed_texts(texts)
        claim_vector = vectors[0]
        scores = [
            cosine_similarity(claim_vector, evidence_vector)
            for evidence_vector in vectors[1:]
        ]
        best_index = max(range(len(scores)), key=scores.__getitem__)
        best_result = evidence[best_index]
        best_score = scores[best_index]

        claim_numbers = {term for term in claim_terms if term.isdigit()}
        evidence_terms = set(_content_terms(best_result.chunk.text))
        missing_numbers = sorted(claim_numbers - evidence_terms)
        if missing_numbers:
            best_score = 0.0

        supported = best_score >= self.support_threshold
        matched_terms = sorted(set(claim_terms) & evidence_terms)
        missing_terms = sorted(set(claim_terms) - evidence_terms)
        return ClaimSupportResult(
            claim=claim,
            supported=supported,
            score=best_score,
            evidence_chunk_id=best_result.chunk.chunk_id,
            matched_terms=matched_terms,
            missing_terms=[] if supported else missing_terms,
        )


def context_precision(results: list[RetrievalResult], reference_chunk_ids: set[str]) -> float:
    """Measure the proportion of retrieved chunks that are relevant."""
    if not results:
        return 0.0
    relevant = sum(1 for result in results if result.chunk.chunk_id in reference_chunk_ids)
    return relevant / len(results)


def context_recall(results: list[RetrievalResult], reference_chunk_ids: set[str]) -> float:
    """Measure how much of the needed context was retrieved."""
    if not reference_chunk_ids:
        return 0.0
    retrieved = {result.chunk.chunk_id for result in results}
    return len(retrieved & reference_chunk_ids) / len(reference_chunk_ids)


def extract_claims(answer_text: str) -> list[str]:
    """Break an answer into sentence-level claims."""
    normalized = re.sub(r"\s+", " ", answer_text.strip())
    claims = [part.strip() for part in re.split(r"(?<=[.!?;])\s+", normalized)]
    return [claim for claim in claims if claim]


def score_claims(
    answer: GeneratedAnswer,
    results: list[RetrievalResult],
    judge: ClaimSupportJudge | None = None,
) -> list[ClaimSupportResult]:
    """Score answer claims against retrieved or cited evidence."""
    claims = extract_claims(answer.answer_text)
    if answer.refusal:
        return [
            ClaimSupportResult(
                claim=claim,
                supported=True,
                score=1.0,
                matched_terms=[],
                missing_terms=[],
            )
            for claim in claims
        ]
    support_judge = judge or LexicalClaimSupportJudge()
    evidence = _evidence_for_answer(answer, results)
    return [support_judge.score_claim(claim, evidence) for claim in claims]


def unsupported_claim_count(answer_text: str, evidence_text: str) -> int:
    """Count claims not supported by lexical claim-evidence overlap."""
    evidence_chunk = _manual_evidence_result(evidence_text)
    answer = GeneratedAnswer(
        question="manual",
        answer_text=answer_text,
        citations=[evidence_chunk.chunk.chunk_id],
        model_name="manual-eval",
        grounded=True,
    )
    return sum(not result.supported for result in score_claims(answer, [evidence_chunk]))


def citation_support(answer: GeneratedAnswer, results: list[RetrievalResult]) -> float:
    """Measure whether cited chunks exist in retrieved context."""
    if answer.refusal:
        return 1.0
    if not answer.citations:
        return 0.0
    retrieved_chunk_ids = {result.chunk.chunk_id for result in results}
    supported = sum(1 for citation in answer.citations if citation in retrieved_chunk_ids)
    return supported / len(answer.citations)


def refusal_correctness(answer: GeneratedAnswer, expected_unanswerable: bool) -> bool:
    """Check whether a refusal matched the scenario."""
    return answer.refusal is expected_unanswerable


def evaluate_answer(
    question: str,
    answer: GeneratedAnswer,
    results: list[RetrievalResult],
    *,
    reference_chunk_ids: list[str] | None = None,
    expected_answer: str = "",
    expected_unanswerable: bool | None = None,
    judge: RelevanceJudge | None = None,
    claim_judge: ClaimSupportJudge | None = None,
) -> EvaluationResult:
    """Evaluate a generated answer."""
    reference_ids = set(reference_chunk_ids or [])
    evidence_text = "\n".join(result.chunk.text for result in results)
    overlap_judge = judge or OverlapJudge()
    claim_support = score_claims(answer, results, claim_judge)
    unsupported_claims = [result.claim for result in claim_support if not result.supported]
    unsupported = len(unsupported_claims)
    claim_count = max(len(claim_support), 1)
    supported_claim_count = len(claim_support) - unsupported
    faithfulness = max(0.0, 1.0 - unsupported / claim_count)
    notes: list[str] = []
    citation_score = citation_support(answer, results)
    if citation_score < 1.0:
        notes.append("One or more citations were unsupported by retrieved context.")
    if unsupported > 0:
        notes.append("Unsupported claims detected in answer text.")
    return EvaluationResult(
        context_precision=context_precision(results, reference_ids),
        context_recall=context_recall(results, reference_ids),
        answer_relevance=overlap_judge.score(
            question, answer.answer_text, expected_answer or evidence_text
        ),
        faithfulness=faithfulness,
        citation_support=citation_score,
        unsupported_claim_count=unsupported,
        claim_count=len(claim_support),
        supported_claim_count=supported_claim_count,
        unsupported_claims=unsupported_claims,
        claim_support=claim_support,
        refusal_correct=(
            refusal_correctness(answer, expected_unanswerable)
            if expected_unanswerable is not None
            else None
        ),
        notes=notes,
    )


def export_evaluation_report_csv(report: EvaluationResult, output_path: Path) -> None:
    """Export a single-row CSV evaluation report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(report.model_dump().keys()))
        writer.writeheader()
        writer.writerow(report.model_dump(mode="json"))


def export_evaluation_report_markdown(report: EvaluationResult, output_path: Path) -> None:
    """Export an evaluation report as Markdown."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Evaluation Report",
        "",
        f"- Context precision: {report.context_precision:.2f}",
        f"- Context recall: {report.context_recall:.2f}",
        f"- Answer relevance: {report.answer_relevance:.2f}",
        f"- Faithfulness: {report.faithfulness:.2f}",
        f"- Citation support: {report.citation_support:.2f}",
        f"- Claims: {report.supported_claim_count}/{report.claim_count} supported",
        f"- Unsupported claims: {report.unsupported_claim_count}",
        f"- Refusal correct: {report.refusal_correct}",
    ]
    if report.unsupported_claims:
        lines.extend(["", "## Unsupported Claims", ""])
        lines.extend(f"- {claim}" for claim in report.unsupported_claims)
    if report.notes:
        lines.extend(["", "## Notes", ""])
        lines.extend(f"- {note}" for note in report.notes)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _evidence_for_answer(
    answer: GeneratedAnswer,
    results: list[RetrievalResult],
) -> list[RetrievalResult]:
    if not answer.citations:
        return results
    cited = set(answer.citations)
    cited_results = [result for result in results if result.chunk.chunk_id in cited]
    return cited_results or results


def _content_terms(text: str) -> list[str]:
    terms = []
    for term in tokenize(text):
        normalized = _normalize_term(term)
        if normalized and (normalized.isdigit() or normalized not in STOPWORDS):
            terms.append(normalized)
    return sorted(set(terms))


def _normalize_term(term: str) -> str:
    if term.isdigit():
        return term
    if len(term) > 4 and term.endswith("ies"):
        return f"{term[:-3]}y"
    for suffix in ("ing", "ed", "es", "s"):
        if len(term) > len(suffix) + 3 and term.endswith(suffix):
            return term[: -len(suffix)]
    return term


def _manual_evidence_result(evidence_text: str) -> RetrievalResult:
    from ragops_lab.domain import DocumentChunk

    chunk = DocumentChunk(
        chunk_id="manual:0",
        document_id="manual",
        text=evidence_text,
        start_offset=0,
        end_offset=len(evidence_text),
        token_count=len(tokenize(evidence_text)),
    )
    return RetrievalResult(chunk=chunk, score=1.0, rank=1, retrieval_method="manual")
