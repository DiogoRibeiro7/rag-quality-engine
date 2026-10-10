"""Second-stage retrieval reranking."""

from __future__ import annotations

from typing import Protocol

from ragops_lab.domain import RetrievalResult

from .tokenizer import tokenize
from .vector import EmbeddingClient, cosine_similarity


class Retriever(Protocol):
    """Minimal retrieval interface used by reranking pipelines."""

    def search(self, query: str, *, top_k: int = 5) -> list[RetrievalResult]:
        """Return ranked retrieval candidates."""


class Reranker(Protocol):
    """Interface for second-stage candidate reranking."""

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        *,
        top_k: int,
    ) -> list[RetrievalResult]:
        """Rerank candidate results for a query."""


class LexicalOverlapReranker:
    """Deterministic offline reranker based on query-token coverage.

    The score is the fraction of unique query tokens present in the candidate
    chunk. The original retrieval score is used only as a deterministic
    tie-breaker, so the reranker remains independent of retriever score scale.
    """

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        *,
        top_k: int,
    ) -> list[RetrievalResult]:
        """Rerank candidates by lexical query coverage."""
        if top_k <= 0:
            raise ValueError("top_k must be positive.")

        query_terms = set(tokenize(query))
        scored: list[tuple[RetrievalResult, float]] = []
        for candidate in candidates:
            chunk_terms = set(tokenize(candidate.chunk.text))
            overlap = len(query_terms & chunk_terms)
            score = overlap / max(len(query_terms), 1)
            scored.append((candidate, score))

        scored.sort(
            key=lambda item: (
                -item[1],
                -item[0].score,
                item[0].rank,
                item[0].chunk.chunk_id,
            )
        )

        return [
            RetrievalResult(
                chunk=candidate.chunk,
                score=score,
                rank=rank,
                retrieval_method=f"reranked-{candidate.retrieval_method}",
                matched_terms=sorted(set(tokenize(query)) & set(tokenize(candidate.chunk.text))),
            )
            for rank, (candidate, score) in enumerate(scored[:top_k], start=1)
        ]


class EmbeddingSimilarityReranker:
    """Semantic reranker backed by embedding similarity."""

    def __init__(self, embedding_client: EmbeddingClient) -> None:
        self.embedding_client = embedding_client

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        *,
        top_k: int,
    ) -> list[RetrievalResult]:
        """Rerank candidates by semantic similarity to the query."""
        if top_k <= 0:
            raise ValueError("top_k must be positive.")
        if not candidates:
            return []

        texts = [query, *[candidate.chunk.text for candidate in candidates]]
        vectors = self.embedding_client.embed_texts(texts)
        query_vector = vectors[0]
        scored = [
            (candidate, cosine_similarity(query_vector, vector))
            for candidate, vector in zip(candidates, vectors[1:], strict=True)
        ]
        scored.sort(
            key=lambda item: (
                -item[1],
                -item[0].score,
                item[0].rank,
                item[0].chunk.chunk_id,
            )
        )
        return [
            RetrievalResult(
                chunk=candidate.chunk,
                score=score,
                rank=rank,
                retrieval_method=f"reranked-{candidate.retrieval_method}",
                matched_terms=[],
            )
            for rank, (candidate, score) in enumerate(scored[:top_k], start=1)
        ]


class RetrieveThenRerank:
    """Retrieve a widened candidate set, then rerank to the requested top-k."""

    def __init__(
        self,
        retriever: Retriever,
        reranker: Reranker,
        *,
        candidate_multiplier: int = 4,
    ) -> None:
        if candidate_multiplier < 1:
            raise ValueError("candidate_multiplier must be at least 1.")
        self.retriever = retriever
        self.reranker = reranker
        self.candidate_multiplier = candidate_multiplier

    def search(self, query: str, *, top_k: int = 5) -> list[RetrievalResult]:
        """Retrieve candidates and apply second-stage reranking."""
        if top_k <= 0:
            raise ValueError("top_k must be positive.")
        candidate_k = top_k * self.candidate_multiplier
        candidates = self.retriever.search(query, top_k=candidate_k)
        return self.reranker.rerank(query, candidates, top_k=top_k)
