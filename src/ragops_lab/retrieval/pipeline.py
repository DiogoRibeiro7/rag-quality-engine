"""Helpers for composing configured retrieval pipelines."""

from __future__ import annotations

from ragops_lab.config import RetrievalProfile
from ragops_lab.domain import RetrievalResult

from .reranking import (
    EmbeddingSimilarityReranker,
    LexicalOverlapReranker,
    Reranker,
    Retriever,
    RetrieveThenRerank,
)
from .vector import EmbeddingClient


def apply_profile_reranking(
    retriever: Retriever,
    profile: RetrievalProfile,
    embedding_client: EmbeddingClient | None = None,
) -> Retriever:
    """Wrap a retriever with second-stage reranking when enabled."""
    if not profile.rerank:
        return retriever
    reranker: Reranker
    if profile.reranker_strategy == "lexical":
        reranker = LexicalOverlapReranker()
    else:
        if embedding_client is None:
            raise ValueError("Embedding reranking requires an embedding client.")
        reranker = EmbeddingSimilarityReranker(embedding_client)
    return RetrieveThenRerank(
        retriever,
        reranker,
        candidate_multiplier=profile.rerank_candidate_multiplier,
    )


def search_with_profile(
    retriever: Retriever,
    profile: RetrievalProfile,
    query: str,
    embedding_client: EmbeddingClient | None = None,
) -> list[RetrievalResult]:
    """Execute configured retrieval and optional reranking."""
    configured = apply_profile_reranking(
        retriever,
        profile,
        embedding_client=embedding_client,
    )
    return configured.search(query, top_k=profile.top_k)
