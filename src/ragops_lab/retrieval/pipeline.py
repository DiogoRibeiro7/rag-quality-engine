"""Helpers for composing configured retrieval pipelines."""

from __future__ import annotations

from ragops_lab.config import RetrievalProfile
from ragops_lab.domain import RetrievalResult

from .reranking import LexicalOverlapReranker, Retriever, RetrieveThenRerank


def apply_profile_reranking(
    retriever: Retriever,
    profile: RetrievalProfile,
) -> Retriever:
    """Wrap a retriever with second-stage reranking when enabled."""
    if not profile.rerank:
        return retriever
    return RetrieveThenRerank(
        retriever,
        LexicalOverlapReranker(),
        candidate_multiplier=profile.rerank_candidate_multiplier,
    )


def search_with_profile(
    retriever: Retriever,
    profile: RetrievalProfile,
    query: str,
) -> list[RetrievalResult]:
    """Execute configured retrieval and optional reranking."""
    configured = apply_profile_reranking(retriever, profile)
    return configured.search(query, top_k=profile.top_k)
