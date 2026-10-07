"""Hybrid retrieval."""

from __future__ import annotations

from ragops_lab.domain import RetrievalResult

from .lexical import BM25Retriever
from .vector import VectorRetriever

SUPPORTED_FUSION_STRATEGIES = frozenset({"weighted", "rrf"})


class HybridRetriever:
    """Combine lexical and vector retrieval results."""

    def __init__(
        self,
        lexical_retriever: BM25Retriever,
        vector_retriever: VectorRetriever,
        *,
        lexical_weight: float = 0.5,
        vector_weight: float = 0.5,
        fusion_strategy: str = "weighted",
        rrf_k: int = 60,
    ) -> None:
        normalized_strategy = fusion_strategy.strip().lower()
        if normalized_strategy not in SUPPORTED_FUSION_STRATEGIES:
            supported = ", ".join(sorted(SUPPORTED_FUSION_STRATEGIES))
            raise ValueError(
                f"Unsupported fusion strategy: {fusion_strategy}. Supported: {supported}."
            )
        if rrf_k <= 0:
            raise ValueError("rrf_k must be positive.")

        self.lexical_retriever = lexical_retriever
        self.vector_retriever = vector_retriever
        self.lexical_weight = lexical_weight
        self.vector_weight = vector_weight
        self.fusion_strategy = normalized_strategy
        self.rrf_k = rrf_k

    def search(self, query: str, *, top_k: int = 5) -> list[RetrievalResult]:
        """Search both retrievers and combine their results."""
        lexical_results = self.lexical_retriever.search(query, top_k=top_k * 2)
        vector_results = self.vector_retriever.search(query, top_k=top_k * 2)
        by_chunk_id = {
            result.chunk.chunk_id: result.chunk for result in [*lexical_results, *vector_results]
        }

        if self.fusion_strategy == "rrf":
            combined = self._rrf_scores(lexical_results, vector_results)
        else:
            combined = self._weighted_scores(lexical_results, vector_results)

        combined.sort(key=lambda item: (-item[1], item[0]))
        return [
            RetrievalResult(
                chunk=by_chunk_id[chunk_id],
                score=score,
                rank=rank,
                retrieval_method="hybrid",
                matched_terms=[],
            )
            for rank, (chunk_id, score) in enumerate(combined[:top_k], start=1)
        ]

    def _weighted_scores(
        self,
        lexical_results: list[RetrievalResult],
        vector_results: list[RetrievalResult],
    ) -> list[tuple[str, float]]:
        """Combine normalized retriever scores using fixed weights."""
        lexical_scores = {result.chunk.chunk_id: result.score for result in lexical_results}
        vector_scores = {result.chunk.chunk_id: result.score for result in vector_results}
        lexical_max = max(lexical_scores.values(), default=1.0)
        vector_max = max(vector_scores.values(), default=1.0)
        chunk_ids = set(lexical_scores) | set(vector_scores)

        combined: list[tuple[str, float]] = []
        for chunk_id in chunk_ids:
            lexical_score = lexical_scores.get(chunk_id, 0.0) / lexical_max
            vector_score = vector_scores.get(chunk_id, 0.0) / vector_max
            final_score = lexical_score * self.lexical_weight + vector_score * self.vector_weight
            if final_score > 0.0:
                combined.append((chunk_id, final_score))
        return combined

    def _rrf_scores(
        self,
        lexical_results: list[RetrievalResult],
        vector_results: list[RetrievalResult],
    ) -> list[tuple[str, float]]:
        """Combine rankings with Reciprocal Rank Fusion."""
        scores: dict[str, float] = {}
        for results in (lexical_results, vector_results):
            for rank, result in enumerate(results, start=1):
                chunk_id = result.chunk.chunk_id
                scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (self.rrf_k + rank)
        return list(scores.items())
