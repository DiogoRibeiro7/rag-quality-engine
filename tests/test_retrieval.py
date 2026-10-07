from __future__ import annotations

from pathlib import Path

from ragops_lab.config import EmbeddingSettings
from ragops_lab.domain import DocumentChunk, RetrievalResult
from ragops_lab.retrieval import (
    BM25Retriever,
    FakeEmbeddingClient,
    HybridRetriever,
    LexicalOverlapReranker,
    LocalVectorIndex,
    RetrievalGoldenExample,
    RetrieveThenRerank,
    VectorRetriever,
    build_embedding_client,
    evaluate_retrieval,
    tokenize,
)


def _chunks() -> list[DocumentChunk]:
    return [
        DocumentChunk(
            chunk_id="apollo:0",
            document_id="apollo",
            text="Apollo 11 landed on the Moon in 1969.",
            start_offset=0,
            end_offset=38,
            token_count=8,
        ),
        DocumentChunk(
            chunk_id="metrics:0",
            document_id="metrics",
            text="Faithfulness and citation support are critical RAG metrics.",
            start_offset=0,
            end_offset=60,
            token_count=9,
        ),
    ]


def test_tokenizer_normalizes_terms() -> None:
    assert tokenize("Apollo-11, MOON!") == ["apollo", "11", "moon"]


def test_lexical_vector_and_hybrid_retrieval_rank_expected_chunk() -> None:
    chunks = _chunks()
    lexical = BM25Retriever(chunks)
    vector = VectorRetriever(
        chunks, FakeEmbeddingClient(["apollo", "moon", "faithfulness", "citation"])
    )
    hybrid = HybridRetriever(lexical, vector, lexical_weight=0.7, vector_weight=0.3)

    assert lexical.search("moon mission", top_k=1)[0].chunk.chunk_id == "apollo:0"
    assert vector.search("citation support", top_k=1)[0].chunk.chunk_id == "metrics:0"
    assert hybrid.search("moon mission", top_k=1)[0].chunk.chunk_id == "apollo:0"


def test_fake_embedding_client_reuses_document_vocabulary_for_queries() -> None:
    vector = VectorRetriever(_chunks(), FakeEmbeddingClient())

    results = vector.search("citation support", top_k=1)

    assert results[0].chunk.chunk_id == "metrics:0"


def test_local_vector_index_round_trips_and_reloads_retriever(tmp_path: Path) -> None:
    index_path = tmp_path / "vector_index.json"
    index = LocalVectorIndex.build(_chunks())

    index.save(index_path)
    loaded = LocalVectorIndex.load(index_path)
    results = loaded.as_retriever().search("citation support", top_k=1)

    assert loaded.embedding_model == "fake-bow"
    assert loaded.embedding_provider == "fake"
    assert loaded.vocabulary
    assert results[0].chunk.chunk_id == "metrics:0"


def test_build_embedding_client_uses_fake_default() -> None:
    client = build_embedding_client(EmbeddingSettings())

    assert isinstance(client, FakeEmbeddingClient)


def test_vector_retriever_rejects_mismatched_precomputed_vectors() -> None:
    try:
        VectorRetriever(_chunks(), FakeEmbeddingClient(), chunk_vectors=[[1.0]])
    except ValueError as exc:
        assert "one vector per chunk" in str(exc)
    else:
        raise AssertionError("Expected mismatched vectors to fail.")


def test_retrieval_evaluation_computes_recall_and_mrr() -> None:
    report = evaluate_retrieval(
        BM25Retriever(_chunks()),
        [RetrievalGoldenExample(query="moon landing", relevant_chunk_ids=["apollo:0"])],
        top_k=2,
    )

    assert report.recall_at_k == 1.0
    assert report.mean_reciprocal_rank == 1.0


class _StaticRetriever:
    """Minimal deterministic retriever used to test fusion behavior."""

    def __init__(self, results: list[RetrievalResult]) -> None:
        self._results = results

    def search(self, query: str, *, top_k: int = 5) -> list[RetrievalResult]:
        del query
        return self._results[:top_k]


def test_rrf_combines_rankings_without_raw_score_calibration() -> None:
    first, second = _chunks()
    lexical_results = [
        RetrievalResult(
            chunk=first,
            score=1000.0,
            rank=1,
            retrieval_method="lexical",
            matched_terms=[],
        ),
        RetrievalResult(
            chunk=second,
            score=999.0,
            rank=2,
            retrieval_method="lexical",
            matched_terms=[],
        ),
    ]
    vector_results = [
        RetrievalResult(
            chunk=second,
            score=0.0002,
            rank=1,
            retrieval_method="vector",
            matched_terms=[],
        ),
        RetrievalResult(
            chunk=first,
            score=0.0001,
            rank=2,
            retrieval_method="vector",
            matched_terms=[],
        ),
    ]

    retriever = HybridRetriever(
        _StaticRetriever(lexical_results),  # type: ignore[arg-type]
        _StaticRetriever(vector_results),  # type: ignore[arg-type]
        fusion_strategy="rrf",
        rrf_k=60,
    )
    results = retriever.search("query", top_k=2)

    assert {result.chunk.chunk_id for result in results} == {"apollo:0", "metrics:0"}
    assert results[0].score == results[1].score
    assert [result.rank for result in results] == [1, 2]


def test_hybrid_retriever_rejects_invalid_fusion_strategy() -> None:
    chunks = _chunks()
    lexical = BM25Retriever(chunks)
    vector = VectorRetriever(chunks, FakeEmbeddingClient())

    try:
        HybridRetriever(lexical, vector, fusion_strategy="unknown")
    except ValueError as exc:
        assert "Unsupported fusion strategy" in str(exc)
    else:
        raise AssertionError("Expected invalid fusion strategy to fail.")


def test_hybrid_retriever_rejects_non_positive_rrf_k() -> None:
    chunks = _chunks()
    lexical = BM25Retriever(chunks)
    vector = VectorRetriever(chunks, FakeEmbeddingClient())

    try:
        HybridRetriever(lexical, vector, fusion_strategy="rrf", rrf_k=0)
    except ValueError as exc:
        assert "rrf_k must be positive" in str(exc)
    else:
        raise AssertionError("Expected non-positive rrf_k to fail.")


def test_lexical_overlap_reranker_can_change_candidate_order() -> None:
    first, second = _chunks()
    candidates = [
        RetrievalResult(
            chunk=first,
            score=10.0,
            rank=1,
            retrieval_method="hybrid",
            matched_terms=[],
        ),
        RetrievalResult(
            chunk=second,
            score=1.0,
            rank=2,
            retrieval_method="hybrid",
            matched_terms=[],
        ),
    ]

    results = LexicalOverlapReranker().rerank(
        "citation support",
        candidates,
        top_k=2,
    )

    assert results[0].chunk.chunk_id == "metrics:0"
    assert results[0].rank == 1
    assert results[0].retrieval_method == "reranked-hybrid"
    assert results[0].matched_terms == ["citation", "support"]


class _RecordingRetriever:
    """Retriever test double that records the requested candidate count."""

    def __init__(self, results: list[RetrievalResult]) -> None:
        self.results = results
        self.requested_top_k: int | None = None

    def search(self, query: str, *, top_k: int = 5) -> list[RetrievalResult]:
        del query
        self.requested_top_k = top_k
        return self.results[:top_k]


def test_retrieve_then_rerank_widens_candidate_pool() -> None:
    first, second = _chunks()
    base_results = [
        RetrievalResult(
            chunk=first,
            score=2.0,
            rank=1,
            retrieval_method="hybrid",
            matched_terms=[],
        ),
        RetrievalResult(
            chunk=second,
            score=1.0,
            rank=2,
            retrieval_method="hybrid",
            matched_terms=[],
        ),
    ]
    retriever = _RecordingRetriever(base_results)
    pipeline = RetrieveThenRerank(
        retriever,
        LexicalOverlapReranker(),
        candidate_multiplier=3,
    )

    results = pipeline.search("citation support", top_k=1)

    assert retriever.requested_top_k == 3
    assert len(results) == 1
    assert results[0].chunk.chunk_id == "metrics:0"


def test_retrieve_then_rerank_rejects_invalid_candidate_multiplier() -> None:
    retriever = _RecordingRetriever([])

    try:
        RetrieveThenRerank(retriever, LexicalOverlapReranker(), candidate_multiplier=0)
    except ValueError as exc:
        assert "candidate_multiplier" in str(exc)
    else:
        raise AssertionError("Expected invalid candidate multiplier to fail.")
