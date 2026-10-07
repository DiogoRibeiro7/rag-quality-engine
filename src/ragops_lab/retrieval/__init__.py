"""Retrieval package."""

from .evaluation import RetrievalEvaluationReport, RetrievalGoldenExample, evaluate_retrieval
from .hybrid import HybridRetriever
from .index import LocalVectorIndex
from .lexical import BM25Retriever
from .reranking import LexicalOverlapReranker, Reranker, Retriever, RetrieveThenRerank
from .tokenizer import tokenize
from .vector import (
    EmbeddingClient,
    FakeEmbeddingClient,
    SentenceTransformerEmbeddingClient,
    VectorRetriever,
    build_embedding_client,
)

__all__ = [
    "tokenize",
    "BM25Retriever",
    "EmbeddingClient",
    "FakeEmbeddingClient",
    "SentenceTransformerEmbeddingClient",
    "VectorRetriever",
    "build_embedding_client",
    "HybridRetriever",
    "LocalVectorIndex",
    "Retriever",
    "Reranker",
    "LexicalOverlapReranker",
    "RetrieveThenRerank",
    "RetrievalGoldenExample",
    "RetrievalEvaluationReport",
    "evaluate_retrieval",
]
