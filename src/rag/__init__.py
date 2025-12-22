"""RAG (Retrieval-Augmented Generation) module for NPC memory system.

This module provides semantic search capabilities for NPC memories using:
- Ollama embeddings (nomic-embed-text) for semantic representation
- Qdrant vector database for efficient similarity search
- BM25 for keyword matching
- Hybrid search with RRF (Reciprocal Rank Fusion) for combining results
"""

from .embeddings import EmbeddingClient, OllamaEmbeddings
from .vector_store import VectorStore, QdrantVectorStore, InMemoryVectorStore
from .retriever import HybridRetriever, SearchResult

__all__ = [
    "EmbeddingClient",
    "OllamaEmbeddings",
    "VectorStore",
    "QdrantVectorStore",
    "InMemoryVectorStore",
    "HybridRetriever",
    "SearchResult",
]
