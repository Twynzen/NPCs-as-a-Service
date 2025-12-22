"""Central configuration for NPC Service."""

import os
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    # Service
    service_name: str = "NPC Service"
    service_version: str = "0.3.0"  # Updated for RAG support
    debug: bool = Field(default=False, alias="DEBUG")

    # Server
    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8000, alias="PORT")

    # LLM - Ollama (local)
    ollama_url: str = Field(default="http://localhost:11434", alias="OLLAMA_URL")
    ollama_model: str = Field(default="phi3.5", alias="OLLAMA_MODEL")

    # LLM - OpenAI (optional, for complex narratives)
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o-mini", alias="OPENAI_MODEL")

    # RAG - Embeddings
    embedding_model: str = Field(
        default="nomic-embed-text",
        alias="EMBEDDING_MODEL",
        description="Ollama embedding model (nomic-embed-text: 768d, mxbai-embed-large: 1024d)"
    )
    embedding_dimensions: int = Field(
        default=768,
        alias="EMBEDDING_DIMENSIONS",
        description="Embedding vector dimensions (must match model)"
    )
    embedding_batch_size: int = Field(
        default=32,
        alias="EMBEDDING_BATCH_SIZE",
        description="Batch size for embedding generation"
    )

    # RAG - Hybrid Search
    use_hybrid_search: bool = Field(
        default=True,
        alias="USE_HYBRID_SEARCH",
        description="Enable BM25 + Vector hybrid search"
    )
    rrf_k: int = Field(
        default=60,
        alias="RRF_K",
        description="RRF constant (higher = less emphasis on top ranks)"
    )
    retrieval_top_k: int = Field(
        default=5,
        alias="RETRIEVAL_TOP_K",
        description="Number of memories to retrieve per query"
    )

    # Memory - Redis (optional, falls back to in-memory)
    redis_url: Optional[str] = Field(default=None, alias="REDIS_URL")

    # Memory - Vector DB (Qdrant)
    qdrant_url: Optional[str] = Field(default=None, alias="QDRANT_URL")
    qdrant_collection: str = Field(default="npc_memories", alias="QDRANT_COLLECTION")
    qdrant_api_key: Optional[str] = Field(default=None, alias="QDRANT_API_KEY")

    # Memory - Recency Decay
    recency_decay_factor: float = Field(
        default=0.995,
        alias="RECENCY_DECAY_FACTOR",
        description="Decay factor per hour (0.995^hours). Half-life ~138 hours"
    )

    # Memory - Importance Scoring
    importance_weight: float = Field(
        default=0.3,
        alias="IMPORTANCE_WEIGHT",
        description="Weight for importance in retrieval scoring"
    )
    recency_weight: float = Field(
        default=0.3,
        alias="RECENCY_WEIGHT",
        description="Weight for recency in retrieval scoring"
    )
    relevance_weight: float = Field(
        default=0.4,
        alias="RELEVANCE_WEIGHT",
        description="Weight for relevance (semantic similarity)"
    )

    # Limits
    max_context_tokens: int = 4096
    max_response_tokens: int = 500
    max_memory_items: int = 100
    max_archival_memories: int = 10000
    session_timeout_minutes: int = 60

    # Paths
    characters_dir: str = "src/characters/templates"
    output_dir: str = "output"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

    def get_llm_config(self) -> dict:
        """Get LLM configuration based on available providers."""
        config = {
            "local": {
                "provider": "ollama",
                "url": self.ollama_url,
                "model": self.ollama_model,
            }
        }

        if self.openai_api_key:
            config["cloud"] = {
                "provider": "openai",
                "api_key": self.openai_api_key,
                "model": self.openai_model,
            }

        return config

    def get_memory_config(self) -> dict:
        """Get memory configuration based on available backends."""
        return {
            "redis": {
                "available": self.redis_url is not None,
                "url": self.redis_url,
            },
            "qdrant": {
                "available": self.qdrant_url is not None,
                "url": self.qdrant_url,
                "collection": self.qdrant_collection,
                "api_key": self.qdrant_api_key,
            },
        }

    def get_rag_config(self) -> dict:
        """Get RAG configuration for embeddings and retrieval."""
        return {
            "embeddings": {
                "model": self.embedding_model,
                "dimensions": self.embedding_dimensions,
                "batch_size": self.embedding_batch_size,
                "provider": "ollama",
                "url": self.ollama_url,
            },
            "retrieval": {
                "use_hybrid": self.use_hybrid_search,
                "rrf_k": self.rrf_k,
                "top_k": self.retrieval_top_k,
            },
            "scoring": {
                "importance_weight": self.importance_weight,
                "recency_weight": self.recency_weight,
                "relevance_weight": self.relevance_weight,
                "recency_decay": self.recency_decay_factor,
            },
            "vector_store": {
                "available": self.qdrant_url is not None,
                "url": self.qdrant_url,
                "collection": self.qdrant_collection,
                "api_key": self.qdrant_api_key,
            },
        }


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get settings instance (for dependency injection)."""
    return settings
