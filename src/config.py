"""Central configuration for NPC Service."""

import os
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    # Service
    service_name: str = "NPC Service"
    service_version: str = "0.2.0"
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

    # Memory - Redis (optional, falls back to in-memory)
    redis_url: Optional[str] = Field(default=None, alias="REDIS_URL")

    # Memory - Vector DB (optional, falls back to local)
    qdrant_url: Optional[str] = Field(default=None, alias="QDRANT_URL")
    qdrant_collection: str = Field(default="npc_memories", alias="QDRANT_COLLECTION")

    # Limits
    max_context_tokens: int = 4096
    max_response_tokens: int = 500
    max_memory_items: int = 100
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
            },
        }


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get settings instance (for dependency injection)."""
    return settings
