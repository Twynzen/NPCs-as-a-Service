"""Ollama API client for LLM interactions."""

import httpx
from typing import Optional, Generator
from pydantic import BaseModel


class Message(BaseModel):
    """Chat message."""
    role: str  # "system", "user", "assistant"
    content: str


class OllamaClient:
    """Client for Ollama API interactions."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "phi3.5",
        timeout: float = 120.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._client = httpx.Client(timeout=timeout)

    def chat(
        self,
        messages: list[Message],
        temperature: float = 0.7,
        max_tokens: int = 500,
        stop: Optional[list[str]] = None,
    ) -> str:
        """
        Send a chat completion request.

        Args:
            messages: List of conversation messages
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens to generate
            stop: Stop sequences

        Returns:
            Generated response text
        """
        payload = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        if stop:
            payload["options"]["stop"] = stop

        response = self._client.post(
            f"{self.base_url}/api/chat",
            json=payload,
        )
        response.raise_for_status()

        return response.json()["message"]["content"]

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 500,
    ) -> str:
        """
        Simple text generation (non-chat format).

        Args:
            prompt: The prompt text
            system: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        if system:
            payload["system"] = system

        response = self._client.post(
            f"{self.base_url}/api/generate",
            json=payload,
        )
        response.raise_for_status()

        return response.json()["response"]

    def chat_stream(
        self,
        messages: list[Message],
        temperature: float = 0.7,
        max_tokens: int = 500,
    ) -> Generator[str, None, None]:
        """
        Stream chat completion responses.

        Yields:
            Response text chunks
        """
        payload = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        with self._client.stream(
            "POST",
            f"{self.base_url}/api/chat",
            json=payload,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    import json
                    data = json.loads(line)
                    if "message" in data and "content" in data["message"]:
                        yield data["message"]["content"]

    def is_available(self) -> bool:
        """Check if Ollama server is available."""
        try:
            response = self._client.get(f"{self.base_url}/api/tags")
            return response.status_code == 200
        except httpx.RequestError:
            return False

    def list_models(self) -> list[str]:
        """List available models."""
        try:
            response = self._client.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            models = response.json().get("models", [])
            return [m["name"] for m in models]
        except httpx.RequestError:
            return []

    def model_exists(self, model_name: Optional[str] = None) -> bool:
        """Check if a specific model is available."""
        model = model_name or self.model
        available = self.list_models()
        # Check both exact match and partial match (without tag)
        return any(
            model == m or model == m.split(":")[0] or m.startswith(f"{model}:")
            for m in available
        )

    def close(self):
        """Close the HTTP client."""
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
