"""Health check endpoints."""

from fastapi import APIRouter
from pydantic import BaseModel

from src.config import settings
from src.server.app import get_engine
from src.characters.loader import list_characters

router = APIRouter()


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    ollama_connected: bool
    ollama_model: str
    available_characters: list[str]
    active_sessions: int


class ComponentStatus(BaseModel):
    """Detailed component status."""
    name: str
    status: str
    details: dict = {}


class DetailedHealthResponse(BaseModel):
    """Detailed health check response."""
    status: str
    version: str
    components: list[ComponentStatus]


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Basic health check.

    Returns service status and connected components.
    """
    engine = get_engine()

    # Check Ollama
    ollama_connected = False
    if engine.llm:
        ollama_connected = engine.llm.is_available()

    # Get available characters
    characters = list_characters()

    # Count active sessions
    active_sessions = len(engine.list_sessions())

    return HealthResponse(
        status="healthy" if ollama_connected else "degraded",
        version=settings.service_version,
        ollama_connected=ollama_connected,
        ollama_model=settings.ollama_model,
        available_characters=characters,
        active_sessions=active_sessions,
    )


@router.get("/health/detailed", response_model=DetailedHealthResponse)
async def detailed_health_check():
    """
    Detailed health check with component status.

    Useful for debugging and monitoring.
    """
    engine = get_engine()
    components = []

    # Ollama status
    ollama_status = "down"
    ollama_details = {}
    if engine.llm:
        if engine.llm.is_available():
            ollama_status = "up"
            ollama_details["url"] = settings.ollama_url
            ollama_details["model"] = settings.ollama_model
            ollama_details["model_available"] = engine.llm.model_exists()
            ollama_details["available_models"] = engine.llm.list_models()[:5]  # First 5
        else:
            ollama_status = "down"
            ollama_details["error"] = f"Cannot connect to {settings.ollama_url}"

    components.append(ComponentStatus(
        name="ollama",
        status=ollama_status,
        details=ollama_details,
    ))

    # Character loader status
    try:
        characters = list_characters()
        components.append(ComponentStatus(
            name="characters",
            status="up",
            details={"available": characters, "count": len(characters)},
        ))
    except Exception as e:
        components.append(ComponentStatus(
            name="characters",
            status="error",
            details={"error": str(e)},
        ))

    # Memory status
    components.append(ComponentStatus(
        name="memory",
        status="up",
        details={"type": "in-memory", "sessions": len(engine.list_sessions())},
    ))

    # Overall status
    overall = "healthy"
    if any(c.status == "down" for c in components):
        overall = "degraded"
    if all(c.status == "down" for c in components):
        overall = "unhealthy"

    return DetailedHealthResponse(
        status=overall,
        version=settings.service_version,
        components=components,
    )


@router.get("/models")
async def list_models():
    """List available LLM models."""
    engine = get_engine()

    if not engine.llm:
        return {"models": [], "error": "LLM client not configured"}

    if not engine.llm.is_available():
        return {"models": [], "error": "Ollama not available"}

    models = engine.llm.list_models()
    current = settings.ollama_model

    return {
        "current_model": current,
        "available_models": models,
        "current_available": any(current in m or m.startswith(f"{current}:") for m in models),
    }
