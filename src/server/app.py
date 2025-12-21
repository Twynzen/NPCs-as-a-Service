"""FastAPI application for NPC Service."""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from src.config import settings
from src.engine.npc_engine import NPCEngine
from src.llm.ollama_client import OllamaClient

# Global engine instance
engine: NPCEngine = None


def get_engine() -> NPCEngine:
    """Get the global NPC engine instance."""
    global engine
    if engine is None:
        raise RuntimeError("Engine not initialized. Server not started properly.")
    return engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup and shutdown."""
    global engine

    # Startup
    print(f"🚀 Starting {settings.service_name} v{settings.service_version}")

    # Initialize LLM client
    llm = OllamaClient(
        base_url=settings.ollama_url,
        model=settings.ollama_model,
    )

    # Check Ollama availability
    if llm.is_available():
        print(f"✓ Connected to Ollama at {settings.ollama_url}")
        if llm.model_exists():
            print(f"✓ Model '{settings.ollama_model}' available")
        else:
            print(f"⚠ Model '{settings.ollama_model}' not found. Pull it with: ollama pull {settings.ollama_model}")
    else:
        print(f"⚠ Ollama not available at {settings.ollama_url}")
        print("  Start Ollama with: ollama serve")

    # Initialize engine
    engine = NPCEngine(llm_client=llm)
    print("✓ NPC Engine initialized")

    yield

    # Shutdown
    print("👋 Shutting down NPC Service")
    if llm:
        llm.close()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title=settings.service_name,
        description="AI-powered NPC conversation service with persistent memory",
        version=settings.service_version,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS middleware for browser access
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # In production: restrict this
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Import and register routes
    from src.server.routes import health, sessions, chat, characters
    app.include_router(health.router, prefix="/v1", tags=["health"])
    app.include_router(sessions.router, prefix="/v1/sessions", tags=["sessions"])
    app.include_router(chat.router, prefix="/v1", tags=["chat"])
    app.include_router(characters.router, prefix="/v1/characters", tags=["characters"])

    # Serve playground static files if they exist
    playground_dir = Path(__file__).parent.parent.parent / "playground"
    if playground_dir.exists():
        app.mount("/playground", StaticFiles(directory=str(playground_dir), html=True), name="playground")

    @app.get("/", include_in_schema=False)
    async def root():
        """Root endpoint with service info."""
        return {
            "service": settings.service_name,
            "version": settings.service_version,
            "docs": "/docs",
            "playground": "/playground",
            "health": "/v1/health",
        }

    return app


# Create app instance
app = create_app()
