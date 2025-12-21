"""Session management endpoints."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from src.server.app import get_engine

router = APIRouter()


class CreateSessionRequest(BaseModel):
    """Request to create a new session."""
    npc_id: str
    player_id: str


class SessionResponse(BaseModel):
    """Session information response."""
    id: str
    npc_id: str
    player_id: str
    created_at: str
    last_activity: str
    turn_count: int
    ended: bool
    first_message: Optional[str] = None


class SessionListResponse(BaseModel):
    """List of sessions."""
    sessions: list[SessionResponse]
    total: int


@router.post("", response_model=SessionResponse, status_code=201)
async def create_session(request: CreateSessionRequest):
    """
    Create a new conversation session.

    This initializes a conversation between a player and an NPC.
    Returns the session ID needed for all subsequent chat requests.

    Example:
    ```
    POST /v1/sessions
    {"npc_id": "zamir", "player_id": "player_123"}
    ```
    """
    engine = get_engine()

    try:
        session = engine.create_session(
            npc_id=request.npc_id,
            player_id=request.player_id,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    # Get first message if available
    first_msg = engine.get_first_message(session.id)

    return SessionResponse(
        id=session.id,
        npc_id=session.npc_id,
        player_id=session.player_id,
        created_at=session.created_at.isoformat(),
        last_activity=session.last_activity.isoformat(),
        turn_count=session.turn_count,
        ended=session.ended,
        first_message=first_msg.content if first_msg else None,
    )


@router.get("/{session_id}", response_model=SessionResponse)
async def get_session(session_id: str):
    """
    Get session information.

    Returns current state of the session including turn count.
    """
    engine = get_engine()
    session = engine.get_session(session_id)

    if not session:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")

    return SessionResponse(
        id=session.id,
        npc_id=session.npc_id,
        player_id=session.player_id,
        created_at=session.created_at.isoformat(),
        last_activity=session.last_activity.isoformat(),
        turn_count=session.turn_count,
        ended=session.ended,
    )


@router.delete("/{session_id}")
async def end_session(session_id: str):
    """
    End a conversation session.

    Archives the conversation to long-term memory and marks the session as ended.
    """
    engine = get_engine()
    session = engine.get_session(session_id)

    if not session:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")

    summary = engine.end_session(session_id)

    return {
        "session_id": session_id,
        "ended": True,
        "archived_summary": summary,
    }


@router.get("", response_model=SessionListResponse)
async def list_sessions(player_id: Optional[str] = None):
    """
    List active sessions.

    Optionally filter by player_id.
    """
    engine = get_engine()
    sessions = engine.list_sessions(player_id=player_id)

    return SessionListResponse(
        sessions=[
            SessionResponse(
                id=s.id,
                npc_id=s.npc_id,
                player_id=s.player_id,
                created_at=s.created_at.isoformat(),
                last_activity=s.last_activity.isoformat(),
                turn_count=s.turn_count,
                ended=s.ended,
            )
            for s in sessions
        ],
        total=len(sessions),
    )


@router.get("/{session_id}/memory")
async def get_session_memory(session_id: str):
    """
    Get memory state for a session.

    Useful for debugging and game integration.
    """
    engine = get_engine()
    session = engine.get_session(session_id)

    if not session:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")

    stats = engine.get_memory_stats(session.npc_id, session.player_id)

    return {
        "session_id": session_id,
        "npc_id": session.npc_id,
        "player_id": session.player_id,
        "memory": stats,
    }
