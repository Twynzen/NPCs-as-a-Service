"""Chat endpoints for NPC conversations."""

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
import json

from src.server.app import get_engine

router = APIRouter()


class ChatRequest(BaseModel):
    """Chat message request."""
    session_id: str
    message: str


class ChatResponse(BaseModel):
    """Chat response from NPC."""
    session_id: str
    npc_id: str
    content: str
    actions: list[dict]
    memory_updated: bool
    turn_number: int
    timestamp: str


class QuickChatRequest(BaseModel):
    """Quick chat without session management."""
    npc_id: str
    player_id: str
    message: str


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Send a message and get NPC response.

    Requires an active session. Use POST /v1/sessions first.

    Example:
    ```
    POST /v1/chat
    {"session_id": "sess_abc123", "message": "Hello, bartender!"}
    ```
    """
    engine = get_engine()

    try:
        response = engine.chat(
            session_id=request.session_id,
            user_message=request.message,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    return ChatResponse(
        session_id=response.session_id,
        npc_id=response.npc_id,
        content=response.content,
        actions=response.actions,
        memory_updated=response.memory_updated,
        turn_number=response.turn_number,
        timestamp=response.timestamp,
    )


@router.post("/chat/quick", response_model=ChatResponse)
async def quick_chat(request: QuickChatRequest):
    """
    One-shot chat without explicit session management.

    Creates or reuses a session automatically. Useful for simple integrations.

    Example:
    ```
    POST /v1/chat/quick
    {"npc_id": "zamir", "player_id": "player_123", "message": "Hello!"}
    ```
    """
    engine = get_engine()

    # Find or create session
    sessions = engine.list_sessions(player_id=request.player_id)
    session = next(
        (s for s in sessions if s.npc_id == request.npc_id and not s.ended),
        None,
    )

    if not session:
        try:
            session = engine.create_session(
                npc_id=request.npc_id,
                player_id=request.player_id,
            )
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))

    try:
        response = engine.chat(
            session_id=session.id,
            user_message=request.message,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    return ChatResponse(
        session_id=response.session_id,
        npc_id=response.npc_id,
        content=response.content,
        actions=response.actions,
        memory_updated=response.memory_updated,
        turn_number=response.turn_number,
        timestamp=response.timestamp,
    )


@router.get("/chat/stream/{session_id}")
async def chat_stream_get(session_id: str, message: str):
    """
    Stream NPC response via Server-Sent Events (GET).

    Useful for browser EventSource connections.

    Example:
    ```
    GET /v1/chat/stream/sess_abc123?message=Hello
    ```
    """
    engine = get_engine()
    session = engine.get_session(session_id)

    if not session:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")

    async def generate():
        try:
            async for chunk in engine.chat_stream(session_id, message):
                yield f"data: {json.dumps({'chunk': chunk})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@router.websocket("/chat/ws/{session_id}")
async def websocket_chat(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for real-time chat.

    Connect, then send JSON messages:
    ```json
    {"message": "Hello!"}
    ```

    Receive streamed responses:
    ```json
    {"type": "chunk", "content": "Hello"}
    {"type": "chunk", "content": " there!"}
    {"type": "done", "actions": [...], "turn": 1}
    ```
    """
    engine = get_engine()
    session = engine.get_session(session_id)

    if not session:
        await websocket.close(code=4004, reason="Session not found")
        return

    await websocket.accept()

    try:
        while True:
            # Receive message
            data = await websocket.receive_json()
            message = data.get("message", "")

            if not message:
                await websocket.send_json({"type": "error", "error": "Empty message"})
                continue

            # Stream response
            full_response = ""
            try:
                async for chunk in engine.chat_stream(session_id, message):
                    full_response += chunk
                    await websocket.send_json({
                        "type": "chunk",
                        "content": chunk,
                    })

                # Parse actions from complete response
                clean, actions = engine.action_parser.parse(full_response)

                await websocket.send_json({
                    "type": "done",
                    "content": clean,
                    "actions": [a.to_dict() for a in actions],
                    "turn": session.turn_count,
                })

            except Exception as e:
                await websocket.send_json({
                    "type": "error",
                    "error": str(e),
                })

    except WebSocketDisconnect:
        pass


@router.post("/memory/inject")
async def inject_memory(
    npc_id: str,
    player_id: str,
    content: str,
    importance: float = 5.0,
):
    """
    Inject a memory into an NPC's knowledge of a player.

    Useful for game events that the NPC should "know" about.

    Example:
    ```
    POST /v1/memory/inject?npc_id=zamir&player_id=player_123
    {"content": "This player helped defend the bar from raiders", "importance": 8}
    ```
    """
    engine = get_engine()

    engine.inject_memory(
        npc_id=npc_id,
        player_id=player_id,
        content=content,
        importance=importance,
    )

    return {
        "status": "injected",
        "npc_id": npc_id,
        "player_id": player_id,
        "content": content,
        "importance": importance,
    }
