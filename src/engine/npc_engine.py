"""NPC Engine - Main orchestrator for NPC interactions.

Coordinates memory, context building, LLM generation, and action execution.
This is the core runtime for the NPC service.
"""

from typing import Optional, AsyncGenerator
from datetime import datetime
from pydantic import BaseModel, Field
import uuid

from src.characters.loader import Character, load_character
from src.memory.manager import MemoryManager
from src.engine.context_builder import ContextBuilder
from src.engine.action_parser import ActionParser, NPCAction
from src.llm.ollama_client import OllamaClient, Message


class NPCResponse(BaseModel):
    """Complete response from NPC engine."""
    session_id: str
    npc_id: str
    content: str
    actions: list[dict] = Field(default_factory=list)
    memory_updated: bool = False
    turn_number: int = 0
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class Session(BaseModel):
    """Active conversation session."""
    id: str
    npc_id: str
    player_id: str
    created_at: datetime = Field(default_factory=datetime.now)
    last_activity: datetime = Field(default_factory=datetime.now)
    turn_count: int = 0
    ended: bool = False


class NPCEngine:
    """
    Main NPC interaction engine.

    Handles the complete flow:
    1. Load/create session
    2. Build context with memory
    3. Generate response
    4. Parse and execute actions
    5. Update memory
    """

    def __init__(
        self,
        llm_client: Optional[OllamaClient] = None,
        memory_manager: Optional[MemoryManager] = None,
    ):
        self.llm = llm_client
        self.memory = memory_manager or MemoryManager()
        self.action_parser = ActionParser()

        # Session storage (in production: Redis)
        self._sessions: dict[str, Session] = {}
        self._characters: dict[str, Character] = {}
        self._context_builders: dict[str, ContextBuilder] = {}

    def set_llm(self, client: OllamaClient) -> None:
        """Set or update the LLM client."""
        self.llm = client

    # === Character Management ===

    def load_character(self, name: str) -> Character:
        """Load a character by name."""
        if name not in self._characters:
            self._characters[name] = load_character(name)
        return self._characters[name]

    def get_context_builder(self, npc_id: str) -> ContextBuilder:
        """Get or create context builder for a character."""
        if npc_id not in self._context_builders:
            character = self.load_character(npc_id)
            self._context_builders[npc_id] = ContextBuilder(
                character=character,
                action_parser=self.action_parser,
            )
        return self._context_builders[npc_id]

    # === Session Management ===

    def create_session(self, npc_id: str, player_id: str) -> Session:
        """Create a new conversation session."""
        # Ensure character exists
        self.load_character(npc_id)

        session = Session(
            id=f"sess_{uuid.uuid4().hex[:12]}",
            npc_id=npc_id,
            player_id=player_id,
        )

        self._sessions[session.id] = session
        return session

    def get_session(self, session_id: str) -> Optional[Session]:
        """Get an existing session."""
        return self._sessions.get(session_id)

    def end_session(self, session_id: str) -> Optional[str]:
        """End a session and archive the conversation."""
        session = self._sessions.get(session_id)
        if not session:
            return None

        # Archive conversation to memory
        summary = self.memory.end_session(
            session.npc_id,
            session.player_id,
            session_id,
        )

        session.ended = True
        return summary

    def list_sessions(self, player_id: Optional[str] = None) -> list[Session]:
        """List active sessions, optionally filtered by player."""
        sessions = [s for s in self._sessions.values() if not s.ended]
        if player_id:
            sessions = [s for s in sessions if s.player_id == player_id]
        return sessions

    # === Chat ===

    def chat(
        self,
        session_id: str,
        user_message: str,
    ) -> NPCResponse:
        """
        Process a chat message and generate NPC response.

        This is the main interaction method.
        """
        if not self.llm:
            raise RuntimeError("LLM client not configured")

        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"Session not found: {session_id}")

        if session.ended:
            raise ValueError(f"Session has ended: {session_id}")

        # Update session activity
        session.last_activity = datetime.now()
        session.turn_count += 1

        # Get context builder
        context_builder = self.get_context_builder(session.npc_id)

        # Get memory context
        memory_context = self.memory.get_memory_context(
            npc_id=session.npc_id,
            player_id=session.player_id,
            session_id=session_id,
            query=user_message,
        )

        # Get conversation history
        recall = self.memory.get_recall(
            session.npc_id,
            session.player_id,
            session_id,
        )

        # Build context
        built = context_builder.build(
            memory_context=memory_context,
            user_message=user_message,
            conversation_history=recall.to_messages(),
        )

        # Generate response
        messages = [Message(role="system", content=built.system_prompt)]
        for msg in built.messages:
            messages.append(Message(role=msg["role"], content=msg["content"]))

        raw_response = self.llm.chat(
            messages=messages,
            temperature=0.7,
            max_tokens=500,
        )

        # Parse actions
        clean_response, actions = self.action_parser.parse(raw_response)

        # Store turn in recall memory
        self.memory.add_turn(
            npc_id=session.npc_id,
            player_id=session.player_id,
            session_id=session_id,
            role="user",
            content=user_message,
        )
        self.memory.add_turn(
            npc_id=session.npc_id,
            player_id=session.player_id,
            session_id=session_id,
            role="assistant",
            content=clean_response,
            actions=[a.to_dict() for a in actions],
        )

        # Execute memory-related actions
        memory_updated = self._execute_memory_actions(
            session.npc_id,
            session.player_id,
            actions,
        )

        return NPCResponse(
            session_id=session_id,
            npc_id=session.npc_id,
            content=clean_response,
            actions=[a.to_dict() for a in actions],
            memory_updated=memory_updated,
            turn_number=session.turn_count,
        )

    async def chat_stream(
        self,
        session_id: str,
        user_message: str,
    ) -> AsyncGenerator[str, None]:
        """
        Stream a chat response token by token.

        Yields content chunks as they're generated.
        Actions are parsed from the complete response at the end.
        """
        if not self.llm:
            raise RuntimeError("LLM client not configured")

        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"Session not found: {session_id}")

        # Update session
        session.last_activity = datetime.now()
        session.turn_count += 1

        # Build context
        context_builder = self.get_context_builder(session.npc_id)
        memory_context = self.memory.get_memory_context(
            npc_id=session.npc_id,
            player_id=session.player_id,
            session_id=session_id,
            query=user_message,
        )

        recall = self.memory.get_recall(
            session.npc_id,
            session.player_id,
            session_id,
        )

        built = context_builder.build(
            memory_context=memory_context,
            user_message=user_message,
            conversation_history=recall.to_messages(),
        )

        # Stream generation
        messages = [Message(role="system", content=built.system_prompt)]
        for msg in built.messages:
            messages.append(Message(role=msg["role"], content=msg["content"]))

        full_response = ""
        for chunk in self.llm.chat_stream(messages):
            full_response += chunk
            yield chunk

        # Parse and handle actions after streaming completes
        clean_response, actions = self.action_parser.parse(full_response)

        # Store in memory
        self.memory.add_turn(
            npc_id=session.npc_id,
            player_id=session.player_id,
            session_id=session_id,
            role="user",
            content=user_message,
        )
        self.memory.add_turn(
            npc_id=session.npc_id,
            player_id=session.player_id,
            session_id=session_id,
            role="assistant",
            content=clean_response,
            actions=[a.to_dict() for a in actions],
        )

        # Execute memory actions
        self._execute_memory_actions(
            session.npc_id,
            session.player_id,
            actions,
        )

    def _execute_memory_actions(
        self,
        npc_id: str,
        player_id: str,
        actions: list[NPCAction],
    ) -> bool:
        """Execute actions that modify memory."""
        memory_updated = False

        for action in actions:
            if action.name == "remember":
                self.memory.add_memory(
                    npc_id=npc_id,
                    player_id=player_id,
                    content=action.parameters.get("fact", ""),
                    importance=float(action.parameters.get("importance", 5)),
                    memory_type="observation",
                )
                memory_updated = True

            elif action.name == "update_relationship":
                delta = int(action.parameters.get("delta", 0))
                self.memory.update_core(
                    npc_id=npc_id,
                    player_id=player_id,
                    updates={"relationship_delta": delta},
                )
                memory_updated = True

            elif action.name == "set_emotion":
                emotion = action.parameters.get("emotion", "neutral")
                self.memory.update_core(
                    npc_id=npc_id,
                    player_id=player_id,
                    updates={"emotion": emotion},
                )
                memory_updated = True

        return memory_updated

    # === First Message ===

    def get_first_message(self, session_id: str) -> Optional[NPCResponse]:
        """Get the NPC's opening message for a new conversation."""
        session = self.get_session(session_id)
        if not session:
            return None

        context_builder = self.get_context_builder(session.npc_id)
        first_msg = context_builder.build_first_message()

        if not first_msg:
            return None

        # Store in memory
        self.memory.add_turn(
            npc_id=session.npc_id,
            player_id=session.player_id,
            session_id=session_id,
            role="assistant",
            content=first_msg,
        )

        return NPCResponse(
            session_id=session_id,
            npc_id=session.npc_id,
            content=first_msg,
            turn_number=0,
        )

    # === Memory Access ===

    def get_memory_stats(self, npc_id: str, player_id: str) -> dict:
        """Get memory statistics for debugging."""
        return self.memory.get_stats(npc_id, player_id)

    def inject_memory(
        self,
        npc_id: str,
        player_id: str,
        content: str,
        importance: float = 5.0,
    ) -> None:
        """Manually inject a memory (for game events)."""
        self.memory.add_memory(
            npc_id=npc_id,
            player_id=player_id,
            content=content,
            importance=importance,
            memory_type="fact",
        )
