"""Conversation simulation loop between NPC and customer."""

from typing import Optional
from pydantic import BaseModel, Field
from datetime import datetime

from src.characters.loader import Character
from src.customers.generator import CustomerProfile, get_customer_system_prompt
from src.llm.ollama_client import OllamaClient, Message


class ConversationTurn(BaseModel):
    """A single turn in the conversation."""
    speaker: str  # "customer" or "npc"
    message: str
    turn_number: int


class ConversationResult(BaseModel):
    """Complete conversation result."""
    id: str
    character: str
    customer_profile: dict
    turns: list[ConversationTurn]
    ended_naturally: bool
    total_turns: int
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class ConversationSimulator:
    """Simulates conversations between an NPC and generated customers."""

    def __init__(
        self,
        llm_client: OllamaClient,
        max_turns: int = 20,
        temperature: float = 0.7,
    ):
        self.llm = llm_client
        self.max_turns = max_turns
        self.temperature = temperature
        self._conversation_counter = 0

    def simulate(
        self,
        character: Character,
        customer: CustomerProfile,
    ) -> ConversationResult:
        """
        Run a complete conversation simulation.

        Args:
            character: The NPC character
            customer: The customer profile

        Returns:
            ConversationResult with full dialogue
        """
        self._conversation_counter += 1
        conversation_id = f"conv_{self._conversation_counter:05d}"

        turns: list[ConversationTurn] = []
        ended_naturally = False

        # Build system prompts
        npc_system = character.build_system_prompt()
        if character.example_dialogues:
            npc_system += "\n" + character.get_few_shot_examples()

        customer_system = get_customer_system_prompt(customer)

        # Customer starts the conversation
        customer_response = self._generate_customer_turn(
            customer_system,
            [],
            is_opening=True,
            opening_line=customer.opening_line,
        )

        turns.append(ConversationTurn(
            speaker="customer",
            message=customer_response,
            turn_number=0,
        ))

        # Conversation loop
        for turn_num in range(1, self.max_turns):
            # NPC responds
            npc_response = self._generate_npc_turn(
                npc_system,
                character.name,
                turns,
            )

            turns.append(ConversationTurn(
                speaker="npc",
                message=npc_response,
                turn_number=turn_num,
            ))

            # Check for natural ending from NPC
            if self._check_conversation_end(npc_response):
                ended_naturally = True
                break

            # Customer responds
            turn_num += 1
            if turn_num >= self.max_turns:
                break

            customer_response = self._generate_customer_turn(
                customer_system,
                turns,
            )

            turns.append(ConversationTurn(
                speaker="customer",
                message=customer_response,
                turn_number=turn_num,
            ))

            # Check for natural ending from customer
            if self._check_conversation_end(customer_response):
                ended_naturally = True
                break

        return ConversationResult(
            id=conversation_id,
            character=character.name,
            customer_profile=customer.model_dump(),
            turns=turns,
            ended_naturally=ended_naturally,
            total_turns=len(turns),
        )

    def _generate_npc_turn(
        self,
        system_prompt: str,
        npc_name: str,
        history: list[ConversationTurn],
    ) -> str:
        """Generate NPC response."""
        messages = [Message(role="system", content=system_prompt)]

        for turn in history:
            if turn.speaker == "customer":
                messages.append(Message(role="user", content=turn.message))
            else:
                messages.append(Message(role="assistant", content=turn.message))

        response = self.llm.chat(
            messages,
            temperature=self.temperature,
            max_tokens=300,
            stop=["\nCustomer:", "\n\nCustomer:", "[END]"],
        )

        return self._clean_response(response, npc_name)

    def _generate_customer_turn(
        self,
        system_prompt: str,
        history: list[ConversationTurn],
        is_opening: bool = False,
        opening_line: Optional[str] = None,
    ) -> str:
        """Generate customer response."""
        # For opening, we can use the provided line with slight variation
        if is_opening and opening_line:
            # Let the LLM add some natural variation
            variation_prompt = f"""You are a customer at a bar. Say this opening line with slight natural variation:

Original: {opening_line}

Your version (keep it similar but natural):"""

            response = self.llm.generate(
                variation_prompt,
                temperature=0.8,
                max_tokens=150,
            )
            return self._clean_response(response, "Customer")

        messages = [Message(role="system", content=system_prompt)]

        for turn in history:
            if turn.speaker == "customer":
                messages.append(Message(role="assistant", content=turn.message))
            else:
                messages.append(Message(role="user", content=turn.message))

        response = self.llm.chat(
            messages,
            temperature=self.temperature,
            max_tokens=200,
            stop=[f"\n{history[0].message.split()[0] if history else 'Zamir'}:", "\n\nZamir:", "[END]"],
        )

        return self._clean_response(response, "Customer")

    def _clean_response(self, response: str, speaker: str) -> str:
        """Clean up the generated response."""
        response = response.strip()

        # Remove any speaker prefixes the model might add
        prefixes_to_remove = [
            f"{speaker}:",
            "Customer:",
            "Zamir:",
            "NPC:",
            "Bartender:",
        ]
        for prefix in prefixes_to_remove:
            if response.startswith(prefix):
                response = response[len(prefix):].strip()

        # Remove markdown code blocks if present
        if response.startswith("```"):
            lines = response.split("\n")
            response = "\n".join(
                line for line in lines
                if not line.startswith("```")
            ).strip()

        return response

    def _check_conversation_end(self, message: str) -> bool:
        """Check if the conversation has reached a natural end."""
        end_indicators = [
            "[END]",
            "[end]",
            "*leaves the bar*",
            "*walks away*",
            "*exits*",
            "*departs*",
            "*turns to leave*",
            "Goodbye",
            "See you around",
            "I should go",
            "I'll be on my way",
            "Time for me to leave",
            "*finishes drink and stands*",
            "*pays and leaves*",
        ]

        message_lower = message.lower()
        return any(indicator.lower() in message_lower for indicator in end_indicators)
