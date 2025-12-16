"""Evaluate conversation quality and character consistency."""

from typing import Optional
from pydantic import BaseModel, Field

from src.simulation.conversation import ConversationResult, ConversationTurn
from src.characters.loader import Character
from src.llm.ollama_client import OllamaClient


class EvaluationMetrics(BaseModel):
    """Metrics for conversation evaluation."""
    character_consistency: float = Field(ge=0, le=1)
    conversation_naturalness: float = Field(ge=0, le=1)
    goal_progress: float = Field(ge=0, le=1)
    ended_naturally: bool
    num_turns: int
    notes: list[str] = Field(default_factory=list)


class ConversationEvaluator:
    """Evaluates conversation quality and character consistency."""

    def __init__(self, llm_client: Optional[OllamaClient] = None):
        self.llm = llm_client

    def evaluate(
        self,
        conversation: ConversationResult,
        character: Character,
    ) -> EvaluationMetrics:
        """
        Evaluate a completed conversation.

        Args:
            conversation: The conversation to evaluate
            character: The NPC character definition

        Returns:
            EvaluationMetrics with scores and notes
        """
        notes = []

        # Basic metrics
        num_turns = conversation.total_turns
        ended_naturally = conversation.ended_naturally

        # Evaluate character consistency
        consistency = self._evaluate_consistency(
            conversation.turns,
            character,
            notes,
        )

        # Evaluate naturalness
        naturalness = self._evaluate_naturalness(
            conversation.turns,
            notes,
        )

        # Evaluate goal progress
        goal_progress = self._evaluate_goal_progress(
            conversation.turns,
            conversation.customer_profile,
            notes,
        )

        return EvaluationMetrics(
            character_consistency=consistency,
            conversation_naturalness=naturalness,
            goal_progress=goal_progress,
            ended_naturally=ended_naturally,
            num_turns=num_turns,
            notes=notes,
        )

    def _evaluate_consistency(
        self,
        turns: list[ConversationTurn],
        character: Character,
        notes: list[str],
    ) -> float:
        """Evaluate character consistency using heuristics."""
        if not turns:
            return 0.0

        npc_turns = [t for t in turns if t.speaker == "npc"]
        if not npc_turns:
            return 0.0

        score = 1.0
        penalties = []

        # Check for "never" violations
        if character.behavioral_boundaries:
            never_keywords = {
                "faction sympathies": ["i support", "i'm with", "my allegiance"],
                "break character": ["as an ai", "i'm an ai", "i cannot", "i'm just a program"],
                "emotional outburst": ["!!!", "i hate", "i love you"],
            }

            for turn in npc_turns:
                message_lower = turn.message.lower()
                for violation, keywords in never_keywords.items():
                    if any(kw in message_lower for kw in keywords):
                        score -= 0.2
                        penalties.append(f"Possible '{violation}' violation")

        # Check for "always" compliance
        if character.behavioral_boundaries and character.behavioral_boundaries.always:
            # Check if first response offers a drink (for Zamir specifically)
            if npc_turns and "drink" not in npc_turns[0].message.lower():
                if "zamir" in character.name.lower():
                    score -= 0.1
                    penalties.append("First response didn't offer a drink")

        # Check for speech pattern consistency
        if character.personality.speech_patterns:
            # Look for action markers (asterisks) which are encouraged
            has_actions = any("*" in t.message for t in npc_turns)
            if not has_actions and len(npc_turns) > 2:
                score -= 0.05
                penalties.append("No action descriptions in responses")

        # Check response lengths are appropriate (2-4 sentences typically)
        for turn in npc_turns:
            # Rough sentence count
            sentence_count = turn.message.count(".") + turn.message.count("!") + turn.message.count("?")
            if sentence_count > 8:
                score -= 0.05
                penalties.append(f"Turn {turn.turn_number}: Response too long")

        if penalties:
            notes.extend(penalties)

        return max(0.0, min(1.0, score))

    def _evaluate_naturalness(
        self,
        turns: list[ConversationTurn],
        notes: list[str],
    ) -> float:
        """Evaluate how natural the conversation flows."""
        if len(turns) < 2:
            notes.append("Conversation too short to evaluate naturalness")
            return 0.5

        score = 1.0
        issues = []

        # Check for repetition
        messages = [t.message.lower() for t in turns]
        for i, msg in enumerate(messages):
            for j, other in enumerate(messages[i+1:], i+1):
                # Check for substantial repetition
                if len(msg) > 20 and msg in other:
                    score -= 0.15
                    issues.append(f"Repetition detected between turns {i} and {j}")
                    break

        # Check for appropriate turn lengths
        for turn in turns:
            words = len(turn.message.split())
            if words < 3:
                score -= 0.05
                issues.append(f"Turn {turn.turn_number}: Very short response")
            elif words > 200:
                score -= 0.1
                issues.append(f"Turn {turn.turn_number}: Excessively long response")

        # Check for conversation flow (responses should relate to previous)
        # This is a simple heuristic - could be enhanced with LLM
        if len(turns) >= 4:
            # Check if there's some word overlap between adjacent turns
            no_connection_count = 0
            for i in range(1, len(turns)):
                prev_words = set(turns[i-1].message.lower().split())
                curr_words = set(turns[i].message.lower().split())
                # Remove common words
                common_words = {"the", "a", "an", "is", "are", "was", "were", "i", "you", "to", "and"}
                prev_words -= common_words
                curr_words -= common_words
                if not prev_words.intersection(curr_words):
                    no_connection_count += 1

            if no_connection_count > len(turns) // 3:
                score -= 0.1
                issues.append("Conversation flow seems disconnected")

        if issues:
            notes.extend(issues)

        return max(0.0, min(1.0, score))

    def _evaluate_goal_progress(
        self,
        turns: list[ConversationTurn],
        customer_profile: dict,
        notes: list[str],
    ) -> float:
        """Evaluate if the customer made progress toward their goal."""
        objective = customer_profile.get("objective", "unknown")

        if not turns:
            return 0.0

        # Simple heuristic based on conversation length and ending
        base_score = 0.5

        # Longer conversations suggest engagement
        if len(turns) >= 6:
            base_score += 0.2
        elif len(turns) >= 4:
            base_score += 0.1

        # Check for goal-related keywords in conversation
        all_text = " ".join(t.message.lower() for t in turns)

        goal_keywords = {
            "information": ["know", "tell", "heard", "rumor", "news", "about"],
            "work": ["job", "work", "hire", "employ", "opportunity", "pay"],
            "refuge": ["safe", "hide", "stay", "help", "danger", "trouble"],
            "social": ["drink", "talk", "how are", "good to see", "friend"],
            "business": ["buy", "sell", "trade", "deal", "offer", "price"],
        }

        keywords = goal_keywords.get(objective, [])
        keyword_matches = sum(1 for kw in keywords if kw in all_text)

        if keyword_matches >= 3:
            base_score += 0.2
            notes.append(f"Good engagement with {objective} objective")
        elif keyword_matches >= 1:
            base_score += 0.1

        return min(1.0, base_score)

    def evaluate_with_llm(
        self,
        conversation: ConversationResult,
        character: Character,
    ) -> EvaluationMetrics:
        """
        Full evaluation using LLM for deeper analysis.
        Requires LLM client to be set.
        """
        if not self.llm:
            return self.evaluate(conversation, character)

        # Get basic metrics first
        basic_metrics = self.evaluate(conversation, character)
        notes = list(basic_metrics.notes)

        # Use LLM for character consistency check
        conv_text = "\n".join(
            f"{'Customer' if t.speaker == 'customer' else character.name}: {t.message}"
            for t in conversation.turns
        )

        prompt = f"""Evaluate this conversation for character consistency.

Character: {character.name}
Role: {character.role}
Key traits: {', '.join(character.personality.traits[:5])}

Conversation:
{conv_text}

Rate the character consistency from 0.0 to 1.0, where:
- 1.0 = Perfect, always in character
- 0.7 = Good, minor inconsistencies
- 0.5 = Mixed, some notable issues
- 0.3 = Poor, frequent breaks
- 0.0 = Completely out of character

Respond with just a number between 0.0 and 1.0:"""

        try:
            response = self.llm.generate(prompt, temperature=0.1, max_tokens=10)
            llm_consistency = float(response.strip())
            llm_consistency = max(0.0, min(1.0, llm_consistency))

            # Average with heuristic score
            final_consistency = (basic_metrics.character_consistency + llm_consistency) / 2
            notes.append(f"LLM consistency score: {llm_consistency:.2f}")
        except (ValueError, Exception):
            final_consistency = basic_metrics.character_consistency

        return EvaluationMetrics(
            character_consistency=final_consistency,
            conversation_naturalness=basic_metrics.conversation_naturalness,
            goal_progress=basic_metrics.goal_progress,
            ended_naturally=basic_metrics.ended_naturally,
            num_turns=basic_metrics.num_turns,
            notes=notes,
        )
