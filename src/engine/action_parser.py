"""Action Parser - Extracts structured actions from NPC responses.

Allows NPCs to execute game actions like giving items, starting quests,
or updating their own memory.
"""

import re
import json
from typing import Optional
from pydantic import BaseModel, Field


class NPCAction(BaseModel):
    """A structured action the NPC wants to execute."""
    name: str
    parameters: dict = Field(default_factory=dict)
    raw_text: Optional[str] = None

    def to_dict(self) -> dict:
        return {"action": self.name, "parameters": self.parameters}


# Default actions available to all NPCs
DEFAULT_ACTIONS = {
    "remember": {
        "description": "Store an important fact about this player",
        "parameters": {
            "fact": {"type": "string", "description": "The fact to remember"},
            "importance": {"type": "integer", "description": "Importance 1-10", "default": 5},
        },
    },
    "update_relationship": {
        "description": "Adjust how you feel about this player",
        "parameters": {
            "delta": {"type": "integer", "description": "-3 to +3 change"},
            "reason": {"type": "string", "description": "Why the change"},
        },
    },
    "set_emotion": {
        "description": "Change your current emotional state",
        "parameters": {
            "emotion": {"type": "string", "description": "New emotion"},
        },
    },
    "give_item": {
        "description": "Give an item to the player",
        "parameters": {
            "item_id": {"type": "string", "description": "Item identifier"},
            "quantity": {"type": "integer", "description": "Amount", "default": 1},
        },
    },
    "start_quest": {
        "description": "Offer or start a quest for the player",
        "parameters": {
            "quest_id": {"type": "string", "description": "Quest identifier"},
        },
    },
    "end_conversation": {
        "description": "Signal that the conversation should end",
        "parameters": {
            "reason": {"type": "string", "description": "Why ending"},
        },
    },
}


class ActionParser:
    """
    Parses NPC responses to extract actions.

    Supports multiple formats:
    - JSON blocks: ```json {"action": "give_item", "params": {...}} ```
    - Inline tags: [ACTION: give_item(item_id="sword", quantity=1)]
    - Natural language hints: *gives the player a sword*
    """

    def __init__(self, custom_actions: Optional[dict] = None):
        self.actions = {**DEFAULT_ACTIONS}
        if custom_actions:
            self.actions.update(custom_actions)

    def get_actions_prompt(self) -> str:
        """Generate prompt text describing available actions."""
        lines = [
            "## Available Actions",
            "You can perform actions by including them in your response.",
            "Format: [ACTION: action_name(param1=\"value1\", param2=value2)]",
            "",
            "Available actions:",
        ]

        for name, spec in self.actions.items():
            params = ", ".join(
                f"{k}: {v['type']}"
                for k, v in spec.get("parameters", {}).items()
            )
            lines.append(f"- {name}({params}): {spec['description']}")

        lines.append("")
        lines.append("Example: [ACTION: remember(fact=\"Player is looking for work\", importance=7)]")

        return "\n".join(lines)

    def parse(self, response: str) -> tuple[str, list[NPCAction]]:
        """
        Parse response and extract actions.

        Returns:
            Tuple of (cleaned_response, list_of_actions)
        """
        actions = []
        cleaned = response

        # Try JSON blocks first
        json_actions, cleaned = self._parse_json_blocks(cleaned)
        actions.extend(json_actions)

        # Try inline action tags
        inline_actions, cleaned = self._parse_inline_actions(cleaned)
        actions.extend(inline_actions)

        # Try natural language patterns
        nl_actions = self._parse_natural_language(response)
        actions.extend(nl_actions)

        # Deduplicate actions by name
        seen = set()
        unique_actions = []
        for action in actions:
            if action.name not in seen:
                seen.add(action.name)
                unique_actions.append(action)

        return cleaned.strip(), unique_actions

    def _parse_json_blocks(self, text: str) -> tuple[list[NPCAction], str]:
        """Extract JSON action blocks."""
        actions = []
        cleaned = text

        # Match ```json ... ``` blocks
        pattern = r'```json\s*(\{[^`]+\})\s*```'

        for match in re.finditer(pattern, text, re.DOTALL):
            try:
                data = json.loads(match.group(1))
                if "action" in data:
                    action = NPCAction(
                        name=data["action"],
                        parameters=data.get("params", data.get("parameters", {})),
                        raw_text=match.group(0),
                    )
                    actions.append(action)
                    cleaned = cleaned.replace(match.group(0), "")
            except json.JSONDecodeError:
                continue

        return actions, cleaned

    def _parse_inline_actions(self, text: str) -> tuple[list[NPCAction], str]:
        """Extract inline action tags like [ACTION: name(params)]."""
        actions = []
        cleaned = text

        # Match [ACTION: name(params)] pattern
        pattern = r'\[ACTION:\s*(\w+)\(([^)]*)\)\]'

        for match in re.finditer(pattern, text):
            name = match.group(1)
            params_str = match.group(2)

            # Parse parameters
            params = self._parse_params_string(params_str)

            action = NPCAction(
                name=name,
                parameters=params,
                raw_text=match.group(0),
            )
            actions.append(action)
            cleaned = cleaned.replace(match.group(0), "")

        return actions, cleaned

    def _parse_params_string(self, params_str: str) -> dict:
        """Parse parameter string like 'fact="hello", importance=5'."""
        params = {}

        if not params_str.strip():
            return params

        # Match key=value or key="value" patterns
        pattern = r'(\w+)\s*=\s*(?:"([^"]*)"|\'([^\']*)\'|(\d+(?:\.\d+)?)|(\w+))'

        for match in re.finditer(pattern, params_str):
            key = match.group(1)
            # Try each capture group
            value = match.group(2) or match.group(3) or match.group(4) or match.group(5)

            # Try to convert to appropriate type
            if value and value.isdigit():
                value = int(value)
            elif value and re.match(r'^\d+\.\d+$', value):
                value = float(value)

            params[key] = value

        return params

    def _parse_natural_language(self, text: str) -> list[NPCAction]:
        """Extract actions from natural language patterns."""
        actions = []

        # Pattern: *gives X to the player* or *hands over X*
        give_patterns = [
            r'\*gives?\s+(?:the\s+)?(?:player\s+)?(?:a\s+)?([^*]+?)\*',
            r'\*hands?\s+(?:over\s+)?(?:a\s+)?([^*]+?)\*',
        ]

        for pattern in give_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                item = match.group(1).strip()
                if item and len(item) < 50:  # Sanity check
                    actions.append(NPCAction(
                        name="give_item",
                        parameters={"item_id": item, "quantity": 1},
                        raw_text=match.group(0),
                    ))

        # Pattern: conversation ending
        end_patterns = [
            r'\*(?:turns away|walks away|leaves|ends the conversation)\*',
            r'\*(?:dismisses|waves off)\*',
        ]

        for pattern in end_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                actions.append(NPCAction(
                    name="end_conversation",
                    parameters={"reason": "natural ending"},
                ))
                break

        return actions

    def validate_action(self, action: NPCAction) -> tuple[bool, Optional[str]]:
        """Validate an action against known action schemas."""
        if action.name not in self.actions:
            return False, f"Unknown action: {action.name}"

        spec = self.actions[action.name]
        required_params = [
            k for k, v in spec.get("parameters", {}).items()
            if "default" not in v
        ]

        for param in required_params:
            if param not in action.parameters:
                return False, f"Missing required parameter: {param}"

        return True, None
