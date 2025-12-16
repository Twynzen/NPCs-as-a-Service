"""Customer profile generator using archetypes."""

import random
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field


class CustomerProfile(BaseModel):
    """Generated customer profile for simulation."""
    id: str
    archetype: str
    faction: str
    emotional_state: str
    objective: str
    trust_level: str
    personality_description: str
    opening_line: str
    conversation_goal: str
    hidden_agenda: Optional[str] = None


class Archetype(BaseModel):
    """Customer archetype definition."""
    name: str
    faction: str
    description: str
    personality_hints: list[str]


class CustomerGenerator:
    """Generates diverse customer profiles for NPC conversation simulation."""

    def __init__(self, archetypes_path: Optional[Path] = None):
        if archetypes_path is None:
            archetypes_path = Path(__file__).parent / "archetypes.yaml"

        with open(archetypes_path, "r", encoding="utf-8") as f:
            self.data = yaml.safe_load(f)

        self.factions = {f["name"]: f for f in self.data["factions"]}
        self.emotional_states = {e["name"]: e for e in self.data["emotional_states"]}
        self.objectives = {o["name"]: o for o in self.data["objectives"]}
        self.trust_levels = {t["name"]: t for t in self.data["trust_levels"]}
        self.archetypes = {a["name"]: Archetype(**a) for a in self.data["archetypes"]}

    def generate(
        self,
        archetype: Optional[str] = None,
        faction: Optional[str] = None,
        emotional_state: Optional[str] = None,
        objective: Optional[str] = None,
        trust_level: Optional[str] = None,
    ) -> CustomerProfile:
        """
        Generate a customer profile with optional constraints.

        All parameters are optional - if not specified, random values are chosen.
        """
        # Select archetype (may constrain faction)
        if archetype:
            arch = self.archetypes[archetype]
        else:
            if faction:
                # Filter archetypes by faction
                valid = [a for a in self.archetypes.values() if a.faction == faction]
                arch = random.choice(valid)
            else:
                arch = random.choice(list(self.archetypes.values()))

        # Use archetype's faction if not specified
        selected_faction = faction or arch.faction

        # Select other attributes
        selected_emotional = emotional_state or random.choice(
            list(self.emotional_states.keys())
        )
        selected_objective = objective or random.choice(list(self.objectives.keys()))
        selected_trust = trust_level or random.choice(list(self.trust_levels.keys()))

        # Build personality description
        personality_desc = self._build_personality(
            arch, selected_faction, selected_emotional, selected_objective, selected_trust
        )

        # Generate opening line
        opening = self._generate_opening(
            arch, selected_emotional, selected_objective, selected_trust
        )

        # Define conversation goal
        goal = self._define_goal(arch, selected_objective, selected_emotional)

        # Maybe add hidden agenda
        hidden = self._maybe_hidden_agenda(arch, selected_faction, selected_objective)

        return CustomerProfile(
            id=f"cust_{random.randint(10000, 99999)}",
            archetype=arch.name,
            faction=selected_faction,
            emotional_state=selected_emotional,
            objective=selected_objective,
            trust_level=selected_trust,
            personality_description=personality_desc,
            opening_line=opening,
            conversation_goal=goal,
            hidden_agenda=hidden,
        )

    def _build_personality(
        self,
        arch: Archetype,
        faction: str,
        emotional: str,
        objective: str,
        trust: str,
    ) -> str:
        """Build a rich personality description for the customer."""
        faction_info = self.factions[faction]
        emotional_info = self.emotional_states[emotional]
        objective_info = self.objectives[objective]
        trust_info = self.trust_levels[trust]

        parts = [
            f"A {arch.name.replace('_', ' ')} who is {emotional_info['behavior']}.",
            f"Affiliated with {faction_info['description']}.",
            f"Here because: {objective_info['description']}.",
            f"Trust level: {trust_info['description']}.",
            "",
            "Personality traits:",
        ]
        for hint in arch.personality_hints:
            parts.append(f"- {hint}")

        parts.append("")
        parts.append("Speech characteristics:")
        for hint in faction_info["speech_hints"]:
            parts.append(f"- {hint}")
        for hint in emotional_info["speech_hints"]:
            parts.append(f"- {hint}")

        return "\n".join(parts)

    def _generate_opening(
        self,
        arch: Archetype,
        emotional: str,
        objective: str,
        trust: str,
    ) -> str:
        """Generate an appropriate opening line."""
        openings = {
            ("stranger", "calm"): [
                "*approaches the bar and takes a seat* A drink, if you're serving.",
                "*slides onto a barstool, looking around* This the place they call The Crossed Keys?",
                "*nods in greeting* Heard this was a good spot for a quiet drink.",
            ],
            ("stranger", "nervous"): [
                "*glances around before sitting* I was told to ask for the house special.",
                "*approaches quickly, speaking low* I need... I need a drink. Something strong.",
                "*sits down, fidgeting* Is it always this busy in here?",
            ],
            ("stranger", "desperate"): [
                "*rushes to the bar* Please, I need help. I don't know who else to ask.",
                "*slams credits on the bar* I'll pay double. I just need information.",
                "*looks around frantically* They said you're the one who knows things.",
            ],
            ("stranger", "suspicious"): [
                "*takes a seat, watching the bartender carefully* What's good here?",
                "*sits down slowly* Interesting place. You the owner?",
                "*studies the bar before speaking* A lot of different faces in here.",
            ],
            ("acquaintance", "calm"): [
                "*takes their usual spot* The usual, Zamir. How's business?",
                "*settles in comfortably* Good to see you again. Quiet night?",
                "*nods in recognition* Zamir. I'll have what I had last time.",
            ],
            ("regular", "calm"): [
                "*sits at their usual spot* Zamir, my friend. It's been too long.",
                "*waves familiarly* You know what I like. And maybe some conversation?",
                "*settles in with a sigh* Rough week. I need your wisdom and your whiskey.",
            ],
            ("regular", "nervous"): [
                "*sits quickly, leaning in* Zamir, something's happening. We need to talk.",
                "*glances at the door before sitting* I might have trouble following me.",
                "*takes a seat, voice low* I've heard things. Things you should know.",
            ],
        }

        # Find best match or use default
        key = (trust, emotional)
        if key in openings:
            return random.choice(openings[key])

        # Fallback openings based on objective
        objective_openings = {
            "information": "*sits down, looking purposeful* I'm looking for information.",
            "work": "*approaches the bar* Word is you know about job opportunities.",
            "refuge": "*looks around nervously before sitting* I need somewhere safe.",
            "social": "*settles in comfortably* Just here for a drink and company.",
            "business": "*takes a seat, placing a small case on the bar* I have a proposition.",
        }

        return objective_openings.get(
            objective, "*approaches the bar* I'll have whatever's good."
        )

    def _define_goal(self, arch: Archetype, objective: str, emotional: str) -> str:
        """Define the customer's conversation goal."""
        objective_info = self.objectives[objective]

        goal_templates = {
            "information": f"Learn about {random.choice(objective_info['typical_topics'])} without revealing too much about themselves",
            "work": "Find work opportunities that match their skills and risk tolerance",
            "refuge": "Find a safe place to lay low without attracting attention",
            "social": "Enjoy a drink, maybe share some gossip, have a pleasant conversation",
            "business": "Complete a transaction or find a buyer/seller for their goods",
        }

        base_goal = goal_templates.get(objective, "Have a conversation")

        if emotional == "desperate":
            base_goal += " - urgently"
        elif emotional == "suspicious":
            base_goal += " - while testing if the bartender can be trusted"
        elif emotional == "drunk":
            base_goal += " - though may get distracted or overshare"

        return base_goal

    def _maybe_hidden_agenda(
        self,
        arch: Archetype,
        faction: str,
        objective: str,
    ) -> Optional[str]:
        """Some customers have hidden agendas."""
        # 30% chance of hidden agenda
        if random.random() > 0.3:
            return None

        hidden_agendas = {
            "qdt": [
                "Actually gathering intel on rebel sympathizers",
                "Testing the bartender's loyalties for QDT",
                "Looking for someone specific to report to superiors",
            ],
            "rebel": [
                "Scouting the bar as a potential safe house",
                "Looking for potential recruits to the cause",
                "Trying to identify any QDT informants",
            ],
            "neutral": [
                "Actually working for a third party",
                "Has information to sell to the highest bidder",
                "Running a personal con on the side",
            ],
        }

        return random.choice(hidden_agendas.get(faction, []))

    def generate_batch(self, count: int, **constraints) -> list[CustomerProfile]:
        """Generate multiple diverse customer profiles."""
        profiles = []
        for _ in range(count):
            profiles.append(self.generate(**constraints))
        return profiles


def get_customer_system_prompt(profile: CustomerProfile) -> str:
    """Build a system prompt for an LLM playing this customer."""
    return f"""You are roleplaying as a customer at The Crossed Keys bar in Nuvaris Station.

## Your Profile
{profile.personality_description}

## Your Goal
{profile.conversation_goal}

{"## Hidden Agenda (don't reveal directly)" + chr(10) + profile.hidden_agenda if profile.hidden_agenda else ""}

## How to Play This Character
- Start with your opening line (or similar)
- Stay in character based on your emotional state and objectives
- React naturally to the bartender's responses
- Don't reveal hidden agendas directly, but let them influence your behavior
- End the conversation naturally when your goal is met or clearly unachievable
- Use *asterisks* for actions and non-verbal cues

## Your Opening Line
{profile.opening_line}

Remember: You are the CUSTOMER. Wait for the bartender to respond. Keep responses conversational (2-4 sentences typically).
When the conversation reaches a natural end, say something like "[END]" or describe leaving the bar.
"""
