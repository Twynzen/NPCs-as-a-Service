# NPC Conversation Simulator

Generate synthetic conversations for training AI-powered NPCs with persistent personality and memory.

## Overview

This tool simulates conversations between NPCs and generated customer profiles, producing training data for NPC systems. It uses Ollama for local LLM inference and supports any character defined via YAML/JSON character cards.

**Features:**
- Generic character card system (YAML/JSON)
- Customer archetype generation with emotional states, factions, objectives
- Experience/memory extraction from conversations
- Quality metrics and character consistency evaluation
- Sequential execution (prototype-friendly)

## Quick Start

### Prerequisites

1. **Python 3.11+**
2. **UV package manager**
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
3. **Ollama** running locally with a model
   ```bash
   # Install Ollama: https://ollama.ai
   ollama serve  # Start the server
   ollama pull phi3.5  # Or llama3.2
   ```

### Installation

```bash
# Clone and enter the directory
cd NPCs-as-a-Service

# Install dependencies
uv sync
```

### Run Your First Simulation

```bash
# Check system requirements
uv run python -m src.main check

# Generate 10 conversations with Zamir
uv run python -m src.main simulate --character zamir --count 10

# Generate with verbose output
uv run python -m src.main simulate -c zamir -n 5 --verbose
```

## Usage

### Commands

```bash
# List available characters
uv run python -m src.main list-npcs

# Generate conversations
uv run python -m src.main simulate [OPTIONS]

# System check
uv run python -m src.main check
```

### Simulate Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--character` | `-c` | `zamir` | Character name |
| `--count` | `-n` | `10` | Number of conversations |
| `--model` | `-m` | `phi3.5` | Ollama model to use |
| `--max-turns` | | `20` | Max turns per conversation |
| `--temperature` | `-t` | `0.7` | LLM sampling temperature |
| `--output` | `-o` | auto | Output file path |
| `--verbose` | `-v` | `false` | Show conversation details |

### Examples

```bash
# Generate 100 conversations with Zamir
uv run python -m src.main simulate --character zamir --count 100

# Use a different model
uv run python -m src.main simulate -c zamir -n 20 --model llama3.2

# Custom output file
uv run python -m src.main simulate -c zamir -n 50 -o my_conversations.json

# Higher creativity
uv run python -m src.main simulate -c zamir -n 10 -t 0.9
```

## Output Format

### Conversations JSON

```json
{
  "metadata": {
    "character": "zamir",
    "model": "phi3.5",
    "generated_at": "2025-12-16T10:30:00",
    "count": 10,
    "summary": {
      "avg_turns": 8.5,
      "avg_consistency": 0.85,
      "natural_endings": 8
    }
  },
  "conversations": [
    {
      "id": "conv_00001",
      "character": "Zamir Talaquett",
      "customer_profile": {
        "archetype": "courier",
        "faction": "rebel",
        "emotional_state": "nervous",
        "objective": "information",
        "trust_level": "stranger"
      },
      "turns": [
        {"speaker": "customer", "message": "..."},
        {"speaker": "npc", "message": "..."}
      ],
      "extracted_experiences": [
        {
          "description": "A nervous rebel courier sought passage information",
          "importance": 7,
          "topics": ["rebel", "passage", "information"]
        }
      ],
      "metrics": {
        "num_turns": 8,
        "character_consistency": 0.85,
        "conversation_naturalness": 0.90,
        "conversation_ended_naturally": true
      }
    }
  ]
}
```

### Experiences JSON

```json
{
  "metadata": {
    "character": "zamir",
    "total_experiences": 25
  },
  "experiences": [
    {
      "description": "A desperate dock worker sought work to pay debts",
      "importance": 6,
      "topics": ["work", "debt", "dock_supervisor"],
      "emotional_context": "desperate",
      "customer_type": "dock_supervisor",
      "conversation_id": "conv_00003"
    }
  ]
}
```

## Adding New NPCs

### 1. Create a Character Card

Create a YAML file in `src/characters/templates/`:

```yaml
# src/characters/templates/my_npc.yaml
name: "Character Name"
role: "Brief description of role"

personality:
  traits:
    - "Key personality trait 1"
    - "Key personality trait 2"
  speech_patterns:
    - "How they speak"
    - "Verbal mannerisms"
  big_five:  # Optional
    openness: 0.5
    conscientiousness: 0.5
    extraversion: 0.5
    agreeableness: 0.5
    neuroticism: 0.5

backstory: |
  Character background and history.
  Multiple lines supported.

knowledge_domains:
  expert:
    - "Area of expertise 1"
  familiar:
    - "Things they know about"
  ignorant:
    - "Things they don't know"

behavioral_boundaries:
  never:
    - "Things the character would NEVER do"
  always:
    - "Things the character ALWAYS does"

first_message: |
  The character's opening message when starting a conversation.

example_dialogues:
  - context: "Situation description"
    customer: "What the customer says"
    npc: "How the NPC responds"
```

### 2. Run Simulations

```bash
uv run python -m src.main simulate --character my_npc --count 100
```

### Character Card Schema

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | ✓ | Character's name |
| `role` | string | ✓ | Brief role description |
| `personality.traits` | list[str] | ✓ | Core personality traits |
| `personality.speech_patterns` | list[str] | | How they speak |
| `personality.big_five` | object | | OCEAN personality scores |
| `backstory` | string | | Character background |
| `knowledge_domains` | object | | Expert/familiar/ignorant areas |
| `behavioral_boundaries` | object | | Never/always rules |
| `first_message` | string | | Opening conversation message |
| `example_dialogues` | list | | Few-shot examples |

## Customizing Customer Archetypes

Edit `src/customers/archetypes.yaml` to modify:

- **Factions**: Groups customers belong to
- **Emotional states**: How customers feel
- **Objectives**: What customers want
- **Trust levels**: Relationship with NPC
- **Archetypes**: Specific customer types

## Project Structure

```
npc-simulator/
├── src/
│   ├── characters/
│   │   ├── loader.py          # Character card loading
│   │   └── templates/
│   │       └── zamir.yaml     # Zamir character card
│   │
│   ├── customers/
│   │   ├── generator.py       # Customer profile generation
│   │   └── archetypes.yaml    # Customer archetypes
│   │
│   ├── simulation/
│   │   ├── conversation.py    # Dialogue loop
│   │   ├── experience_extractor.py  # Memory extraction
│   │   └── evaluator.py       # Quality metrics
│   │
│   ├── llm/
│   │   └── ollama_client.py   # Ollama API client
│   │
│   └── main.py                # CLI entry point
│
├── output/
│   ├── conversations/         # Generated conversations
│   └── experiences/           # Extracted experiences
│
├── pyproject.toml             # UV/Python config
└── README.md
```

## Troubleshooting

### Ollama not available
```bash
# Start the Ollama server
ollama serve

# In another terminal, verify it's running
curl http://localhost:11434/api/tags
```

### Model not found
```bash
# Pull the recommended model
ollama pull phi3.5

# Or use an alternative
ollama pull llama3.2
```

### Character not found
```bash
# List available characters
uv run python -m src.main list-npcs

# Check the templates directory
ls src/characters/templates/
```

## Recommended Models

| Model | Size | Quality | Speed | Notes |
|-------|------|---------|-------|-------|
| `phi3.5` | 3.8B | Good | Fast | Recommended |
| `llama3.2` | 3.2B | Good | Fast | Alternative |
| `mistral` | 7B | Better | Slower | Higher quality |

## License

MIT License - See LICENSE file for details.

---

Built for the NPCs-as-a-Service project. See the whitepaper for theoretical foundations.
