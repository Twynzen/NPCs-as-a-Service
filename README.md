# NPC Service

AI-powered NPCs with persistent memory for games. Provides a REST/WebSocket API that any game can consume to have intelligent, memorable NPC conversations.

## Features

- **REST + WebSocket API** - Easy integration with any game engine
- **Persistent Memory** - NPCs remember players across sessions
- **Character Cards** - Define any NPC via YAML/JSON
- **Action System** - NPCs can give items, start quests, remember facts
- **Conversation Simulator** - Generate training data for fine-tuning
- **Web Playground** - Test NPCs without writing code

## Quick Start

### Prerequisites

1. **Python 3.11+**
2. **UV** package manager:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
3. **Ollama** with a model:
   ```bash
   ollama serve
   ollama pull phi3.5
   ```

### Installation

```bash
git clone <repo>
cd NPCs-as-a-Service
uv sync
```

### Start the Server

```bash
# Start the API server
uv run python -m src.main serve

# Or with auto-reload for development
uv run python -m src.main serve --reload
```

Open http://localhost:8000/playground to test NPCs in your browser.

## API Usage

### 1. Create a Session

```bash
curl -X POST http://localhost:8000/v1/sessions \
  -H "Content-Type: application/json" \
  -d '{"npc_id": "zamir", "player_id": "player_123"}'
```

Response:
```json
{
  "id": "sess_abc123",
  "npc_id": "zamir",
  "player_id": "player_123",
  "first_message": "*Zamir looks up from polishing a glass*..."
}
```

### 2. Send Messages

```bash
curl -X POST http://localhost:8000/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "sess_abc123", "message": "I need information about QDT"}'
```

Response:
```json
{
  "content": "*Zamir's polishing slows* QDT, you say...",
  "actions": [
    {"action": "remember", "parameters": {"fact": "Player seeks QDT info", "importance": 6}}
  ],
  "turn_number": 1
}
```

### 3. WebSocket Streaming

```javascript
const ws = new WebSocket('ws://localhost:8000/v1/chat/ws/sess_abc123');

ws.send(JSON.stringify({ message: "Hello!" }));

ws.onmessage = (e) => {
  const data = JSON.parse(e.data);
  if (data.type === 'chunk') {
    console.log(data.content);  // Streaming token
  } else if (data.type === 'done') {
    console.log('Actions:', data.actions);
  }
};
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/health` | GET | Service health check |
| `/v1/characters` | GET | List available NPCs |
| `/v1/characters/{id}` | GET | Get NPC details |
| `/v1/sessions` | POST | Create conversation session |
| `/v1/sessions/{id}` | GET | Get session info |
| `/v1/sessions/{id}` | DELETE | End session |
| `/v1/chat` | POST | Send message, get response |
| `/v1/chat/quick` | POST | One-shot chat (auto session) |
| `/v1/chat/ws/{id}` | WS | WebSocket streaming |
| `/v1/memory/inject` | POST | Inject memory into NPC |

Full API docs: http://localhost:8000/docs

## Memory System

NPCs have three memory layers:

### Core Memory (Always Active)
- Player name and known facts
- Relationship level (-10 hostile to +10 trusted)
- NPC's current emotional state

### Recall Memory (Session)
- Last 20 conversation turns
- Provides immediate context

### Archival Memory (Persistent)
- Long-term memories with importance scores
- Semantic search for relevant memories
- Survives across sessions

### Memory Actions

NPCs can execute memory actions:

```json
{"action": "remember", "parameters": {"fact": "Player helped defend the bar", "importance": 8}}
{"action": "update_relationship", "parameters": {"delta": 2}}
{"action": "set_emotion", "parameters": {"emotion": "grateful"}}
{"action": "give_item", "parameters": {"item_id": "special_drink", "quantity": 1}}
```

## Adding NPCs

Create a YAML file in `src/characters/templates/`:

```yaml
name: "Elena the Merchant"
role: "Traveling merchant specializing in rare artifacts"

personality:
  traits:
    - "Shrewd businesswoman"
    - "Knows the value of everything"
  speech_patterns:
    - "Speaks in terms of trades and deals"

backstory: |
  Elena has traveled the trade routes for twenty years.

knowledge_domains:
  expert: ["Artifact valuation", "Trade routes"]
  ignorant: ["Magic", "Military tactics"]

behavioral_boundaries:
  never: ["Give items for free"]
  always: ["Quote a price for everything"]

first_message: |
  *Elena looks up* "Ah, a customer! What treasures seek you today?"
```

## Conversation Simulator

Generate training data:

```bash
# Generate 100 conversations
uv run python -m src.main simulate -c zamir -n 100

# With verbose output
uv run python -m src.main simulate -c zamir -n 50 --verbose
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                       GAME CLIENT                            │
│            (Unity, Unreal, Godot, Web, etc.)                │
└─────────────────────────┬───────────────────────────────────┘
                          │ REST / WebSocket
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                      NPC SERVICE                             │
│  ┌──────────┐  ┌──────────┐  ┌────────────────────────────┐ │
│  │ FastAPI  │──│  Engine  │──│         Memory             │ │
│  │  Server  │  │  (NPC)   │  │  Core │ Recall │ Archival  │ │
│  └──────────┘  └────┬─────┘  └────────────────────────────┘ │
│                     │                                        │
│  ┌──────────────────┴────────────────────────────────────┐  │
│  │                   LLM Layer                            │  │
│  │           Ollama (local)  │  OpenAI (cloud)            │  │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Project Structure

```
npc-service/
├── src/
│   ├── server/           # FastAPI application
│   │   ├── app.py
│   │   └── routes/
│   ├── engine/           # NPC runtime
│   │   ├── npc_engine.py
│   │   ├── context_builder.py
│   │   └── action_parser.py
│   ├── memory/           # Memory system
│   │   ├── manager.py
│   │   ├── core_memory.py
│   │   ├── recall_memory.py
│   │   └── archival_memory.py
│   ├── characters/       # Character loading
│   │   └── templates/
│   ├── simulation/       # Training data generation
│   ├── llm/              # LLM clients
│   ├── config.py
│   └── main.py
├── playground/           # Web UI
├── output/
├── pyproject.toml
└── README.md
```

## CLI Commands

```bash
# Start API server
uv run python -m src.main serve [--port 8000] [--reload]

# Generate training conversations
uv run python -m src.main simulate -c CHARACTER -n COUNT

# List available NPCs
uv run python -m src.main list-npcs

# Check system status
uv run python -m src.main check
```

## Configuration

Environment variables (or `.env` file):

```bash
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=phi3.5
OPENAI_API_KEY=sk-...      # Optional
HOST=0.0.0.0
PORT=8000
```

## Integration Example (JavaScript)

```javascript
class NPCClient {
  constructor(baseUrl = 'http://localhost:8000') {
    this.baseUrl = baseUrl;
  }

  async startSession(npcId, playerId) {
    const res = await fetch(`${this.baseUrl}/v1/sessions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ npc_id: npcId, player_id: playerId })
    });
    return res.json();
  }

  async chat(sessionId, message) {
    const res = await fetch(`${this.baseUrl}/v1/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: sessionId, message })
    });
    return res.json();
  }
}

// Usage
const npc = new NPCClient();
const session = await npc.startSession('zamir', 'player_1');
const response = await npc.chat(session.id, 'Hello!');
console.log(response.content);
```

## License

MIT License
