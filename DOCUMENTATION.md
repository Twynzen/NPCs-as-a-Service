# NPC Service - Documentación Completa

## Índice

1. [Visión General](#visión-general)
2. [Arquitectura del Sistema](#arquitectura-del-sistema)
3. [Instalación y Configuración](#instalación-y-configuración)
4. [API Server](#api-server)
5. [Sistema de Memoria](#sistema-de-memoria)
6. [Motor de NPC](#motor-de-npc)
7. [Simulador de Conversaciones](#simulador-de-conversaciones)
8. [Character Cards](#character-cards)
9. [Sistema de Acciones](#sistema-de-acciones)
10. [Integración con Juegos](#integración-con-juegos)
11. [Próximos Pasos](#próximos-pasos)
12. [Referencia de API](#referencia-de-api)

---

## Visión General

**NPC Service** es un sistema completo para crear NPCs con inteligencia artificial que:

- **Recuerdan** a los jugadores entre sesiones
- **Mantienen** personalidad consistente
- **Ejecutan acciones** en el juego (dar items, iniciar quests)
- **Generan** datos de entrenamiento para fine-tuning

### Dos Modos de Uso

```
┌─────────────────────────────────────────────────────────────┐
│                      NPC SERVICE                             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   MODO SERVIDOR (Producción)        MODO SIMULADOR (Datos)  │
│   ─────────────────────────         ──────────────────────  │
│                                                              │
│   uv run python -m src.main serve   uv run python -m src.main simulate
│                                                              │
│   • API REST + WebSocket            • Genera conversaciones │
│   • Memoria persistente             • Extrae experiencias   │
│   • Integración con juegos          • Evalúa consistencia   │
│   • Playground web                  • Output JSON           │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Arquitectura del Sistema

### Diagrama de Componentes

```
┌─────────────────────────────────────────────────────────────────────────┐
│                            CLIENTES                                      │
│    ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐              │
│    │  Unity   │  │ Unreal   │  │  Godot   │  │   Web    │              │
│    └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘              │
└─────────┼─────────────┼─────────────┼─────────────┼─────────────────────┘
          │             │             │             │
          └─────────────┴──────┬──────┴─────────────┘
                               │ HTTP / WebSocket
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         API LAYER (FastAPI)                              │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  /v1/sessions    /v1/chat    /v1/characters    /v1/health       │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         NPC ENGINE                                       │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────────────────┐   │
│  │    Context    │  │    Action     │  │      Character            │   │
│  │    Builder    │  │    Parser     │  │      Loader               │   │
│  └───────┬───────┘  └───────┬───────┘  └───────────┬───────────────┘   │
│          │                  │                      │                    │
│          └──────────────────┼──────────────────────┘                    │
│                             ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    MEMORY MANAGER                                │   │
│  │   ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐     │   │
│  │   │    Core     │  │   Recall    │  │      Archival       │     │   │
│  │   │   Memory    │  │   Memory    │  │      Memory         │     │   │
│  │   │             │  │             │  │                     │     │   │
│  │   │ • Nombre    │  │ • Últimos   │  │ • Memorias largo    │     │   │
│  │   │ • Relación  │  │   20 turnos │  │   plazo             │     │   │
│  │   │ • Emoción   │  │ • Contexto  │  │ • Búsqueda          │     │   │
│  │   │ • Hechos    │  │   inmediato │  │   semántica         │     │   │
│  │   └─────────────┘  └─────────────┘  └─────────────────────┘     │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          LLM LAYER                                       │
│  ┌───────────────────────────────┐  ┌───────────────────────────────┐   │
│  │         Ollama (Local)        │  │      OpenAI (Cloud)           │   │
│  │   • phi3.5, llama3.2          │  │   • gpt-4o-mini               │   │
│  │   • Sin costo por request     │  │   • Para narrativa compleja   │   │
│  │   • Latencia ~1-2s            │  │   • Fallback opcional         │   │
│  └───────────────────────────────┘  └───────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

### Estructura de Archivos

```
npc-service/
│
├── src/
│   │
│   ├── server/                    # ═══ API SERVER ═══
│   │   ├── __init__.py
│   │   ├── app.py                 # FastAPI app, lifespan, CORS
│   │   └── routes/
│   │       ├── health.py          # GET /v1/health, /v1/health/detailed
│   │       ├── sessions.py        # CRUD sesiones de conversación
│   │       ├── chat.py            # POST /v1/chat, WS /v1/chat/ws
│   │       └── characters.py      # Gestión de NPCs
│   │
│   ├── engine/                    # ═══ NPC ENGINE ═══
│   │   ├── __init__.py
│   │   ├── npc_engine.py          # Orquestador principal
│   │   ├── context_builder.py     # Construye prompts con memoria
│   │   └── action_parser.py       # Extrae acciones de respuestas
│   │
│   ├── memory/                    # ═══ MEMORY SYSTEM ═══
│   │   ├── __init__.py
│   │   ├── manager.py             # Orquestador de memoria
│   │   ├── core_memory.py         # Memoria siempre en contexto
│   │   ├── recall_memory.py       # Historial de conversación
│   │   └── archival_memory.py     # Memorias largo plazo
│   │
│   ├── characters/                # ═══ CHARACTER SYSTEM ═══
│   │   ├── __init__.py
│   │   ├── loader.py              # Carga YAML/JSON
│   │   └── templates/
│   │       └── zamir.yaml         # Zamir (ejemplo)
│   │
│   ├── customers/                 # ═══ CUSTOMER GENERATOR ═══
│   │   ├── __init__.py
│   │   ├── generator.py           # Genera perfiles aleatorios
│   │   └── archetypes.yaml        # Arquetipos configurables
│   │
│   ├── simulation/                # ═══ CONVERSATION SIMULATOR ═══
│   │   ├── __init__.py
│   │   ├── conversation.py        # Loop de diálogo
│   │   ├── experience_extractor.py # Extrae memorias
│   │   └── evaluator.py           # Métricas de calidad
│   │
│   ├── llm/                       # ═══ LLM CLIENTS ═══
│   │   ├── __init__.py
│   │   └── ollama_client.py       # Cliente Ollama API
│   │
│   ├── config.py                  # Configuración centralizada
│   └── main.py                    # CLI (serve, simulate, check)
│
├── playground/                    # ═══ WEB UI ═══
│   └── index.html                 # Interfaz de prueba
│
├── output/                        # ═══ GENERATED DATA ═══
│   ├── conversations/             # JSON de conversaciones
│   └── experiences/               # Experiencias extraídas
│
├── pyproject.toml                 # Dependencias UV
└── README.md                      # Documentación básica
```

---

## Instalación y Configuración

### Requisitos

| Requisito | Versión | Notas |
|-----------|---------|-------|
| Python | 3.11+ | Requerido |
| UV | Latest | Package manager |
| Ollama | Latest | LLM local |
| Modelo LLM | phi3.5 / llama3.2 | Recomendado |

### Instalación Paso a Paso

```bash
# 1. Clonar repositorio
git clone <repo-url>
cd NPCs-as-a-Service

# 2. Instalar UV (si no lo tienes)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. Instalar dependencias
uv sync

# 4. Instalar Ollama (si no lo tienes)
# macOS/Linux: curl -fsSL https://ollama.com/install.sh | sh
# Windows: Descargar de https://ollama.com

# 5. Iniciar Ollama y descargar modelo
ollama serve  # En una terminal
ollama pull phi3.5  # En otra terminal

# 6. Verificar instalación
uv run python -m src.main check
```

### Variables de Entorno

Crear archivo `.env` en la raíz (opcional):

```bash
# LLM Local
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=phi3.5

# LLM Cloud (opcional)
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini

# Servidor
HOST=0.0.0.0
PORT=8000
DEBUG=false

# Memoria (opcional - futuro)
REDIS_URL=redis://localhost:6379
QDRANT_URL=http://localhost:6333
```

---

## API Server

### Iniciar el Servidor

```bash
# Modo producción
uv run python -m src.main serve

# Modo desarrollo (auto-reload)
uv run python -m src.main serve --reload

# Puerto personalizado
uv run python -m src.main serve --port 3000
```

### URLs Disponibles

| URL | Descripción |
|-----|-------------|
| `http://localhost:8000` | Información del servicio |
| `http://localhost:8000/docs` | Swagger UI (API interactiva) |
| `http://localhost:8000/redoc` | ReDoc (documentación) |
| `http://localhost:8000/playground` | Web UI para probar NPCs |

### Flujo de Conversación

```
┌─────────────────────────────────────────────────────────────────┐
│                    FLUJO DE CONVERSACIÓN                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. CREAR SESIÓN                                                │
│     ─────────────                                               │
│     POST /v1/sessions                                           │
│     {"npc_id": "zamir", "player_id": "player_123"}             │
│                    │                                            │
│                    ▼                                            │
│     Response: {"id": "sess_abc", "first_message": "..."}       │
│                                                                  │
│  2. ENVIAR MENSAJES (loop)                                      │
│     ──────────────────────                                      │
│     POST /v1/chat                                               │
│     {"session_id": "sess_abc", "message": "Hola"}              │
│                    │                                            │
│                    ▼                                            │
│     Response: {                                                 │
│       "content": "Respuesta del NPC...",                       │
│       "actions": [{"action": "remember", ...}],                │
│       "turn_number": 1                                          │
│     }                                                           │
│                                                                  │
│  3. CERRAR SESIÓN                                               │
│     ──────────────                                              │
│     DELETE /v1/sessions/sess_abc                                │
│                    │                                            │
│                    ▼                                            │
│     Response: {"archived_summary": "..."}                       │
│     (La conversación se guarda en memoria a largo plazo)        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Endpoints Detallados

#### Health Check

```bash
# Básico
GET /v1/health

# Response
{
  "status": "healthy",
  "version": "0.2.0",
  "ollama_connected": true,
  "ollama_model": "phi3.5",
  "available_characters": ["zamir"],
  "active_sessions": 3
}
```

```bash
# Detallado
GET /v1/health/detailed

# Response
{
  "status": "healthy",
  "components": [
    {"name": "ollama", "status": "up", "details": {...}},
    {"name": "characters", "status": "up", "details": {...}},
    {"name": "memory", "status": "up", "details": {...}}
  ]
}
```

#### Sesiones

```bash
# Crear sesión
POST /v1/sessions
Content-Type: application/json
{
  "npc_id": "zamir",
  "player_id": "player_123"
}

# Response
{
  "id": "sess_abc123def456",
  "npc_id": "zamir",
  "player_id": "player_123",
  "created_at": "2025-12-21T10:30:00",
  "turn_count": 0,
  "first_message": "*Zamir looks up...*"
}
```

```bash
# Obtener info de sesión
GET /v1/sessions/{session_id}

# Listar sesiones activas
GET /v1/sessions?player_id=player_123

# Terminar sesión
DELETE /v1/sessions/{session_id}
```

#### Chat

```bash
# Chat simple
POST /v1/chat
{
  "session_id": "sess_abc123",
  "message": "¿Qué sabes sobre QDT?"
}

# Response
{
  "session_id": "sess_abc123",
  "npc_id": "zamir",
  "content": "*Zamir's polishing slows* QDT, dices...",
  "actions": [
    {
      "action": "remember",
      "parameters": {"fact": "Pregunta sobre QDT", "importance": 6}
    }
  ],
  "memory_updated": true,
  "turn_number": 1,
  "timestamp": "2025-12-21T10:31:00"
}
```

```bash
# Chat rápido (sin gestión manual de sesión)
POST /v1/chat/quick
{
  "npc_id": "zamir",
  "player_id": "player_123",
  "message": "Hola"
}
```

#### WebSocket Streaming

```javascript
// Conectar
const ws = new WebSocket('ws://localhost:8000/v1/chat/ws/sess_abc123');

// Enviar mensaje
ws.send(JSON.stringify({ message: "Cuéntame sobre tu pasado" }));

// Recibir respuesta (streaming)
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);

  switch(data.type) {
    case 'chunk':
      // Token individual
      process.stdout.write(data.content);
      break;

    case 'done':
      // Respuesta completa
      console.log('\nAcciones:', data.actions);
      break;

    case 'error':
      console.error('Error:', data.error);
      break;
  }
};
```

#### Personajes

```bash
# Listar NPCs disponibles
GET /v1/characters

# Response
{
  "characters": [
    {"name": "Zamir Talaquett", "role": "Bartender...", "traits": [...]}
  ],
  "total": 1
}
```

```bash
# Detalles de un NPC
GET /v1/characters/zamir

# Ver system prompt generado
GET /v1/characters/zamir/system-prompt

# Ver mensaje inicial
GET /v1/characters/zamir/first-message
```

#### Memoria

```bash
# Inyectar memoria manualmente (para eventos del juego)
POST /v1/memory/inject?npc_id=zamir&player_id=player_123
{
  "content": "Este jugador salvó el bar de unos ladrones",
  "importance": 9.0
}
```

---

## Sistema de Memoria

### Tres Capas de Memoria

El sistema implementa el patrón de Stanford Generative Agents / MemGPT:

```
┌─────────────────────────────────────────────────────────────────┐
│                    SISTEMA DE MEMORIA                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                    CORE MEMORY                              │ │
│  │                  (~500 tokens max)                          │ │
│  │                                                              │ │
│  │  Siempre en el prompt. El NPC puede editarla.              │ │
│  │                                                              │ │
│  │  • Nombre del jugador                                       │ │
│  │  • Hechos conocidos (máx 10)                               │ │
│  │  • Nivel de relación (-10 a +10)                           │ │
│  │  • Cantidad de interacciones                                │ │
│  │  • Estado emocional actual del NPC                         │ │
│  │  • Objetivo actual del NPC                                  │ │
│  │                                                              │ │
│  └────────────────────────────────────────────────────────────┘ │
│                           │                                      │
│                           ▼                                      │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                   RECALL MEMORY                             │ │
│  │                 (últimos 20 turnos)                         │ │
│  │                                                              │ │
│  │  Contexto de la conversación actual.                       │ │
│  │                                                              │ │
│  │  Turn 1: Player: "Hola"                                     │ │
│  │  Turn 2: NPC: "Bienvenido..."                              │ │
│  │  Turn 3: Player: "Busco información"                        │ │
│  │  ...                                                        │ │
│  │                                                              │ │
│  └────────────────────────────────────────────────────────────┘ │
│                           │                                      │
│                           ▼                                      │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                  ARCHIVAL MEMORY                            │ │
│  │               (ilimitada, persistente)                      │ │
│  │                                                              │ │
│  │  Memorias a largo plazo con importancia y búsqueda.        │ │
│  │                                                              │ │
│  │  {                                                          │ │
│  │    "content": "El jugador busca trabajo para pagar deudas", │ │
│  │    "importance": 7,                                         │ │
│  │    "topics": ["trabajo", "deudas"],                        │ │
│  │    "emotional_context": "desesperado",                      │ │
│  │    "created_at": "2025-12-20T..."                          │ │
│  │  }                                                          │ │
│  │                                                              │ │
│  │  Búsqueda: recency × importance × relevance                │ │
│  │                                                              │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Fórmula de Recuperación

Cuando el jugador dice algo, el sistema busca memorias relevantes:

```python
score = (recency * 0.3) + (importance * 0.3) + (relevance * 0.4)

# recency: Decae exponencialmente (0.995^horas)
# importance: Normalizado 0-1 (de puntuación 1-10)
# relevance: Overlap de keywords o similaridad semántica
```

### Niveles de Relación

```
-10 ────────────────── 0 ────────────────── +10
 │                     │                     │
 Hostil            Neutral               Confianza
 │                     │                     │
 "Profundamente     "Extraño o          "Amigo de
  desconfía"         conocido"           confianza"
```

El NPC puede cambiar la relación via acciones:
```json
{"action": "update_relationship", "parameters": {"delta": 2}}
```

---

## Motor de NPC

### Context Builder

Ensambla el prompt final combinando:

```
┌─────────────────────────────────────────────────────────────────┐
│                     PROMPT FINAL                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  # You are Zamir Talaquett                                      │
│  Bartender and information broker at 'The Crossed Keys'...      │
│                                                                  │
│  ## Background                                                   │
│  Former merchant sailor who lost his ship...                    │
│                                                                  │
│  ## Personality                                                  │
│  Neutral between factions, Information broker...                │
│                                                                  │
│  ## How you speak                                               │
│  - Measured, unhurried cadence                                  │
│  - Uses metaphors related to drinks and sailing                 │
│                                                                  │
│  ## Your knowledge                                              │
│  Expert in: Local rumors, Station gossip...                     │
│  Don't know: Technical engineering, Military ops...             │
│                                                                  │
│  ## Absolute rules                                              │
│  - NEVER: Reveal faction sympathies                             │
│  - ALWAYS: Offer a drink before business                        │
│                                                                  │
│  ═══════════════════════════════════════════════════════════    │
│                                                                  │
│  ## What you know about this person:                            │
│  - Name: Carlos                                                 │
│  - They are looking for work                                    │
│  - Relationship: This person is a regular you somewhat trust    │
│  - They have visited 5 times                                    │
│                                                                  │
│  ## Your current state:                                         │
│  - Mood: curious                                                │
│                                                                  │
│  ═══════════════════════════════════════════════════════════    │
│                                                                  │
│  ## Relevant memories:                                          │
│  - [observation] Carlos mentioned having debts last week        │
│  - [observation] Carlos asked about QDT shipping schedules      │
│                                                                  │
│  ═══════════════════════════════════════════════════════════    │
│                                                                  │
│  ## Available Actions                                           │
│  - remember(fact, importance): Store important fact             │
│  - update_relationship(delta): Change relationship              │
│  - give_item(item_id, quantity): Give item to player            │
│  ...                                                            │
│                                                                  │
│  ═══════════════════════════════════════════════════════════    │
│                                                                  │
│  ## Response Guidelines                                         │
│  - Stay in character at all times                               │
│  - Keep responses 2-4 sentences                                 │
│  - Use *asterisks* for actions                                  │
│  ...                                                            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Action Parser

Extrae acciones estructuradas de las respuestas del NPC:

**Formatos soportados:**

```
1. JSON Block:
   ```json
   {"action": "give_item", "params": {"item_id": "ale", "quantity": 1}}
   ```

2. Inline Tag:
   [ACTION: remember(fact="Player seeks work", importance=7)]

3. Lenguaje Natural:
   *le entrega una cerveza al jugador*
   → Detecta: give_item
```

---

## Simulador de Conversaciones

### Propósito

Genera datos de entrenamiento sintéticos para:
- Fine-tuning de modelos
- Validación de character cards
- Testing de consistencia

### Uso

```bash
# Generar 100 conversaciones con Zamir
uv run python -m src.main simulate -c zamir -n 100

# Con output detallado
uv run python -m src.main simulate -c zamir -n 50 --verbose

# Modelo diferente
uv run python -m src.main simulate -c zamir -n 20 --model llama3.2

# Temperatura alta (más creatividad)
uv run python -m src.main simulate -c zamir -n 10 -t 0.9
```

### Flujo del Simulador

```
┌─────────────────────────────────────────────────────────────────┐
│                 SIMULADOR DE CONVERSACIONES                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. CARGAR CHARACTER CARD                                       │
│     └─→ zamir.yaml                                              │
│                                                                  │
│  2. GENERAR PERFIL DE CLIENTE (aleatorio)                       │
│     ├─ Arquetipo: courier, merchant, security_officer...       │
│     ├─ Facción: qdt, rebel, neutral                            │
│     ├─ Estado emocional: calm, nervous, desperate...           │
│     ├─ Objetivo: information, work, refuge, social...          │
│     └─ Nivel de confianza: stranger, acquaintance, regular     │
│                                                                  │
│  3. LOOP DE DIÁLOGO                                             │
│     ┌─────────────────────────────────────────────────────┐    │
│     │  Cliente: *se acerca nervioso* Necesito información  │    │
│     │                      ↓                               │    │
│     │  NPC: *deja de pulir el vaso* Información, dices... │    │
│     │                      ↓                               │    │
│     │  Cliente: Sobre los horarios de QDT...              │    │
│     │                      ↓                               │    │
│     │  NPC: *baja la voz* Eso tiene un precio...          │    │
│     │                      ↓                               │    │
│     │  ... (hasta fin natural o max_turns)                │    │
│     └─────────────────────────────────────────────────────┘    │
│                                                                  │
│  4. EXTRAER EXPERIENCIAS                                        │
│     └─→ "Un rebelde nervioso buscó información sobre QDT"      │
│         Importancia: 7, Tópicos: [qdt, información, rebelde]   │
│                                                                  │
│  5. EVALUAR CALIDAD                                             │
│     ├─ Consistencia del personaje: 0.85                        │
│     ├─ Naturalidad: 0.90                                       │
│     └─ Progreso del objetivo: 0.75                             │
│                                                                  │
│  6. GUARDAR JSON                                                │
│     └─→ output/conversations/zamir_20251221_103000.json        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Arquetipos de Clientes

| Facción | Arquetipos |
|---------|------------|
| **QDT** | corporate_suit, security_officer, dock_supervisor |
| **Rebel** | courier, recruiter, veteran_fighter |
| **Neutral** | merchant_captain, station_worker, bounty_hunter, refugee, entertainer |

| Estado Emocional | Comportamiento |
|------------------|----------------|
| calm | Relajado, paciente, conversacional |
| nervous | Ansioso, mira alrededor, habla rápido |
| desperate | Urgente, ofrece más de lo normal |
| suspicious | Cauteloso, pregunta más de lo que responde |
| drunk | Revelador, emocional, impredecible |

### Output JSON

```json
{
  "metadata": {
    "character": "zamir",
    "model": "phi3.5",
    "generated_at": "2025-12-21T10:30:00",
    "count": 100,
    "summary": {
      "avg_turns": 8.5,
      "avg_consistency": 0.85,
      "natural_endings": 82
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
        {"speaker": "customer", "message": "*se acerca nervioso* ..."},
        {"speaker": "npc", "message": "*levanta la vista* ..."}
      ],
      "extracted_experiences": [
        {
          "description": "Un courier rebelde nervioso buscó rutas seguras",
          "importance": 7,
          "topics": ["rebel", "courier", "routes"]
        }
      ],
      "metrics": {
        "num_turns": 8,
        "character_consistency": 0.87,
        "conversation_naturalness": 0.92,
        "conversation_ended_naturally": true
      }
    }
  ]
}
```

---

## Character Cards

### Esquema Completo

```yaml
# Obligatorios
name: "Nombre del Personaje"
role: "Descripción breve del rol"

personality:
  traits:                        # Lista de rasgos de personalidad
    - "Rasgo 1"
    - "Rasgo 2"

  speech_patterns:               # Cómo habla el personaje
    - "Patrón de habla 1"
    - "Patrón de habla 2"

  big_five:                      # Opcional: personalidad OCEAN
    openness: 0.6                # 0-1
    conscientiousness: 0.7
    extraversion: 0.5
    agreeableness: 0.4
    neuroticism: 0.3

# Opcionales pero recomendados
backstory: |
  Historia del personaje.
  Puede ser multilínea.

knowledge_domains:
  expert:                        # Áreas de expertise
    - "Tema experto 1"
  familiar:                      # Conocimiento básico
    - "Tema familiar 1"
  ignorant:                      # No sabe nada de esto
    - "Tema ignorado 1"

behavioral_boundaries:
  never:                         # NUNCA hace esto
    - "Comportamiento prohibido 1"
  always:                        # SIEMPRE hace esto
    - "Comportamiento obligatorio 1"

first_message: |
  Mensaje inicial cuando empieza una conversación.

example_dialogues:               # Few-shot examples
  - context: "Descripción del contexto"
    customer: "Lo que dice el cliente"
    npc: "Respuesta del NPC"
```

### Ejemplo: Zamir

Ver archivo completo en `src/characters/templates/zamir.yaml`

```yaml
name: "Zamir Talaquett"
role: "Bartender and information broker at 'The Crossed Keys' tavern"

personality:
  traits:
    - "Neutral between QDT and Rebels - survival through impartiality"
    - "Information broker who trades secrets as currency"
    - "World-weary but not cynical"

  speech_patterns:
    - "Measured, unhurried cadence - never rushes words"
    - "Uses metaphors related to drinks and sailing"
    - "Deflects personal questions with humor"

behavioral_boundaries:
  never:
    - "Reveal personal faction sympathies"
    - "Give information freely to strangers"
    - "Break confidence of trusted regulars"
  always:
    - "Offer a drink before business"
    - "Remember returning customers"
```

### Crear un Nuevo NPC

1. Crear archivo YAML en `src/characters/templates/mi_npc.yaml`
2. Definir al menos: name, role, personality.traits
3. Probar: `uv run python -m src.main simulate -c mi_npc -n 5`

---

## Sistema de Acciones

### Acciones Disponibles

| Acción | Parámetros | Descripción |
|--------|------------|-------------|
| `remember` | fact (str), importance (1-10) | Guardar un hecho sobre el jugador |
| `update_relationship` | delta (-3 to +3), reason (str) | Cambiar nivel de relación |
| `set_emotion` | emotion (str) | Cambiar estado emocional del NPC |
| `give_item` | item_id (str), quantity (int) | Dar un item al jugador |
| `start_quest` | quest_id (str) | Iniciar una quest |
| `end_conversation` | reason (str) | Terminar la conversación |

### Cómo el NPC Usa Acciones

El NPC incluye acciones en su respuesta:

```
*Zamir asiente lentamente* Pareces alguien de fiar. Ten, esto te
ayudará en tu camino.
[ACTION: give_item(item_id="old_map", quantity=1)]
[ACTION: update_relationship(delta=1)]
[ACTION: remember(fact="Ayudé al jugador con un mapa", importance=5)]
```

### Acciones Personalizadas

En `src/engine/action_parser.py`:

```python
CUSTOM_ACTIONS = {
    "call_guards": {
        "description": "Call the tavern guards",
        "parameters": {
            "reason": {"type": "string"}
        }
    },
    "offer_discount": {
        "description": "Offer a discount on services",
        "parameters": {
            "percentage": {"type": "integer", "default": 10}
        }
    }
}
```

---

## Integración con Juegos

### JavaScript / Web

```javascript
class NPCClient {
  constructor(baseUrl = 'http://localhost:8000') {
    this.baseUrl = baseUrl;
    this.sessionId = null;
  }

  async startSession(npcId, playerId) {
    const res = await fetch(`${this.baseUrl}/v1/sessions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ npc_id: npcId, player_id: playerId })
    });
    const data = await res.json();
    this.sessionId = data.id;
    return data;
  }

  async chat(message) {
    const res = await fetch(`${this.baseUrl}/v1/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: this.sessionId, message })
    });
    return res.json();
  }

  async endSession() {
    await fetch(`${this.baseUrl}/v1/sessions/${this.sessionId}`, {
      method: 'DELETE'
    });
  }

  // WebSocket para streaming
  connectStreaming(onChunk, onDone) {
    const ws = new WebSocket(`ws://${this.baseUrl.replace('http://', '')}/v1/chat/ws/${this.sessionId}`);

    ws.onmessage = (e) => {
      const data = JSON.parse(e.data);
      if (data.type === 'chunk') onChunk(data.content);
      else if (data.type === 'done') onDone(data);
    };

    return {
      send: (message) => ws.send(JSON.stringify({ message })),
      close: () => ws.close()
    };
  }
}

// Uso
const npc = new NPCClient();
const session = await npc.startSession('zamir', 'player_1');
console.log('NPC dice:', session.first_message);

const response = await npc.chat('Hola, busco trabajo');
console.log('NPC responde:', response.content);

// Procesar acciones
for (const action of response.actions) {
  if (action.action === 'give_item') {
    gameInventory.add(action.parameters.item_id, action.parameters.quantity);
  }
}
```

### Unity (C#)

```csharp
using System;
using System.Collections;
using UnityEngine;
using UnityEngine.Networking;

[Serializable]
public class SessionRequest { public string npc_id; public string player_id; }
[Serializable]
public class ChatRequest { public string session_id; public string message; }
[Serializable]
public class ChatResponse { public string content; public Action[] actions; public int turn_number; }
[Serializable]
public class Action { public string action; public ActionParams parameters; }

public class NPCService : MonoBehaviour
{
    private string baseUrl = "http://localhost:8000";
    private string sessionId;

    public IEnumerator StartSession(string npcId, string playerId, System.Action<string> onFirstMessage)
    {
        var request = new SessionRequest { npc_id = npcId, player_id = playerId };
        var json = JsonUtility.ToJson(request);

        using (var www = new UnityWebRequest($"{baseUrl}/v1/sessions", "POST"))
        {
            www.uploadHandler = new UploadHandlerRaw(System.Text.Encoding.UTF8.GetBytes(json));
            www.downloadHandler = new DownloadHandlerBuffer();
            www.SetRequestHeader("Content-Type", "application/json");

            yield return www.SendWebRequest();

            if (www.result == UnityWebRequest.Result.Success)
            {
                var response = JsonUtility.FromJson<SessionResponse>(www.downloadHandler.text);
                sessionId = response.id;
                onFirstMessage?.Invoke(response.first_message);
            }
        }
    }

    public IEnumerator Chat(string message, System.Action<ChatResponse> onResponse)
    {
        var request = new ChatRequest { session_id = sessionId, message = message };
        var json = JsonUtility.ToJson(request);

        using (var www = new UnityWebRequest($"{baseUrl}/v1/chat", "POST"))
        {
            www.uploadHandler = new UploadHandlerRaw(System.Text.Encoding.UTF8.GetBytes(json));
            www.downloadHandler = new DownloadHandlerBuffer();
            www.SetRequestHeader("Content-Type", "application/json");

            yield return www.SendWebRequest();

            if (www.result == UnityWebRequest.Result.Success)
            {
                var response = JsonUtility.FromJson<ChatResponse>(www.downloadHandler.text);
                ProcessActions(response.actions);
                onResponse?.Invoke(response);
            }
        }
    }

    private void ProcessActions(Action[] actions)
    {
        foreach (var action in actions)
        {
            switch (action.action)
            {
                case "give_item":
                    Inventory.Instance.AddItem(action.parameters.item_id, action.parameters.quantity);
                    break;
                case "start_quest":
                    QuestManager.Instance.StartQuest(action.parameters.quest_id);
                    break;
            }
        }
    }
}
```

### Godot (GDScript)

```gdscript
extends Node

var base_url = "http://localhost:8000"
var session_id = ""

func start_session(npc_id: String, player_id: String) -> Dictionary:
    var http = HTTPRequest.new()
    add_child(http)

    var body = JSON.stringify({"npc_id": npc_id, "player_id": player_id})
    var headers = ["Content-Type: application/json"]

    http.request(base_url + "/v1/sessions", headers, HTTPClient.METHOD_POST, body)
    var response = await http.request_completed
    http.queue_free()

    var json = JSON.parse_string(response[3].get_string_from_utf8())
    session_id = json["id"]
    return json

func chat(message: String) -> Dictionary:
    var http = HTTPRequest.new()
    add_child(http)

    var body = JSON.stringify({"session_id": session_id, "message": message})
    var headers = ["Content-Type: application/json"]

    http.request(base_url + "/v1/chat", headers, HTTPClient.METHOD_POST, body)
    var response = await http.request_completed
    http.queue_free()

    var json = JSON.parse_string(response[3].get_string_from_utf8())
    process_actions(json["actions"])
    return json

func process_actions(actions: Array):
    for action in actions:
        match action["action"]:
            "give_item":
                Inventory.add_item(action["parameters"]["item_id"])
            "start_quest":
                QuestManager.start(action["parameters"]["quest_id"])
```

---

## Próximos Pasos

### Fase Actual: MVP Funcional ✅

- [x] API REST con FastAPI
- [x] WebSocket streaming
- [x] Sistema de memoria (Core, Recall, Archival)
- [x] Motor de NPC con acciones
- [x] Simulador de conversaciones
- [x] Playground web
- [x] Character card de Zamir

### Fase 2: Persistencia

- [ ] **Redis** para memoria entre reinicios
  ```bash
  # Instalar Redis
  docker run -d -p 6379:6379 redis

  # Configurar
  REDIS_URL=redis://localhost:6379
  ```

- [ ] **Qdrant** para búsqueda semántica
  ```bash
  # Instalar Qdrant
  docker run -d -p 6333:6333 qdrant/qdrant

  # Configurar
  QDRANT_URL=http://localhost:6333
  ```

### Fase 3: Mejoras de Calidad

- [ ] Embeddings para archival memory (sentence-transformers)
- [ ] Evaluación LLM de consistencia (no solo heurísticas)
- [ ] Reflection engine (generar insights de alto nivel)
- [ ] Importancia automática via LLM

### Fase 4: Escalabilidad

- [ ] Rate limiting por API key
- [ ] Multi-tenancy (múltiples juegos)
- [ ] Métricas y dashboard
- [ ] Caché de respuestas frecuentes

### Fase 5: Routing Inteligente

- [ ] Clasificador de complejidad
- [ ] FunctionsGemma para acciones simples
- [ ] Cloud API para narrativa compleja
- [ ] Fallback graceful

---

## Referencia de API

### Resumen de Endpoints

```
┌─────────────────────────────────────────────────────────────────┐
│                     API ENDPOINTS                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  HEALTH                                                         │
│  GET    /v1/health              Estado básico                   │
│  GET    /v1/health/detailed     Estado detallado                │
│  GET    /v1/models              Modelos LLM disponibles         │
│                                                                  │
│  CHARACTERS                                                      │
│  GET    /v1/characters          Listar NPCs                     │
│  GET    /v1/characters/{id}     Detalles de NPC                 │
│  GET    /v1/characters/{id}/system-prompt    Ver prompt         │
│  GET    /v1/characters/{id}/first-message    Mensaje inicial    │
│                                                                  │
│  SESSIONS                                                        │
│  POST   /v1/sessions            Crear sesión                    │
│  GET    /v1/sessions            Listar sesiones                 │
│  GET    /v1/sessions/{id}       Info de sesión                  │
│  DELETE /v1/sessions/{id}       Terminar sesión                 │
│  GET    /v1/sessions/{id}/memory    Ver memoria de sesión       │
│                                                                  │
│  CHAT                                                           │
│  POST   /v1/chat                Chat con NPC                    │
│  POST   /v1/chat/quick          Chat sin sesión manual          │
│  GET    /v1/chat/stream/{id}    SSE streaming                   │
│  WS     /v1/chat/ws/{id}        WebSocket streaming             │
│                                                                  │
│  MEMORY                                                          │
│  POST   /v1/memory/inject       Inyectar memoria                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Códigos de Estado

| Código | Significado |
|--------|-------------|
| 200 | Éxito |
| 201 | Creado (sesión nueva) |
| 400 | Bad request (parámetros inválidos) |
| 404 | No encontrado (sesión, NPC) |
| 503 | Servicio no disponible (Ollama caído) |

### Errores Comunes

```json
// NPC no encontrado
{"detail": "Character 'unknown' not found. Available: zamir"}

// Sesión no encontrada
{"detail": "Session not found: sess_invalid"}

// Ollama no disponible
{"detail": "LLM client not configured"}
```

---

## Licencia

MIT License

---

*Documentación generada el 2025-12-21*
*NPC Service v0.2.0*
