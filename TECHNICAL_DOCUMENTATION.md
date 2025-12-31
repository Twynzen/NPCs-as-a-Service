# Documentación Técnica de NPC Service

> **Versión del Sistema:** 0.3.0 (RAG-Enabled)
> **Última Actualización:** Diciembre 2025

Este documento proporciona la referencia técnica definitiva del **NPC Service**, una plataforma de IA para videojuegos que transforma personajes estáticos en entidades dinámicas con memoria persistente, razonamiento contextual y capacidades de acción.

La versión 0.3.0 introduce un cambio de paradigma: el paso de una memoria basada en palabras clave a una arquitectura **RAG (Retrieval-Augmented Generation) Híbrida**, permitiendo a los NPCs comprender el significado semántico detrás de las interacciones.

---

## 1. Arquitectura de Alto Nivel

El sistema actúa como un middleware inteligente entre el Cliente de Juego y los Modelos de Lenguaje (LLM).

```mermaid
graph TD
    Client[Game Client / Web] <--> API[FastAPI Gateway]
    API <--> Engine[NPC Engine]
    
    subgraph "Knowledge & Memory (RAG Core)"
        Engine --> RAG[Hybrid Retriever]
        RAG --> VectorDB[(Vector Store: Qdrant)]
        RAG --> BM25[BM25 Index]
        RAG --> Embed[Embedding Service]
    end
    
    subgraph "Runtime Logic"
        Engine --> Ctx[Context Builder]
        Engine --> Parser[Action Parser]
        Engine --> State[Session Manager]
    end

    subgraph "Inference Layer"
        Embed <--> Ollama_Embed[Ollama (nomic-embed-text)]
        Engine <--> Ollama_Chat[Ollama (phi3.5)]
    end
```

---

## 2. Subsistema de Memoria RAG (Retrieval-Augmented Generation)

Este es el núcleo de la inteligencia del sistema. A diferencia de los chatbots tradicionales que solo ven los últimos mensajes, NPC Service utiliza un sistema de recuperación híbrida para acceder a recuerdos a largo plazo.

### 2.1 El Problema de la Memoria
Los sistemas anteriores usaban búsqueda por palabras clave (`keyword matching`).
- **Fallo:** Si el usuario decía "mi hogar", el sistema no encontraba recuerdos sobre "mi casa".
- **Solución:** Implementar vectores semánticos que entienden que "hogar" ≈ "casa".

### 2.2 Estrategia de Búsqueda Híbrida
Para garantizar la máxima precisión, el sistema no confía solo en vectores. Utiliza una estrategia dual implementada en `src/rag/retriever.py`:

1.  **Búsqueda Vectorial (Semántica):**
    -   Convierte la consulta del usuario en un vector de 768 dimensiones (usando `nomic-embed-text`).
    -   Busca en la base de datos vectores cercanos (similitud de coseno).
    -   *Ventaja:* Entiende sinónimos, paráfrasis y conceptos abstractos.
2.  **Búsqueda BM25 (Léxica):**
    -   Utiliza algoritmos probabilísticos para encontrar coincidencias exactas de términos raros.
    -   *Ventaja:* Crucial para nombres propios inventados ("Zarathos", "Korrath"), IDs o terminología específica del lore que el modelo de embeddings puede desconcoer.
3.  **Fusión RRF (Reciprocal Rank Fusion):**
    -   Combina ambos rankings usando la fórmula:
        $$Score(d) = \sum \frac{1}{k + rank(d)}$$
    -   Esto asegura que un documento que es relevante en *ambos* métodos suba a la cima.

### 2.3 Algoritmo de Puntuación de Memoria
Una vez recuperados los candidatos, la `ArchivalMemory` aplica un filtro final basado en la "Fórmula de Relevancia" inspirada en la arquitectura cognitiva humana:

$$Score = (w_1 \cdot Recencia) + (w_2 \cdot Importancia) + (w_3 \cdot Relevancia)$$

-   **Recencia:** Decaimiento exponencial ($0.995^h$). Los recuerdos recientes son más vívidos.
-   **Importancia:** Valor intrínseco (1-10) asignado por el LLM al momento de crear el recuerdo.
-   **Relevancia:** El score calculado por el sistema híbrido RAG.

---

## 3. Componentes de Infraestructura

### 3.1 Almacenamiento Vectorial (`src/rag/vector_store.py`)
El sistema abstrae el almacenamiento vectorial para permitir flexibilidad:

*   **Qdrant (Producción):**
    *   Utiliza indexación HNSW (Hierarchical Navigable Small World) para búsquedas en tiempo logarítmico $O(\log n)$.
    *   Habilita filtrado nativo por Payload (Metadata) para aislar memorias por `npc_id` y `player_id`.
*   **InMemory (Desarrollo):**
    *   Implementación ligera usando cálculos de coseno con NumPy/Python puro.
    *   Ideal para pruebas unitarias o despliegues sin dependencias de Docker.

### 3.2 Servicio de Embeddings (`src/rag/embeddings.py`)
*   **Modelo:** Estandarizado en `nomic-embed-text` por su ventana de contexto de 8192 tokens y alta calidad de compresión semántica.
*   **Caché Inteligente:** Implementa un sistema de caché LRU (Least Recently Used) basado en el hash SHA-256 del contenido. Esto evita latencia innecesaria re-calculando vectores para frases comunes.

---

## 4. API & Endpoints (Referencia V1)

### Gestión de Sesiones
| Método | Endpoint | Descripción | Payload |
|--------|----------|-------------|---------|
| `POST` | `/v1/sessions` | Crea sesión persistente | `{"npc_id": "marcus", "player_id": "p1"}` |
| `GET` | `/v1/sessions` | Lista sesiones activas | `?player_id=p1` |

### Chat & Interacción
| Método | Endpoint | Descripción | Payload |
|--------|----------|-------------|---------|
| `POST` | `/v1/chat` | Chat síncrono (JSON) | `{"session_id": "...", "message": "Hi"}` |
| `POST` | `/v1/chat/quick` | Chat "One-Shot" (Auto-session) | `{"npc_id": "...", "player_id": "...", "message": "..."}` |
| `GET` | `/v1/chat/stream/{id}` | Streaming (Server-Sent Events) | `?message=Hello` |
| `WS` | `/v1/chat/ws/{id}` | Real-time WebSocket | `{ "message": "..." }` |

### Gestión de Memoria & Lore
| Método | Endpoint | Descripción | Payload |
|--------|----------|-------------|---------|
| `POST` | `/v1/memory/inject` | Inyectar "God Knowledge" | `{"npc_id": "...", "content": "War started", "importance": 10}` |
| `GET` | `/v1/characters` | Catálogo de NPCs | - |
| `GET` | `/v1/health/detailed` | Diagnóstico profundo | - |

---

## 5. Guía de Simulación y Entrenamiento

El sistema incluye un entorno de simulación (`src/simulation`) para "entrenar" la memoria de los NPCs antes de lanzarlos a producción.

### Ciclo de Simulación
1.  **Generación de Cliente:** `CustomerGenerator` crea un perfil psicológico aleatorio (ej. "Rebelde paranoico buscando refugio").
2.  **Interacción:** Se ejecuta un bucle de conversación NPC-Cliente hasta llegar a un final natural (detectado por patrones como `[END]` o `*leaves*`).
3.  **Extracción de Experiencias:**
    *   El módulo `ExperienceExtractor` analiza el transcript completo.
    *   Usa el LLM para identificar "Memorias Episódicas" clave.
    *   Estas memorias se vectorizan y guardan en la `ArchivalMemory` del NPC.
    
*Resultado:* Un NPC que ya tiene "experiencia de vida" y recuerdos antes de hablar con el primer jugador real.

---

## 6. Configuración del Sistema

Variables de entorno críticas en `.env` o `src/config.py`:

| Variable | Uso | Recomendado |
|----------|-----|-------------|
| `OLLAMA_URL` | Endpoint de inferencia | `http://localhost:11434` |
| `OLLAMA_MODEL` | Modelo de Chat | `phi3.5` (rápido) o `llama3` (calidad) |
| `EMBEDDING_MODEL` | Modelo Vectorial | `nomic-embed-text` |
| `QDRANT_URL` | Base de datos vectorial | `http://localhost:6333` (si se usa) |
| `REDIS_URL` | Caché de sesiones | `redis://localhost:6379` |

---

## 7. Extensibilidad: Agnóstico del Personaje

El sistema **NO** está hardcodeado para ningún personaje.
-   El motor carga dinámicamente cualquier archivo `.yaml` o `.json` ubicado en `src/characters/templates/`.
-   **Ejemplo de creación rápida:**
    Cree `src/characters/templates/golem.yaml`:
    ```yaml
    name: "Grok"
    role: "Golem de Piedra Guardián"
    personality:
      traits: ["Lento", "Literal", "Protector"]
      speech_patterns: ["Habla en tercera persona", "Usa palabras de geología"]
    behavioral_boundaries:
      always: ["Obedecer al creador"]
    first_message: "*Rumble* Grok vigila. Tú... ¿quién?"
    ```
    El sistema indexará automáticamente a "Grok" en el siguiente reinicio, creando su propio espacio vectorial y memoria aislada.

---

## 8. Comandos de CLI (Cheatsheet)

| Comando | Acción |
|---------|--------|
| `uv run python -m src.main check` | Verifica conexión con Ollama y estado de modelos |
| `uv run python -m src.main serve --reload` | Inicia el servidor API en modo desarrollo |
| `uv run python -m src.main list-npcs` | Muestra todos los NPCs cargados válidos |
| `uv run python -m src.main simulate -c [id] -n 5` | Genera 5 conversaciones sintéticas para el NPC [id] |

---

> **Nota de Implementación:** Si Qdrant no está disponible, el sistema hará un *fallback* automático a `InMemoryVectorStore`. Esto permite probar el sistema RAG completo simplemente clonando el repo, sin necesidad de configurar contenedores Docker adicionales.