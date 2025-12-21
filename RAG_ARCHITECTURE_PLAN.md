# RAG Architecture Plan para NPC Service

> **Documento de Investigación y Planificación**
> Versión: 1.0
> Fecha: 2024
> Objetivo: Guía completa para implementar RAG en el sistema de NPCs

---

## Tabla de Contenidos

1. [Resumen Ejecutivo](#resumen-ejecutivo)
2. [Arquitectura Actual](#arquitectura-actual)
3. [Arquitectura Propuesta con RAG](#arquitectura-propuesta-con-rag)
4. [Comparativa Detallada](#comparativa-detallada)
5. [Plan de Implementación](#plan-de-implementación)
6. [Tecnologías a Investigar](#tecnologías-a-investigar)
7. [Recursos y Documentación](#recursos-y-documentación)
8. [Ejemplos de Código de Referencia](#ejemplos-de-código-de-referencia)
9. [Decisiones Arquitectónicas](#decisiones-arquitectónicas)
10. [Métricas de Éxito](#métricas-de-éxito)

---

## Resumen Ejecutivo

### El Problema
El sistema actual de memoria usa **búsqueda por palabras clave**, lo cual:
- No entiende sinónimos ni contexto semántico
- Falla cuando el jugador parafrasea
- No puede conectar conceptos relacionados
- Limita la "inteligencia" percibida del NPC

### La Solución
Implementar **RAG (Retrieval-Augmented Generation)** para:
- Búsqueda semántica de memorias
- Recuperación de contexto relevante por significado
- NPCs que "entienden" lo que el jugador quiere decir
- Memoria asociativa similar a la humana

### Impacto Esperado
| Métrica | Actual | Con RAG |
|---------|--------|---------|
| Precisión de recuperación de memorias | ~30% | ~85% |
| Relevancia del contexto | Baja | Alta |
| Coherencia en conversaciones largas | Degrada | Mantiene |
| Experiencia del jugador | Robótica | Natural |

---

## Arquitectura Actual

### Diagrama de Componentes Actual

```
┌─────────────────────────────────────────────────────────────────────┐
│                         NPC SERVICE v0.2.0                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   ┌──────────────┐     ┌──────────────┐     ┌──────────────┐        │
│   │   FastAPI    │────▶│  NPC Engine  │────▶│  Ollama LLM  │        │
│   │   Server     │     │              │     │              │        │
│   └──────────────┘     └──────┬───────┘     └──────────────┘        │
│                               │                                      │
│                               ▼                                      │
│                    ┌──────────────────────┐                         │
│                    │   Memory Manager     │                         │
│                    └──────────┬───────────┘                         │
│                               │                                      │
│          ┌────────────────────┼────────────────────┐                │
│          ▼                    ▼                    ▼                │
│   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐          │
│   │    Core     │     │   Recall    │     │  Archival   │          │
│   │   Memory    │     │   Memory    │     │   Memory    │          │
│   │ (In-Memory) │     │ (In-Memory) │     │ (In-Memory) │          │
│   │             │     │             │     │             │          │
│   │ - Player    │     │ - Last 20   │     │ - Keyword   │          │
│   │   info      │     │   turns     │     │   search    │          │
│   │ - Relations │     │ - Session   │     │ - No embed  │          │
│   │ - State     │     │   history   │     │ - Scoring   │          │
│   └─────────────┘     └─────────────┘     └─────────────┘          │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Flujo de Búsqueda Actual

```
Player Message: "¿Recuerdas lo que te conté sobre mi familia?"
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │      TOKENIZACIÓN SIMPLE      │
                    │                               │
                    │  ["recuerdas", "lo", "que",   │
                    │   "te", "conté", "sobre",     │
                    │   "mi", "familia"]            │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │     BÚSQUEDA POR KEYWORDS     │
                    │                               │
                    │  for memory in memories:      │
                    │    words = memory.split()     │
                    │    overlap = query ∩ words    │
                    │    score = len(overlap)       │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │         RESULTADO             │
                    │                               │
                    │  Memoria: "Tu familia vive    │
                    │  en el norte" ✓ (match:       │
                    │  "familia")                   │
                    │                               │
                    │  Memoria: "Pedro, tu hermano, │
                    │  es minero" ✗ (no match,      │
                    │  aunque ES sobre familia)     │
                    └───────────────────────────────┘
```

### Archivos Relevantes Actuales

```
src/
├── memory/
│   ├── __init__.py
│   ├── core_memory.py      # Memoria siempre en contexto
│   ├── recall_memory.py    # Últimos N turnos
│   ├── archival_memory.py  # ⚠️ AQUÍ ESTÁ EL PROBLEMA
│   └── manager.py          # Orquestador
├── engine/
│   ├── npc_engine.py       # Motor principal
│   ├── context_builder.py  # Construye prompts
│   └── action_parser.py    # Parsea acciones
└── server/
    └── ...
```

### Código Problemático Actual

**`src/memory/archival_memory.py` líneas 118-143:**

```python
def _keyword_search(self, query: str, top_k: int, min_importance: float) -> list[Memory]:
    """Simple keyword-based search with scoring."""
    query_words = set(query.lower().split())  # ← Solo divide por espacios

    scored_memories = []
    for memory in self._local_memories:
        if memory.importance < min_importance:
            continue

        # ← PROBLEMA: Solo compara palabras exactas
        memory_words = set(memory.content.lower().split())
        memory_words.update(t.lower() for t in memory.topics)
        overlap = len(query_words & memory_words)
        relevance = overlap / max(len(query_words), 1)

        # ...scoring...
```

### Limitaciones Identificadas

| Limitación | Impacto | Ejemplo |
|------------|---------|---------|
| **Sin sinónimos** | No encuentra memorias con palabras diferentes | "casa" no encuentra "hogar", "vivienda" |
| **Sin contexto semántico** | No entiende relaciones conceptuales | "hermano" no relaciona con "familia" |
| **Sin embeddings** | Campo `embedding` existe pero siempre es `None` | Desperdicio de estructura |
| **Sin persistencia** | Memorias se pierden al reiniciar | Solo viven en RAM |
| **Sin indexación** | Búsqueda O(n) lineal | Lento con muchas memorias |

---

## Arquitectura Propuesta con RAG

### Diagrama de Componentes con RAG

```
┌────────────────────────────────────────────────────────────────────────────┐
│                         NPC SERVICE v0.3.0 (con RAG)                       │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌──────────────┐     ┌──────────────┐     ┌──────────────┐               │
│   │   FastAPI    │────▶│  NPC Engine  │────▶│  Ollama LLM  │               │
│   │   Server     │     │              │     │  (chat)      │               │
│   └──────────────┘     └──────┬───────┘     └──────────────┘               │
│                               │                                             │
│                               ▼                                             │
│                    ┌──────────────────────┐     ┌──────────────┐           │
│                    │   RAG Orchestrator   │────▶│ Ollama Embed │           │
│                    │      (NUEVO)         │     │ (embeddings) │           │
│                    └──────────┬───────────┘     └──────────────┘           │
│                               │                                             │
│          ┌────────────────────┼────────────────────┐                       │
│          ▼                    ▼                    ▼                       │
│   ┌─────────────┐     ┌─────────────┐     ┌─────────────────────┐         │
│   │    Core     │     │   Recall    │     │   Archival Memory   │         │
│   │   Memory    │     │   Memory    │     │     (MEJORADO)      │         │
│   │  (Redis)    │     │  (Redis)    │     │                     │         │
│   │             │     │             │     │  ┌───────────────┐  │         │
│   │ - Player    │     │ - Last 20   │     │  │ Vector Store  │  │         │
│   │   info      │     │   turns     │     │  │ (ChromaDB/    │  │         │
│   │ - Relations │     │ - Session   │     │  │  Qdrant)      │  │         │
│   │ - State     │     │   history   │     │  └───────────────┘  │         │
│   └─────────────┘     └─────────────┘     │                     │         │
│                                           │  ┌───────────────┐  │         │
│                                           │  │  Hybrid       │  │         │
│                                           │  │  Search       │  │         │
│                                           │  │  (BM25+Vector)│  │         │
│                                           │  └───────────────┘  │         │
│                                           └─────────────────────┘         │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                     KNOWLEDGE BASES (Indexadas)                      │  │
│   │                                                                      │  │
│   │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │  │
│   │  │  NPC Lore    │  │   World      │  │   Player     │               │  │
│   │  │  Knowledge   │  │   Events     │  │   Histories  │               │  │
│   │  │              │  │              │  │              │               │  │
│   │  │ - Backstory  │  │ - News       │  │ - Per-player │               │  │
│   │  │ - Opinions   │  │ - Changes    │  │   memories   │               │  │
│   │  │ - Secrets    │  │ - Rumors     │  │ - Summaries  │               │  │
│   │  └──────────────┘  └──────────────┘  └──────────────┘               │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└────────────────────────────────────────────────────────────────────────────┘
```

### Flujo de Búsqueda con RAG

```
Player Message: "¿Recuerdas lo que te conté sobre mi familia?"
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │      EMBEDDING GENERATION     │
                    │                               │
                    │  Ollama nomic-embed-text:     │
                    │  [0.023, -0.156, 0.891, ...]  │
                    │  (768 dimensiones)            │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │      HYBRID SEARCH            │
                    │                               │
                    │  1. Vector similarity (0.7)   │
                    │  2. BM25 keyword (0.3)        │
                    │                               │
                    │  Combina ambos scores         │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │    RESULTADOS SEMÁNTICOS      │
                    │                               │
                    │  ✓ "Tu familia vive en el     │
                    │     norte" (sim: 0.89)        │
                    │                               │
                    │  ✓ "Pedro, tu hermano, es     │
                    │     minero" (sim: 0.85)       │
                    │                               │
                    │  ✓ "Mencionaste que tu madre  │
                    │     está enferma" (sim: 0.82) │
                    │                               │
                    │  ✓ "Tienes una hermana menor  │
                    │     llamada Luna" (sim: 0.79) │
                    └───────────────────────────────┘
```

### Nuevos Componentes Necesarios

```
src/
├── memory/
│   ├── __init__.py
│   ├── core_memory.py
│   ├── recall_memory.py
│   ├── archival_memory.py      # ← REFACTORIZAR
│   ├── manager.py              # ← REFACTORIZAR
│   └── vector_store.py         # ← NUEVO
├── rag/                        # ← NUEVO MÓDULO
│   ├── __init__.py
│   ├── embeddings.py           # Generación de embeddings
│   ├── retriever.py            # Lógica de recuperación
│   ├── indexer.py              # Indexación de documentos
│   ├── chunker.py              # División de textos largos
│   └── reranker.py             # Re-ranking de resultados
├── knowledge/                  # ← NUEVO MÓDULO
│   ├── __init__.py
│   ├── npc_lore.py             # Conocimiento del NPC
│   ├── world_events.py         # Eventos del mundo
│   └── loader.py               # Carga de knowledge bases
├── engine/
│   ├── npc_engine.py           # ← MODIFICAR
│   ├── context_builder.py      # ← MODIFICAR
│   └── action_parser.py
└── server/
    └── ...
```

---

## Comparativa Detallada

### Tabla Comparativa General

| Aspecto | Arquitectura Actual | Arquitectura con RAG |
|---------|--------------------|--------------------|
| **Búsqueda** | Keyword matching | Semantic + Hybrid |
| **Almacenamiento** | In-memory dict | Vector DB + KV Store |
| **Embeddings** | No implementado | Ollama/OpenAI |
| **Persistencia** | No (se pierde) | Sí (persistente) |
| **Escalabilidad** | ~1000 memorias | ~1M+ memorias |
| **Latencia búsqueda** | O(n) lineal | O(log n) indexado |
| **Relevancia** | ~30% accuracy | ~85% accuracy |
| **Dependencias** | Mínimas | Vector DB + Embed model |

### Comparativa de Flujos

#### Flujo Actual
```
┌─────────┐    ┌──────────┐    ┌──────────┐    ┌─────────┐
│ Player  │───▶│ Tokenize │───▶│ Keyword  │───▶│ Results │
│ Message │    │ (split)  │    │ Match    │    │ (poor)  │
└─────────┘    └──────────┘    └──────────┘    └─────────┘
     │                              │
     │         Simple, rápido       │
     │         pero impreciso       │
     └──────────────────────────────┘
```

#### Flujo con RAG
```
┌─────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌─────────┐
│ Player  │───▶│ Embed    │───▶│ Vector   │───▶│ Rerank   │───▶│ Results │
│ Message │    │ (Ollama) │    │ Search   │    │ (opt)    │    │ (good)  │
└─────────┘    └──────────┘    └──────────┘    └──────────┘    └─────────┘
     │                                              │
     │         Más complejo, pero                   │
     │         semánticamente preciso               │
     └──────────────────────────────────────────────┘
```

### Comparativa de Código

#### Búsqueda Actual
```python
def search(self, query: str, top_k: int = 5) -> list[Memory]:
    query_words = set(query.lower().split())

    scored = []
    for memory in self._local_memories:
        memory_words = set(memory.content.lower().split())
        overlap = len(query_words & memory_words)
        relevance = overlap / max(len(query_words), 1)
        scored.append((memory, relevance))

    return sorted(scored, key=lambda x: x[1], reverse=True)[:top_k]
```

#### Búsqueda con RAG (Propuesta)
```python
async def search(self, query: str, top_k: int = 5) -> list[Memory]:
    # 1. Generar embedding del query
    query_embedding = await self.embeddings.embed(query)

    # 2. Búsqueda vectorial
    vector_results = await self.vector_store.similarity_search(
        embedding=query_embedding,
        top_k=top_k * 2,  # Over-fetch for reranking
        filter={"npc_id": self.npc_id, "player_id": self.player_id}
    )

    # 3. Búsqueda por keywords (BM25) - opcional
    keyword_results = self.bm25_search(query, top_k=top_k)

    # 4. Combinar y re-rankear (Hybrid Search)
    combined = self.hybrid_merge(vector_results, keyword_results)

    # 5. Re-ranking final (opcional, mejora calidad)
    if self.reranker:
        combined = await self.reranker.rerank(query, combined)

    return combined[:top_k]
```

### Comparativa de Escenarios

#### Escenario 1: Referencia a Conversación Pasada

**Input:** "¿Te acuerdas de lo que hablamos ayer?"

| Sistema | Comportamiento | Resultado |
|---------|----------------|-----------|
| **Actual** | Busca "acuerdas", "hablamos", "ayer" | ❌ No encuentra nada relevante |
| **RAG** | Embedding captura "referencia temporal pasada + conversación" | ✅ Recupera resumen de sesión anterior |

#### Escenario 2: Sinónimos y Paráfrasis

**Input:** "Mi hogar fue destruido por los orcos"

Memoria almacenada: "El jugador perdió su casa en un ataque"

| Sistema | Comportamiento | Resultado |
|---------|----------------|-----------|
| **Actual** | "hogar" ≠ "casa", "destruido" ≠ "perdió" | ❌ No hay match |
| **RAG** | Vectores de "hogar destruido" ≈ "casa perdida" | ✅ Alta similitud semántica |

#### Escenario 3: Conexiones Conceptuales

**Input:** "Necesito dinero urgente"

Memorias almacenadas:
- "El jugador tiene una deuda con el gremio de ladrones"
- "Pedro (hermano) trabaja en las minas de Korrath"

| Sistema | Comportamiento | Resultado |
|---------|----------------|-----------|
| **Actual** | Solo busca "dinero", "urgente" | ❌ No encuentra conexión |
| **RAG** | "dinero urgente" → relacionado con "deuda", "trabajo" | ✅ Encuentra ambas memorias relevantes |

---

## Plan de Implementación

### Fase 0: Preparación (Investigación)
**Duración estimada:** Investigación previa

```
□ Estudiar documentación de ChromaDB/Qdrant
□ Estudiar embeddings de Ollama (nomic-embed-text)
□ Revisar patrones de RAG en LangChain/LlamaIndex
□ Entender hybrid search (vector + BM25)
□ Investigar chunking strategies
□ Revisar MemGPT paper para memoria jerárquica
```

### Fase 1: Infraestructura de Embeddings
**Objetivo:** Poder generar embeddings localmente

```
Tareas:
├── Crear src/rag/embeddings.py
│   ├── Clase EmbeddingClient
│   ├── Soporte Ollama (nomic-embed-text)
│   ├── Soporte OpenAI (fallback)
│   └── Caché de embeddings
│
├── Actualizar src/config.py
│   ├── EMBEDDING_MODEL
│   ├── EMBEDDING_DIMENSIONS
│   └── EMBEDDING_PROVIDER
│
└── Tests
    ├── test_embedding_generation.py
    └── test_embedding_similarity.py

Entregable: Poder hacer embed("texto") → [0.1, 0.2, ...]
```

### Fase 2: Vector Store Integration
**Objetivo:** Almacenar y buscar por vectores

```
Tareas:
├── Elegir e instalar Vector DB
│   ├── Opción A: ChromaDB (simple, embedded)
│   └── Opción B: Qdrant (robusto, ya en config)
│
├── Crear src/rag/vector_store.py
│   ├── Interfaz abstracta VectorStore
│   ├── ChromaDBStore implementation
│   ├── QdrantStore implementation
│   └── InMemoryStore (fallback/testing)
│
├── Métodos requeridos:
│   ├── add(id, embedding, metadata)
│   ├── search(embedding, top_k, filters)
│   ├── delete(id)
│   └── update(id, embedding, metadata)
│
└── Tests
    ├── test_vector_store_crud.py
    └── test_vector_search_accuracy.py

Entregable: CRUD completo de vectores con filtros
```

### Fase 3: Retriever y Hybrid Search
**Objetivo:** Búsqueda inteligente combinada

```
Tareas:
├── Crear src/rag/retriever.py
│   ├── Clase RAGRetriever
│   ├── Vector search method
│   ├── BM25/keyword search method
│   └── Hybrid merge con pesos configurables
│
├── Crear src/rag/chunker.py
│   ├── Estrategias de chunking
│   ├── Overlap handling
│   └── Metadata preservation
│
├── Actualizar scoring formula:
│   │
│   │  score = (α × vector_sim) +
│   │          (β × keyword_score) +
│   │          (γ × recency) +
│   │          (δ × importance)
│   │
│   │  donde α + β + γ + δ = 1
│   │
│
└── Tests
    ├── test_hybrid_search.py
    └── test_retriever_relevance.py

Entregable: Retriever que combina vector + keyword search
```

### Fase 4: Refactorizar Archival Memory
**Objetivo:** Integrar RAG en el sistema existente

```
Tareas:
├── Modificar src/memory/archival_memory.py
│   ├── Usar RAGRetriever internamente
│   ├── Mantener interfaz actual (backwards compatible)
│   ├── Auto-embed al agregar memorias
│   └── Migración de memorias existentes
│
├── Modificar src/memory/manager.py
│   ├── Inicializar RAG components
│   ├── Pasar embedding client
│   └── Configurar vector store
│
├── Actualizar Memory model:
│   │
│   │  class Memory(BaseModel):
│   │      ...
│   │      embedding: list[float]  # Ya no Optional
│   │      embedding_model: str    # Tracking
│   │
│
└── Tests de integración
    ├── test_archival_with_rag.py
    └── test_memory_migration.py

Entregable: Sistema de memoria usando RAG internamente
```

### Fase 5: Knowledge Bases
**Objetivo:** Indexar lore y conocimiento del NPC

```
Tareas:
├── Crear src/knowledge/
│   ├── loader.py - Carga YAML/JSON/MD
│   ├── npc_lore.py - Lore específico del NPC
│   └── world_events.py - Eventos del mundo
│
├── Estructura de Knowledge Base:
│   │
│   │  knowledge/
│   │  ├── npcs/
│   │  │   └── zamir/
│   │  │       ├── backstory.md
│   │  │       ├── opinions.yaml
│   │  │       └── secrets.yaml
│   │  └── world/
│   │      ├── nuvaris/
│   │      │   ├── history.md
│   │      │   └── factions.yaml
│   │      └── events/
│   │          └── current.yaml
│   │
│
├── Indexación automática al inicio
│   ├── Detectar cambios en archivos
│   ├── Re-indexar solo lo modificado
│   └── Versioning de knowledge
│
└── Tests
    ├── test_knowledge_loading.py
    └── test_knowledge_retrieval.py

Entregable: Knowledge bases indexadas y buscables
```

### Fase 6: Context Builder Mejorado
**Objetivo:** Construir prompts con contexto RAG

```
Tareas:
├── Modificar src/engine/context_builder.py
│   ├── Integrar RAG retrieval
│   ├── Budget de tokens por sección
│   ├── Priorización inteligente
│   └── Formato optimizado
│
├── Nueva estructura de contexto:
│   │
│   │  ## Core Memory (siempre incluido)
│   │  {player_info, relationship, npc_state}
│   │
│   │  ## Retrieved Knowledge (RAG)
│   │  {relevant_lore, world_context}
│   │
│   │  ## Retrieved Memories (RAG)
│   │  {semantic_memories, important_facts}
│   │
│   │  ## Recent Conversation
│   │  {last_n_turns}
│   │
│
└── Tests
    ├── test_context_building.py
    └── test_token_budgeting.py

Entregable: Prompts enriquecidos con contexto relevante
```

### Fase 7: Persistencia
**Objetivo:** Memorias que sobreviven reinicios

```
Tareas:
├── Integrar Redis para Core/Recall
│   ├── Serialización de memorias
│   ├── TTL para sesiones
│   └── Pub/sub para sync (futuro)
│
├── Configurar persistencia de Vector DB
│   ├── ChromaDB: persist_directory
│   ├── Qdrant: colecciones persistentes
│   └── Backups automáticos
│
├── Migración de datos
│   ├── Script de exportación
│   ├── Script de importación
│   └── Validación de integridad
│
└── Tests
    ├── test_persistence.py
    └── test_recovery.py

Entregable: Sistema que persiste entre reinicios
```

### Fase 8: Optimización y Caché
**Objetivo:** Rendimiento en producción

```
Tareas:
├── Implementar caché de embeddings
│   ├── LRU cache para queries frecuentes
│   ├── Pre-compute embeddings comunes
│   └── Invalidación inteligente
│
├── Optimizar búsquedas
│   ├── Índices HNSW en vector store
│   ├── Filtrado pre-búsqueda
│   └── Batch processing
│
├── Métricas y monitoring
│   ├── Latencia de embeddings
│   ├── Latencia de búsqueda
│   ├── Hit rate de caché
│   └── Relevancia de resultados
│
└── Tests de performance
    ├── test_latency.py
    └── test_throughput.py

Entregable: Sistema optimizado para producción
```

### Resumen Visual del Plan

```
Fase 0   Fase 1    Fase 2    Fase 3    Fase 4    Fase 5    Fase 6    Fase 7    Fase 8
  │        │         │         │         │         │         │         │         │
  ▼        ▼         ▼         ▼         ▼         ▼         ▼         ▼         ▼
┌────┐  ┌─────┐   ┌─────┐   ┌─────┐   ┌─────┐   ┌─────┐   ┌─────┐   ┌─────┐   ┌─────┐
│Inv │─▶│Embed│──▶│Vec  │──▶│Retr │──▶│Arch │──▶│Know │──▶│Ctx  │──▶│Pers │──▶│Opt  │
│est │  │ding │   │Store│   │iever│   │ival │   │ledge│   │Build│   │ist  │   │imiz │
└────┘  └─────┘   └─────┘   └─────┘   └─────┘   └─────┘   └─────┘   └─────┘   └─────┘
  │        │         │         │         │         │         │         │         │
  │        └─────────┴─────────┴─────────┘         │         │         │         │
  │                    │                           │         │         │         │
  │              CORE RAG STACK                    │         │         │         │
  │                                                │         │         │         │
  │                    ┌───────────────────────────┘         │         │         │
  │                    │                                     │         │         │
  │              NPC KNOWLEDGE                               │         │         │
  │                                                          │         │         │
  │                    ┌─────────────────────────────────────┘         │         │
  │                    │                                               │         │
  │              INTEGRATION                                           │         │
  │                                                                    │         │
  │                    ┌───────────────────────────────────────────────┘         │
  │                    │                                                         │
  │              PRODUCTION READY                                                │
  │                                                                              │
  └──────────────────────────────────────────────────────────────────────────────┘
                                      TÚ ESTÁS AQUÍ
```

---

## Tecnologías a Investigar

### 1. Vector Databases

#### ChromaDB
```yaml
Tipo: Embedded vector database
Uso: Desarrollo local, proyectos pequeños-medianos
Ventajas:
  - Fácil de usar (pip install)
  - No requiere servidor separado
  - Persistencia a disco simple
  - Buena documentación
Desventajas:
  - No escala horizontalmente
  - Menos features que Qdrant/Pinecone

Documentación: https://docs.trychroma.com/
GitHub: https://github.com/chroma-core/chroma
Ejemplos: https://docs.trychroma.com/guides

Instalación:
  pip install chromadb

Código ejemplo:
  import chromadb
  client = chromadb.Client()
  collection = client.create_collection("memories")
  collection.add(
      documents=["memory text"],
      embeddings=[[0.1, 0.2, ...]],
      ids=["mem_001"]
  )
```

#### Qdrant
```yaml
Tipo: Vector database (cloud/self-hosted)
Uso: Producción, alta escala
Ventajas:
  - Alto rendimiento
  - Filtrado avanzado
  - Payload storage
  - Clustering/sharding
Desventajas:
  - Requiere servidor separado (Docker)
  - Más complejo de configurar

Documentación: https://qdrant.tech/documentation/
GitHub: https://github.com/qdrant/qdrant
Python Client: https://github.com/qdrant/qdrant-client

Instalación:
  pip install qdrant-client
  docker run -p 6333:6333 qdrant/qdrant

Código ejemplo:
  from qdrant_client import QdrantClient
  from qdrant_client.models import Distance, VectorParams

  client = QdrantClient("localhost", port=6333)
  client.create_collection(
      collection_name="memories",
      vectors_config=VectorParams(size=768, distance=Distance.COSINE)
  )
```

#### FAISS (Facebook AI Similarity Search)
```yaml
Tipo: Library (no database)
Uso: Búsqueda de similitud pura, muy rápido
Ventajas:
  - Extremadamente rápido
  - Soporta billones de vectores
  - GPU support
Desventajas:
  - No es una DB (no persistencia nativa)
  - No tiene filtrado por metadata
  - Más bajo nivel

Documentación: https://faiss.ai/
GitHub: https://github.com/facebookresearch/faiss

Instalación:
  pip install faiss-cpu  # o faiss-gpu
```

### 2. Embedding Models

#### Ollama Embeddings
```yaml
Modelos disponibles:
  - nomic-embed-text (768 dims, recomendado)
  - mxbai-embed-large (1024 dims)
  - all-minilm (384 dims, más rápido)

Documentación: https://ollama.com/blog/embedding-models
API: https://github.com/ollama/ollama/blob/main/docs/api.md#generate-embeddings

Instalación:
  ollama pull nomic-embed-text

Código ejemplo:
  import httpx

  response = httpx.post(
      "http://localhost:11434/api/embeddings",
      json={
          "model": "nomic-embed-text",
          "prompt": "texto a embebir"
      }
  )
  embedding = response.json()["embedding"]
```

#### OpenAI Embeddings
```yaml
Modelos:
  - text-embedding-3-small (1536 dims, barato)
  - text-embedding-3-large (3072 dims, mejor)
  - text-embedding-ada-002 (legacy)

Documentación: https://platform.openai.com/docs/guides/embeddings
Pricing: https://openai.com/pricing

Código ejemplo:
  from openai import OpenAI

  client = OpenAI()
  response = client.embeddings.create(
      input="texto a embebir",
      model="text-embedding-3-small"
  )
  embedding = response.data[0].embedding
```

#### Sentence Transformers
```yaml
Tipo: Local models (HuggingFace)
Modelos recomendados:
  - all-MiniLM-L6-v2 (rápido, 384 dims)
  - all-mpnet-base-v2 (mejor calidad, 768 dims)
  - paraphrase-multilingual-MiniLM-L12-v2 (multilenguaje)

Documentación: https://www.sbert.net/
GitHub: https://github.com/UKPLab/sentence-transformers

Instalación:
  pip install sentence-transformers

Código ejemplo:
  from sentence_transformers import SentenceTransformer

  model = SentenceTransformer('all-MiniLM-L6-v2')
  embeddings = model.encode(["texto 1", "texto 2"])
```

### 3. RAG Frameworks

#### LangChain
```yaml
Tipo: Framework completo para LLM apps
Uso: Referencia de patrones RAG
Ventajas:
  - Muchos ejemplos
  - Integraciones con todo
  - Bien documentado
Desventajas:
  - Puede ser overkill
  - Abstracción pesada

Documentación: https://python.langchain.com/docs/
RAG Tutorial: https://python.langchain.com/docs/tutorials/rag/
GitHub: https://github.com/langchain-ai/langchain

Instalación:
  pip install langchain langchain-community
```

#### LlamaIndex
```yaml
Tipo: Framework especializado en RAG
Uso: Data indexing y retrieval
Ventajas:
  - Diseñado para RAG desde cero
  - Excelente para documentos
  - Múltiples índices
Desventajas:
  - Curva de aprendizaje

Documentación: https://docs.llamaindex.ai/
Starter Tutorial: https://docs.llamaindex.ai/en/stable/getting_started/starter_example/
GitHub: https://github.com/run-llama/llama_index

Instalación:
  pip install llama-index
```

### 4. Hybrid Search

#### BM25 (Best Matching 25)
```yaml
Tipo: Algoritmo de ranking para keyword search
Uso: Complemento a vector search
Por qué importa:
  - Vector search puede fallar con términos específicos
  - BM25 es excelente para nombres propios, códigos
  - Hybrid = mejor de ambos mundos

Implementación en Python:
  pip install rank-bm25

Código ejemplo:
  from rank_bm25 import BM25Okapi

  corpus = ["doc 1 text", "doc 2 text", "doc 3 text"]
  tokenized = [doc.split() for doc in corpus]
  bm25 = BM25Okapi(tokenized)

  query = "search terms"
  scores = bm25.get_scores(query.split())
```

#### Reciprocal Rank Fusion (RRF)
```yaml
Tipo: Algoritmo para combinar rankings
Uso: Fusionar resultados de vector + keyword search

Documentación:
  - Paper: https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf
  - Explicación: https://www.elastic.co/guide/en/elasticsearch/reference/current/rrf.html

Fórmula:
  RRF(d) = Σ 1/(k + rank(d))
  donde k = 60 (típicamente)

Código ejemplo:
  def reciprocal_rank_fusion(rankings: list[list], k: int = 60) -> dict:
      scores = {}
      for ranking in rankings:
          for rank, doc_id in enumerate(ranking):
              if doc_id not in scores:
                  scores[doc_id] = 0
              scores[doc_id] += 1 / (k + rank + 1)
      return sorted(scores.items(), key=lambda x: x[1], reverse=True)
```

### 5. Key-Value Store (Persistencia)

#### Redis
```yaml
Tipo: In-memory data store
Uso: Core memory, recall memory, caché

Documentación: https://redis.io/docs/
Python Client: https://redis-py.readthedocs.io/

Instalación:
  pip install redis
  docker run -p 6379:6379 redis

Código ejemplo:
  import redis

  r = redis.Redis(host='localhost', port=6379, db=0)
  r.set('core:npc1:player1', json.dumps(core_memory))
  data = json.loads(r.get('core:npc1:player1'))
```

### 6. Papers y Conceptos Clave

```yaml
Papers fundamentales:
  - "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
    https://arxiv.org/abs/2005.11401

  - "MemGPT: Towards LLMs as Operating Systems"
    https://arxiv.org/abs/2310.08560

  - "Generative Agents: Interactive Simulacra of Human Behavior"
    https://arxiv.org/abs/2304.03442
    (Stanford - el que inspiró este proyecto)

  - "Dense Passage Retrieval for Open-Domain Question Answering"
    https://arxiv.org/abs/2004.04906

Conceptos a entender:
  - Cosine similarity vs Dot product
  - HNSW (Hierarchical Navigable Small World) indexing
  - Chunking strategies (fixed, semantic, recursive)
  - Query expansion
  - Hypothetical Document Embeddings (HyDE)
  - Cross-encoder reranking
```

---

## Recursos y Documentación

### Tutoriales Recomendados

```yaml
RAG desde cero:
  1. "Building RAG from Scratch"
     https://www.youtube.com/watch?v=BrsocJb-fAo

  2. "RAG Tutorial" (LangChain)
     https://python.langchain.com/docs/tutorials/rag/

  3. "Vector Databases Explained"
     https://www.pinecone.io/learn/vector-database/

Embeddings:
  1. "Text Embeddings Visually Explained"
     https://www.youtube.com/watch?v=OATCgQtNX2o

  2. "Sentence Embeddings - SBERT"
     https://www.sbert.net/docs/quickstart.html

Memory Systems para LLMs:
  1. "MemGPT: Memory Management for LLMs"
     https://memgpt.ai/

  2. "LangGraph Memory"
     https://langchain-ai.github.io/langgraph/concepts/memory/
```

### Repositorios de Referencia

```yaml
Implementaciones similares:
  - MemGPT: https://github.com/cpacker/MemGPT
  - LangChain RAG: https://github.com/langchain-ai/langchain/tree/master/cookbook
  - LlamaIndex Examples: https://github.com/run-llama/llama_index/tree/main/docs/examples
  - ChromaDB Examples: https://github.com/chroma-core/chroma/tree/main/examples
  - Qdrant Examples: https://github.com/qdrant/examples
```

### Blogs y Artículos

```yaml
Arquitectura RAG:
  - "RAG Architecture Deep Dive"
    https://blog.langchain.dev/retrieval/

  - "Advanced RAG Techniques"
    https://www.pinecone.io/learn/advanced-rag-techniques/

  - "Hybrid Search Explained"
    https://www.elastic.co/blog/improving-information-retrieval-elastic-learned-sparse-encoder

NPC AI específico:
  - "Building AI NPCs with Memory"
    https://blog.character.ai/

  - "Generative Agents Paper Summary"
    https://lilianweng.github.io/posts/2023-06-23-agent/
```

---

## Ejemplos de Código de Referencia

### Ejemplo 1: Embedding Client Básico

```python
"""
Referencia: Cliente de embeddings con soporte multi-provider
Archivo futuro: src/rag/embeddings.py
"""
from abc import ABC, abstractmethod
from typing import Optional
import httpx
from functools import lru_cache

class EmbeddingProvider(ABC):
    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        pass

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        pass

class OllamaEmbeddings(EmbeddingProvider):
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "nomic-embed-text"
    ):
        self.base_url = base_url
        self.model = model
        self.client = httpx.AsyncClient(timeout=30.0)

    async def embed(self, text: str) -> list[float]:
        response = await self.client.post(
            f"{self.base_url}/api/embeddings",
            json={"model": self.model, "prompt": text}
        )
        response.raise_for_status()
        return response.json()["embedding"]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        # Ollama no tiene batch nativo, procesamos secuencialmente
        return [await self.embed(text) for text in texts]

class OpenAIEmbeddings(EmbeddingProvider):
    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small"
    ):
        self.api_key = api_key
        self.model = model
        self.client = httpx.AsyncClient(
            base_url="https://api.openai.com/v1",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=30.0
        )

    async def embed(self, text: str) -> list[float]:
        response = await self.client.post(
            "/embeddings",
            json={"model": self.model, "input": text}
        )
        response.raise_for_status()
        return response.json()["data"][0]["embedding"]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        response = await self.client.post(
            "/embeddings",
            json={"model": self.model, "input": texts}
        )
        response.raise_for_status()
        data = response.json()["data"]
        return [item["embedding"] for item in sorted(data, key=lambda x: x["index"])]

# Factory
def create_embedding_provider(
    provider: str = "ollama",
    **kwargs
) -> EmbeddingProvider:
    providers = {
        "ollama": OllamaEmbeddings,
        "openai": OpenAIEmbeddings,
    }
    return providers[provider](**kwargs)
```

### Ejemplo 2: Vector Store con ChromaDB

```python
"""
Referencia: Wrapper de ChromaDB para memorias
Archivo futuro: src/rag/vector_store.py
"""
import chromadb
from chromadb.config import Settings
from typing import Optional
from pydantic import BaseModel

class SearchResult(BaseModel):
    id: str
    content: str
    metadata: dict
    score: float

class ChromaVectorStore:
    def __init__(
        self,
        persist_directory: Optional[str] = None,
        collection_name: str = "npc_memories"
    ):
        if persist_directory:
            self.client = chromadb.PersistentClient(path=persist_directory)
        else:
            self.client = chromadb.Client()

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # Cosine similarity
        )

    def add(
        self,
        id: str,
        content: str,
        embedding: list[float],
        metadata: Optional[dict] = None
    ) -> None:
        self.collection.add(
            ids=[id],
            documents=[content],
            embeddings=[embedding],
            metadatas=[metadata or {}]
        )

    def add_batch(
        self,
        ids: list[str],
        contents: list[str],
        embeddings: list[list[float]],
        metadatas: Optional[list[dict]] = None
    ) -> None:
        self.collection.add(
            ids=ids,
            documents=contents,
            embeddings=embeddings,
            metadatas=metadatas or [{} for _ in ids]
        )

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        where: Optional[dict] = None,
        where_document: Optional[dict] = None
    ) -> list[SearchResult]:
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            where_document=where_document,
            include=["documents", "metadatas", "distances"]
        )

        search_results = []
        for i in range(len(results["ids"][0])):
            # ChromaDB returns distances, convert to similarity
            distance = results["distances"][0][i]
            similarity = 1 - distance  # For cosine distance

            search_results.append(SearchResult(
                id=results["ids"][0][i],
                content=results["documents"][0][i],
                metadata=results["metadatas"][0][i],
                score=similarity
            ))

        return search_results

    def delete(self, ids: list[str]) -> None:
        self.collection.delete(ids=ids)

    def update(
        self,
        id: str,
        content: Optional[str] = None,
        embedding: Optional[list[float]] = None,
        metadata: Optional[dict] = None
    ) -> None:
        update_kwargs = {"ids": [id]}
        if content:
            update_kwargs["documents"] = [content]
        if embedding:
            update_kwargs["embeddings"] = [embedding]
        if metadata:
            update_kwargs["metadatas"] = [metadata]

        self.collection.update(**update_kwargs)

    def count(self) -> int:
        return self.collection.count()
```

### Ejemplo 3: Hybrid Retriever

```python
"""
Referencia: Retriever híbrido (vector + BM25)
Archivo futuro: src/rag/retriever.py
"""
from dataclasses import dataclass
from typing import Optional
from rank_bm25 import BM25Okapi
import numpy as np

@dataclass
class RetrievedDocument:
    id: str
    content: str
    metadata: dict
    vector_score: float
    keyword_score: float
    final_score: float

class HybridRetriever:
    def __init__(
        self,
        vector_store,
        embedding_provider,
        vector_weight: float = 0.7,
        keyword_weight: float = 0.3
    ):
        self.vector_store = vector_store
        self.embeddings = embedding_provider
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight

        # BM25 index (se construye dinámicamente)
        self._bm25_corpus: list[str] = []
        self._bm25_ids: list[str] = []
        self._bm25_index: Optional[BM25Okapi] = None

    def _tokenize(self, text: str) -> list[str]:
        """Simple tokenization - mejorar según idioma"""
        return text.lower().split()

    def build_bm25_index(self, documents: list[tuple[str, str]]) -> None:
        """
        Construir índice BM25 desde lista de (id, content)
        """
        self._bm25_ids = [doc[0] for doc in documents]
        self._bm25_corpus = [doc[1] for doc in documents]
        tokenized = [self._tokenize(doc) for doc in self._bm25_corpus]
        self._bm25_index = BM25Okapi(tokenized)

    async def search(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[dict] = None
    ) -> list[RetrievedDocument]:
        # 1. Vector search
        query_embedding = await self.embeddings.embed(query)
        vector_results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k * 2,  # Over-fetch para fusión
            where=filters
        )

        # 2. BM25 search (si está disponible)
        keyword_scores = {}
        if self._bm25_index:
            tokenized_query = self._tokenize(query)
            bm25_scores = self._bm25_index.get_scores(tokenized_query)

            # Normalizar scores
            max_score = max(bm25_scores) if max(bm25_scores) > 0 else 1
            for idx, score in enumerate(bm25_scores):
                keyword_scores[self._bm25_ids[idx]] = score / max_score

        # 3. Fusionar resultados (Hybrid)
        results = []
        for vr in vector_results:
            kw_score = keyword_scores.get(vr.id, 0.0)

            final_score = (
                self.vector_weight * vr.score +
                self.keyword_weight * kw_score
            )

            results.append(RetrievedDocument(
                id=vr.id,
                content=vr.content,
                metadata=vr.metadata,
                vector_score=vr.score,
                keyword_score=kw_score,
                final_score=final_score
            ))

        # 4. Ordenar por score final
        results.sort(key=lambda x: x.final_score, reverse=True)

        return results[:top_k]

class RRFRetriever:
    """
    Reciprocal Rank Fusion - alternativa a weighted combination
    """
    def __init__(self, k: int = 60):
        self.k = k

    def fuse(
        self,
        rankings: list[list[str]],  # Lista de listas de IDs ordenados
        documents: dict[str, dict]   # ID -> {content, metadata}
    ) -> list[tuple[str, float]]:
        """
        Fusiona múltiples rankings usando RRF
        """
        scores = {}

        for ranking in rankings:
            for rank, doc_id in enumerate(ranking):
                if doc_id not in scores:
                    scores[doc_id] = 0.0
                scores[doc_id] += 1.0 / (self.k + rank + 1)

        # Ordenar por score
        sorted_results = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return sorted_results
```

### Ejemplo 4: Integración con Memory Manager

```python
"""
Referencia: MemoryManager actualizado con RAG
Archivo futuro: Modificación de src/memory/manager.py
"""
from typing import Optional

class RAGMemoryManager:
    """
    Memory Manager mejorado con RAG
    """
    def __init__(
        self,
        embedding_provider,
        vector_store,
        kv_store=None  # Redis para core/recall
    ):
        self.embeddings = embedding_provider
        self.vector_store = vector_store
        self.kv_store = kv_store

        # Retriever híbrido
        self.retriever = HybridRetriever(
            vector_store=vector_store,
            embedding_provider=embedding_provider
        )

        # Memorias en RAM (fallback si no hay kv_store)
        self._core_memories: dict[str, CoreMemory] = {}
        self._recall_memories: dict[str, RecallMemory] = {}

    async def add_memory(
        self,
        npc_id: str,
        player_id: str,
        content: str,
        importance: float = 5.0,
        memory_type: str = "observation",
        topics: Optional[list[str]] = None,
    ) -> Memory:
        """
        Agregar memoria con embedding automático
        """
        # Crear memoria
        memory = Memory.create(
            npc_id=npc_id,
            player_id=player_id,
            content=content,
            importance=importance,
            memory_type=memory_type,
            topics=topics,
        )

        # Generar embedding
        embedding = await self.embeddings.embed(content)
        memory.embedding = embedding

        # Guardar en vector store
        self.vector_store.add(
            id=memory.id,
            content=content,
            embedding=embedding,
            metadata={
                "npc_id": npc_id,
                "player_id": player_id,
                "importance": importance,
                "memory_type": memory_type,
                "topics": topics or [],
                "created_at": memory.created_at.isoformat(),
            }
        )

        return memory

    async def search_memories(
        self,
        npc_id: str,
        player_id: str,
        query: str,
        top_k: int = 5,
        min_importance: float = 0.0,
    ) -> list[Memory]:
        """
        Búsqueda semántica de memorias
        """
        results = await self.retriever.search(
            query=query,
            top_k=top_k,
            filters={
                "npc_id": npc_id,
                "player_id": player_id,
            }
        )

        # Filtrar por importancia
        memories = []
        for r in results:
            if r.metadata.get("importance", 0) >= min_importance:
                memories.append(Memory(
                    id=r.id,
                    npc_id=r.metadata["npc_id"],
                    player_id=r.metadata["player_id"],
                    content=r.content,
                    importance=r.metadata["importance"],
                    memory_type=r.metadata["memory_type"],
                    topics=r.metadata.get("topics", []),
                    # ... otros campos
                ))

        return memories

    async def get_memory_context(
        self,
        npc_id: str,
        player_id: str,
        session_id: str,
        query: str,
        max_archival: int = 5,
    ) -> MemoryContext:
        """
        Construir contexto completo con RAG
        """
        # Core memory (siempre incluido)
        core = self.get_core(npc_id, player_id)

        # Recall memory (conversación reciente)
        recall = self.get_recall(npc_id, player_id, session_id)

        # Archival memory (RAG search)
        archival_memories = await self.search_memories(
            npc_id=npc_id,
            player_id=player_id,
            query=query,
            top_k=max_archival,
        )

        # Formatear para prompt
        archival_text = "\n".join([
            f"- [{m.memory_type}] {m.content}"
            for m in archival_memories
        ])

        return MemoryContext(
            core=core.to_prompt_text(),
            recall=recall.to_prompt_text(),
            archival=archival_text,
            total_tokens_estimate=self._estimate_tokens(
                core, recall, archival_text
            ),
        )
```

---

## Decisiones Arquitectónicas

### Decisión 1: Vector Database

```yaml
Opciones:
  A) ChromaDB
     - Pros: Simple, embedded, sin setup
     - Cons: Menos escalable
     - Recomendado para: Desarrollo, proyectos pequeños

  B) Qdrant
     - Pros: Robusto, escalable, features avanzadas
     - Cons: Requiere Docker/servidor
     - Recomendado para: Producción, alta escala

Recomendación: Empezar con ChromaDB, migrar a Qdrant si necesario
Interfaz abstracta permite cambiar sin refactorizar
```

### Decisión 2: Embedding Model

```yaml
Opciones:
  A) Ollama (nomic-embed-text)
     - Pros: Local, gratis, privado, 768 dims
     - Cons: Necesita Ollama corriendo
     - Latencia: ~50-100ms

  B) OpenAI (text-embedding-3-small)
     - Pros: Muy bueno, rápido
     - Cons: Costo, dependencia externa
     - Latencia: ~200-500ms (red)

  C) Sentence Transformers (local)
     - Pros: Local, muchos modelos
     - Cons: Setup adicional
     - Latencia: ~20-50ms (GPU), ~100-200ms (CPU)

Recomendación: Ollama como default, OpenAI como fallback
Ya usamos Ollama para LLM, mantiene stack consistente
```

### Decisión 3: Hybrid Search

```yaml
Opciones:
  A) Solo Vector Search
     - Pros: Simple
     - Cons: Falla con términos específicos

  B) Solo BM25
     - Pros: Bueno para keywords
     - Cons: No entiende semántica

  C) Hybrid (Vector + BM25)
     - Pros: Mejor de ambos mundos
     - Cons: Más complejo, más lento

Recomendación: Hybrid con pesos configurables
Default: 70% vector, 30% BM25
Permite ajustar según caso de uso
```

### Decisión 4: Persistencia

```yaml
Opciones:
  A) Todo en memoria (actual)
     - Pros: Simple, rápido
     - Cons: Se pierde al reiniciar

  B) SQLite + Vector Store
     - Pros: Simple, archivo único
     - Cons: No escala bien

  C) Redis + Vector Store
     - Pros: Rápido, escalable
     - Cons: Otro servicio que mantener

Recomendación:
- ChromaDB con persist_directory para vectors
- Redis opcional para core/recall
- Fallback a memoria si Redis no disponible
```

---

## Métricas de Éxito

### Métricas de Calidad

```yaml
Retrieval Accuracy:
  - Métrica: Recall@K (¿las memorias relevantes están en top-K?)
  - Baseline actual: ~30%
  - Objetivo: >80%
  - Medición: Dataset de test con memorias etiquetadas

Semantic Relevance:
  - Métrica: MRR (Mean Reciprocal Rank)
  - Baseline actual: ~0.4
  - Objetivo: >0.8
  - Medición: Queries de prueba con respuestas conocidas

User Experience:
  - Métrica: Coherencia percibida (1-5)
  - Baseline actual: ~2.5
  - Objetivo: >4.0
  - Medición: Evaluación manual de conversaciones
```

### Métricas de Performance

```yaml
Latencia de Embedding:
  - Objetivo: <100ms por query
  - Medición: p50, p95, p99

Latencia de Búsqueda:
  - Objetivo: <50ms para top-5
  - Medición: p50, p95, p99

Throughput:
  - Objetivo: >100 queries/segundo
  - Medición: Load testing

Uso de Memoria:
  - Objetivo: <500MB para 100k memorias
  - Medición: Memory profiling
```

### Tests de Validación

```yaml
Test de Sinónimos:
  Query: "Mi hogar fue destruido"
  Memoria: "El jugador perdió su casa"
  Esperado: Match con score > 0.8

Test de Conceptos:
  Query: "¿Recuerdas a mi familia?"
  Memorias: ["Pedro es tu hermano", "Tu madre está enferma"]
  Esperado: Ambas recuperadas en top-3

Test de Paráfrasis:
  Query: "Lo que te conté ayer"
  Memoria: "En nuestra conversación anterior mencionaste..."
  Esperado: Match con score > 0.7

Test de Nombres Propios:
  Query: "¿Conoces a Zarathos?"
  Memoria: "Zarathos es el líder del gremio de magos"
  Esperado: Match exacto (BM25 contribution)
```

---

## Checklist de Investigación

### Antes de Implementar

```
□ Leer documentación de ChromaDB (conceptos, API)
□ Leer documentación de Ollama embeddings
□ Entender cosine similarity vs dot product
□ Revisar paper de RAG (al menos abstract y methodology)
□ Revisar paper de MemGPT (arquitectura de memoria)
□ Ejecutar ejemplos de LangChain RAG
□ Probar ChromaDB localmente con datos dummy
□ Probar generación de embeddings con Ollama
□ Entender BM25 y cuándo es útil
□ Revisar RRF (Reciprocal Rank Fusion)
```

### Durante la Implementación

```
□ Empezar con tests (TDD si posible)
□ Implementar interfaz abstracta primero
□ Validar embeddings con queries conocidas
□ Medir latencias desde el inicio
□ Documentar decisiones en código
□ Mantener backwards compatibility
```

### Después de Implementar

```
□ Benchmark con dataset de prueba
□ Comparar métricas vs baseline
□ Documentar configuración óptima
□ Crear guía de troubleshooting
□ Actualizar DOCUMENTATION.md
```

---

## Siguiente Paso

Una vez completada esta investigación:

1. **Crear branch** para RAG implementation
2. **Fase 1**: Implementar embedding client
3. **Fase 2**: Integrar ChromaDB
4. **Fase 3**: Implementar retriever híbrido
5. **Fase 4**: Refactorizar archival memory
6. **Testing**: Validar mejoras con métricas

---

> **Nota**: Este documento es una guía de investigación. Los ejemplos de código son referenciales y pueden requerir ajustes durante la implementación real.
