"""Vector store abstraction for semantic memory storage.

Provides implementations for:
- Qdrant: Production-ready vector database with HNSW indexing
- InMemory: Development/testing fallback using numpy

Key concepts:
- HNSW (Hierarchical Navigable Small World): O(log n) search via graph traversal
- Cosine similarity: Measures angle between vectors (ideal for normalized embeddings)
- Payload indexing: Required for efficient filtering (Qdrant doesn't index by default)
"""

import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class VectorPoint:
    """A point in the vector store."""
    id: str
    vector: list[float]
    payload: dict[str, Any] = field(default_factory=dict)
    score: float = 0.0  # Similarity score (set during search)


class VectorStore(ABC):
    """Abstract base class for vector stores."""

    @abstractmethod
    async def create_collection(
        self,
        name: str,
        dimensions: int,
        recreate: bool = False,
    ) -> None:
        """Create a collection for storing vectors.

        Args:
            name: Collection name
            dimensions: Vector dimensionality (e.g., 768 for nomic-embed-text)
            recreate: If True, drop existing collection first
        """
        pass

    @abstractmethod
    async def add(
        self,
        collection: str,
        points: list[VectorPoint],
    ) -> None:
        """Add points to a collection.

        Args:
            collection: Collection name
            points: List of VectorPoint objects
        """
        pass

    @abstractmethod
    async def search(
        self,
        collection: str,
        query_vector: list[float],
        top_k: int = 10,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[VectorPoint]:
        """Search for similar vectors.

        Args:
            collection: Collection name
            query_vector: Query embedding
            top_k: Number of results to return
            filters: Payload filter conditions

        Returns:
            List of VectorPoint with scores (highest similarity first)
        """
        pass

    @abstractmethod
    async def delete(
        self,
        collection: str,
        ids: list[str],
    ) -> None:
        """Delete points by ID.

        Args:
            collection: Collection name
            ids: List of point IDs to delete
        """
        pass

    @abstractmethod
    async def count(self, collection: str) -> int:
        """Get total point count in collection."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close the connection."""
        pass


class QdrantVectorStore(VectorStore):
    """Qdrant vector database implementation.

    Features:
    - HNSW indexing for O(log n) search
    - Payload filtering during graph traversal (efficient)
    - Cosine distance (optimal for normalized embeddings)
    - Scalar quantization option for memory efficiency

    Configuration:
        m=16, ef_construct=200: Good balance of recall (~95%) and speed
    """

    def __init__(
        self,
        url: str = "http://localhost:6333",
        api_key: Optional[str] = None,
        prefer_grpc: bool = True,
    ):
        """Initialize Qdrant client.

        Args:
            url: Qdrant server URL
            api_key: Optional API key for authentication
            prefer_grpc: Use gRPC for better performance
        """
        self.url = url
        self.api_key = api_key
        self.prefer_grpc = prefer_grpc
        self._client = None
        self._initialized = False

    async def _get_client(self):
        """Get or create Qdrant client (lazy initialization)."""
        if self._client is None:
            try:
                from qdrant_client import AsyncQdrantClient
                self._client = AsyncQdrantClient(
                    url=self.url,
                    api_key=self.api_key,
                    prefer_grpc=self.prefer_grpc,
                )
                self._initialized = True
            except ImportError:
                raise RuntimeError(
                    "qdrant-client not installed. "
                    "Install with: pip install qdrant-client"
                )
        return self._client

    async def create_collection(
        self,
        name: str,
        dimensions: int,
        recreate: bool = False,
    ) -> None:
        """Create a Qdrant collection with optimal configuration.

        Uses:
        - Cosine distance (best for normalized embeddings)
        - HNSW params: m=16, ef_construct=200
        - Payload indexing for npc_id and player_id
        """
        from qdrant_client.models import (
            Distance,
            VectorParams,
            HnswConfigDiff,
            PayloadSchemaType,
        )

        client = await self._get_client()

        # Check if collection exists
        collections = await client.get_collections()
        exists = any(c.name == name for c in collections.collections)

        if exists and recreate:
            await client.delete_collection(name)
            exists = False

        if not exists:
            await client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(
                    size=dimensions,
                    distance=Distance.COSINE,
                ),
                hnsw_config=HnswConfigDiff(
                    m=16,  # Connections per node
                    ef_construct=200,  # Build-time search depth
                ),
            )

            # Create payload indexes for efficient filtering
            # CRITICAL: Without these, every filter scans all points
            await client.create_payload_index(
                collection_name=name,
                field_name="npc_id",
                field_schema=PayloadSchemaType.KEYWORD,
            )
            await client.create_payload_index(
                collection_name=name,
                field_name="player_id",
                field_schema=PayloadSchemaType.KEYWORD,
            )
            await client.create_payload_index(
                collection_name=name,
                field_name="importance",
                field_schema=PayloadSchemaType.FLOAT,
            )
            await client.create_payload_index(
                collection_name=name,
                field_name="created_at",
                field_schema=PayloadSchemaType.DATETIME,
            )

            logger.info(f"Created Qdrant collection '{name}' with {dimensions} dimensions")

    async def add(
        self,
        collection: str,
        points: list[VectorPoint],
    ) -> None:
        """Add points to Qdrant collection."""
        if not points:
            return

        from qdrant_client.models import PointStruct

        client = await self._get_client()

        qdrant_points = [
            PointStruct(
                id=point.id,
                vector=point.vector,
                payload=point.payload,
            )
            for point in points
        ]

        await client.upsert(
            collection_name=collection,
            points=qdrant_points,
        )

        logger.debug(f"Added {len(points)} points to collection '{collection}'")

    async def search(
        self,
        collection: str,
        query_vector: list[float],
        top_k: int = 10,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[VectorPoint]:
        """Search for similar vectors with optional filtering.

        Qdrant uses "Filterable HNSW" - filters are checked during graph
        traversal, not as a post-processing step.
        """
        from qdrant_client.models import Filter, FieldCondition, MatchValue, Range

        client = await self._get_client()

        # Build filter conditions
        qdrant_filter = None
        if filters:
            conditions = []
            for key, value in filters.items():
                if isinstance(value, dict):
                    # Range filter (e.g., {"gte": 0.5})
                    range_params = {}
                    if "gte" in value:
                        range_params["gte"] = value["gte"]
                    if "lte" in value:
                        range_params["lte"] = value["lte"]
                    if "gt" in value:
                        range_params["gt"] = value["gt"]
                    if "lt" in value:
                        range_params["lt"] = value["lt"]
                    conditions.append(
                        FieldCondition(key=key, range=Range(**range_params))
                    )
                else:
                    # Exact match
                    conditions.append(
                        FieldCondition(key=key, match=MatchValue(value=value))
                    )
            qdrant_filter = Filter(must=conditions)

        results = await client.search(
            collection_name=collection,
            query_vector=query_vector,
            limit=top_k,
            query_filter=qdrant_filter,
            with_payload=True,
        )

        return [
            VectorPoint(
                id=str(r.id),
                vector=r.vector if r.vector else [],
                payload=r.payload or {},
                score=r.score,
            )
            for r in results
        ]

    async def delete(
        self,
        collection: str,
        ids: list[str],
    ) -> None:
        """Delete points by ID."""
        if not ids:
            return

        from qdrant_client.models import PointIdsList

        client = await self._get_client()
        await client.delete(
            collection_name=collection,
            points_selector=PointIdsList(points=ids),
        )

        logger.debug(f"Deleted {len(ids)} points from collection '{collection}'")

    async def count(self, collection: str) -> int:
        """Get point count in collection."""
        client = await self._get_client()
        info = await client.get_collection(collection)
        return info.points_count

    async def scroll(
        self,
        collection: str,
        filters: Optional[dict[str, Any]] = None,
        limit: int = 100,
    ) -> list[VectorPoint]:
        """Scroll through points (no ranking, just filtered retrieval).

        Use for "get all memories for this NPC" type queries.
        """
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        client = await self._get_client()

        # Build filter
        qdrant_filter = None
        if filters:
            conditions = [
                FieldCondition(key=k, match=MatchValue(value=v))
                for k, v in filters.items()
                if not isinstance(v, dict)
            ]
            if conditions:
                qdrant_filter = Filter(must=conditions)

        results, _ = await client.scroll(
            collection_name=collection,
            scroll_filter=qdrant_filter,
            limit=limit,
            with_payload=True,
            with_vectors=True,
        )

        return [
            VectorPoint(
                id=str(r.id),
                vector=r.vector if r.vector else [],
                payload=r.payload or {},
                score=0.0,
            )
            for r in results
        ]

    async def close(self) -> None:
        """Close Qdrant client."""
        if self._client:
            await self._client.close()
            self._client = None

    async def is_available(self) -> bool:
        """Check if Qdrant server is available."""
        try:
            client = await self._get_client()
            await client.get_collections()
            return True
        except Exception:
            return False


class InMemoryVectorStore(VectorStore):
    """In-memory vector store for development and testing.

    Uses brute-force cosine similarity search (O(n)).
    Suitable for small datasets (<10k vectors).
    """

    def __init__(self):
        """Initialize in-memory store."""
        self._collections: dict[str, dict[str, VectorPoint]] = {}
        self._dimensions: dict[str, int] = {}

    async def create_collection(
        self,
        name: str,
        dimensions: int,
        recreate: bool = False,
    ) -> None:
        """Create an in-memory collection."""
        if recreate or name not in self._collections:
            self._collections[name] = {}
            self._dimensions[name] = dimensions
            logger.info(f"Created in-memory collection '{name}' with {dimensions} dimensions")

    async def add(
        self,
        collection: str,
        points: list[VectorPoint],
    ) -> None:
        """Add points to collection."""
        if collection not in self._collections:
            raise ValueError(f"Collection '{collection}' does not exist")

        for point in points:
            self._collections[collection][point.id] = point

    async def search(
        self,
        collection: str,
        query_vector: list[float],
        top_k: int = 10,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[VectorPoint]:
        """Search using brute-force cosine similarity."""
        if collection not in self._collections:
            return []

        results = []
        for point in self._collections[collection].values():
            # Apply filters
            if filters:
                match = True
                for key, value in filters.items():
                    if isinstance(value, dict):
                        # Range filter
                        point_value = point.payload.get(key)
                        if point_value is not None:
                            if "gte" in value and point_value < value["gte"]:
                                match = False
                            if "lte" in value and point_value > value["lte"]:
                                match = False
                    else:
                        if point.payload.get(key) != value:
                            match = False
                            break
                if not match:
                    continue

            # Calculate cosine similarity
            score = self._cosine_similarity(query_vector, point.vector)
            results.append(VectorPoint(
                id=point.id,
                vector=point.vector,
                payload=point.payload,
                score=score,
            ))

        # Sort by score descending
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if len(a) != len(b):
            return 0.0

        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    async def delete(
        self,
        collection: str,
        ids: list[str],
    ) -> None:
        """Delete points by ID."""
        if collection not in self._collections:
            return

        for id_ in ids:
            self._collections[collection].pop(id_, None)

    async def count(self, collection: str) -> int:
        """Get point count."""
        return len(self._collections.get(collection, {}))

    async def scroll(
        self,
        collection: str,
        filters: Optional[dict[str, Any]] = None,
        limit: int = 100,
    ) -> list[VectorPoint]:
        """Scroll through points with optional filtering."""
        if collection not in self._collections:
            return []

        results = []
        for point in self._collections[collection].values():
            if filters:
                match = all(
                    point.payload.get(k) == v
                    for k, v in filters.items()
                    if not isinstance(v, dict)
                )
                if not match:
                    continue
            results.append(point)
            if len(results) >= limit:
                break

        return results

    async def close(self) -> None:
        """No-op for in-memory store."""
        pass

    async def is_available(self) -> bool:
        """Always available."""
        return True

    def clear(self, collection: Optional[str] = None) -> None:
        """Clear data (for testing)."""
        if collection:
            self._collections[collection] = {}
        else:
            self._collections.clear()
            self._dimensions.clear()
