from __future__ import annotations

from dataclasses import dataclass, field

from .embedding import BaseEmbeddingProvider
from .graph_store import BaseGraphStore
from .vector_store import BaseVectorStore


@dataclass(slots=True)
class RetrievedChunk:
    text: str
    score: float | None = None
    metadata: dict = field(default_factory=dict)


class RAGService:
    def __init__(
        self,
        embedding_provider: BaseEmbeddingProvider,
        vector_store: BaseVectorStore,
        graph_store: BaseGraphStore | None = None,
    ) -> None:
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        self.graph_store = graph_store

    def index_text(
        self,
        document_id: str,
        text: str,
        metadata: dict | None = None,
    ) -> None:
        embedding = self.embedding_provider.embed_text(text)
        self.vector_store.add_text(
            document_id=document_id,
            text=text,
            embedding=embedding,
            metadata=metadata,
        )

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        embedding = self.embedding_provider.embed_text(query)
        results = self.vector_store.search(embedding=embedding, top_k=top_k)

        chunks: list[RetrievedChunk] = []
        for result in results:
            chunks.append(
                RetrievedChunk(
                    text=result.get("text", ""),
                    score=result.get("score"),
                    metadata=result.get("metadata", {}),
                )
            )
        return chunks

    def query_graph(self, cypher: str, params: dict | None = None) -> list[dict]:
        if self.graph_store is None:
            raise ValueError("graph store is not configured")
        return self.graph_store.query(cypher=cypher, params=params)
