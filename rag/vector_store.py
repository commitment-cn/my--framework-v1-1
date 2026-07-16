from __future__ import annotations

from abc import ABC, abstractmethod


class BaseVectorStore(ABC):
    @abstractmethod
    def add_text(
        self,
        document_id: str,
        text: str,
        embedding: list[float],
        metadata: dict | None = None,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def search(
        self,
        embedding: list[float],
        top_k: int = 5,
    ) -> list[dict]:
        raise NotImplementedError


class QdrantVectorStore(BaseVectorStore):
    def __init__(
        self,
        collection_name: str,
        host: str = "localhost",
        port: int = 6333,
        api_key: str | None = None,
    ) -> None:
        self.collection_name = collection_name
        self.host = host
        self.port = port
        self.api_key = api_key

    def add_text(
        self,
        document_id: str,
        text: str,
        embedding: list[float],
        metadata: dict | None = None,
    ) -> None:
        raise NotImplementedError(
            "QdrantVectorStore is a first-pass placeholder. "
            "Implement the actual Qdrant upsert here."
        )

    def search(
        self,
        embedding: list[float],
        top_k: int = 5,
    ) -> list[dict]:
        raise NotImplementedError(
            "QdrantVectorStore is a first-pass placeholder. "
            "Implement the actual Qdrant search here."
        )
