from __future__ import annotations

from abc import ABC, abstractmethod


class BaseGraphStore(ABC):
    @abstractmethod
    def query(self, cypher: str, params: dict | None = None) -> list[dict]:
        raise NotImplementedError


class Neo4jGraphStore(BaseGraphStore):
    def __init__(
        self,
        uri: str,
        username: str,
        password: str,
        database: str | None = None,
    ) -> None:
        self.uri = uri
        self.username = username
        self.password = password
        self.database = database

    def query(self, cypher: str, params: dict | None = None) -> list[dict]:
        raise NotImplementedError(
            "Neo4jGraphStore is a first-pass placeholder. "
            "Implement the actual Neo4j query logic here."
        )
