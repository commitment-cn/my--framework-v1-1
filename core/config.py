from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class LLMConfig:
    # Backend source label such as openai, deepseek, ollama, or lmstudio.
    # The runtime still speaks a single OpenAI-compatible protocol.
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    temperature: float = 0.7
    max_tokens: int | None = None
    timeout: int = 30


@dataclass(slots=True)
class AgentConfig:
    max_history_length: int = 20
    max_tool_steps: int = 3
    debug: bool = False


@dataclass(slots=True)
class RAGConfig:
    embedding_provider: str = "bailian"
    embedding_model: str = "text-embedding-v4"
    vector_store: str = "qdrant"
    vector_collection: str = "agent_knowledge"
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    graph_store: str = "neo4j"
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_database: str = "neo4j"


@dataclass(slots=True)
class Config:
    llm: LLMConfig = field(default_factory=LLMConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)

    @classmethod
    def from_yaml(cls, path: str = "config.yaml") -> "Config":
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"config file not found: {config_path}")

        with config_path.open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file) or {}

        llm_data = data.get("llm", {})
        agent_data = data.get("agent", {})
        rag_data = data.get("rag", {})

        return cls(
            llm=LLMConfig(
                provider=llm_data.get("provider", "openai"),
                model=llm_data.get("model", "gpt-4o-mini"),
                temperature=llm_data.get("temperature", 0.7),
                max_tokens=llm_data.get("max_tokens"),
                timeout=llm_data.get("timeout", 30),
            ),
            agent=AgentConfig(
                max_history_length=agent_data.get("max_history_length", 20),
                max_tool_steps=agent_data.get("max_tool_steps", 3),
                debug=agent_data.get("debug", False),
            ),
            rag=RAGConfig(
                embedding_provider=rag_data.get("embedding_provider", "bailian"),
                embedding_model=rag_data.get("embedding_model", "text-embedding-v4"),
                vector_store=rag_data.get("vector_store", "qdrant"),
                vector_collection=rag_data.get("vector_collection", "agent_knowledge"),
                qdrant_host=rag_data.get("qdrant_host", "localhost"),
                qdrant_port=rag_data.get("qdrant_port", 6333),
                graph_store=rag_data.get("graph_store", "neo4j"),
                neo4j_uri=rag_data.get("neo4j_uri", "bolt://localhost:7687"),
                neo4j_database=rag_data.get("neo4j_database", "neo4j"),
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
