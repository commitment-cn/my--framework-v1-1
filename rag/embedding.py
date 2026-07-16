from __future__ import annotations

from abc import ABC, abstractmethod


class BaseEmbeddingProvider(ABC):
    @abstractmethod
    def embed_text(self, text: str) -> list[float]:
        raise NotImplementedError

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_text(text) for text in texts]


class BailianEmbeddingProvider(BaseEmbeddingProvider):
    def __init__(
        self,
        api_key: str,
        model_name: str,
        base_url: str | None = None,
    ) -> None:
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = base_url

    def embed_text(self, text: str) -> list[float]:
        raise NotImplementedError(
            "BailianEmbeddingProvider is a first-pass placeholder. "
            "Implement the actual BaiLian embedding call here."
        )
