from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from openai import OpenAI

from .config import LLMConfig


OpenAIMessage = dict[str, Any]
OpenAIToolSpec = dict[str, Any]


class BaseLLM(ABC):
    def __init__(
        self,
        *,
        provider: str,
        api_style: str,
        model_name: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> None:
        self.provider = provider
        self.api_style = api_style
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

    @abstractmethod
    def create_chat_completion(
        self,
        messages: list[OpenAIMessage],
        tools: list[OpenAIToolSpec] | None = None,
    ) -> Any:
        raise NotImplementedError


class OpenAICompatibleLLM(BaseLLM):
    def __init__(
        self,
        *,
        provider: str,
        config: LLMConfig,
        api_key: str,
        base_url: str | None = None,
    ) -> None:
        super().__init__(
            provider=provider,
            api_style="openai_compatible",
            model_name=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = config.timeout
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
        )

    def create_chat_completion(
        self,
        messages: list[OpenAIMessage],
        tools: list[OpenAIToolSpec] | None = None,
    ) -> Any:
        request: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
        }
        if self.max_tokens is not None:
            request["max_tokens"] = self.max_tokens
        if tools:
            request["tools"] = tools
        return self.client.chat.completions.create(**request)
