from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from .config import LLMConfig


OpenAIMessage = dict[str, Any]
OpenAIToolSpec = dict[str, Any]


@dataclass(slots=True)
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0

    def add(self, other: "TokenUsage") -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.total_tokens += other.total_tokens
        self.cached_tokens += other.cached_tokens
        self.reasoning_tokens += other.reasoning_tokens

    def to_dict(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cached_tokens": self.cached_tokens,
            "reasoning_tokens": self.reasoning_tokens,
        }


@dataclass(slots=True, frozen=True)
class ModelToolCall:
    call_id: str
    name: str
    arguments: str = "{}"


@dataclass(slots=True)
class ModelResponse:
    content: str = ""
    tool_calls: list[ModelToolCall] = field(default_factory=list)
    finish_reason: str | None = None
    usage: TokenUsage = field(default_factory=TokenUsage)
    response_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseLLM(ABC):
    def __init__(
        self,
        *,
        provider: str,
        api_style: str,
        model_name: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        timeout: int | float | None = None,
    ) -> None:
        self.provider = provider
        self.api_style = api_style
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

    @abstractmethod
    def create_chat_completion(
        self,
        messages: list[OpenAIMessage],
        tools: list[OpenAIToolSpec] | None = None,
    ) -> ModelResponse:
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
        try:
            from openai import OpenAI
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "openai package is required for OpenAICompatibleLLM"
            ) from exc

        super().__init__(
            provider=provider,
            api_style="openai_compatible",
            model_name=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            timeout=config.timeout,
        )
        self.api_key = api_key
        self.base_url = base_url
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
        )

    def create_chat_completion(
        self,
        messages: list[OpenAIMessage],
        tools: list[OpenAIToolSpec] | None = None,
    ) -> ModelResponse:
        request: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
        }
        if self.max_tokens is not None:
            request["max_tokens"] = self.max_tokens
        if tools:
            request["tools"] = tools
        response = self.client.chat.completions.create(**request)
        return self._normalize_response(response)

    @staticmethod
    def _normalize_response(response: Any) -> ModelResponse:
        choices = getattr(response, "choices", None) or []
        if not choices:
            raise ValueError("model response contained no choices")

        choice = choices[0]
        message = getattr(choice, "message", None)
        if message is None:
            raise ValueError("model response contained no message")

        tool_calls: list[ModelToolCall] = []
        for tool_call in getattr(message, "tool_calls", None) or []:
            function = getattr(tool_call, "function", None)
            if function is None:
                raise ValueError("model tool call contained no function")
            tool_calls.append(
                ModelToolCall(
                    call_id=getattr(tool_call, "id", "") or "",
                    name=getattr(function, "name", "") or "",
                    arguments=getattr(function, "arguments", "") or "{}",
                )
            )

        usage = getattr(response, "usage", None)
        input_tokens = OpenAICompatibleLLM._usage_value(usage, "prompt_tokens")
        output_tokens = OpenAICompatibleLLM._usage_value(usage, "completion_tokens")
        total_tokens = OpenAICompatibleLLM._usage_value(usage, "total_tokens")
        if total_tokens == 0 and (input_tokens or output_tokens):
            total_tokens = input_tokens + output_tokens

        prompt_details = getattr(usage, "prompt_tokens_details", None)
        completion_details = getattr(usage, "completion_tokens_details", None)
        response_metadata: dict[str, Any] = {}
        for source_name, target_name in (
            ("model", "response_model"),
            ("system_fingerprint", "system_fingerprint"),
        ):
            value = getattr(response, source_name, None)
            if value is not None:
                response_metadata[target_name] = value

        return ModelResponse(
            content=getattr(message, "content", None) or "",
            tool_calls=tool_calls,
            finish_reason=getattr(choice, "finish_reason", None),
            usage=TokenUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                cached_tokens=OpenAICompatibleLLM._usage_value(
                    prompt_details,
                    "cached_tokens",
                ),
                reasoning_tokens=OpenAICompatibleLLM._usage_value(
                    completion_details,
                    "reasoning_tokens",
                ),
            ),
            response_id=getattr(response, "id", None),
            metadata=response_metadata,
        )

    @staticmethod
    def _usage_value(source: Any, name: str) -> int:
        value = getattr(source, name, 0) if source is not None else 0
        return value if isinstance(value, int) and not isinstance(value, bool) else 0
