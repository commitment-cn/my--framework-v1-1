from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


MessageRole = Literal["system", "user", "assistant", "tool"]


@dataclass(slots=True)
class Message:
    role: MessageRole
    content: str = ""
    tool_call_id: str | None = None
    name: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.content is None:
            raise ValueError("message content must not be None")
        if not isinstance(self.content, str):
            raise TypeError("message content must be a string")
        if self.role == "tool" and not self.tool_call_id:
            raise ValueError("tool messages require tool_call_id")
        if self.tool_calls is not None and self.role != "assistant":
            raise ValueError("tool_calls are only valid on assistant messages")

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "role": self.role,
            "content": self.content,
        }
        if self.tool_call_id is not None:
            payload["tool_call_id"] = self.tool_call_id
        if self.name is not None:
            payload["name"] = self.name
        if self.tool_calls is not None:
            payload["tool_calls"] = self.tool_calls
        if self.metadata is not None:
            payload["metadata"] = self.metadata
        return payload

    def to_openai_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "role": self.role,
            "content": self.content,
        }
        if self.tool_call_id is not None:
            payload["tool_call_id"] = self.tool_call_id
        if self.name is not None:
            payload["name"] = self.name
        if self.tool_calls is not None:
            payload["tool_calls"] = self.tool_calls
        return payload

    def to_anthropic_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.name is not None:
            payload["name"] = self.name
        if self.tool_call_id is not None:
            payload["tool_call_id"] = self.tool_call_id
        if self.metadata is not None:
            payload["metadata"] = self.metadata
        return payload

    def __str__(self) -> str:
        return f"[{self.role}] {self.content}"
