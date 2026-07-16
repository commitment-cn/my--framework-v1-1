from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


MessageRole = Literal["system", "user", "assistant", "tool"]


@dataclass(slots=True)
class Message:
    role: MessageRole
    content: str
    tool_call_id: str | None = None
    name: str | None = None

    def __post_init__(self) -> None:
        if self.content is None:
            raise ValueError("message content must not be None")
        if not isinstance(self.content, str):
            raise TypeError("message content must be a string")
        if self.role == "tool" and not self.tool_call_id:
            raise ValueError("tool messages require tool_call_id")

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "role": self.role,
            "content": self.content,
        }
        if self.tool_call_id is not None:
            payload["tool_call_id"] = self.tool_call_id
        if self.name is not None:
            payload["name"] = self.name
        return payload

    def __str__(self) -> str:
        return f"[{self.role}] {self.content}"