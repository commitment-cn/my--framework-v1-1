from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

TOOL_SOURCE_LOCAL = "local"
TOOL_SOURCE_NATIVE = "native"
TOOL_SOURCE_MCP = "mcp"


@dataclass(slots=True)
class ToolParameter:
    type: str
    required: bool = True
    description: str = ""
    default: Any = None
    enum: list[Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "type": self.type,
            "required": self.required,
            "description": self.description,
        }
        if self.default is not None:
            payload["default"] = self.default
        if self.enum:
            payload["enum"] = self.enum
        return payload


@dataclass(slots=True)
class ToolCall:
    tool_name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    call_id: str | None = None

    def __post_init__(self) -> None:
        if not self.tool_name.strip():
            raise ValueError("tool_name must not be empty")
        if not isinstance(self.arguments, dict):
            raise TypeError("arguments must be a dict")


@dataclass(slots=True)
class ToolResult:
    success: bool
    content: str
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def ok(
        cls,
        *,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> ToolResult:
        return cls(
            success=True,
            content=content,
            metadata=metadata or {},
        )

    @classmethod
    def fail(
        cls,
        *,
        error: str,
        metadata: dict[str, Any] | None = None,
        content: str = "",
    ) -> ToolResult:
        return cls(
            success=False,
            content=content,
            error=error,
            metadata=metadata or {},
        )


@dataclass(slots=True)
class Tool:
    name: str
    description: str
    func: Callable[..., Any]
    parameters: dict[str, ToolParameter] = field(default_factory=dict)
    source: str = TOOL_SOURCE_LOCAL

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("tool name must not be empty")
        if not self.description.strip():
            raise ValueError("tool description must not be empty")
        if not callable(self.func):
            raise TypeError("tool func must be callable")

    def run(self, **kwargs: Any) -> ToolResult:
        result = self.func(**kwargs)

        if isinstance(result, ToolResult):
            return result
        if isinstance(result, str):
            return ToolResult.ok(content=result)

        return ToolResult.ok(content=str(result))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "source": self.source,
            "parameters": {
                name: parameter.to_dict()
                for name, parameter in self.parameters.items()
            },
        }

    def to_openai_tool(self) -> dict[str, Any]:
        properties: dict[str, Any] = {}
        required: list[str] = []

        for name, parameter in self.parameters.items():
            properties[name] = {
                "type": parameter.type,
                "description": parameter.description,
            }
            if parameter.enum:
                properties[name]["enum"] = parameter.enum
            if parameter.default is not None:
                properties[name]["default"] = parameter.default
            if parameter.required:
                required.append(name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                    "additionalProperties": False,
                },
            },
        }
