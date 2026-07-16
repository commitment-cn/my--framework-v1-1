from __future__ import annotations

from typing import Any, Callable

from .base import (
    TOOL_SOURCE_LOCAL,
    Tool,
    ToolCall,
    ToolParameter,
    ToolResult,
)


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def register_callable(
        self,
        *,
        name: str,
        description: str,
        func: Callable[..., Any],
        parameters: dict[str, ToolParameter | dict[str, Any]] | None = None,
        source: str = TOOL_SOURCE_LOCAL,
    ) -> Tool:
        tool = Tool(
            name=name,
            description=description,
            func=func,
            parameters=self._normalize_parameters(parameters),
            source=source,
        )
        self.register(tool)
        return tool

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise KeyError(f"tool not found: {name}") from exc

    def list_tools(self) -> list[Tool]:
        return list(self._tools.values())

    def describe_tools(self) -> list[dict[str, Any]]:
        return [tool.to_dict() for tool in self.list_tools()]

    def describe_openai_tools(self) -> list[dict[str, Any]]:
        return [tool.to_openai_tool() for tool in self.list_tools()]

    def execute(self, tool_call: ToolCall) -> ToolResult:
        tool = self.get(tool_call.tool_name)
        self._validate_arguments(tool, tool_call.arguments)

        try:
            result = tool.run(**tool_call.arguments)
        except Exception as exc:
            return ToolResult.fail(error=str(exc))

        return result

    def call(self, name: str, **kwargs: Any) -> ToolResult:
        return self.execute(
            ToolCall(
                tool_name=name,
                arguments=kwargs,
            )
        )

    @staticmethod
    def _normalize_parameters(
        parameters: dict[str, ToolParameter | dict[str, Any]] | None,
    ) -> dict[str, ToolParameter]:
        if not parameters:
            return {}

        normalized: dict[str, ToolParameter] = {}
        for name, parameter in parameters.items():
            if isinstance(parameter, ToolParameter):
                normalized[name] = parameter
                continue
            if not isinstance(parameter, dict):
                raise TypeError("tool parameter definition must be ToolParameter or dict")
            normalized[name] = ToolParameter(
                type=parameter["type"],
                required=parameter.get("required", True),
                description=parameter.get("description", ""),
                default=parameter.get("default"),
                enum=parameter.get("enum"),
            )
        return normalized

    @staticmethod
    def _validate_arguments(tool: Tool, arguments: dict[str, Any]) -> None:
        for name, parameter in tool.parameters.items():
            if parameter.required and name not in arguments:
                raise ValueError(f"missing required argument '{name}' for tool '{tool.name}'")
            if name not in arguments:
                continue
            if not ToolRegistry._matches_type(arguments[name], parameter.type):
                raise TypeError(
                    f"argument '{name}' for tool '{tool.name}' must be of type {parameter.type}"
                )
        unknown_arguments = set(arguments) - set(tool.parameters)
        if unknown_arguments:
            unknown = ", ".join(sorted(unknown_arguments))
            raise ValueError(f"unknown arguments for tool '{tool.name}': {unknown}")

    @staticmethod
    def _matches_type(value: Any, declared_type: str) -> bool:
        if declared_type == "string":
            return isinstance(value, str)
        if declared_type == "integer":
            return isinstance(value, int) and not isinstance(value, bool)
        if declared_type == "number":
            return (isinstance(value, int) or isinstance(value, float)) and not isinstance(value, bool)
        if declared_type == "boolean":
            return isinstance(value, bool)
        if declared_type == "object":
            return isinstance(value, dict)
        if declared_type == "array":
            return isinstance(value, list)
        return True
