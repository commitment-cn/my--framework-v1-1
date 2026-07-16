from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .base import TOOL_SOURCE_MCP, Tool, ToolParameter, ToolResult
from .registry import ToolRegistry


@dataclass(slots=True)
class MCPToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any] = field(default_factory=dict)


class MCPToolAdapter:
    def __init__(
        self,
        server_name: str,
        list_tools: Callable[[], list[MCPToolSpec | dict[str, Any]]],
        call_tool: Callable[[str, dict[str, Any]], Any],
    ) -> None:
        self.server_name = server_name
        self._list_tools = list_tools
        self._call_tool = call_tool

    def export_tools(self, prefix: str | None = None) -> list[Tool]:
        tools: list[Tool] = []
        for spec in self._normalize_specs(self._list_tools()):
            tool_name = f"{prefix}.{spec.name}" if prefix else spec.name
            tools.append(
                Tool(
                    name=tool_name,
                    description=spec.description,
                    func=self._build_executor(spec.name),
                    parameters=self._schema_to_parameters(spec.input_schema),
                    source=TOOL_SOURCE_MCP,
                )
            )
        return tools

    def register_to(self, registry: ToolRegistry, prefix: str | None = None) -> list[Tool]:
        tools = self.export_tools(prefix=prefix)
        for tool in tools:
            registry.register(tool)
        return tools

    def _build_executor(self, mcp_tool_name: str) -> Callable[..., ToolResult]:
        def executor(**kwargs: Any) -> ToolResult:
            result = self._call_tool(mcp_tool_name, kwargs)
            if isinstance(result, ToolResult):
                return result
            if isinstance(result, str):
                return ToolResult.ok(
                    content=result,
                    metadata={"mcp_server": self.server_name},
                )
            return ToolResult.ok(
                content=str(result),
                metadata={"mcp_server": self.server_name},
            )

        return executor

    @staticmethod
    def _normalize_specs(items: list[MCPToolSpec | dict[str, Any]]) -> list[MCPToolSpec]:
        specs: list[MCPToolSpec] = []
        for item in items:
            if isinstance(item, MCPToolSpec):
                specs.append(item)
                continue
            specs.append(
                MCPToolSpec(
                    name=item["name"],
                    description=item.get("description", ""),
                    input_schema=item.get("inputSchema", item.get("input_schema", {})),
                )
            )
        return specs

    @staticmethod
    def _schema_to_parameters(schema: dict[str, Any]) -> dict[str, ToolParameter]:
        if not schema:
            return {}

        properties = schema.get("properties", {})
        required = set(schema.get("required", []))
        parameters: dict[str, ToolParameter] = {}

        for name, value in properties.items():
            parameters[name] = ToolParameter(
                type=value.get("type", "string"),
                required=name in required,
                description=value.get("description", ""),
                default=value.get("default"),
                enum=value.get("enum"),
            )

        return parameters
