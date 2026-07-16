from __future__ import annotations

import os
from datetime import datetime

from dotenv import load_dotenv

from agent.codeagent import CodeAgent
from core.config import Config
from core.llm import OpenAICompatibleLLM
from my_tools.bash import run_bash
from tools.mcp import MCPToolAdapter
from tools.registry import ToolRegistry


def get_current_time() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def build_demo_mcp_adapter() -> MCPToolAdapter:
    def list_tools() -> list[dict[str, object]]:
        return [
            {
                "name": "echo",
                "description": "Echo the given text from an MCP server.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "Text to echo back.",
                        }
                    },
                    "required": ["text"],
                },
            }
        ]

    def call_tool(name: str, arguments: dict[str, object]) -> str:
        if name != "echo":
            raise ValueError(f"unsupported demo MCP tool: {name}")
        return f"MCP Echo: {arguments['text']}"

    return MCPToolAdapter(
        server_name="demo-server",
        list_tools=list_tools,
        call_tool=call_tool,
    )


def build_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register_callable(
        name="get_current_time",
        description="Get the current local time.",
        func=get_current_time,
    )
    registry.register_callable(
        name="bash",
        description="Run a bash command and return stdout and stderr.",
        func=run_bash,
        parameters={
            "command": {
                "type": "string",
                "description": "The bash command to execute.",
            },
            "timeout_seconds": {
                "type": "integer",
                "required": False,
                "description": "Maximum execution time in seconds.",
                "default": 30,
            },
        },
    )

    demo_mcp = build_demo_mcp_adapter()
    demo_mcp.register_to(registry, prefix="demo_mcp")
    return registry


def build_llm(config: Config) -> OpenAICompatibleLLM:
    api_key = os.getenv("LLM_API_KEY")
    base_url = os.getenv("LLM_BASE_URL")

    if not api_key:
        raise ValueError("missing LLM_API_KEY")

    return OpenAICompatibleLLM(
        provider=config.llm.provider,
        config=config.llm,
        api_key=api_key,
        base_url=base_url,
    )


def main() -> None:
    load_dotenv()

    config = Config.from_yaml()
    llm = build_llm(config)
    registry = build_registry()

    agent = CodeAgent(
        name="code-agent",
        llm=llm,
        config=config,
        tool_registry=registry,
    )

    print("=== OpenAI-Compatible Tool Calling Demo ===")
    print("Available tools:")
    for tool in registry.describe_tools():
        print(f"- {tool['name']} ({tool['source']})")

    print("\nType 'exit' to quit.")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break
        if not user_input:
            continue

        try:
            response = agent.run(user_input)
        except Exception as exc:
            print(f"Error: {exc}")
            continue

        print(f"Agent: {response}")


if __name__ == "__main__":
    main()
