from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from core.config import Config
from core.agent import Agent
from core.llm import OpenAICompatibleLLM
from my_tools.bash import build_run_bash_tool
from runtime import Runtime
from tools.registry import ToolRegistry


def load_prompt(filename: str) -> str:
    prompt_path = Path(__file__).parent / "prompt" / filename
    return prompt_path.read_text(encoding="utf-8")


def build_llm(config: Config) -> OpenAICompatibleLLM:
    api_key = os.getenv("LLM_API_KEY")
    if not api_key:
        raise ValueError("missing LLM_API_KEY")
    return OpenAICompatibleLLM(
        provider=config.llm.provider,
        config=config.llm,
        api_key=api_key,
        base_url=os.getenv("LLM_BASE_URL"),
    )


def build_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(build_run_bash_tool())
    return registry


def main() -> None:
    load_dotenv()
    config = Config.from_yaml()
    agent = Agent(
        name="CodeAgent",
        llm=build_llm(config),
        system_prompt=load_prompt("code_agent_system.txt"),
        config=config,
        tool_registry=build_registry(),
    )
    result = Runtime().run(
        agent,
        "Please run a bash command to list files in the current directory.",
    )
    if result.ok:
        print(f"Agent Response: {result.output}")
    else:
        error = result.state.error.message if result.state.error else result.stop_reason.value
        print(f"Agent stopped ({result.stop_reason.value}): {error}")


if __name__ == "__main__":
    main()
