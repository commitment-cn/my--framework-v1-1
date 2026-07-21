from __future__ import annotations

from typing import Any, Optional

from .config import Config
from .llm import BaseLLM
from .message import Message
from memory import ConversationMemory
from tools.registry import ToolRegistry


class Agent:
    def __init__(
        self,
        name: str,
        llm: BaseLLM,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        tool_registry: ToolRegistry | None = None,
    ) -> None:
        self.name = name
        self.llm = llm
        self.system_prompt = system_prompt
        self.config = config or Config()
        self.memory = ConversationMemory(max_length=self.config.agent.max_history_length)
        self.tool_registry = tool_registry or ToolRegistry()

    def add_message(self, message: Message) -> None:
        self.memory.add(message)

    def build_messages(self) -> list[dict[str, Any]]:
        messages = [message.to_openai_dict() for message in self.get_history()]
        if self.system_prompt:
            messages.insert(0, {"role": "system", "content": self.system_prompt})
        return messages

    def trim_history(self) -> None:
        self.memory.trim()

    def clear_history(self) -> None:
        self.memory.clear()

    def get_history(self) -> list[Message]:
        return self.memory.get_messages()

    def __str__(self) -> str:
        return f"Agent(name={self.name}, api_style={self.llm.api_style})"
