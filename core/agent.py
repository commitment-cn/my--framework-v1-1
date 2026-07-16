from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from .config import Config
from .llm import BaseLLM
from .message import Message
from memory import ConversationMemory


class Agent(ABC):
    def __init__(
        self,
        name: str,
        llm: BaseLLM,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
    ) -> None:
        self.name = name
        self.llm = llm
        self.system_prompt = system_prompt
        self.config = config or Config()
        self.memory = ConversationMemory(max_length=self.config.agent.max_history_length)

    @abstractmethod
    def run(self, input_text: str) -> str:
        raise NotImplementedError

    def add_message(self, message: Message) -> None:
        self.memory.add(message)

    def trim_history(self) -> None:
        self.memory.trim()

    def clear_history(self) -> None:
        self.memory.clear()

    def get_history(self) -> list[Message]:
        return self.memory.get_messages()

    def __str__(self) -> str:
        return f"Agent(name={self.name}, api_style={self.llm.api_style})"
