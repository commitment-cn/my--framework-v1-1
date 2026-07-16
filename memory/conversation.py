from __future__ import annotations

from core.message import Message

from .base import BaseMemory


class ConversationMemory(BaseMemory):
    def __init__(self, max_length: int | None = None) -> None:
        self.max_length = max_length
        self._messages: list[Message] = []

    def add(self, message: Message) -> None:
        self._messages.append(message)
        self.trim()

    def extend(self, messages: list[Message]) -> None:
        self._messages.extend(messages)
        self.trim()

    def get_messages(self) -> list[Message]:
        return self._messages.copy()

    def clear(self) -> None:
        self._messages.clear()

    def trim(self) -> None:
        if self.max_length is None or self.max_length <= 0:
            return
        if len(self._messages) > self.max_length:
            self._messages = self._messages[-self.max_length:]
