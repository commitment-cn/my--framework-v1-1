from __future__ import annotations

from abc import ABC, abstractmethod

from core.message import Message


class BaseMemory(ABC):
    @abstractmethod
    def add(self, message: Message) -> None:
        raise NotImplementedError

    @abstractmethod
    def extend(self, messages: list[Message]) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_messages(self) -> list[Message]:
        raise NotImplementedError

    @abstractmethod
    def clear(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def trim(self) -> None:
        raise NotImplementedError
