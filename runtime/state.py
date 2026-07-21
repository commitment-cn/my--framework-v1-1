from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from threading import Event
from typing import Any
from uuid import uuid4

from core.llm import TokenUsage
from core.message import Message

from .events import RunTrace


class RunStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    STEP_LIMIT = "step_limit"


class StopReason(str, Enum):
    COMPLETED = "completed"
    MODEL_ERROR = "model_error"
    TOOL_ERROR = "tool_error"
    STEP_LIMIT = "step_limit"
    CANCELLED = "cancelled"


@dataclass(slots=True)
class RuntimeErrorInfo:
    error_type: str
    message: str


class CancellationToken:
    """Cooperative cancellation for calls between model and tool operations."""

    def __init__(self) -> None:
        self._event = Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def is_cancelled(self) -> bool:
        return self._event.is_set()


@dataclass(slots=True)
class RunState:
    run_id: str = field(default_factory=lambda: str(uuid4()))
    agent_name: str = ""
    status: RunStatus = RunStatus.RUNNING
    stop_reason: StopReason | None = None
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    ended_at: datetime | None = None
    model_calls: int = 0
    tool_steps: int = 0
    tool_calls: int = 0
    usage: TokenUsage = field(default_factory=TokenUsage)
    messages: list[Message] = field(default_factory=list)
    error: RuntimeErrorInfo | None = None

    def finish(
        self,
        *,
        status: RunStatus,
        reason: StopReason,
        error: RuntimeErrorInfo | None = None,
    ) -> None:
        self.status = status
        self.stop_reason = reason
        self.error = error
        self.ended_at = datetime.now(timezone.utc)


@dataclass(slots=True)
class RunResult:
    run_id: str
    status: RunStatus
    stop_reason: StopReason
    output: str | None
    state: RunState
    trace: RunTrace

    @property
    def ok(self) -> bool:
        return self.status == RunStatus.COMPLETED
