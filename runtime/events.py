from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4


class RunEventType(str, Enum):
    RUN_STARTED = "run.started"
    RUN_COMPLETED = "run.completed"
    RUN_FAILED = "run.failed"
    RUN_CANCELLED = "run.cancelled"
    RUN_STEP_LIMIT = "run.step_limit"
    MODEL_CALL_STARTED = "model_call.started"
    MODEL_CALL_COMPLETED = "model_call.completed"
    MODEL_CALL_FAILED = "model_call.failed"
    TOOL_CALL_STARTED = "tool_call.started"
    TOOL_CALL_COMPLETED = "tool_call.completed"
    TOOL_CALL_FAILED = "tool_call.failed"


@dataclass(slots=True, frozen=True)
class RunEvent:
    run_id: str
    event_type: RunEventType
    sequence: int
    parent_event_id: str | None = None
    event_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    duration_ms: float | None = None
    status: str = "ok"
    attempt: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)
    error_type: str | None = None
    error_message: str | None = None


class EventSink(ABC):
    """Backend-neutral destination for runtime events."""

    @abstractmethod
    def emit(self, event: RunEvent) -> None:
        raise NotImplementedError


@dataclass(slots=True)
class RunTrace(EventSink):
    """In-memory trace recorder used by default for a completed run."""

    run_id: str
    events: list[RunEvent] = field(default_factory=list)
    sink_errors: list[str] = field(default_factory=list)

    def emit(self, event: RunEvent) -> None:
        if event.run_id != self.run_id:
            raise ValueError("event belongs to a different run")
        if event.sequence != len(self.events):
            raise ValueError("event sequence must be contiguous")
        self.events.append(event)

    def copy_events(self) -> list[RunEvent]:
        return list(self.events)
