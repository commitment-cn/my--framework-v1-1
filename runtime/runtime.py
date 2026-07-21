from __future__ import annotations

import json
from time import perf_counter
from typing import Any

from core.agent import Agent
from core.llm import ModelResponse, ModelToolCall
from core.message import Message
from tools.base import ToolCall, ToolResult

from .events import EventSink, RunEvent, RunEventType, RunTrace
from .state import (
    CancellationToken,
    RunResult,
    RunState,
    RunStatus,
    RuntimeErrorInfo,
    StopReason,
)


class Runtime:
    """Execute an Agent strategy through a bounded model/tool loop."""

    def __init__(self, event_sink: EventSink | None = None) -> None:
        self.event_sink = event_sink

    def run(
        self,
        agent: Agent,
        input_text: str,
        *,
        cancellation_token: CancellationToken | None = None,
    ) -> RunResult:
        token = cancellation_token or CancellationToken()
        state = RunState(agent_name=agent.name)
        trace = RunTrace(run_id=state.run_id)
        root_event = self._emit(
            trace,
            RunEventType.RUN_STARTED,
            metadata={"agent_name": agent.name},
        )

        try:
            if token.is_cancelled:
                return self._finish(
                    state,
                    trace,
                    root_event.event_id,
                    status=RunStatus.CANCELLED,
                    reason=StopReason.CANCELLED,
                )

            user_message = Message(role="user", content=input_text)
            agent.add_message(user_message)
            state.messages.append(user_message)
            tools = agent.tool_registry.describe_openai_tools()

            while True:
                if token.is_cancelled:
                    return self._finish(
                        state,
                        trace,
                        root_event.event_id,
                        status=RunStatus.CANCELLED,
                        reason=StopReason.CANCELLED,
                    )

                model_messages = agent.build_messages()
                model_request_metadata = self._model_request_metadata(
                    agent,
                    model_messages,
                    tools,
                )
                model_started_at = perf_counter()
                model_started = self._emit(
                    trace,
                    RunEventType.MODEL_CALL_STARTED,
                    parent_event_id=root_event.event_id,
                    metadata=model_request_metadata,
                )

                try:
                    response = agent.llm.create_chat_completion(model_messages, tools=tools)
                except Exception as exc:
                    error = self._error_info(exc)
                    self._emit(
                        trace,
                        RunEventType.MODEL_CALL_FAILED,
                        parent_event_id=model_started.event_id,
                        duration_ms=self._elapsed_ms(model_started_at),
                        status="error",
                        error=error,
                        metadata=model_request_metadata,
                    )
                    return self._finish(
                        state,
                        trace,
                        root_event.event_id,
                        status=RunStatus.FAILED,
                        reason=StopReason.MODEL_ERROR,
                        error=error,
                    )

                state.model_calls += 1
                state.usage.add(response.usage)
                tool_calls = response.tool_calls
                model_completed = self._emit(
                    trace,
                    RunEventType.MODEL_CALL_COMPLETED,
                    parent_event_id=model_started.event_id,
                    duration_ms=self._elapsed_ms(model_started_at),
                    metadata=self._model_response_metadata(response),
                )

                if token.is_cancelled:
                    return self._finish(
                        state,
                        trace,
                        root_event.event_id,
                        status=RunStatus.CANCELLED,
                        reason=StopReason.CANCELLED,
                    )

                if not tool_calls:
                    output = response.content
                    assistant_message = Message(role="assistant", content=output)
                    agent.add_message(assistant_message)
                    state.messages.append(assistant_message)
                    return self._finish(
                        state,
                        trace,
                        root_event.event_id,
                        status=RunStatus.COMPLETED,
                        reason=StopReason.COMPLETED,
                        output=output,
                    )

                if state.tool_steps >= max(agent.config.agent.max_tool_steps, 0):
                    return self._finish(
                        state,
                        trace,
                        root_event.event_id,
                        status=RunStatus.STEP_LIMIT,
                        reason=StopReason.STEP_LIMIT,
                    )

                assistant_message = self._build_assistant_tool_message(response)
                tool_messages: list[Message] = []
                state.tool_steps += 1

                for tool_call in tool_calls:
                    state.tool_calls += 1
                    tool_messages.append(
                        self._run_tool_call(
                            agent,
                            trace,
                            model_completed.event_id,
                            tool_call,
                            token,
                        )
                    )

                # Keep an assistant tool request and all of its results together.
                agent.memory.extend([assistant_message, *tool_messages])
                state.messages.extend([assistant_message, *tool_messages])

        except Exception as exc:
            error = self._error_info(exc)
            return self._finish(
                state,
                trace,
                root_event.event_id,
                status=RunStatus.FAILED,
                reason=StopReason.TOOL_ERROR,
                error=error,
            )

    def _run_tool_call(
        self,
        agent: Agent,
        trace: RunTrace,
        parent_event_id: str,
        tool_call: ModelToolCall,
        token: CancellationToken,
    ) -> Message:
        name = tool_call.name
        call_id = tool_call.call_id
        started_at = perf_counter()
        started = self._emit(
            trace,
            RunEventType.TOOL_CALL_STARTED,
            parent_event_id=parent_event_id,
            metadata={"tool_name": name, "call_id": call_id},
        )

        result: ToolResult
        error: RuntimeErrorInfo | None = None
        try:
            if token.is_cancelled:
                result = ToolResult.fail(error="run cancelled before tool execution")
            else:
                arguments = json.loads(tool_call.arguments or "{}")
                result = agent.tool_registry.execute(
                    ToolCall(tool_name=name, arguments=arguments, call_id=call_id)
                )
        except Exception as exc:
            error = self._error_info(exc)
            result = ToolResult.fail(error=str(exc))

        event_type = (
            RunEventType.TOOL_CALL_COMPLETED
            if result.success
            else RunEventType.TOOL_CALL_FAILED
        )
        if not result.success and error is None:
            error = RuntimeErrorInfo(
                error_type="ToolExecutionError",
                message=result.error or "tool execution failed",
            )
        self._emit(
            trace,
            event_type,
            parent_event_id=started.event_id,
            duration_ms=self._elapsed_ms(started_at),
            status="ok" if result.success else "error",
            error=error,
            metadata={"tool_name": name, "call_id": call_id},
        )

        content = result.content
        if not result.success:
            content = f"ERROR: {result.error}\n{result.content}".strip()
        return Message(role="tool", content=content, tool_call_id=call_id)

    def _finish(
        self,
        state: RunState,
        trace: RunTrace,
        parent_event_id: str,
        *,
        status: RunStatus,
        reason: StopReason,
        output: str | None = None,
        error: RuntimeErrorInfo | None = None,
    ) -> RunResult:
        state.finish(status=status, reason=reason, error=error)
        terminal_type = {
            RunStatus.COMPLETED: RunEventType.RUN_COMPLETED,
            RunStatus.CANCELLED: RunEventType.RUN_CANCELLED,
            RunStatus.STEP_LIMIT: RunEventType.RUN_STEP_LIMIT,
            RunStatus.FAILED: RunEventType.RUN_FAILED,
        }[status]
        self._emit(
            trace,
            terminal_type,
            parent_event_id=parent_event_id,
            status="ok" if status == RunStatus.COMPLETED else status.value,
            error=error,
            metadata={
                "model_calls": state.model_calls,
                "tool_steps": state.tool_steps,
                "tool_calls": state.tool_calls,
                "stop_reason": reason.value,
                **state.usage.to_dict(),
            },
        )
        return RunResult(
            run_id=state.run_id,
            status=status,
            stop_reason=reason,
            output=output,
            state=state,
            trace=trace,
        )

    def _emit(
        self,
        trace: RunTrace,
        event_type: RunEventType,
        *,
        parent_event_id: str | None = None,
        duration_ms: float | None = None,
        status: str = "ok",
        attempt: int = 1,
        metadata: dict[str, Any] | None = None,
        error: RuntimeErrorInfo | None = None,
    ) -> RunEvent:
        event = RunEvent(
            run_id=trace.run_id,
            event_type=event_type,
            sequence=len(trace.events),
            parent_event_id=parent_event_id,
            duration_ms=duration_ms,
            status=status,
            attempt=attempt,
            metadata=metadata or {},
            error_type=error.error_type if error else None,
            error_message=error.message if error else None,
        )
        trace.emit(event)
        if self.event_sink is not None:
            try:
                self.event_sink.emit(event)
            except Exception as exc:
                trace.sink_errors.append(f"{type(exc).__name__}: {exc}")
        return event

    @staticmethod
    def _build_assistant_tool_message(response: ModelResponse) -> Message:
        tool_calls = []
        for tool_call in response.tool_calls:
            tool_calls.append(
                {
                    "id": tool_call.call_id,
                    "type": "function",
                    "function": {
                        "name": tool_call.name,
                        "arguments": tool_call.arguments,
                    },
                }
            )
        return Message(
            role="assistant",
            content=response.content,
            tool_calls=tool_calls,
        )

    @staticmethod
    def _model_request_metadata(
        agent: Agent,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "provider": agent.llm.provider,
            "api_style": agent.llm.api_style,
            "model": agent.llm.model_name,
            "temperature": agent.llm.temperature,
            "max_tokens": agent.llm.max_tokens,
            "timeout": agent.llm.timeout,
            "message_count": len(messages),
            "tool_count": len(tools),
        }

    @staticmethod
    def _model_response_metadata(response: ModelResponse) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "response_id": response.response_id,
            "tool_call_count": len(response.tool_calls),
            "finish_reason": response.finish_reason,
            **response.usage.to_dict(),
            **response.metadata,
        }
        return metadata

    @staticmethod
    def _error_info(exc: Exception) -> RuntimeErrorInfo:
        return RuntimeErrorInfo(error_type=type(exc).__name__, message=str(exc))

    @staticmethod
    def _elapsed_ms(started_at: float) -> float:
        return round((perf_counter() - started_at) * 1000, 3)
