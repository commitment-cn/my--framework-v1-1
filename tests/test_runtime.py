from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import Any

from core.agent import Agent
from core.config import AgentConfig, Config
from core.llm import (
    BaseLLM,
    ModelResponse,
    ModelToolCall,
    OpenAICompatibleLLM,
    TokenUsage,
)
from runtime import (
    CancellationToken,
    EventSink,
    RunEvent,
    RunEventType,
    RunStatus,
    Runtime,
    StopReason,
)
from tools.base import ToolResult
from tools.registry import ToolRegistry


def tool_call(call_id: str, name: str, arguments: str) -> ModelToolCall:
    return ModelToolCall(call_id=call_id, name=name, arguments=arguments)


def response(
    content: str | None = None,
    *,
    tool_calls: list[ModelToolCall] | None = None,
    finish_reason: str = "stop",
) -> ModelResponse:
    return ModelResponse(
        content=content or "",
        tool_calls=tool_calls or [],
        finish_reason=finish_reason,
        usage=TokenUsage(input_tokens=10, output_tokens=2, total_tokens=12),
        response_id="response-id",
    )


class FakeLLM(BaseLLM):
    def __init__(self, outcomes: list[ModelResponse | Exception]) -> None:
        super().__init__(
            provider="test",
            api_style="openai_compatible",
            model_name="fake-model",
            temperature=0.25,
            max_tokens=512,
            timeout=15,
        )
        self.outcomes = list(outcomes)
        self.requests: list[dict[str, Any]] = []

    def create_chat_completion(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> ModelResponse:
        self.requests.append({"messages": messages, "tools": tools})
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class CollectingSink(EventSink):
    def __init__(self) -> None:
        self.events: list[RunEvent] = []

    def emit(self, event: RunEvent) -> None:
        self.events.append(event)


class FailingSink(EventSink):
    def emit(self, event: RunEvent) -> None:
        raise OSError("export unavailable")


class CancellingLLM(FakeLLM):
    def __init__(
        self,
        outcomes: list[ModelResponse | Exception],
        token: CancellationToken,
    ) -> None:
        super().__init__(outcomes)
        self.token = token

    def create_chat_completion(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> ModelResponse:
        response_value = super().create_chat_completion(messages, tools)
        self.token.cancel()
        return response_value


def build_agent(
    outcomes: list[ModelResponse | Exception],
    *,
    max_tool_steps: int = 3,
) -> Agent:
    registry = ToolRegistry()
    registry.register_callable(
        name="echo",
        description="Echo a value.",
        func=lambda value: value,
        parameters={"value": {"type": "string"}},
    )
    config = Config(agent=AgentConfig(max_history_length=20, max_tool_steps=max_tool_steps))
    return Agent(
        name="test-agent",
        llm=FakeLLM(outcomes),
        system_prompt="Test system prompt.",
        config=config,
        tool_registry=registry,
    )


class RuntimeTests(unittest.TestCase):
    def test_runtime_is_reusable_across_agent_definitions(self) -> None:
        sink = CollectingSink()
        runtime = Runtime(event_sink=sink)
        first_agent = build_agent([response("first")])
        second_agent = build_agent([response("second")])
        second_agent.name = "second-agent"
        second_agent.system_prompt = "A different strategy."

        first = runtime.run(first_agent, "one")
        second = runtime.run(second_agent, "two")

        self.assertEqual(first.output, "first")
        self.assertEqual(second.output, "second")
        self.assertNotEqual(first.run_id, second.run_id)
        self.assertEqual(
            [event for event in sink.events if event.run_id == first.run_id],
            first.trace.events,
        )
        self.assertEqual(
            [event for event in sink.events if event.run_id == second.run_id],
            second.trace.events,
        )

    def test_completed_run_persists_messages_and_usage_trace(self) -> None:
        agent = build_agent([response("done")])

        result = Runtime().run(agent, "hello")

        self.assertTrue(result.ok)
        self.assertEqual(result.output, "done")
        self.assertEqual(result.stop_reason, StopReason.COMPLETED)
        self.assertEqual([message.role for message in agent.get_history()], ["user", "assistant"])
        completed = next(
            event
            for event in result.trace.events
            if event.event_type == RunEventType.MODEL_CALL_COMPLETED
        )
        self.assertEqual(completed.metadata["total_tokens"], 12)
        self.assertEqual(completed.metadata["input_tokens"], 10)
        started = next(
            event
            for event in result.trace.events
            if event.event_type == RunEventType.MODEL_CALL_STARTED
        )
        self.assertEqual(started.metadata["provider"], "test")
        self.assertEqual(started.metadata["temperature"], 0.25)
        self.assertEqual(started.metadata["max_tokens"], 512)
        self.assertEqual(started.metadata["timeout"], 15)
        self.assertEqual(started.metadata["tool_count"], 1)
        self.assertEqual(result.state.usage.total_tokens, 12)
        self.assertEqual(
            [event.sequence for event in result.trace.events],
            list(range(len(result.trace.events))),
        )

    def test_multiple_tool_steps_use_one_runtime_loop(self) -> None:
        agent = build_agent(
            [
                response(
                    tool_calls=[tool_call("call-1", "echo", '{"value": "one"}')],
                    finish_reason="tool_calls",
                ),
                response(
                    tool_calls=[tool_call("call-2", "echo", '{"value": "two"}')],
                    finish_reason="tool_calls",
                ),
                response("finished"),
            ]
        )

        result = Runtime().run(agent, "run tools")

        self.assertEqual(result.status, RunStatus.COMPLETED)
        self.assertEqual(result.state.model_calls, 3)
        self.assertEqual(result.state.tool_steps, 2)
        self.assertEqual(result.state.tool_calls, 2)
        self.assertEqual(result.state.usage.input_tokens, 30)
        self.assertEqual(result.state.usage.output_tokens, 6)
        self.assertEqual(result.state.usage.total_tokens, 36)
        self.assertEqual(
            [message.role for message in agent.get_history()],
            ["user", "assistant", "tool", "assistant", "tool", "assistant"],
        )
        self.assertEqual(agent.get_history()[2].content, "one")
        self.assertEqual(agent.get_history()[4].content, "two")
        self.assertEqual(result.trace.events[-1].metadata["total_tokens"], 36)

    def test_step_limit_does_not_persist_dangling_tool_request(self) -> None:
        agent = build_agent(
            [
                response(
                    tool_calls=[tool_call("call-1", "echo", '{"value": "one"}')],
                    finish_reason="tool_calls",
                ),
                response(
                    tool_calls=[tool_call("call-2", "echo", '{"value": "two"}')],
                    finish_reason="tool_calls",
                ),
            ],
            max_tool_steps=1,
        )

        result = Runtime().run(agent, "run tools")

        self.assertEqual(result.status, RunStatus.STEP_LIMIT)
        self.assertEqual(result.stop_reason, StopReason.STEP_LIMIT)
        self.assertEqual(
            [message.role for message in agent.get_history()],
            ["user", "assistant", "tool"],
        )
        self.assertEqual(result.trace.events[-1].event_type, RunEventType.RUN_STEP_LIMIT)

    def test_invalid_tool_arguments_are_returned_to_the_model(self) -> None:
        agent = build_agent(
            [
                response(
                    tool_calls=[tool_call("call-1", "echo", "not-json")],
                    finish_reason="tool_calls",
                ),
                response("recovered"),
            ]
        )

        result = Runtime().run(agent, "run tool")

        self.assertTrue(result.ok)
        self.assertIn("ERROR:", agent.get_history()[2].content)
        self.assertIn(
            RunEventType.TOOL_CALL_FAILED,
            [event.event_type for event in result.trace.events],
        )

    def test_unknown_tool_is_returned_to_the_model(self) -> None:
        agent = build_agent(
            [
                response(
                    tool_calls=[tool_call("call-1", "missing", "{}")],
                    finish_reason="tool_calls",
                ),
                response("recovered"),
            ]
        )

        result = Runtime().run(agent, "run missing tool")

        self.assertTrue(result.ok)
        self.assertIn("tool not found", agent.get_history()[2].content)
        self.assertIn(
            RunEventType.TOOL_CALL_FAILED,
            [event.event_type for event in result.trace.events],
        )

    def test_tool_exception_is_returned_to_the_model(self) -> None:
        agent = build_agent(
            [
                response(
                    tool_calls=[tool_call("call-1", "explode", "{}")],
                    finish_reason="tool_calls",
                ),
                response("recovered"),
            ]
        )

        def explode() -> str:
            raise RuntimeError("boom")

        agent.tool_registry.register_callable(
            name="explode",
            description="Raise an error.",
            func=explode,
        )

        result = Runtime().run(agent, "run exploding tool")

        self.assertTrue(result.ok)
        self.assertIn("boom", agent.get_history()[2].content)
        self.assertEqual(result.state.tool_calls, 1)

    def test_explicit_tool_failure_is_returned_to_the_model(self) -> None:
        agent = build_agent(
            [
                response(
                    tool_calls=[tool_call("call-1", "fail", "{}")],
                    finish_reason="tool_calls",
                ),
                response("handled"),
            ]
        )
        agent.tool_registry.register_callable(
            name="fail",
            description="Return a typed failure.",
            func=lambda: ToolResult.fail(error="expected failure"),
        )

        result = Runtime().run(agent, "run failing tool")

        self.assertTrue(result.ok)
        self.assertIn("expected failure", agent.get_history()[2].content)
        failed = next(
            event
            for event in result.trace.events
            if event.event_type == RunEventType.TOOL_CALL_FAILED
        )
        self.assertEqual(failed.error_type, "ToolExecutionError")

    def test_model_error_returns_typed_failure(self) -> None:
        agent = build_agent([TimeoutError("model timed out")])

        result = Runtime().run(agent, "hello")

        self.assertEqual(result.status, RunStatus.FAILED)
        self.assertEqual(result.stop_reason, StopReason.MODEL_ERROR)
        self.assertEqual(result.state.error.error_type, "TimeoutError")
        self.assertEqual(result.trace.events[-1].event_type, RunEventType.RUN_FAILED)

    def test_pre_cancelled_run_does_not_mutate_memory(self) -> None:
        agent = build_agent([response("unused")])
        token = CancellationToken()
        token.cancel()

        result = Runtime().run(agent, "hello", cancellation_token=token)

        self.assertEqual(result.status, RunStatus.CANCELLED)
        self.assertEqual(result.stop_reason, StopReason.CANCELLED)
        self.assertEqual(agent.get_history(), [])
        self.assertEqual(result.trace.events[-1].event_type, RunEventType.RUN_CANCELLED)

    def test_cancellation_after_model_call_stops_before_tool_execution(self) -> None:
        token = CancellationToken()
        agent = build_agent([])
        agent.llm = CancellingLLM(
            [
                response(
                    tool_calls=[tool_call("call-1", "echo", '{"value": "unused"}')],
                    finish_reason="tool_calls",
                )
            ],
            token,
        )

        result = Runtime().run(agent, "cancel during model call", cancellation_token=token)

        self.assertEqual(result.status, RunStatus.CANCELLED)
        self.assertEqual(result.state.model_calls, 1)
        self.assertEqual(result.state.tool_calls, 0)
        self.assertEqual([message.role for message in agent.get_history()], ["user"])

    def test_consecutive_runs_reuse_agent_conversation_memory(self) -> None:
        agent = build_agent([response("My name is Ming."), response("Ming")])
        runtime = Runtime()

        first = runtime.run(agent, "My name is Ming.")
        second = runtime.run(agent, "What is my name?")

        self.assertTrue(first.ok)
        self.assertTrue(second.ok)
        second_request = agent.llm.requests[1]["messages"]
        self.assertEqual(
            [message["role"] for message in second_request],
            ["system", "user", "assistant", "user"],
        )
        self.assertEqual(second_request[1]["content"], "My name is Ming.")
        self.assertEqual(second_request[2]["content"], "My name is Ming.")

    def test_missing_usage_defaults_to_zero(self) -> None:
        agent = build_agent([ModelResponse(content="done", finish_reason="stop")])

        result = Runtime().run(agent, "hello")

        self.assertTrue(result.ok)
        self.assertEqual(result.state.usage.total_tokens, 0)
        self.assertEqual(result.trace.events[-1].metadata["total_tokens"], 0)

    def test_event_sink_failure_does_not_fail_the_run(self) -> None:
        agent = build_agent([response("done")])

        result = Runtime(event_sink=FailingSink()).run(agent, "hello")

        self.assertTrue(result.ok)
        self.assertGreaterEqual(len(result.trace.sink_errors), 1)
        self.assertIn("export unavailable", result.trace.sink_errors[0])

    def test_openai_response_is_normalized_with_detailed_usage(self) -> None:
        raw_tool_call = SimpleNamespace(
            id="call-1",
            function=SimpleNamespace(name="echo", arguments='{"value": "ok"}'),
        )
        raw_response = SimpleNamespace(
            id="response-1",
            model="provider-model",
            system_fingerprint="fingerprint-1",
            choices=[
                SimpleNamespace(
                    finish_reason="tool_calls",
                    message=SimpleNamespace(content=None, tool_calls=[raw_tool_call]),
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=20,
                completion_tokens=5,
                total_tokens=25,
                prompt_tokens_details=SimpleNamespace(cached_tokens=8),
                completion_tokens_details=SimpleNamespace(reasoning_tokens=3),
            ),
        )

        normalized = OpenAICompatibleLLM._normalize_response(raw_response)

        self.assertEqual(normalized.response_id, "response-1")
        self.assertEqual(normalized.finish_reason, "tool_calls")
        self.assertEqual(normalized.tool_calls[0], tool_call("call-1", "echo", '{"value": "ok"}'))
        self.assertEqual(normalized.usage.input_tokens, 20)
        self.assertEqual(normalized.usage.output_tokens, 5)
        self.assertEqual(normalized.usage.cached_tokens, 8)
        self.assertEqual(normalized.usage.reasoning_tokens, 3)
        self.assertEqual(normalized.metadata["response_model"], "provider-model")


if __name__ == "__main__":
    unittest.main()
