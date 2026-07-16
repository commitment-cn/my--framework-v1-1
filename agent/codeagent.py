from __future__ import annotations

import json
from typing import Any

from core.agent import Agent
from core.config import Config
from core.llm import BaseLLM
from core.message import Message
from tools.base import ToolCall
from tools.registry import ToolRegistry


DEFAULT_CODE_AGENT_PROMPT = """You are a pragmatic code agent.

Use available tools when they help you produce a better answer.
If a tool is not needed, answer directly.
Keep answers concise and concrete.
"""


class CodeAgent(Agent):
    def __init__(
        self,
        name: str,
        llm: BaseLLM,
        system_prompt: str | None = None,
        config: Config | None = None,
        tool_registry: ToolRegistry | None = None,
    ) -> None:
        super().__init__(
            name=name,
            llm=llm,
            system_prompt=system_prompt or DEFAULT_CODE_AGENT_PROMPT,
            config=config,
        )
        self.tool_registry = tool_registry or ToolRegistry()

    def build_messages(self, input_text: str) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self.system_prompt or DEFAULT_CODE_AGENT_PROMPT}
        ]
        messages.extend(message.to_dict() for message in self.get_history())
        messages.append({"role": "user", "content": input_text})
        return messages

    def run(self, input_text: str) -> str:
        messages: list[dict[str, object]] = self.build_messages(input_text)
        tools = self.tool_registry.describe_openai_tools()

        for _ in range(self.config.agent.max_tool_steps):
            response = self.llm.create_chat_completion(messages=messages, tools=tools)
            message = self._extract_message(response)
            tool_calls = getattr(message, "tool_calls", None) or []

            if not tool_calls:
                final_text = self._coerce_content(getattr(message, "content", None)).strip()
                if not final_text:
                    raise ValueError("llm returned neither tool calls nor final content")
                self.add_message(Message(role="user", content=input_text))
                self.add_message(Message(role="assistant", content=final_text))
                return final_text

            messages.append(self._build_assistant_tool_call_message(message))

            for tool_call in tool_calls:
                parse_result = self._parse_tool_call(tool_call)
                if parse_result["error"] is not None:
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": parse_result["id"],
                            "content": parse_result["error"],
                        }
                    )
                    continue

                result = self.tool_registry.execute(
                    ToolCall(
                        tool_name=parse_result["name"],
                        arguments=parse_result["arguments"],
                        call_id=parse_result["id"],
                    )
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": parse_result["id"],
                        "content": self._format_tool_result(result),
                    }
                )

        raise RuntimeError("agent exceeded max tool steps")

    @staticmethod
    def _extract_message(response: Any) -> Any:
        try:
            return response.choices[0].message
        except (AttributeError, IndexError, KeyError, TypeError) as exc:
            raise ValueError("failed to extract llm response message") from exc

    @staticmethod
    def _coerce_content(content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts: list[str] = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text = part.get("text")
                    if isinstance(text, str):
                        text_parts.append(text)
                elif hasattr(part, "type") and getattr(part, "type") == "text":
                    text = getattr(part, "text", None)
                    if isinstance(text, str):
                        text_parts.append(text)
            return "".join(text_parts)
        raise TypeError("llm response content must be a string or text parts")

    @staticmethod
    def _build_assistant_tool_call_message(message: Any) -> dict[str, object]:
        return {
            "role": "assistant",
            "content": CodeAgent._coerce_content(getattr(message, "content", None)),
            "tool_calls": [
                {
                    "id": getattr(tool_call, "id", ""),
                    "type": "function",
                    "function": {
                        "name": getattr(getattr(tool_call, "function", None), "name", ""),
                        "arguments": getattr(getattr(tool_call, "function", None), "arguments", None) or "{}",
                    },
                }
                for tool_call in getattr(message, "tool_calls", None) or []
                if getattr(tool_call, "type", None) == "function"
            ],
        }

    @staticmethod
    def _parse_tool_call(tool_call: Any) -> dict[str, Any]:
        function = getattr(tool_call, "function", None)
        tool_id = getattr(tool_call, "id", "")
        tool_name = getattr(function, "name", "")
        raw_arguments = getattr(function, "arguments", None) or "{}"

        try:
            arguments = json.loads(raw_arguments)
        except json.JSONDecodeError as exc:
            return {
                "id": tool_id,
                "name": tool_name,
                "arguments": {},
                "error": f"failed to decode tool arguments for tool '{tool_name}': {exc}",
            }

        if not isinstance(arguments, dict):
            return {
                "id": tool_id,
                "name": tool_name,
                "arguments": {},
                "error": f"tool arguments for tool '{tool_name}' must decode to an object",
            }

        return {
            "id": tool_id,
            "name": tool_name,
            "arguments": arguments,
            "error": None,
        }

    @staticmethod
    def _format_tool_result(result: object) -> str:
        if hasattr(result, "success") and hasattr(result, "content") and hasattr(result, "error"):
            if result.success:
                return result.content
            return result.error or result.content or "tool execution failed"
        return str(result)
