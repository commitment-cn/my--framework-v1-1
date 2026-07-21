from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from tools.base import Tool, ToolParameter, ToolResult


def _find_bash() -> str | None:
    bash_path = shutil.which("bash")
    if bash_path:
        return bash_path

    candidates = (
        Path(r"C:\Program Files\Git\bin\bash.exe"),
        Path(r"C:\Program Files\Git\usr\bin\bash.exe"),
    )
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    return None


def run_bash(
    command: str,
    timeout_seconds: int = 30,
) -> ToolResult:
    bash_path = _find_bash()
    if bash_path is None:
        return ToolResult.fail(
            error="bash executable not found. Install Git Bash or add bash.exe to PATH."
        )

    try:
        completed = subprocess.run(
            [bash_path, "-lc", command],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return ToolResult.fail(
            error=f"bash command timed out after {timeout_seconds} seconds"
        )
    except OSError as exc:
        return ToolResult.fail(error=f"failed to start bash: {exc}")

    output = (completed.stdout or "").strip()
    error_output = (completed.stderr or "").strip()
    content_parts: list[str] = []

    if output:
        content_parts.append(output)
    if error_output:
        content_parts.append(f"stderr:\n{error_output}")
    if not content_parts:
        content_parts.append("(no output)")

    content = "\n\n".join(content_parts)
    if completed.returncode != 0:
        return ToolResult.fail(
            error=f"bash exited with code {completed.returncode}",
            content=content,
            metadata={"returncode": completed.returncode},
        )

    return ToolResult.ok(
        content=content,
        metadata={"returncode": completed.returncode},
    )


def run_bash_text(command: str, timeout_seconds: int = 30) -> str:
    result = run_bash(command=command, timeout_seconds=timeout_seconds)
    if result.success:
        return result.content
    return f"ERROR: {result.error}\n{result.content}"


def build_run_bash_tool() -> Tool:
    return Tool(
        name="run_bash",
        description="Run a bash command in the local shell and return stdout and stderr.",
        func=run_bash,
        parameters={
            "command": ToolParameter(
                type="string",
                required=True,
                description="The bash command to execute.",
            ),
            "timeout_seconds": ToolParameter(
                type="integer",
                required=False,
                description="Maximum time to wait before stopping the command.",
                default=30,
            ),
        },
    )


RUN_BASH_TOOL = build_run_bash_tool()
