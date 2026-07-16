from __future__ import annotations

import shutil
import subprocess

from tools.base import ToolResult


def run_bash(
    command: str,
    timeout_seconds: int = 30,
) -> ToolResult:
    bash_path = shutil.which("bash")
    if bash_path is None:
        return ToolResult.fail(error="bash executable not found on PATH")

    try:
        completed = subprocess.run(
            [bash_path, "-lc", command],
            capture_output=True,
            text=True,
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
