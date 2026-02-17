from __future__ import annotations

import os
import shutil
import subprocess
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Default models (upgraded)
# ---------------------------------------------------------------------------
DEFAULT_OPENAI_MODEL = "gpt-5.2"
DEFAULT_CLAUDE_MODEL = "claude-sonnet-4-5"  # alias -> always latest Sonnet 4.5


def build_prompt(mode: str, query: str, passages: List[Dict[str, Any]]) -> str:
    """
    Build a single text prompt used across providers.
    `passages` items: {"text": "...", "meta": {...}}
    """
    mode = (mode or "study_guide").strip()

    context_blocks = []
    for i, p in enumerate(passages[:12], start=1):
        meta = p.get("meta") or {}
        source = meta.get("source", "unknown")
        kind = meta.get("kind", "doc")
        url = meta.get("url", "")
        header = f"[{i}] source={source} kind={kind}"
        if url:
            header += f" url={url}"
        text = (p.get("text") or "").strip()
        if text:
            context_blocks.append(f"{header}\n{text}")

    context = "\n\n".join(context_blocks).strip()

    instructions = {
        "study_guide": (
            "You are OpenSift, an AI study buddy. Using ONLY the provided context, "
            "explain clearly, structure into sections, and include a short summary + key terms."
        ),
        "quiz": (
            "You are OpenSift. Using ONLY the provided context, create a quiz (mix of MCQ and short answer) "
            "with an answer key."
        ),
        "explain": (
            "You are OpenSift. Using ONLY the provided context, explain the concept simply, then more deeply, "
            "and include common misconceptions."
        ),
    }.get(mode, "You are OpenSift. Use ONLY the provided context to answer as helpfully as possible.")

    prompt = f"""{instructions}

CONTEXT:
{context if context else "(no context provided)"}

QUESTION:
{query}

RULES:
- If the context does not contain enough information, say so and suggest what to ingest next.
- When helpful, cite sources by [number] (e.g., [1], [2]).
"""
    return prompt


def generate_with_openai(prompt: str, model: Optional[str] = None) -> str:
    """
    Non-streaming generation via OpenAI SDK.
    Default model: GPT-5.2
    """
    model = (model or DEFAULT_OPENAI_MODEL).strip()

    try:
        from openai import OpenAI  # type: ignore
    except Exception as e:
        raise RuntimeError("openai package not installed. Run: pip install openai") from e

    client = OpenAI()
    resp = client.responses.create(
        model=model,
        input=prompt,
    )

    # Best-effort extract output text
    if hasattr(resp, "output_text"):
        return (resp.output_text or "").strip()  # type: ignore[attr-defined]

    parts: List[str] = []
    for item in getattr(resp, "output", []) or []:
        for c in getattr(item, "content", []) or []:
            if getattr(c, "type", "") == "output_text":
                parts.append(getattr(c, "text", ""))
    return "".join(parts).strip()


def generate_with_claude(prompt: str, model: Optional[str] = None) -> str:
    """
    Non-streaming generation via Anthropic SDK.
    Default model: Claude Sonnet 4.5 (alias, always latest)
    """
    model = (model or DEFAULT_CLAUDE_MODEL).strip()

    try:
        import anthropic  # type: ignore
    except Exception as e:
        raise RuntimeError("anthropic package not installed. Run: pip install anthropic") from e

    client = anthropic.Anthropic()
    msg = client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )

    out: List[str] = []
    for block in getattr(msg, "content", []) or []:
        if getattr(block, "type", "") == "text":
            out.append(getattr(block, "text", ""))
    return "".join(out).strip()


def _which_cmd(cmd: str) -> Optional[str]:
    return shutil.which(cmd)


def _run_subprocess(args: List[str], stdin_text: str, timeout_s: int = 180) -> str:
    proc = subprocess.run(
        args,
        input=stdin_text,
        text=True,
        capture_output=True,
        timeout=timeout_s,
        env=os.environ.copy(),
    )
    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        stdout = (proc.stdout or "").strip()
        detail = stderr or stdout or f"exit_code={proc.returncode}"
        raise RuntimeError(f"Claude Code command failed: {detail}")
    return (proc.stdout or "").strip()


def generate_with_claude_code(prompt: str, model: Optional[str] = None) -> str:
    """
    Best-effort Claude Code integration.

    Assumptions:
      - You have a Claude Code CLI installed (default command: `claude`)
      - You've authenticated it already (e.g., setup-token / long-lived token)
      - The CLI can accept prompt via stdin or an argument

    Configuration:
      - OPENSIFT_CLAUDE_CODE_CMD: override CLI command name/path (default: claude)
      - OPENSIFT_CLAUDE_CODE_ARGS: extra args (space-separated) to append (optional)

    Model:
      - We try to pass `--model <model>` first (common pattern)
      - If that fails, we retry without model flags
    """
    cmd = os.environ.get("OPENSIFT_CLAUDE_CODE_CMD", "claude").strip()
    extra_args_raw = os.environ.get("OPENSIFT_CLAUDE_CODE_ARGS", "").strip()
    extra_args = [a for a in extra_args_raw.split(" ") if a] if extra_args_raw else []

    exe = _which_cmd(cmd)
    if not exe:
        raise RuntimeError(
            f"Claude Code CLI not found: '{cmd}'. "
            "Install Claude Code or set OPENSIFT_CLAUDE_CODE_CMD to the correct executable."
        )

    chosen_model = (model or DEFAULT_CLAUDE_MODEL).strip()

    # Try common invocation patterns:
    # 1) `claude --model <model>` reading prompt from stdin
    # 2) `claude` reading prompt from stdin (no model flag)
    #
    # If your CLI uses a different flag (like `-m`), set OPENSIFT_CLAUDE_CODE_ARGS accordingly.
    try:
        args = [exe, "--model", chosen_model, *extra_args]
        return _run_subprocess(args, prompt)
    except Exception:
        # Retry without the model flag to avoid hard coupling to a specific CLI interface
        args = [exe, *extra_args]
        return _run_subprocess(args, prompt)