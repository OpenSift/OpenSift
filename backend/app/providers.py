from __future__ import annotations

import os
import subprocess
from typing import List, Dict, Any, Literal, Optional

from openai import OpenAI
from anthropic import Anthropic

def generate_with_claude_code(prompt: str, model: str | None = None) -> str:
    """
    Uses Claude Code (subscription / setup-token) instead of Anthropic API.
    Requires:
      - `claude` CLI installed
      - CLAUDE_CODE_OAUTH_TOKEN set (from `claude setup-token`)
    Note: Ensure ANTHROPIC_API_KEY is NOT set if you want subscription billing,
    since Claude Code prioritizes env API keys over subscription auth.
    """
    token = os.environ.get("CLAUDE_CODE_OAUTH_TOKEN")
    if not token:
        raise RuntimeError("CLAUDE_CODE_OAUTH_TOKEN not set")

    env = os.environ.copy()
    env["CLAUDE_CODE_OAUTH_TOKEN"] = token

    # Avoid accidentally forcing API-key billing through env precedence
    env.pop("ANTHROPIC_API_KEY", None)

    # Claude Code CLI flags can vary by version; we keep this minimal:
    # - feed prompt via stdin
    # - read stdout as the answer
    cmd = ["claude"]
    if model:
        cmd += ["--model", model]

    proc = subprocess.run(
        cmd,
        input=prompt.encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        check=False,
    )

    if proc.returncode != 0:
        raise RuntimeError(
            f"Claude Code CLI failed (exit {proc.returncode}): {proc.stderr.decode('utf-8', errors='ignore')}"
        )

    return proc.stdout.decode("utf-8", errors="ignore").strip()

OpenSiftMode = Literal["key_points", "study_guide", "quiz"]
Provider = Literal["openai", "claude"]

def build_prompt(mode: OpenSiftMode, query: str, passages: List[Dict[str, Any]]) -> str:
    context = "\n\n".join(f"[Source {i+1}] {p['text']}" for i, p in enumerate(passages))
    if mode == "key_points":
        instruction = "Return 8–15 crisp, test-relevant bullets."
    elif mode == "quiz":
        instruction = "Return: 10 multiple-choice (A–D) + answer key + 5 short-answer."
    else:
        instruction = "Make a study guide: key concepts, definitions, formulas (if any), pitfalls, 5 recall prompts."
    return f"""You are OpenSift, an AI study assistant.
Student request: {query}
Instructions: {instruction}
Use ONLY the provided sources. If missing info, say what's missing.

SOURCES:
{context}
""".strip()

def generate_with_openai(prompt: str, model: str = "gpt-4.1-mini") -> str:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set")
    client = OpenAI(api_key=key)
    resp = client.responses.create(
        model=model,
        input=[{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
        temperature=0.3,
    )
    return resp.output_text

def generate_with_claude(prompt: str, model: str = "claude-3-5-sonnet-latest") -> str:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    client = Anthropic(api_key=key)
    msg = client.messages.create(
        model=model,
        max_tokens=1200,
        temperature=0.3,
        messages=[{"role": "user", "content": prompt}],
    )
    # Anthropic returns content blocks
    return "".join([blk.text for blk in msg.content if getattr(blk, "text", None)])