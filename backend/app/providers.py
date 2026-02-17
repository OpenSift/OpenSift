from __future__ import annotations

from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Default models (upgraded)
# ---------------------------------------------------------------------------
DEFAULT_OPENAI_MODEL = "gpt-5.2"
DEFAULT_CLAUDE_MODEL = "claude-sonnet-4-5"  # alias -> latest Sonnet 4.5
DEFAULT_CLAUDE_MODEL_PINNED = "claude-sonnet-4-5"  # pinned version


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
    Default model upgraded to GPT-5.2.
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
    try:
        return resp.output_text  # type: ignore[attr-defined]
    except Exception:
        # Fallback: traverse output items if present
        parts: List[str] = []
        for item in getattr(resp, "output", []) or []:
            for c in getattr(item, "content", []) or []:
                if getattr(c, "type", "") == "output_text":
                    parts.append(getattr(c, "text", ""))
        return "".join(parts).strip()


def generate_with_claude(prompt: str, model: Optional[str] = None) -> str:
    """
    Non-streaming generation via Anthropic SDK.
    Default model upgraded to Claude Sonnet 4.5 alias.
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

    # Anthropic returns a list of content blocks
    out = []
    for block in getattr(msg, "content", []) or []:
        if getattr(block, "type", "") == "text":
            out.append(getattr(block, "text", ""))
    return "".join(out).strip()


def generate_with_claude_code(prompt: str, model: Optional[str] = None) -> str:
    """
    Claude Code / local integration (environment-specific).
    We keep it as a best-effort fallback; model defaults to Sonnet 4.5 alias.
    """
    # If your Claude Code wrapper uses env vars, this value may be ignored.
    _ = (model or DEFAULT_CLAUDE_MODEL).strip()

    # Placeholder: keep your existing implementation hook here.
    # If you already have working Claude Code integration, preserve it.
    # For now, just raise a helpful error if not implemented.
    raise RuntimeError(
        "Claude Code provider is not configured in app/providers.py. "
        "If you already have a working Claude Code implementation elsewhere, keep using it."
    )