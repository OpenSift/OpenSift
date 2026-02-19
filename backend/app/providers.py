from __future__ import annotations

import asyncio
import json
import os
import shlex
import shutil
import subprocess
from typing import Any, AsyncGenerator, Dict, List, Optional


# ---------------------------------------------------------------------------
# Default models (upgraded)
# ---------------------------------------------------------------------------
DEFAULT_OPENAI_MODEL = "gpt-5.2"
DEFAULT_CODEX_MODEL = "gpt-5.3-codex"
DEFAULT_CLAUDE_MODEL = "claude-sonnet-4-6"
SUPPORTED_CLAUDE_MODELS = (
    "claude-sonnet-4-6",
    "claude-sonnet-4-5",
)
SUPPORTED_OPENAI_MODELS = (
    "gpt-5.3-codex",
    "gpt-5.2",
)
SUPPORTED_THINKING_LEVELS = ("low", "medium", "high", "extra_high")
THINKING_BUDGETS = {
    "low": 1024,
    "medium": 2048,
    "high": 4096,
    "extra_high": 8192,
}


def normalize_thinking_level(level: str) -> str:
    s = (level or "medium").strip().lower().replace("-", "_")
    if s not in SUPPORTED_THINKING_LEVELS:
        return "medium"
    return s


def claude_thinking_budget(level: str) -> int:
    return int(THINKING_BUDGETS.get(normalize_thinking_level(level), THINKING_BUDGETS["medium"]))


def build_prompt(mode: str, query: str, passages: List[Dict[str, Any]], study_style: str = "") -> str:
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

    style_block = (study_style or "").strip()
    if style_block:
        style_text = f"STUDY STYLE PREFERENCES:\n{style_block}\n"
    else:
        style_text = ""

    prompt = f"""{instructions}

{style_text}

CONTEXT:
{context if context else "(no context provided)"}

QUESTION:
{query}

RULES:
- If the context does not contain enough information, say so and suggest what to ingest next.
- When helpful, cite sources by [number] (e.g., [1], [2]).
- Follow STUDY STYLE PREFERENCES when present, but never invent facts beyond CONTEXT.
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


def generate_with_claude(
    prompt: str,
    model: Optional[str] = None,
    thinking_enabled: bool = False,
    thinking_level: str = "medium",
) -> str:
    """
    Non-streaming generation via Anthropic SDK.
    Default model: Claude Sonnet 4.6
    """
    model = (model or DEFAULT_CLAUDE_MODEL).strip()

    try:
        import anthropic  # type: ignore
    except Exception as e:
        raise RuntimeError("anthropic package not installed. Run: pip install anthropic") from e

    client = anthropic.Anthropic()
    params: Dict[str, Any] = {
        "model": model,
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": prompt}],
    }
    if thinking_enabled:
        params["thinking"] = {"type": "enabled", "budget_tokens": claude_thinking_budget(thinking_level)}

    try:
        msg = client.messages.create(**params)
    except Exception:
        # Some model/tooling combinations do not support thinking flags yet.
        if thinking_enabled:
            params.pop("thinking", None)
            msg = client.messages.create(**params)
        else:
            raise

    out: List[str] = []
    for block in getattr(msg, "content", []) or []:
        if getattr(block, "type", "") == "text":
            out.append(getattr(block, "text", ""))
    return "".join(out).strip()


def _which_cmd(cmd: str) -> Optional[str]:
    return shutil.which(cmd)


def claude_code_cli_available() -> bool:
    cmd = os.environ.get("OPENSIFT_CLAUDE_CODE_CMD", "claude").strip() or "claude"
    return bool(_which_cmd(cmd))


def codex_cli_available() -> bool:
    cmd = os.environ.get("OPENSIFT_CODEX_CMD", "codex").strip() or "codex"
    exe = _which_cmd(cmd)
    if not exe:
        return False
    if _looks_like_wrong_codex_cli(cmd):
        return False
    return True


def _looks_like_wrong_codex_cli(cmd: str) -> bool:
    """
    Detect the unrelated legacy/npm `codex` package ("Render your codex.")
    which does not support ChatGPT Codex login/generation flows.
    """
    exe = _which_cmd(cmd)
    if not exe:
        return False
    try:
        proc = subprocess.run(
            [exe, "--help"],
            text=True,
            capture_output=True,
            timeout=4,
            env=os.environ.copy(),
        )
        out = ((proc.stdout or "") + "\n" + (proc.stderr or "")).lower()
        return ("render your codex" in out) or ("codex build" in out and "template" in out)
    except Exception:
        return False


def _parse_cli_args(raw: str) -> List[str]:
    try:
        return shlex.split(raw or "")
    except Exception:
        return [a for a in (raw or "").split(" ") if a]


def _codex_exec_supports_skip_git_repo_check(exe: str) -> bool:
    try:
        proc = subprocess.run(
            [exe, "exec", "--help"],
            text=True,
            capture_output=True,
            timeout=4,
            env=os.environ.copy(),
        )
        out = ((proc.stdout or "") + "\n" + (proc.stderr or "")).lower()
        return "--skip-git-repo-check" in out
    except Exception:
        return False


def _run_subprocess(args: List[str], stdin_text: str, timeout_s: int = 180, env: Optional[Dict[str, str]] = None) -> str:
    proc = subprocess.run(
        args,
        input=stdin_text,
        text=True,
        capture_output=True,
        timeout=timeout_s,
        env=(env or os.environ.copy()),
    )
    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        stdout = (proc.stdout or "").strip()
        detail = stderr or stdout or f"exit_code={proc.returncode}"
        raise RuntimeError(f"CLI command failed: {detail}")
    return (proc.stdout or "").strip()


def _coerce_str(v: Any) -> str:
    return v.strip() if isinstance(v, str) else ""


def _find_token_in_obj(obj: Any) -> str:
    """
    Conservative token extraction from unknown auth.json shape.
    Prefers explicit token-like keys first.
    """
    preferred_keys = (
        "chatgpt_codex_oauth_token",
        "codex_oauth_token",
        "oauth_token",
        "access_token",
        "id_token",
        "token",
    )

    if isinstance(obj, dict):
        lower_map = {str(k).lower(): v for k, v in obj.items()}
        for k in preferred_keys:
            v = lower_map.get(k)
            s = _coerce_str(v)
            if s:
                return s
        for v in obj.values():
            s = _find_token_in_obj(v)
            if s:
                return s
        return ""

    if isinstance(obj, list):
        for v in obj:
            s = _find_token_in_obj(v)
            if s:
                return s
        return ""

    return ""


def _load_codex_oauth_token() -> str:
    # 1) explicit env always wins
    env_token = _coerce_str(os.environ.get("CHATGPT_CODEX_OAUTH_TOKEN", ""))
    if env_token:
        return env_token

    # 2) auto-discover from Codex auth file(s).
    # Docker-first default for OpenSift containerized usage, then user home fallback.
    override_path = _coerce_str(os.environ.get("OPENSIFT_CODEX_AUTH_PATH", ""))
    paths: List[str] = []
    if override_path:
        paths.append(os.path.expanduser(override_path))
    paths.extend(
        [
            "/app/.codex/auth.json",
            os.path.join(os.path.expanduser("~"), ".codex", "auth.json"),
        ]
    )

    seen: set[str] = set()
    for auth_path in paths:
        p = os.path.abspath(auth_path)
        if p in seen:
            continue
        seen.add(p)
        if not os.path.exists(p):
            continue
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            token = _find_token_in_obj(data)
            if token:
                return token
        except Exception:
            continue
    return ""


def _codex_subprocess_env() -> Dict[str, str]:
    env = os.environ.copy()
    token = _load_codex_oauth_token()
    if token:
        env["CHATGPT_CODEX_OAUTH_TOKEN"] = token
    return env


def codex_auth_detected() -> bool:
    return bool(_load_codex_oauth_token())


def generate_with_claude_code(prompt: str, model: Optional[str] = None) -> str:
    """
    Best-effort Claude Code integration.

    We try a sequence of non-interactive invocations first, because invoking
    `claude` without print/prompt flags can enter interactive mode and return an
    empty stdout even with exit code 0.
    """
    cmd = os.environ.get("OPENSIFT_CLAUDE_CODE_CMD", "claude").strip()
    extra_args = _parse_cli_args(os.environ.get("OPENSIFT_CLAUDE_CODE_ARGS", "").strip())

    exe = _which_cmd(cmd)
    if not exe:
        raise RuntimeError(
            f"Claude Code CLI not found: '{cmd}'. "
            "Install Claude Code or set OPENSIFT_CLAUDE_CODE_CMD to the correct executable."
        )

    chosen_model = (model or DEFAULT_CLAUDE_MODEL).strip()

    # Prefer direct prompt flags (no TTY required), then fall back to stdin forms.
    # Keep extra args before model flags so users can override behavior explicitly.
    attempts: List[tuple[List[str], str]] = [
        ([exe, *extra_args, "--model", chosen_model, "-p", prompt], ""),
        ([exe, *extra_args, "--model", chosen_model, "--print", prompt], ""),
        ([exe, *extra_args, "-p", prompt], ""),
        ([exe, *extra_args, "--print", prompt], ""),
        ([exe, *extra_args, "--model", chosen_model], prompt),
        ([exe, *extra_args], prompt),
    ]

    failures: List[str] = []
    env = os.environ.copy()
    for args, stdin_text in attempts:
        try:
            proc = subprocess.run(
                args,
                input=stdin_text,
                text=True,
                capture_output=True,
                timeout=180,
                env=env,
            )
        except Exception as e:
            failures.append(f"{' '.join(args[:4])}... -> spawn failed: {e}")
            continue

        stdout = (proc.stdout or "").strip()
        stderr = (proc.stderr or "").strip()
        if proc.returncode != 0:
            detail = stderr or stdout or f"exit_code={proc.returncode}"
            failures.append(f"{' '.join(args[:4])}... -> {detail[:220]}")
            continue
        if not stdout:
            # Treat empty output as a hard failure so caller can fall back to a
            # working provider path instead of silently returning empty text.
            detail = stderr or "no stdout produced"
            failures.append(f"{' '.join(args[:4])}... -> empty output ({detail[:220]})")
            continue
        return stdout

    detail_blob = " | ".join(failures[-6:]) if failures else "unknown failure"
    raise RuntimeError(
        "Claude Code CLI command returned no usable output. "
        f"Tried multiple invocation modes. Details: {detail_blob}"
    )


def generate_with_codex(prompt: str, model: Optional[str] = None) -> str:
    """
    Best-effort Codex CLI integration.

    Configuration:
      - CHATGPT_CODEX_OAUTH_TOKEN: optional token configured during setup
      - OPENSIFT_CODEX_CMD: override command (default: codex)
      - OPENSIFT_CODEX_ARGS: extra args (space-separated)

    Invocation:
      - Tries: codex exec --model <model> [args] -
      - Falls back to: codex exec [args] -
    """
    cmd = os.environ.get("OPENSIFT_CODEX_CMD", "codex").strip()
    env = _codex_subprocess_env()
    for args in build_codex_cli_invocations(model=model):
        try:
            return _run_subprocess(args, prompt, env=env)
        except Exception:
            continue
    raise RuntimeError(
        f"Codex CLI command failed for '{cmd}'. "
        "Check Codex auth (/app/.codex/auth.json or ~/.codex/auth.json) "
        "or CHATGPT_CODEX_OAUTH_TOKEN, plus OPENSIFT_CODEX_CMD/OPENSIFT_CODEX_ARGS."
    )


def build_codex_cli_invocations(model: Optional[str] = None) -> List[List[str]]:
    cmd = os.environ.get("OPENSIFT_CODEX_CMD", "codex").strip()
    extra_args = _parse_cli_args(os.environ.get("OPENSIFT_CODEX_ARGS", "").strip())

    exe = _which_cmd(cmd)
    if not exe:
        raise RuntimeError(
            f"Codex CLI not found: '{cmd}'. "
            "Install Codex or set OPENSIFT_CODEX_CMD to the correct executable."
        )
    if _looks_like_wrong_codex_cli(cmd):
        raise RuntimeError(
            f"'{cmd}' appears to be the unrelated npm 'codex' tool (site generator), not ChatGPT Codex CLI. "
            "Install the correct ChatGPT Codex CLI and set OPENSIFT_CODEX_CMD to that executable."
        )

    chosen_model = (model or DEFAULT_CODEX_MODEL).strip()
    skip_git_repo_check = (os.environ.get("OPENSIFT_CODEX_SKIP_GIT_REPO_CHECK", "true").strip().lower() in ("1", "true", "yes", "on"))
    exec_supports_skip = _codex_exec_supports_skip_git_repo_check(exe)
    # Use non-interactive `exec` to avoid TTY-only interactive mode.
    # "-" means prompt comes from stdin.
    invocations: List[List[str]] = []

    if skip_git_repo_check and exec_supports_skip:
        invocations.extend(
            [
                [exe, "exec", "--skip-git-repo-check", "--model", chosen_model, *extra_args, "-"],
                [exe, "exec", "--skip-git-repo-check", *extra_args, "-"],
            ]
        )

    invocations.extend(
        [
            [exe, "exec", "--model", chosen_model, *extra_args, "-"],
            [exe, "exec", *extra_args, "-"],
        ]
    )
    return invocations


async def stream_with_codex(prompt: str, model: Optional[str] = None) -> AsyncGenerator[str, None]:
    """
    Stream text from Codex CLI stdout.
    Falls back through invocation variants for compatibility.
    """
    last_err: Optional[Exception] = None
    env = _codex_subprocess_env()
    for args in build_codex_cli_invocations(model=model):
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=env,
        )
        if proc.stdin is None or proc.stdout is None:
            last_err = RuntimeError("Codex stream could not open process pipes.")
            continue

        proc.stdin.write(prompt.encode("utf-8"))
        await proc.stdin.drain()
        proc.stdin.close()

        seen: List[str] = []
        while True:
            chunk = await proc.stdout.read(256)
            if not chunk:
                break
            text = chunk.decode("utf-8", errors="ignore")
            if text:
                seen.append(text)
                yield text

        rc = await proc.wait()
        if rc == 0:
            return

        detail = ("".join(seen).strip() or f"exit_code={rc}")[-1200:]
        last_err = RuntimeError(f"Codex stream command failed: {detail}")

    if last_err:
        raise last_err
    raise RuntimeError("Codex stream command failed.")
