from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import ipaddress
import json
import os
import re
import secrets
import shutil
import subprocess
import threading
import time
import tempfile
from contextlib import asynccontextmanager
from urllib.parse import urlparse
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Deque, DefaultDict, Dict, List, Optional, Tuple
from uuid import uuid4

import anyio
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.status import HTTP_303_SEE_OTHER

from app.chunking import chunk_text
from app.ingest import extract_text_from_pdf, extract_text_from_txt, fetch_url_text
from app.llm import embed_texts, local_embedding_model_loaded, using_local_embeddings, warmup_local_embeddings
from app.atomic_io import atomic_write_json, path_lock
from app.logging_utils import configure_logging
from app.providers import (
    claude_code_cli_available,
    claude_thinking_budget,
    codex_auth_detected,
    codex_cli_available,
    DEFAULT_CLAUDE_MODEL,
    DEFAULT_CODEX_MODEL,
    DEFAULT_OPENAI_MODEL,
    SUPPORTED_CLAUDE_MODELS,
    SUPPORTED_OPENAI_MODELS,
    SUPPORTED_THINKING_LEVELS,
    normalize_thinking_level,
    build_prompt,
    generate_with_claude,
    generate_with_claude_code,
    generate_with_codex,
    generate_with_openai,
    stream_with_codex,
)
from app.soul import get_global_style, set_global_style
from app.vectordb import VectorDB
from app.wellness import build_break_reminder, get_wellness_settings, set_wellness_settings, should_add_break_reminder

from session_store import DEFAULT_DIR as SESSION_DIR, delete_session, list_sessions, load_session, save_session
from study_store import (
    DEFAULT_DIR as STUDY_DIR,
    add_item as add_study_item,
    delete_item as delete_study_item,
    get_item as get_study_item,
    load_library as load_study_library,
)
from quiz_store import DEFAULT_DIR as QUIZ_DIR, add_attempt, delete_attempt, load_attempts
from flashcard_store import DEFAULT_DIR as FLASHCARD_DIR, add_card, delete_card, due_cards, load_cards, review_card
from source_store import (
    DEFAULT_DIR as SOURCE_DIR,
    add_item as add_source_item,
    delete_item as delete_source_item,
    get_item as get_source_item,
    list_owners as list_source_owners,
    load_items as load_source_items,
    new_source_id,
    read_text_blob as read_source_text_blob,
    remove_file as remove_source_file,
    update_item as update_source_item,
    write_binary_blob as write_source_binary_blob,
    write_text_blob as write_source_text_blob,
)

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------
os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)

AUTH_STATE_PATH = os.path.join(os.getcwd(), ".opensift_auth.json")
ENV_FILE_PATH = os.path.join(os.getcwd(), ".env")
OPENSIFT_VERSION = "1.5.0-alpha"
CLI_TOOLS_PREFIX = os.path.join(os.getcwd(), ".opensift_tools")
CLI_TOOLS_BIN_DIR = os.path.join(CLI_TOOLS_PREFIX, "bin")
CLI_INSTALL_TIMEOUT_SECONDS = 420

logger = configure_logging("opensift.ui")
_EMBED_WARMUP_STARTED = False
_EMBED_WARMUP_LOCK = threading.Lock()


def _start_embedding_warmup_if_enabled() -> None:
    global _EMBED_WARMUP_STARTED
    enabled = os.getenv("OPENSIFT_PRELOAD_EMBEDDINGS", "true").strip().lower() in ("1", "true", "yes", "on")
    if not enabled:
        return
    if not using_local_embeddings():
        return
    if local_embedding_model_loaded():
        return
    with _EMBED_WARMUP_LOCK:
        if _EMBED_WARMUP_STARTED:
            return
        _EMBED_WARMUP_STARTED = True

    def _run() -> None:
        try:
            logger.info("embedding_warmup_start provider=local")
            warmup_local_embeddings()
            logger.info("embedding_warmup_done provider=local")
        except Exception:
            logger.exception("embedding_warmup_failed provider=local")

    threading.Thread(target=_run, name="opensift-embed-warmup", daemon=True).start()


def _extract_text_from_image_bytes(data: bytes) -> str:
    try:
        from PIL import Image  # type: ignore
        import pytesseract  # type: ignore
    except Exception:
        return ""
    try:
        import io

        img = Image.open(io.BytesIO(data))
        text = pytesseract.image_to_string(img, lang=os.getenv("OPENSIFT_OCR_LANG", "eng")) or ""
        return text.strip()
    except Exception:
        return ""


@asynccontextmanager
async def _app_lifespan(_app: FastAPI):
    await _print_startup_token()
    _start_embedding_warmup_if_enabled()
    yield


app = FastAPI(title="OpenSift UI", version=OPENSIFT_VERSION, lifespan=_app_lifespan)

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

db = VectorDB()

# History persisted per owner (loaded on demand)
CHAT_HISTORY: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)
CHAT_HISTORY_LOCK = threading.RLock()

# Default history settings for UI
DEFAULT_HISTORY_TURNS = 10


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _normalize_owner(owner: str) -> str:
    owner = (owner or "").strip() or "default"
    owner = re.sub(r"[^a-zA-Z0-9._-]+", "_", owner)
    return owner[:128]


def _ndjson(obj: Dict[str, Any]) -> bytes:
    return (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")


def _context_only_answer(query: str, results: List[Dict[str, Any]]) -> str:
    """
    Deterministic fallback when external generation providers are unavailable.
    Builds a concise answer from retrieved passages only.
    """
    q = (query or "").strip()
    texts: List[str] = []
    for r in results[:8]:
        t = (r.get("text") or "").strip()
        if t:
            texts.append(t.replace("\n", " "))
    if not texts:
        return (
            "I couldn't generate with a model provider, and there was no retrieved context to answer from. "
            "Try ingesting a document or URL first."
        )

    tokens = [w.lower() for w in re.findall(r"[a-zA-Z]{4,}", q)]
    matches: List[str] = []
    for t in texts:
        for sent in re.split(r"(?<=[.!?])\s+", t):
            s = sent.strip()
            if len(s) < 40:
                continue
            sl = s.lower()
            if tokens and not any(tok in sl for tok in tokens):
                continue
            matches.append(s)
            if len(matches) >= 3:
                break
        if len(matches) >= 3:
            break

    if not matches:
        for sent in re.split(r"(?<=[.!?])\s+", texts[0]):
            s = sent.strip()
            if len(s) >= 40:
                matches.append(s)
            if len(matches) >= 2:
                break

    body = "\n".join([f"- {m}" for m in matches[:3]]) if matches else "- No extractable passage found."
    return (
        "Provider generation is currently unavailable, so here is a context-only answer:\n\n"
        f"{body}\n\n"
        "Tip: add a working provider credential in Settings for full generated responses."
    )


def _ensure_owner_loaded(owner: str) -> None:
    with CHAT_HISTORY_LOCK:
        if owner not in CHAT_HISTORY or not CHAT_HISTORY[owner]:
            CHAT_HISTORY[owner] = load_session(owner, SESSION_DIR)


def _history_snapshot(owner: str) -> List[Dict[str, Any]]:
    _ensure_owner_loaded(owner)
    with CHAT_HISTORY_LOCK:
        return [dict(m) for m in CHAT_HISTORY[owner]]


def _history_append(owner: str, msg: Dict[str, Any]) -> None:
    with CHAT_HISTORY_LOCK:
        _ensure_owner_loaded(owner)
        CHAT_HISTORY[owner].append(msg)
        save_session(owner, CHAT_HISTORY[owner], SESSION_DIR)


def _history_extend(owner: str, messages: List[Dict[str, Any]]) -> None:
    with CHAT_HISTORY_LOCK:
        _ensure_owner_loaded(owner)
        CHAT_HISTORY[owner].extend(messages)
        save_session(owner, CHAT_HISTORY[owner], SESSION_DIR)


def _history_replace(owner: str, messages: List[Dict[str, Any]]) -> None:
    with CHAT_HISTORY_LOCK:
        CHAT_HISTORY[owner] = list(messages)
        save_session(owner, CHAT_HISTORY[owner], SESSION_DIR)


def _history_clear(owner: str) -> None:
    with CHAT_HISTORY_LOCK:
        _ensure_owner_loaded(owner)
        CHAT_HISTORY[owner].clear()
        save_session(owner, CHAT_HISTORY[owner], SESSION_DIR)


def _history_delete_owner(owner: str) -> None:
    with CHAT_HISTORY_LOCK:
        CHAT_HISTORY.pop(owner, None)


def _last_generated_assistant(owner: str) -> Optional[Dict[str, Any]]:
    history = _history_snapshot(owner)
    for msg in reversed(history):
        if msg.get("role") != "assistant":
            continue
        if not isinstance(msg.get("text"), str) or not msg.get("text"):
            continue
        if isinstance(msg.get("mode"), str) and msg.get("mode"):
            return msg
    return None


def _build_history_block(history: List[Dict[str, Any]], turns: int) -> str:
    h = history[-max(0, turns):]
    lines = []
    for m in h:
        role = m.get("role")
        text = (m.get("text") or "").strip()
        if not text:
            continue
        if role == "user":
            lines.append(f"User: {text}")
        elif role == "assistant":
            lines.append(f"Assistant: {text}")
    return "\n".join(lines).strip()


# -----------------------------------------------------------------------------
# Localhost-only middleware
# -----------------------------------------------------------------------------
ALLOWED_HOSTS = {"127.0.0.1", "localhost", "0.0.0.0", "::1", "[::1]", "[::]"}


def _is_private_or_loopback_ip(host: str) -> bool:
    try:
        ip = ipaddress.ip_address((host or "").strip())
    except ValueError:
        return False
    return bool(ip.is_loopback or ip.is_private)


def _parse_trusted_client_networks() -> List[ipaddress._BaseNetwork]:
    raw_items: List[str] = []
    raw_ips = os.getenv("OPENSIFT_TRUSTED_CLIENT_IPS", "").strip()
    raw_cidrs = os.getenv("OPENSIFT_TRUSTED_CLIENT_CIDRS", "").strip()
    if raw_ips:
        raw_items.extend([x.strip() for x in raw_ips.split(",") if x.strip()])
    if raw_cidrs:
        raw_items.extend([x.strip() for x in raw_cidrs.split(",") if x.strip()])

    nets: List[ipaddress._BaseNetwork] = []
    for item in raw_items:
        try:
            if "/" in item:
                nets.append(ipaddress.ip_network(item, strict=False))
            else:
                ip = ipaddress.ip_address(item)
                nets.append(ipaddress.ip_network(f"{ip}/{ip.max_prefixlen}", strict=False))
        except ValueError:
            continue
    return nets


def _client_ip_is_trusted(host: str) -> bool:
    try:
        ip = ipaddress.ip_address((host or "").strip())
    except ValueError:
        return False
    for net in _parse_trusted_client_networks():
        if ip.version != net.version:
            continue
        if ip in net:
            return True
    return False


class LocalhostOnlyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        client_host = request.client.host if request.client else ""
        host_header = request.headers.get("host", "")
        host_only = host_header.split(":")[0].strip().lower()

        is_loopback_ip = client_host in ("127.0.0.1", "::1")
        allow_private_clients = (
            os.path.exists("/.dockerenv")
            or os.getenv("OPENSIFT_ALLOW_PRIVATE_CLIENTS", "").strip().lower() in ("1", "true", "yes", "on")
        )
        is_private_client = _is_private_or_loopback_ip(client_host)
        is_trusted_client = _client_ip_is_trusted(client_host)
        is_allowed_host = host_only in ALLOWED_HOSTS

        if not (is_allowed_host and (is_loopback_ip or (allow_private_clients and is_private_client) or is_trusted_client)):
            logger.warning(
                "localhost_blocked client_host=%s host=%s allow_private_clients=%s trusted_client=%s",
                client_host,
                host_only,
                allow_private_clients,
                is_trusted_client,
            )
            message = (
                "OpenSift is configured for localhost-only access.\n"
                "If you are using Safari, Private Relay / 'Hide IP address' may proxy requests and break local checks.\n"
                "Use http://127.0.0.1:8001 directly, and disable Private Relay/Hide IP for local development.\n"
                "If relay/proxy access is required, allowlist trusted egress IPs via OPENSIFT_TRUSTED_CLIENT_IPS or OPENSIFT_TRUSTED_CLIENT_CIDRS."
            )
            return PlainTextResponse(
                message,
                status_code=403,
            )

        return await call_next(request)


app.add_middleware(LocalhostOnlyMiddleware)


class AccessLogMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("x-request-id") or uuid4().hex[:12]
        start = time.perf_counter()
        method = request.method.upper()
        path = request.url.path
        client_ip = request.client.host if request.client else "unknown"

        try:
            response = await call_next(request)
            status = response.status_code
        except Exception:
            dur_ms = (time.perf_counter() - start) * 1000.0
            logger.exception(
                "request_failed id=%s method=%s path=%s client=%s duration_ms=%.2f",
                request_id,
                method,
                path,
                client_ip,
                dur_ms,
            )
            raise

        dur_ms = (time.perf_counter() - start) * 1000.0
        response.headers["X-Request-Id"] = request_id
        logger.info(
            "request id=%s method=%s path=%s status=%d client=%s duration_ms=%.2f",
            request_id,
            method,
            path,
            status,
            client_ip,
            dur_ms,
        )
        return response


app.add_middleware(AccessLogMiddleware)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "same-origin")
        response.headers.setdefault("Permissions-Policy", "geolocation=(), microphone=(), camera=()")
        return response


app.add_middleware(SecurityHeadersMiddleware)

# -----------------------------------------------------------------------------
# Rate limiting (in-memory sliding window)
# -----------------------------------------------------------------------------
RL_WINDOW_SECONDS = 60
RL_CHAT_STREAM_PER_WINDOW = 30
RL_CHAT_INGEST_PER_WINDOW = 12
RL_CHAT_OTHER_PER_WINDOW = 30
RL_LOGIN_PER_WINDOW = 12

_RL: Dict[Tuple[str, str], Deque[float]] = {}
_RL_MAX_KEYS = 2048

MAX_UPLOAD_MB = max(1, int(os.getenv("OPENSIFT_MAX_UPLOAD_MB", "25")))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
MAX_CHAT_MESSAGE_CHARS = max(256, int(os.getenv("OPENSIFT_MAX_CHAT_MESSAGE_CHARS", "8000")))
MAX_SESSION_IMPORT_CHARS = max(1024, int(os.getenv("OPENSIFT_MAX_SESSION_IMPORT_CHARS", "2000000")))
MAX_HISTORY_TURNS = max(1, int(os.getenv("OPENSIFT_MAX_HISTORY_TURNS", "30")))
MAX_RETRIEVAL_K = max(1, int(os.getenv("OPENSIFT_MAX_RETRIEVAL_K", "20")))
try:
    RETRIEVAL_TIMEOUT_SECONDS = max(5.0, float(os.getenv("OPENSIFT_RETRIEVAL_TIMEOUT_SECONDS", "300")))
except Exception:
    RETRIEVAL_TIMEOUT_SECONDS = 300.0


def _rl_bucket(path: str, method: str) -> Optional[Tuple[str, int]]:
    if method != "POST":
        return None
    if path == "/chat/stream":
        return ("chat_stream", RL_CHAT_STREAM_PER_WINDOW)
    if path.startswith("/chat/ingest/"):
        return ("chat_ingest", RL_CHAT_INGEST_PER_WINDOW)
    if path.startswith("/chat/"):
        return ("chat_other", RL_CHAT_OTHER_PER_WINDOW)
    if path in ("/login", "/set-password"):
        return ("auth", RL_LOGIN_PER_WINDOW)
    return None


def _rl_prune_global() -> None:
    if len(_RL) <= _RL_MAX_KEYS:
        return
    keys = list(_RL.keys())[: len(_RL) - _RL_MAX_KEYS]
    for k in keys:
        _RL.pop(k, None)


def _rl_check(client: str, bucket: str, limit: int, now: float) -> Tuple[bool, int, int]:
    key = (client, bucket)
    q = _RL.get(key)
    if q is None:
        q = deque()
        _RL[key] = q

    cutoff = now - RL_WINDOW_SECONDS
    while q and q[0] < cutoff:
        q.popleft()

    if len(q) >= limit:
        retry_after = max(1, int((q[0] + RL_WINDOW_SECONDS) - now))
        return (False, 0, retry_after)

    q.append(now)
    remaining = max(0, limit - len(q))
    return (True, remaining, 0)


def _is_api_path(path: str) -> bool:
    return path.startswith("/chat/")


async def _read_upload_limited(file: UploadFile, max_bytes: int) -> bytes:
    chunks: List[bytes] = []
    total = 0
    while True:
        part = await file.read(1024 * 1024)
        if not part:
            break
        total += len(part)
        if total > max_bytes:
            raise ValueError(f"File exceeds upload limit ({max_bytes} bytes).")
        chunks.append(part)
    return b"".join(chunks)


def _sanitize_post_params(mode: str, provider: str, k: int, history_turns: int) -> Tuple[str, str, int, int]:
    mode_clean = (mode or "study_guide").strip().lower()
    provider_clean = (provider or "claude_code").strip().lower()
    if mode_clean not in ("study_guide", "key_points", "quiz", "explain"):
        raise ValueError("invalid_mode")
    if provider_clean not in ("openai", "claude", "claude_code", "codex"):
        raise ValueError("invalid_provider")
    k_clean = max(1, min(int(k), MAX_RETRIEVAL_K))
    turns_clean = max(0, min(int(history_turns), MAX_HISTORY_TURNS))
    return mode_clean, provider_clean, k_clean, turns_clean


def _preferred_provider_default() -> str:
    if claude_code_cli_available():
        return "claude_code"
    if os.getenv("ANTHROPIC_API_KEY", "").strip():
        return "claude"
    if codex_auth_detected() and codex_cli_available():
        return "codex"
    if os.getenv("OPENAI_API_KEY", "").strip():
        return "openai"
    if os.getenv("CLAUDE_CODE_OAUTH_TOKEN", "").strip():
        return "claude_code"
    return "claude_code"


def _provider_runtime_caps() -> Dict[str, Any]:
    env_data = _read_env_file(ENV_FILE_PATH)
    openai_key = (os.getenv("OPENAI_API_KEY") or env_data.get("OPENAI_API_KEY") or "").strip()
    anthropic_key = (os.getenv("ANTHROPIC_API_KEY") or env_data.get("ANTHROPIC_API_KEY") or "").strip()

    claude_cli_ok = claude_code_cli_available()
    codex_cli_ok = codex_cli_available()
    codex_auth_ok = codex_auth_detected()

    claude_code_ready = bool(claude_cli_ok or anthropic_key)
    codex_ready = bool((codex_cli_ok and codex_auth_ok) or openai_key)
    claude_ready = bool(anthropic_key)
    openai_ready = bool(openai_key)

    return {
        "claude_code_cli_available": claude_cli_ok,
        "codex_cli_available": codex_cli_ok,
        "codex_auth_detected": codex_auth_ok,
        "anthropic_api_key_set": bool(anthropic_key),
        "openai_api_key_set": bool(openai_key),
        "claude_code_ready": claude_code_ready,
        "codex_ready": codex_ready,
        "claude_ready": claude_ready,
        "openai_ready": openai_ready,
        "any_provider_ready": any((claude_code_ready, codex_ready, claude_ready, openai_ready)),
    }


def _resolve_provider_model_pair(provider: str, model: str) -> Tuple[str, str, str]:
    """
    Handle mismatches like "ChatGPT + Sonnet" or "Claude + GPT".
    Returns: (provider, model, note)
    """
    p = (provider or "").strip().lower()
    m = (model or "").strip()
    if not m:
        return p, m, ""

    is_claude_model = m.startswith("claude-")
    is_openai_model = m.startswith("gpt-")

    if p in ("openai", "codex") and is_claude_model:
        return "claude", m, f"Model '{m}' is Claude-family; switching provider to Claude API."
    if p in ("claude", "claude_code") and is_openai_model:
        target = "codex" if (codex_auth_detected() and codex_cli_available()) else "openai"
        return target, m, f"Model '{m}' is GPT-family; switching provider to {target}."
    return p, m, ""


class RateLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        method = request.method.upper()
        b = _rl_bucket(path, method)
        if not b:
            return await call_next(request)

        bucket_name, limit = b
        client_ip = request.client.host if request.client else "unknown"
        now = time.time()

        _rl_prune_global()
        allowed, remaining, retry_after = _rl_check(client_ip, bucket_name, limit, now)

        if not allowed:
            headers = {
                "Retry-After": str(retry_after),
                "X-RateLimit-Limit": str(limit),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Window": str(RL_WINDOW_SECONDS),
                "X-RateLimit-Bucket": bucket_name,
            }
            msg = f"Rate limit exceeded for {bucket_name}. Try again in {retry_after}s."
            if _is_api_path(path):
                return JSONResponse(
                    {"ok": False, "error": "rate_limited", "message": msg},
                    status_code=429,
                    headers=headers,
                )
            return PlainTextResponse(msg, status_code=429, headers=headers)

        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(limit)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Window"] = str(RL_WINDOW_SECONDS)
        response.headers["X-RateLimit-Bucket"] = bucket_name
        return response


app.add_middleware(RateLimitMiddleware)

# -----------------------------------------------------------------------------
# Auth: generated token OR user password
# -----------------------------------------------------------------------------
GEN_TOKEN = secrets.token_urlsafe(24)
SESSION_SIGNING_SECRET = secrets.token_bytes(32)

AUTH_COOKIE_NAME = "opensift_auth"
AUTH_TTL_SECONDS = 60 * 60 * 24 * 7  # 7 days


def _rotate_login_token() -> str:
    global GEN_TOKEN
    GEN_TOKEN = secrets.token_urlsafe(24)
    logger.info("generated_login_token_rotated suffix=%s", GEN_TOKEN[-6:])
    return GEN_TOKEN


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("utf-8").rstrip("=")


def _b64url_decode(s: str) -> bytes:
    pad = "=" * ((4 - (len(s) % 4)) % 4)
    return base64.urlsafe_b64decode((s + pad).encode("utf-8"))


def _sign(message: bytes) -> str:
    sig = hmac.new(SESSION_SIGNING_SECRET, message, hashlib.sha256).digest()
    return _b64url(sig)


def _make_session_cookie() -> str:
    payload = {"ts": int(time.time()), "exp": int(time.time()) + AUTH_TTL_SECONDS}
    raw = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    body = _b64url(raw)
    sig = _sign(body.encode("utf-8"))
    return f"{body}.{sig}"


def _verify_session_cookie(value: str) -> bool:
    try:
        body, sig = value.split(".", 1)
        expected = _sign(body.encode("utf-8"))
        if not hmac.compare_digest(sig, expected):
            return False
        raw = _b64url_decode(body)
        payload = json.loads(raw.decode("utf-8"))
        exp = int(payload.get("exp", 0))
        return int(time.time()) <= exp
    except Exception:
        return False


def _load_auth_state() -> Dict[str, Any]:
    with path_lock(AUTH_STATE_PATH):
        if not os.path.exists(AUTH_STATE_PATH):
            return {"has_password": False}
        try:
            with open(AUTH_STATE_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                return {"has_password": False}
            return data
        except Exception:
            return {"has_password": False}


def _save_auth_state(state: Dict[str, Any]) -> None:
    atomic_write_json(AUTH_STATE_PATH, state)


def _read_env_file(path: str) -> Dict[str, str]:
    data: Dict[str, str] = {}
    if not os.path.exists(path):
        return data
    try:
        with open(path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                key = k.strip()
                if key:
                    data[key] = v.strip()
    except Exception:
        return {}
    return data


def _write_env_file(path: str, values: Dict[str, str]) -> None:
    lines = [
        "# OpenSift local environment",
        "# Updated by OpenSift settings UI",
        "",
    ]
    for k in sorted(values.keys()):
        v = (values.get(k, "") or "").replace("\n", "\\n")
        lines.append(f"{k}={v}")
    lines.append("")
    text = "\n".join(lines)
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    with path_lock(path):
        fd, tmp_path = tempfile.mkstemp(prefix=".env.", dir=directory, text=True)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as tmpf:
                tmpf.write(text)
                tmpf.flush()
                os.fsync(tmpf.fileno())
            os.replace(tmp_path, path)
        finally:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass
        try:
            os.chmod(path, 0o600)
        except Exception:
            pass


def _secret_hint(v: str) -> str:
    s = (v or "").strip()
    if not s:
        return ""
    if len(s) <= 6:
        return "******"
    return f"{s[:2]}...{s[-4:]}"


def _provider_settings_snapshot() -> Dict[str, Any]:
    env_data = _read_env_file(ENV_FILE_PATH)
    openai_key = (os.getenv("OPENAI_API_KEY") or env_data.get("OPENAI_API_KEY") or "").strip()
    anthropic_key = (os.getenv("ANTHROPIC_API_KEY") or env_data.get("ANTHROPIC_API_KEY") or "").strip()
    claude_code_token = (os.getenv("CLAUDE_CODE_OAUTH_TOKEN") or env_data.get("CLAUDE_CODE_OAUTH_TOKEN") or "").strip()
    codex_token = (os.getenv("CHATGPT_CODEX_OAUTH_TOKEN") or env_data.get("CHATGPT_CODEX_OAUTH_TOKEN") or "").strip()
    claude_cmd = (os.getenv("OPENSIFT_CLAUDE_CODE_CMD") or env_data.get("OPENSIFT_CLAUDE_CODE_CMD") or "claude").strip() or "claude"
    codex_cmd = (os.getenv("OPENSIFT_CODEX_CMD") or env_data.get("OPENSIFT_CODEX_CMD") or "codex").strip() or "codex"
    return {
        "ok": True,
        "env_file": ENV_FILE_PATH,
        "cli_tools_prefix": CLI_TOOLS_PREFIX,
        "cli_tools_bin_dir": CLI_TOOLS_BIN_DIR,
        "openai_api_key_set": bool(openai_key),
        "openai_api_key_hint": _secret_hint(openai_key),
        "anthropic_api_key_set": bool(anthropic_key),
        "anthropic_api_key_hint": _secret_hint(anthropic_key),
        "claude_code_oauth_token_set": bool(claude_code_token),
        "claude_code_oauth_token_hint": _secret_hint(claude_code_token),
        "chatgpt_codex_oauth_token_set": bool(codex_token),
        "chatgpt_codex_oauth_token_hint": _secret_hint(codex_token),
        "claude_code_cmd": claude_cmd,
        "claude_code_cli_available": claude_code_cli_available(),
        "codex_cmd": codex_cmd,
        "codex_cli_available": codex_cli_available(),
    }


def _provider_cli_install_spec(target: str) -> Dict[str, str]:
    t = (target or "").strip().lower()
    if t in ("claude", "claude_code"):
        pkg = (os.getenv("CLAUDE_CODE_NPM_PACKAGE", "") or "").strip() or "@anthropic-ai/claude-code"
        return {
            "target": t,
            "package": pkg,
            "binary": "claude",
            "env_key": "OPENSIFT_CLAUDE_CODE_CMD",
        }
    if t == "codex":
        pkg = (os.getenv("CODEX_NPM_PACKAGE", "") or "").strip() or "@openai/codex"
        return {
            "target": t,
            "package": pkg,
            "binary": "codex",
            "env_key": "OPENSIFT_CODEX_CMD",
        }
    raise ValueError("invalid_target")


def _install_provider_cli(target: str) -> Dict[str, str]:
    spec = _provider_cli_install_spec(target)
    npm = shutil.which("npm")
    if not npm:
        raise RuntimeError("npm_not_found")

    os.makedirs(CLI_TOOLS_PREFIX, exist_ok=True)
    env = os.environ.copy()
    env["NPM_CONFIG_PREFIX"] = CLI_TOOLS_PREFIX
    env["HOME"] = env.get("HOME") or "/tmp"
    env["NPM_CONFIG_CACHE"] = env.get("NPM_CONFIG_CACHE") or "/tmp/.npm"
    os.makedirs(env["NPM_CONFIG_CACHE"], exist_ok=True)

    proc = subprocess.run(
        [npm, "install", "-g", spec["package"]],
        text=True,
        capture_output=True,
        timeout=CLI_INSTALL_TIMEOUT_SECONDS,
        env=env,
    )

    out = ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()
    if proc.returncode != 0:
        tail = "\n".join(out.splitlines()[-20:]).strip()
        raise RuntimeError(f"install_failed:{tail or f'exit_code={proc.returncode}'}")

    cmd_path = os.path.join(CLI_TOOLS_BIN_DIR, spec["binary"])
    if not os.path.exists(cmd_path):
        fallback = shutil.which(spec["binary"], path=CLI_TOOLS_BIN_DIR)
        if fallback:
            cmd_path = fallback
    if not os.path.exists(cmd_path):
        raise RuntimeError("binary_not_found_after_install")

    return {
        "target": spec["target"],
        "package": spec["package"],
        "binary": spec["binary"],
        "env_key": spec["env_key"],
        "cmd_path": cmd_path,
        "install_log_tail": "\n".join(out.splitlines()[-12:]).strip(),
    }


def _persist_installed_provider_cli(result: Dict[str, str]) -> Dict[str, Any]:
    env_data = _read_env_file(ENV_FILE_PATH)
    env_data[result["env_key"]] = result["cmd_path"]
    os.environ[result["env_key"]] = result["cmd_path"]
    _write_env_file(ENV_FILE_PATH, env_data)

    snap = _provider_settings_snapshot()
    snap.update(
        {
            "installed_target": result["target"],
            "installed_package": result["package"],
            "installed_binary": result["binary"],
            "installed_cmd_path": result["cmd_path"],
            "install_log_tail": result["install_log_tail"],
        }
    )
    logger.info(
        "provider_cli_installed target=%s package=%s cmd=%s",
        result["target"],
        result["package"],
        result["cmd_path"],
    )
    return snap


async def _install_provider_cli_stream_events(target: str) -> AsyncGenerator[Dict[str, Any], None]:
    spec = _provider_cli_install_spec(target)
    npm = shutil.which("npm")
    if not npm:
        yield {
            "type": "error",
            "error": "npm_not_found",
            "message": "npm is not available in this container image.",
        }
        return

    os.makedirs(CLI_TOOLS_PREFIX, exist_ok=True)
    env = os.environ.copy()
    env["NPM_CONFIG_PREFIX"] = CLI_TOOLS_PREFIX
    env["HOME"] = env.get("HOME") or "/tmp"
    env["NPM_CONFIG_CACHE"] = env.get("NPM_CONFIG_CACHE") or "/tmp/.npm"
    os.makedirs(env["NPM_CONFIG_CACHE"], exist_ok=True)

    expected_s = max(30, int(os.getenv("OPENSIFT_CLI_INSTALL_EXPECTED_SECONDS", "120")))
    start = time.perf_counter()
    last_percent = -1

    yield {
        "type": "start",
        "target": spec["target"],
        "package": spec["package"],
        "binary": spec["binary"],
        "expected_seconds": expected_s,
    }

    proc = await asyncio.create_subprocess_exec(
        npm,
        "install",
        "-g",
        spec["package"],
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        env=env,
    )
    if proc.stdout is None:
        yield {"type": "error", "error": "install_failed", "message": "Unable to capture installer output."}
        return

    logs: List[str] = []
    eof = False
    while not eof:
        try:
            line = await asyncio.wait_for(proc.stdout.readline(), timeout=1.0)
        except asyncio.TimeoutError:
            line = b""

        if line:
            text = line.decode("utf-8", errors="ignore").rstrip()
            if text:
                logs.append(text)
                yield {"type": "log", "text": text}
        else:
            if proc.returncode is not None:
                eof = True

        elapsed = max(0.0, time.perf_counter() - start)
        percent = min(95, max(5, int((elapsed / expected_s) * 100)))
        eta = max(0, int(expected_s - elapsed))
        if percent != last_percent:
            last_percent = percent
            yield {"type": "progress", "percent": percent, "eta_seconds": eta}

    rc = await proc.wait()
    if rc != 0:
        tail = "\n".join(logs[-20:]).strip()
        yield {
            "type": "error",
            "error": "install_failed",
            "message": tail or f"CLI install failed (exit code {rc}).",
        }
        return

    cmd_path = os.path.join(CLI_TOOLS_BIN_DIR, spec["binary"])
    if not os.path.exists(cmd_path):
        fallback = shutil.which(spec["binary"], path=CLI_TOOLS_BIN_DIR)
        if fallback:
            cmd_path = fallback
    if not os.path.exists(cmd_path):
        yield {
            "type": "error",
            "error": "binary_not_found_after_install",
            "message": "Install completed but installed binary was not found in tools bin path.",
        }
        return

    result = {
        "target": spec["target"],
        "package": spec["package"],
        "binary": spec["binary"],
        "env_key": spec["env_key"],
        "cmd_path": cmd_path,
        "install_log_tail": "\n".join(logs[-12:]).strip(),
    }
    snap = _persist_installed_provider_cli(result)
    yield {"type": "progress", "percent": 100, "eta_seconds": 0}
    yield {
        "type": "done",
        "target": result["target"],
        "package": result["package"],
        "binary": result["binary"],
        "cmd_path": result["cmd_path"],
        "install_log_tail": result["install_log_tail"],
        "snapshot": snap,
    }


def _clean_setting_value(raw: str, max_len: int = 4096) -> str:
    s = (raw or "").strip()
    if not s:
        return ""
    if "\n" in s or "\r" in s:
        raise ValueError("invalid_newline")
    if len(s) > max_len:
        raise ValueError("too_long")
    return s


def _pbkdf2_hash_password(password: str, salt: bytes, iters: int) -> bytes:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iters)


def _set_password(password: str) -> None:
    salt = secrets.token_bytes(16)
    iters = 200_000
    pw_hash = _pbkdf2_hash_password(password, salt, iters)
    state = {
        "has_password": True,
        "salt_b64": _b64url(salt),
        "hash_b64": _b64url(pw_hash),
        "algo": "pbkdf2_hmac_sha256",
        "iters": iters,
        "updated_at": _now(),
    }
    _save_auth_state(state)


def _has_password() -> bool:
    return bool(_load_auth_state().get("has_password"))


def _verify_password(password: str) -> bool:
    state = _load_auth_state()
    if not state.get("has_password"):
        return False
    try:
        salt = _b64url_decode(state["salt_b64"])
        expected = _b64url_decode(state["hash_b64"])
        iters = int(state.get("iters", 200_000))
        actual = _pbkdf2_hash_password(password, salt, iters)
        return hmac.compare_digest(expected, actual)
    except Exception:
        return False


def _is_authenticated(request: Request) -> bool:
    cookie = request.cookies.get(AUTH_COOKIE_NAME)
    return bool(cookie and _verify_session_cookie(cookie))


EXEMPT_PATHS = {"/login", "/set-password", "/health"}
EXEMPT_PREFIXES = ("/static",)


class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if path in EXEMPT_PATHS or any(path.startswith(p) for p in EXEMPT_PREFIXES):
            return await call_next(request)
        if _is_authenticated(request):
            return await call_next(request)
        if _is_api_path(path):
            return JSONResponse({"ok": False, "error": "unauthorized"}, status_code=401)
        return RedirectResponse(url="/login", status_code=HTTP_303_SEE_OTHER)


app.add_middleware(AuthMiddleware)

# -----------------------------------------------------------------------------
# CSRF protection (double-submit cookie + custom header)
# -----------------------------------------------------------------------------
CSRF_COOKIE_NAME = "opensift_csrf"
CSRF_HEADER_NAME = "x-csrf-token"
CSRF_EXEMPT_PATHS = {"/health", "/login", "/set-password"}
CSRF_EXEMPT_PREFIXES = ("/static",)


def _csrf_token_for_request(request: Request) -> str:
    current = (request.cookies.get(CSRF_COOKIE_NAME) or "").strip()
    if current:
        return current
    return secrets.token_urlsafe(32)


def _set_csrf_cookie(response: Any, request: Request, token: str) -> None:
    response.set_cookie(
        CSRF_COOKIE_NAME,
        token,
        httponly=True,
        samesite="lax",
        secure=(request.url.scheme == "https"),
        max_age=AUTH_TTL_SECONDS,
    )


def _csrf_invalid(request: Request) -> bool:
    cookie_token = (request.cookies.get(CSRF_COOKIE_NAME) or "").strip()
    header_token = (request.headers.get(CSRF_HEADER_NAME) or "").strip()
    if not cookie_token or not header_token:
        return True
    return not secrets.compare_digest(cookie_token, header_token)


class CsrfMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        method = request.method.upper()
        path = request.url.path
        if method not in ("POST", "PUT", "PATCH", "DELETE"):
            return await call_next(request)
        if path in CSRF_EXEMPT_PATHS or any(path.startswith(p) for p in CSRF_EXEMPT_PREFIXES):
            return await call_next(request)
        if _csrf_invalid(request):
            msg = "CSRF validation failed."
            if _is_api_path(path):
                return JSONResponse({"ok": False, "error": "csrf_failed", "message": msg}, status_code=403)
            return PlainTextResponse(msg, status_code=403)
        return await call_next(request)


app.add_middleware(CsrfMiddleware)


async def _print_startup_token():
    logger.info("OpenSift Local Auth")
    logger.info("generated_login_token valid_until=restart token_present=true")
    logger.info("generated_login_token_hint suffix=%s", GEN_TOKEN[-6:])
    logger.info("password_set=%s", "YES" if _has_password() else "NO")
    logger.info("sessions_dir=%s", SESSION_DIR)
    logger.info("study_library_dir=%s", STUDY_DIR)
    logger.info("quiz_attempts_dir=%s", QUIZ_DIR)
    logger.info("flashcards_dir=%s", FLASHCARD_DIR)
    logger.info("sources_dir=%s", SOURCE_DIR)


# -----------------------------------------------------------------------------
# Provider streaming helpers
# -----------------------------------------------------------------------------
async def _stream_openai(prompt: str, model: str) -> AsyncGenerator[str, None]:
    from openai import OpenAI  # type: ignore

    client = OpenAI()
    stream = client.responses.stream(model=model, input=prompt)
    with stream as s:
        for event in s:
            if getattr(event, "type", "") == "response.output_text.delta":
                delta = getattr(event, "delta", "")
                if delta:
                    yield delta
        _ = s.get_final_response()


async def _stream_anthropic(
    prompt: str,
    model: str,
    thinking_enabled: bool = False,
    thinking_level: str = "medium",
) -> AsyncGenerator[str, None]:
    import anthropic  # type: ignore

    client = anthropic.Anthropic()
    params: Dict[str, Any] = {
        "model": model,
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": prompt}],
    }
    if thinking_enabled:
        params["thinking"] = {
            "type": "enabled",
            "budget_tokens": claude_thinking_budget(thinking_level),
        }

    try:
        with client.messages.stream(**params) as stream:
            for text in stream.text_stream:
                if text:
                    yield text
    except Exception:
        if not thinking_enabled:
            raise
        params.pop("thinking", None)
        with client.messages.stream(**params) as stream:
            for text in stream.text_stream:
                if text:
                    yield text


async def _stream_codex(prompt: str, model: str) -> AsyncGenerator[str, None]:
    async for text in stream_with_codex(prompt, model=model):
        if text:
            yield text


def _run_generate(
    provider: str,
    prompt: str,
    model: str,
    thinking_enabled: bool = False,
    thinking_level: str = "medium",
) -> str:
    provider, model, _note = _resolve_provider_model_pair(provider, model)
    provider = (provider or "").strip().lower()
    model = (model or "").strip()

    if provider == "openai":
        m = model or DEFAULT_OPENAI_MODEL
        return generate_with_openai(prompt, model=m)

    if provider == "claude":
        m = model or DEFAULT_CLAUDE_MODEL
        return generate_with_claude(
            prompt,
            model=m,
            thinking_enabled=thinking_enabled,
            thinking_level=thinking_level,
        )

    if provider == "claude_code":
        # Prefer Claude Code CLI; if unavailable and Anthropic API key exists, fallback to Claude API.
        m = model or DEFAULT_CLAUDE_MODEL
        if not claude_code_cli_available():
            if os.getenv("ANTHROPIC_API_KEY", "").strip():
                logger.warning("claude_code_cli_unavailable_fallback_to_claude_api model=%s", m)
                return generate_with_claude(
                    prompt,
                    model=m,
                    thinking_enabled=thinking_enabled,
                    thinking_level=thinking_level,
                )
            if codex_cli_available() and codex_auth_detected():
                logger.warning("claude_code_cli_unavailable_fallback_to_codex model=%s", DEFAULT_CODEX_MODEL)
                return generate_with_codex(prompt, model=DEFAULT_CODEX_MODEL)
            if os.getenv("OPENAI_API_KEY", "").strip():
                logger.warning("claude_code_cli_unavailable_fallback_to_openai_api model=%s", DEFAULT_OPENAI_MODEL)
                return generate_with_openai(prompt, model=DEFAULT_OPENAI_MODEL)
            raise RuntimeError(
                "Claude Code CLI not found and no fallback provider configured. "
                "Install Claude Code or set OPENSIFT_CLAUDE_CODE_CMD, or configure ANTHROPIC_API_KEY/OPENAI_API_KEY."
            )
        try:
            return generate_with_claude_code(prompt, model=m)
        except Exception:
            # Claude CLI can return success with no usable output in headless environments.
            if os.getenv("ANTHROPIC_API_KEY", "").strip():
                logger.warning("claude_code_generation_failed_fallback_to_claude_api model=%s", m)
                return generate_with_claude(
                    prompt,
                    model=m,
                    thinking_enabled=thinking_enabled,
                    thinking_level=thinking_level,
                )
            if codex_cli_available() and codex_auth_detected():
                logger.warning("claude_code_generation_failed_fallback_to_codex model=%s", DEFAULT_CODEX_MODEL)
                return generate_with_codex(prompt, model=DEFAULT_CODEX_MODEL)
            if os.getenv("OPENAI_API_KEY", "").strip():
                logger.warning("claude_code_generation_failed_fallback_to_openai_api model=%s", DEFAULT_OPENAI_MODEL)
                return generate_with_openai(prompt, model=DEFAULT_OPENAI_MODEL)
            raise

    if provider == "codex":
        m = model or DEFAULT_CODEX_MODEL
        if not codex_cli_available():
            if os.getenv("OPENAI_API_KEY", "").strip():
                logger.warning("codex_cli_unavailable_fallback_to_openai_api model=%s", m)
                return generate_with_openai(prompt, model=m or DEFAULT_OPENAI_MODEL)
            raise RuntimeError(
                "Codex CLI not found and no OPENAI_API_KEY configured. "
                "Install Codex CLI or set OPENSIFT_CODEX_CMD, or configure OpenAI API key."
            )
        try:
            return generate_with_codex(prompt, model=m)
        except Exception:
            if os.getenv("OPENAI_API_KEY", "").strip():
                logger.warning("codex_generation_failed_fallback_to_openai_api model=%s", m)
                return generate_with_openai(prompt, model=m or DEFAULT_OPENAI_MODEL)
            raise

    raise RuntimeError(f"Unknown provider: {provider}")


async def _run_generate_resilient(
    request: Request,
    provider: str,
    prompt: str,
    model: str,
    thinking_enabled: bool = False,
    thinking_level: str = "medium",
) -> str:
    task: "asyncio.Task[str]" = asyncio.create_task(
        anyio.to_thread.run_sync(
            lambda: _run_generate(
                provider,
                prompt,
                model,
                thinking_enabled=thinking_enabled,
                thinking_level=thinking_level,
            ),
            abandon_on_cancel=False,
        )
    )
    try:
        while True:
            if task.done():
                return await task
            if await request.is_disconnected():
                task.cancel()
                raise asyncio.CancelledError()
            await asyncio.sleep(0.25)
    except asyncio.CancelledError:
        task.cancel()
        raise


# -----------------------------------------------------------------------------
# Auth pages
# -----------------------------------------------------------------------------
@app.get("/health")
async def health():
    return JSONResponse(
        {
            "ok": True,
            "time": _now(),
            "diagnostics": {
                "codex_auth_detected": codex_auth_detected(),
                "codex_cli_available": codex_cli_available(),
            },
        }
    )


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    response = templates.TemplateResponse(
        "login.html",
        {"request": request, "mode": "login", "has_password": _has_password(), "token": GEN_TOKEN, "error": None},
    )
    _set_csrf_cookie(response, request, _csrf_token_for_request(request))
    return response


@app.post("/login")
async def login_submit(request: Request, password: str = Form(""), token: str = Form("")):
    token = (token or "").strip()
    password = (password or "").strip()

    ok = False
    if token and secrets.compare_digest(token, GEN_TOKEN):
        ok = True
    elif password and _verify_password(password):
        ok = True

    if not ok:
        response = templates.TemplateResponse(
            "login.html",
            {
                "request": request,
                "mode": "login",
                "has_password": _has_password(),
                "token": GEN_TOKEN,
                "error": "Invalid token or password.",
            },
            status_code=401,
        )
        _set_csrf_cookie(response, request, _csrf_token_for_request(request))
        return response

    resp = RedirectResponse(url="/chat", status_code=HTTP_303_SEE_OTHER)
    resp.set_cookie(
        AUTH_COOKIE_NAME,
        _make_session_cookie(),
        httponly=True,
        samesite="lax",
        secure=(request.url.scheme == "https"),
        max_age=AUTH_TTL_SECONDS,
    )
    return resp


@app.get("/set-password", response_class=HTMLResponse)
async def set_password_page(request: Request):
    response = templates.TemplateResponse(
        "login.html",
        {"request": request, "mode": "set_password", "has_password": _has_password(), "token": GEN_TOKEN, "error": None},
    )
    _set_csrf_cookie(response, request, _csrf_token_for_request(request))
    return response


@app.post("/set-password")
async def set_password_submit(
    request: Request,
    token: str = Form(...),
    new_password: str = Form(...),
    confirm_password: str = Form(...),
):
    token = (token or "").strip()
    new_password = (new_password or "").strip()
    confirm_password = (confirm_password or "").strip()

    if not secrets.compare_digest(token, GEN_TOKEN):
        response = templates.TemplateResponse(
            "login.html",
            {
                "request": request,
                "mode": "set_password",
                "has_password": _has_password(),
                "token": GEN_TOKEN,
                "error": "Invalid token. Copy/paste exactly.",
            },
            status_code=401,
        )
        _set_csrf_cookie(response, request, _csrf_token_for_request(request))
        return response

    if len(new_password) < 8:
        response = templates.TemplateResponse(
            "login.html",
            {
                "request": request,
                "mode": "set_password",
                "has_password": _has_password(),
                "token": GEN_TOKEN,
                "error": "Password must be at least 8 characters.",
            },
            status_code=400,
        )
        _set_csrf_cookie(response, request, _csrf_token_for_request(request))
        return response

    if new_password != confirm_password:
        response = templates.TemplateResponse(
            "login.html",
            {
                "request": request,
                "mode": "set_password",
                "has_password": _has_password(),
                "token": GEN_TOKEN,
                "error": "Passwords do not match.",
            },
            status_code=400,
        )
        _set_csrf_cookie(response, request, _csrf_token_for_request(request))
        return response

    _set_password(new_password)

    resp = RedirectResponse(url="/chat", status_code=HTTP_303_SEE_OTHER)
    resp.set_cookie(
        AUTH_COOKIE_NAME,
        _make_session_cookie(),
        httponly=True,
        samesite="lax",
        secure=(request.url.scheme == "https"),
        max_age=AUTH_TTL_SECONDS,
    )
    return resp


@app.post("/logout")
async def logout():
    resp = RedirectResponse(url="/login", status_code=HTTP_303_SEE_OTHER)
    resp.delete_cookie(AUTH_COOKIE_NAME)
    resp.delete_cookie(CSRF_COOKIE_NAME)
    return resp


# -----------------------------------------------------------------------------
# UI pages
# -----------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def root(request: Request, owner: str = "default"):
    owner = _normalize_owner(owner)
    history = _history_snapshot(owner)
    csrf_token = _csrf_token_for_request(request)
    provider_caps = _provider_runtime_caps()
    response = templates.TemplateResponse(
        "chat.html",
        {
            "request": request,
            "owner": owner,
            "history": history,
            "csrf_token": csrf_token,
            "default_claude_model": DEFAULT_CLAUDE_MODEL,
            "default_openai_model": DEFAULT_OPENAI_MODEL,
            "default_codex_model": DEFAULT_CODEX_MODEL,
            "supported_claude_models": SUPPORTED_CLAUDE_MODELS,
            "supported_openai_models": SUPPORTED_OPENAI_MODELS,
            "supported_thinking_levels": SUPPORTED_THINKING_LEVELS,
            "preferred_provider": _preferred_provider_default(),
            "provider_caps": provider_caps,
        },
    )
    _set_csrf_cookie(response, request, csrf_token)
    return response


@app.get("/chat", response_class=HTMLResponse)
async def chat_page(request: Request, owner: str = "default"):
    owner = _normalize_owner(owner)
    history = _history_snapshot(owner)
    csrf_token = _csrf_token_for_request(request)
    provider_caps = _provider_runtime_caps()
    response = templates.TemplateResponse(
        "chat.html",
        {
            "request": request,
            "owner": owner,
            "history": history,
            "csrf_token": csrf_token,
            "default_claude_model": DEFAULT_CLAUDE_MODEL,
            "default_openai_model": DEFAULT_OPENAI_MODEL,
            "default_codex_model": DEFAULT_CODEX_MODEL,
            "supported_claude_models": SUPPORTED_CLAUDE_MODELS,
            "supported_openai_models": SUPPORTED_OPENAI_MODELS,
            "supported_thinking_levels": SUPPORTED_THINKING_LEVELS,
            "preferred_provider": _preferred_provider_default(),
            "provider_caps": provider_caps,
        },
    )
    _set_csrf_cookie(response, request, csrf_token)
    return response


@app.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request, owner: str = "default"):
    owner = _normalize_owner(owner)
    csrf_token = _csrf_token_for_request(request)
    response = templates.TemplateResponse(
        "settings.html",
        {
            "request": request,
            "owner": owner,
            "csrf_token": csrf_token,
            "token_hint": GEN_TOKEN[-6:],
            "has_password": _has_password(),
        },
    )
    _set_csrf_cookie(response, request, csrf_token)
    return response


@app.get("/library", response_class=HTMLResponse)
async def library_page(request: Request, owner: str = "default"):
    owner = _normalize_owner(owner)
    csrf_token = _csrf_token_for_request(request)
    response = templates.TemplateResponse(
        "library.html",
        {
            "request": request,
            "owner": owner,
            "csrf_token": csrf_token,
        },
    )
    _set_csrf_cookie(response, request, csrf_token)
    return response


@app.post("/chat/clear")
async def chat_clear(owner: str = Form("default")):
    owner = _normalize_owner(owner)
    _history_clear(owner)
    return JSONResponse({"ok": True})


@app.get("/chat/settings/auth")
async def settings_auth_get():
    return JSONResponse({"ok": True, "has_password": _has_password(), "token_hint": GEN_TOKEN[-6:]})


@app.post("/chat/settings/auth/password")
async def settings_auth_password_set(new_password: str = Form(...), confirm_password: str = Form(...)):
    new_password = (new_password or "").strip()
    confirm_password = (confirm_password or "").strip()
    if len(new_password) < 8:
        return JSONResponse({"ok": False, "error": "password_too_short"}, status_code=400)
    if new_password != confirm_password:
        return JSONResponse({"ok": False, "error": "password_mismatch"}, status_code=400)
    _set_password(new_password)
    return JSONResponse({"ok": True, "has_password": True})


@app.post("/chat/settings/auth/token/rotate")
async def settings_auth_token_rotate():
    token = _rotate_login_token()
    return JSONResponse({"ok": True, "token": token, "token_hint": token[-6:]})


@app.get("/chat/settings/providers")
async def settings_providers_get():
    return JSONResponse(_provider_settings_snapshot())


@app.post("/chat/settings/providers")
async def settings_providers_set(
    openai_api_key: str = Form(""),
    anthropic_api_key: str = Form(""),
    claude_code_oauth_token: str = Form(""),
    chatgpt_codex_oauth_token: str = Form(""),
    claude_code_cmd: str = Form(""),
    codex_cmd: str = Form(""),
):
    updates: Dict[str, str] = {}
    removals: List[str] = []

    def _apply_secret(key: str, raw: str) -> None:
        cleaned = _clean_setting_value(raw)
        if not cleaned:
            return
        if cleaned.lower() == "none":
            removals.append(key)
            return
        updates[key] = cleaned

    def _apply_path(key: str, raw: str) -> None:
        cleaned = _clean_setting_value(raw, max_len=512)
        if not cleaned:
            return
        if cleaned.lower() == "none":
            removals.append(key)
            return
        # Keep command strings simple; reject shell-control chars.
        if re.search(r"[;&|`$<>]", cleaned):
            raise ValueError(f"invalid_{key}")
        updates[key] = cleaned

    try:
        _apply_secret("OPENAI_API_KEY", openai_api_key)
        _apply_secret("ANTHROPIC_API_KEY", anthropic_api_key)
        _apply_secret("CLAUDE_CODE_OAUTH_TOKEN", claude_code_oauth_token)
        _apply_secret("CHATGPT_CODEX_OAUTH_TOKEN", chatgpt_codex_oauth_token)
        _apply_path("OPENSIFT_CLAUDE_CODE_CMD", claude_code_cmd)
        _apply_path("OPENSIFT_CODEX_CMD", codex_cmd)
    except ValueError as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)

    if not updates and not removals:
        return JSONResponse(_provider_settings_snapshot())

    env_data = _read_env_file(ENV_FILE_PATH)
    for k in removals:
        env_data.pop(k, None)
        os.environ.pop(k, None)
    for k, v in updates.items():
        env_data[k] = v
        os.environ[k] = v

    _write_env_file(ENV_FILE_PATH, env_data)
    logger.info(
        "provider_settings_updated keys=%s cleared=%s",
        ",".join(sorted(updates.keys())) or "none",
        ",".join(sorted(removals)) or "none",
    )
    return JSONResponse(_provider_settings_snapshot())


@app.post("/chat/settings/providers/install")
async def settings_providers_install(target: str = Form(...)):
    try:
        result = await anyio.to_thread.run_sync(lambda: _install_provider_cli(target))
    except ValueError as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)
    except Exception as e:
        logger.exception("provider_cli_install_failed target=%s", target)
        msg = str(e)
        if msg == "npm_not_found":
            return JSONResponse(
                {
                    "ok": False,
                    "error": "npm_not_found",
                    "message": "npm is not available in this container image.",
                },
                status_code=500,
            )
        return JSONResponse({"ok": False, "error": "install_failed", "message": msg}, status_code=500)

    return JSONResponse(_persist_installed_provider_cli(result))


@app.post("/chat/settings/providers/install/stream")
async def settings_providers_install_stream(target: str = Form(...)):
    async def _gen() -> AsyncGenerator[bytes, None]:
        try:
            async for event in _install_provider_cli_stream_events(target):
                yield _ndjson(event)
        except Exception:
            logger.exception("provider_cli_install_stream_failed target=%s", target)
            yield _ndjson({"type": "error", "error": "install_stream_failed", "message": "Installer stream failed."})

    return StreamingResponse(_gen(), media_type="application/x-ndjson")


# -----------------------------------------------------------------------------
# Session endpoints
# -----------------------------------------------------------------------------
def _session_summaries() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for owner in list_sessions(SESSION_DIR):
        history = load_session(owner, SESSION_DIR)
        last_ts = ""
        if history:
            last_ts = str(history[-1].get("ts") or "")
        out.append({"owner": owner, "count": len(history), "last_ts": last_ts})
    out.sort(key=lambda x: (x.get("last_ts") or "", x.get("owner") or ""), reverse=True)
    return out


@app.get("/chat/session/list")
async def session_list():
    return JSONResponse({"ok": True, "sessions": _session_summaries()})


@app.get("/chat/session/export")
async def session_export(owner: str = "default"):
    owner = _normalize_owner(owner)
    return JSONResponse({"ok": True, "owner": owner, "history": _history_snapshot(owner)})


@app.post("/chat/session/import")
async def session_import(owner: str = Form("default"), payload: str = Form(...), merge: bool = Form(False)):
    """
    payload: JSON array of chat messages
    merge: if true, append; else replace
    """
    owner = _normalize_owner(owner)
    if len((payload or "").strip()) > MAX_SESSION_IMPORT_CHARS:
        return JSONResponse({"ok": False, "error": "payload_too_large"}, status_code=413)
    try:
        data = json.loads(payload)
        if not isinstance(data, list):
            raise ValueError("payload must be a JSON list")
        cleaned: List[Dict[str, Any]] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            role = item.get("role")
            text = item.get("text")
            if role in ("user", "assistant") and isinstance(text, str):
                cleaned.append(item)
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"invalid_payload: {e}"}, status_code=400)

    if merge:
        _history_extend(owner, cleaned)
    else:
        _history_replace(owner, cleaned)
    return JSONResponse({"ok": True, "owner": owner, "count": len(_history_snapshot(owner))})


@app.post("/chat/session/new")
async def session_new(owner: str = Form("default")):
    owner = _normalize_owner(owner)
    _history_replace(owner, [])
    return JSONResponse({"ok": True, "owner": owner})


@app.post("/chat/session/delete")
async def session_delete(owner: str = Form(...), delete_library_items: bool = Form(False)):
    owner = _normalize_owner(owner)
    if not owner:
        return JSONResponse({"ok": False, "error": "owner_required"}, status_code=400)
    _history_delete_owner(owner)
    ok = delete_session(owner, SESSION_DIR)
    if not ok:
        return JSONResponse({"ok": False, "error": "not_found"}, status_code=404)

    deleted_library_count = 0
    if delete_library_items:
        items = load_source_items(owner, SOURCE_DIR)
        for item in items:
            item_id = str(item.get("id") or "").strip()
            if not item_id:
                continue
            removed = delete_source_item(owner, item_id, SOURCE_DIR)
            if not removed:
                continue
            chunk_ids = [x for x in (removed.get("chunk_ids") or []) if isinstance(x, str)]
            if chunk_ids:
                try:
                    await anyio.to_thread.run_sync(lambda: db.delete(chunk_ids))
                except Exception:
                    logger.exception("session_delete_vector_chunks_failed owner=%s item_id=%s", owner, item_id)
            remove_source_file(str(removed.get("binary_path") or ""))
            remove_source_file(str(removed.get("text_path") or ""))
            deleted_library_count += 1

    return JSONResponse(
        {
            "ok": True,
            "owner": owner,
            "delete_library_items": bool(delete_library_items),
            "deleted_library_count": deleted_library_count,
        }
    )


@app.get("/chat/soul")
async def soul_get(owner: str = "default"):
    _ = owner
    style = get_global_style()
    return JSONResponse({"ok": True, "scope": "global", "style": style})


@app.post("/chat/soul/set")
async def soul_set(owner: str = Form("default"), style: str = Form("")):
    _ = owner
    set_global_style(style)
    logger.info("soul_set scope=global chars=%d", len((style or "").strip()))
    return JSONResponse({"ok": True, "scope": "global", "style": get_global_style()})


@app.get("/chat/wellness")
async def wellness_get():
    return JSONResponse({"ok": True, "settings": get_wellness_settings()})


@app.post("/chat/wellness/set")
async def wellness_set(
    enabled: str = Form("true"),
    every_user_msgs: int = Form(6),
    min_minutes: int = Form(45),
):
    enabled_bool = str(enabled).strip().lower() not in ("0", "false", "off", "no")
    settings = set_wellness_settings(
        enabled=enabled_bool,
        every_user_msgs=max(1, int(every_user_msgs)),
        min_minutes=max(1, int(min_minutes)),
    )
    logger.info(
        "wellness_set enabled=%s every_user_msgs=%d min_minutes=%d",
        settings["enabled"],
        settings["every_user_msgs"],
        settings["min_minutes"],
    )
    return JSONResponse({"ok": True, "settings": settings})


# -----------------------------------------------------------------------------
# Study library endpoints
# -----------------------------------------------------------------------------
@app.get("/chat/study/list")
async def study_list(owner: str = "default"):
    items = load_study_library(owner, STUDY_DIR)
    summaries = [
        {
            "id": item.get("id"),
            "title": item.get("title") or "Saved Study Item",
            "mode": item.get("mode") or "",
            "created_at": item.get("created_at") or "",
            "preview": (item.get("text") or "")[:180],
        }
        for item in reversed(items)
    ]
    return JSONResponse({"ok": True, "owner": owner, "items": summaries})


@app.get("/chat/study/get")
async def study_get(owner: str = "default", item_id: str = ""):
    item_id = (item_id or "").strip()
    if not item_id:
        return JSONResponse({"ok": False, "error": "item_id_required"}, status_code=400)
    item = get_study_item(owner, item_id, STUDY_DIR)
    if not item:
        return JSONResponse({"ok": False, "error": "not_found"}, status_code=404)
    return JSONResponse({"ok": True, "owner": owner, "item": item})


@app.post("/chat/study/save-last")
async def study_save_last(owner: str = Form("default"), title: str = Form("")):
    msg = _last_generated_assistant(owner)
    if not msg:
        return JSONResponse({"ok": False, "error": "no_generated_answer_found"}, status_code=404)

    mode = (msg.get("mode") or "").strip()
    ts = msg.get("ts") or _now()
    default_title = f"{(mode or 'study').replace('_', ' ').title()} - {ts[:10]}"
    saved = add_study_item(
        owner=owner,
        title=title.strip() or default_title,
        text=msg.get("text", ""),
        mode=mode,
        sources=msg.get("sources") if isinstance(msg.get("sources"), list) else [],
        created_at=ts,
        base_dir=STUDY_DIR,
    )
    return JSONResponse({"ok": True, "owner": owner, "item": saved})


@app.post("/chat/study/delete")
async def study_delete(owner: str = Form("default"), item_id: str = Form(...)):
    ok = delete_study_item(owner, item_id, STUDY_DIR)
    if not ok:
        return JSONResponse({"ok": False, "error": "not_found"}, status_code=404)
    return JSONResponse({"ok": True, "owner": owner, "item_id": item_id})


# -----------------------------------------------------------------------------
# Quiz attempt endpoints
# -----------------------------------------------------------------------------
@app.get("/chat/quiz/history")
async def quiz_history(owner: str = "default"):
    attempts = load_attempts(owner, QUIZ_DIR)
    attempts_rev = list(reversed(attempts))

    avg_pct = 0.0
    if attempts:
        avg_pct = round(sum(float(a.get("pct") or 0.0) for a in attempts) / len(attempts), 2)

    return JSONResponse(
        {
            "ok": True,
            "owner": owner,
            "stats": {"count": len(attempts), "avg_pct": avg_pct},
            "attempts": attempts_rev,
        }
    )


@app.post("/chat/quiz/add")
async def quiz_add(
    owner: str = Form("default"),
    title: str = Form(""),
    score: int = Form(...),
    total: int = Form(...),
    notes: str = Form(""),
):
    if total <= 0:
        return JSONResponse({"ok": False, "error": "total_must_be_positive"}, status_code=400)
    if score < 0 or score > total:
        return JSONResponse({"ok": False, "error": "score_must_be_between_0_and_total"}, status_code=400)

    item = add_attempt(
        owner=owner,
        title=title,
        score=score,
        total=total,
        notes=notes,
        created_at=_now(),
        base_dir=QUIZ_DIR,
    )
    return JSONResponse({"ok": True, "owner": owner, "attempt": item})


@app.post("/chat/quiz/delete")
async def quiz_delete(owner: str = Form("default"), attempt_id: str = Form(...)):
    ok = delete_attempt(owner, attempt_id, QUIZ_DIR)
    if not ok:
        return JSONResponse({"ok": False, "error": "not_found"}, status_code=404)
    return JSONResponse({"ok": True, "owner": owner, "attempt_id": attempt_id})


# -----------------------------------------------------------------------------
# Flashcard endpoints
# -----------------------------------------------------------------------------
@app.get("/chat/flashcards/list")
async def flashcards_list(owner: str = "default", due_only: bool = False):
    cards = due_cards(owner, FLASHCARD_DIR) if due_only else load_cards(owner, FLASHCARD_DIR)
    cards_sorted = sorted(cards, key=lambda c: c.get("due_at") or "")
    return JSONResponse({"ok": True, "owner": owner, "cards": cards_sorted})


@app.post("/chat/flashcards/add")
async def flashcards_add(owner: str = Form("default"), front: str = Form(...), back: str = Form(...), tags: str = Form("")):
    front = (front or "").strip()
    back = (back or "").strip()
    if not front or not back:
        return JSONResponse({"ok": False, "error": "front_and_back_required"}, status_code=400)

    card = add_card(owner=owner, front=front, back=back, tags=tags, created_at=_now(), base_dir=FLASHCARD_DIR)
    return JSONResponse({"ok": True, "owner": owner, "card": card})


@app.post("/chat/flashcards/review")
async def flashcards_review(owner: str = Form("default"), card_id: str = Form(...), rating: str = Form("good")):
    if rating not in ("again", "hard", "good", "easy"):
        return JSONResponse({"ok": False, "error": "invalid_rating"}, status_code=400)

    card = review_card(owner=owner, card_id=card_id, rating=rating, reviewed_at=_now(), base_dir=FLASHCARD_DIR)
    if not card:
        return JSONResponse({"ok": False, "error": "not_found"}, status_code=404)
    return JSONResponse({"ok": True, "owner": owner, "card": card})


@app.post("/chat/flashcards/delete")
async def flashcards_delete(owner: str = Form("default"), card_id: str = Form(...)):
    ok = delete_card(owner, card_id, FLASHCARD_DIR)
    if not ok:
        return JSONResponse({"ok": False, "error": "not_found"}, status_code=404)
    return JSONResponse({"ok": True, "owner": owner, "card_id": card_id})


def _library_index_text(
    owner: str,
    kind: str,
    title: str,
    text: str,
    url: str = "",
    original_name: str = "",
    binary_path: str = "",
    source_id: str = "",
    folder: str = "",
    tags: str = "",
) -> Dict[str, Any]:
    owner = _normalize_owner(owner)
    clean_text = (text or "").strip()
    if not clean_text:
        raise RuntimeError("No extractable text found for this source.")

    source_id = (source_id or "").strip() or new_source_id()
    prefix = f"{owner}::source::{source_id}"
    chunks = chunk_text(clean_text, prefix=prefix)
    if not chunks:
        raise RuntimeError("No chunkable text extracted from source.")

    texts = [c.text for c in chunks]
    ids = [c.chunk_id for c in chunks]
    metas = [
        {
            "source": title,
            "source_id": source_id,
            "kind": kind,
            "owner": owner,
            "url": url,
            "original_name": original_name,
            "start": c.start,
            "end": c.end,
        }
        for c in chunks
    ]

    embs = embed_texts(texts)
    db.add(ids=ids, documents=texts, metadatas=metas, embeddings=embs)

    text_path = write_source_text_blob(owner, source_id, clean_text, SOURCE_DIR)
    item = {
        "id": source_id,
        "owner": owner,
        "kind": kind,
        "title": (title or "").strip() or f"Untitled {kind}",
        "url": (url or "").strip(),
        "original_name": (original_name or "").strip(),
        "binary_path": (binary_path or "").strip(),
        "text_path": text_path,
        "preview": clean_text[:320],
        "text_chars": len(clean_text),
        "chunk_count": len(chunks),
        "chunk_ids": ids,
        "folder": (folder or "").strip(),
        "tags": (tags or "").strip(),
        "created_at": _now(),
    }
    add_source_item(owner, item, SOURCE_DIR)
    return item


# -----------------------------------------------------------------------------
# Library endpoints (separate from chat UI, but shared retrieval DB)
# -----------------------------------------------------------------------------
@app.get("/chat/library/list")
async def library_list(
    owner: str = "default",
    all_owners: bool = False,
    q: str = "",
    kind: str = "",
    folder: str = "",
    tags: str = "",
    page: int = 1,
    page_size: int = 20,
    sort_by: str = "created_at",
    sort_dir: str = "desc",
):
    owner = _normalize_owner(owner)
    owners: List[str]
    if all_owners:
        owners = list_source_owners(SOURCE_DIR)
    else:
        owners = [owner]

    items: List[Dict[str, Any]] = []
    for own in owners:
        own_items = load_source_items(own, SOURCE_DIR)
        for item in own_items:
            if not isinstance(item, dict):
                continue
            if not item.get("owner"):
                item = dict(item)
                item["owner"] = own
            items.append(item)
    qq = (q or "").strip().lower()
    kk = (kind or "").strip().lower()
    ff = (folder or "").strip().lower()
    tt = [x.strip().lower() for x in (tags or "").split(",") if x.strip()]
    page = max(1, int(page or 1))
    page_size = max(1, min(int(page_size or 20), 100))
    sort_by = (sort_by or "created_at").strip().lower()
    sort_dir = (sort_dir or "desc").strip().lower()

    if qq:
        items = [
            x
            for x in items
            if qq in (str(x.get("title", "")).lower())
            or qq in (str(x.get("preview", "")).lower())
            or qq in (str(x.get("original_name", "")).lower())
            or qq in (str(x.get("url", "")).lower())
        ]
    if kk:
        items = [x for x in items if str(x.get("kind", "")).lower() == kk]
    if ff:
        items = [x for x in items if str(x.get("folder", "")).lower() == ff]
    if tt:
        def _item_tags(item: Dict[str, Any]) -> List[str]:
            return [t.strip().lower() for t in str(item.get("tags", "")).split(",") if t.strip()]
        items = [x for x in items if all(tag in _item_tags(x) for tag in tt)]

    reverse = sort_dir != "asc"
    if sort_by == "title":
        items.sort(key=lambda x: str(x.get("title", "")).lower(), reverse=reverse)
    elif sort_by == "kind":
        items.sort(key=lambda x: str(x.get("kind", "")).lower(), reverse=reverse)
    elif sort_by == "text_chars":
        items.sort(key=lambda x: int(x.get("text_chars") or 0), reverse=reverse)
    else:
        items.sort(key=lambda x: str(x.get("created_at", "")), reverse=reverse)

    total = len(items)
    start = (page - 1) * page_size
    end = start + page_size
    page_items = items[start:end]
    folders = sorted({str(x.get("folder", "")).strip() for x in items if str(x.get("folder", "")).strip()})
    all_owner_names = sorted({str(x.get("owner", "")).strip() for x in items if str(x.get("owner", "")).strip()})
    return JSONResponse(
        {
            "ok": True,
            "owner": owner,
            "all_owners": all_owners,
            "items": page_items,
            "pagination": {
                "page": page,
                "page_size": page_size,
                "total": total,
                "total_pages": max(1, (total + page_size - 1) // page_size),
            },
            "folders": folders,
            "owners": all_owner_names,
        }
    )


@app.get("/chat/library/get")
async def library_get(owner: str = "default", item_owner: str = "", item_id: str = ""):
    owner = _normalize_owner(owner)
    effective_owner = _normalize_owner(item_owner or owner)
    item = get_source_item(effective_owner, item_id, SOURCE_DIR)
    if not item:
        return JSONResponse({"ok": False, "error": "not_found"}, status_code=404)
    text = read_source_text_blob(str(item.get("text_path") or ""))
    if not item.get("owner"):
        item = dict(item)
        item["owner"] = effective_owner
    return JSONResponse({"ok": True, "owner": owner, "item": item, "text": text})


@app.post("/chat/library/note")
async def library_note(
    owner: str = Form("default"),
    title: str = Form(""),
    note: str = Form(...),
    folder: str = Form(""),
    tags: str = Form(""),
):
    owner = _normalize_owner(owner)
    text = (note or "").strip()
    if not text:
        return JSONResponse({"ok": False, "error": "note_required"}, status_code=400)
    item = await anyio.to_thread.run_sync(
        lambda: _library_index_text(
            owner,
            "note",
            title or "Quick Note",
            text,
            folder=folder,
            tags=tags,
        )
    )
    return JSONResponse({"ok": True, "owner": owner, "item": item})


@app.post("/chat/library/url")
async def library_url(
    owner: str = Form("default"),
    url: str = Form(...),
    title: str = Form(""),
    folder: str = Form(""),
    tags: str = Form(""),
):
    owner = _normalize_owner(owner)
    raw_url = (url or "").strip()
    if not raw_url:
        return JSONResponse({"ok": False, "error": "url_required"}, status_code=400)
    if "://" not in raw_url:
        raw_url = f"https://{raw_url}"
    parsed = urlparse(raw_url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return JSONResponse({"ok": False, "error": "invalid_url"}, status_code=400)

    page_title, text = await fetch_url_text(raw_url)
    source_title = (title or "").strip() or page_title or raw_url
    item = await anyio.to_thread.run_sync(
        lambda: _library_index_text(
            owner,
            "url",
            source_title,
            text,
            url=raw_url,
            folder=folder,
            tags=tags,
        )
    )
    return JSONResponse({"ok": True, "owner": owner, "item": item})


@app.post("/chat/library/upload")
async def library_upload(
    owner: str = Form("default"),
    title: str = Form(""),
    folder: str = Form(""),
    tags: str = Form(""),
    file: UploadFile = File(...),
):
    owner = _normalize_owner(owner)
    try:
        data = await _read_upload_limited(file, MAX_UPLOAD_BYTES)
    except ValueError:
        return JSONResponse({"ok": False, "error": "file_too_large"}, status_code=413)

    filename = (file.filename or "upload").strip() or "upload"
    lower = filename.lower()
    kind = "file"
    text = ""
    if lower.endswith(".pdf"):
        kind = "pdf"
        text = await anyio.to_thread.run_sync(lambda: extract_text_from_pdf(data))
    elif lower.endswith((".txt", ".md")):
        kind = "text"
        text = await anyio.to_thread.run_sync(lambda: extract_text_from_txt(data))
    elif lower.endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff", ".tif")):
        kind = "image"
        text = await anyio.to_thread.run_sync(lambda: _extract_text_from_image_bytes(data))
    else:
        return JSONResponse({"ok": False, "error": "unsupported_file_type"}, status_code=400)

    if not (text or "").strip():
        return JSONResponse({"ok": False, "error": "no_text_extracted"}, status_code=400)

    source_id = new_source_id()
    binary_path = write_source_binary_blob(owner, source_id, filename, data, SOURCE_DIR)
    # Reuse ID for indexed source for easier file mapping.
    item = await anyio.to_thread.run_sync(
        lambda: _library_index_text(
            owner=owner,
            kind=kind,
            title=(title or "").strip() or filename,
            text=text,
            original_name=filename,
            binary_path=binary_path,
            source_id=source_id,
            folder=folder,
            tags=tags,
        )
    )
    return JSONResponse({"ok": True, "owner": owner, "item": item})


@app.post("/chat/library/delete")
async def library_delete(owner: str = Form("default"), item_owner: str = Form(""), item_id: str = Form(...)):
    owner = _normalize_owner(owner)
    effective_owner = _normalize_owner(item_owner or owner)
    removed = delete_source_item(effective_owner, item_id, SOURCE_DIR)
    if not removed:
        return JSONResponse({"ok": False, "error": "not_found"}, status_code=404)

    chunk_ids = [x for x in (removed.get("chunk_ids") or []) if isinstance(x, str)]
    if chunk_ids:
        try:
            await anyio.to_thread.run_sync(lambda: db.delete(chunk_ids))
        except Exception:
            logger.exception("library_delete_vector_chunks_failed owner=%s item_id=%s", effective_owner, item_id)

    remove_source_file(str(removed.get("binary_path") or ""))
    remove_source_file(str(removed.get("text_path") or ""))
    return JSONResponse({"ok": True, "owner": effective_owner, "item_id": item_id})


@app.post("/chat/library/update")
async def library_update(
    owner: str = Form("default"),
    item_owner: str = Form(""),
    item_id: str = Form(...),
    title: str = Form(""),
    folder: str = Form(""),
    tags: str = Form(""),
):
    owner = _normalize_owner(owner)
    effective_owner = _normalize_owner(item_owner or owner)
    patch = {
        "title": (title or "").strip(),
        "folder": (folder or "").strip(),
        "tags": (tags or "").strip(),
    }
    item = update_source_item(effective_owner, item_id, patch, SOURCE_DIR)
    if not item:
        return JSONResponse({"ok": False, "error": "not_found"}, status_code=404)
    if not item.get("owner"):
        item = dict(item)
        item["owner"] = effective_owner
    return JSONResponse({"ok": True, "owner": effective_owner, "item": item})


@app.get("/chat/library/download")
async def library_download(owner: str = "default", item_owner: str = "", item_id: str = ""):
    owner = _normalize_owner(owner)
    effective_owner = _normalize_owner(item_owner or owner)
    item = get_source_item(effective_owner, item_id, SOURCE_DIR)
    if not item:
        return JSONResponse({"ok": False, "error": "not_found"}, status_code=404)

    binary_path = str(item.get("binary_path") or "").strip()
    if not binary_path or not os.path.isfile(binary_path):
        return JSONResponse({"ok": False, "error": "binary_not_available"}, status_code=404)

    original_name = (item.get("original_name") or item.get("title") or "source.bin").strip()
    return FileResponse(path=binary_path, filename=original_name)


@app.get("/chat/library/preview")
async def library_preview(owner: str = "default", item_owner: str = "", item_id: str = ""):
    owner = _normalize_owner(owner)
    effective_owner = _normalize_owner(item_owner or owner)
    item = get_source_item(effective_owner, item_id, SOURCE_DIR)
    if not item:
        return JSONResponse({"ok": False, "error": "not_found"}, status_code=404)

    binary_path = str(item.get("binary_path") or "").strip()
    if not binary_path or not os.path.isfile(binary_path):
        return JSONResponse({"ok": False, "error": "binary_not_available"}, status_code=404)

    kind = str(item.get("kind") or "").strip().lower()
    original_name = (item.get("original_name") or item.get("title") or "source.pdf").strip()
    file_ext = os.path.splitext(binary_path)[1].lower()
    name_ext = os.path.splitext(original_name)[1].lower()
    if kind != "pdf" and file_ext != ".pdf" and name_ext != ".pdf":
        return JSONResponse({"ok": False, "error": "preview_not_supported_for_kind"}, status_code=400)

    return FileResponse(
        path=binary_path,
        media_type="application/pdf",
        headers={"Content-Disposition": f'inline; filename="{original_name}"'},
    )


# -----------------------------------------------------------------------------
# Ingest endpoints
# -----------------------------------------------------------------------------
@app.post("/chat/ingest/url")
async def chat_ingest_url(owner: str = Form("default"), url: str = Form(...), source_title: str = Form("")):
    owner = _normalize_owner(owner)
    t0 = time.perf_counter()

    raw_url = (url or "").strip()
    if not raw_url:
        assistant_msg = {
            "role": "assistant",
            "ts": _now(),
            "text": "⚠️ URL is required.",
            "sources": [],
        }
        _history_append(owner, assistant_msg)
        return JSONResponse({"ok": False, "assistant": assistant_msg}, status_code=400)

    if "://" not in raw_url:
        raw_url = f"https://{raw_url}"

    parsed = urlparse(raw_url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        assistant_msg = {
            "role": "assistant",
            "ts": _now(),
            "text": "⚠️ Invalid URL format. Please use a full http/https URL.",
            "sources": [],
        }
        _history_append(owner, assistant_msg)
        return JSONResponse({"ok": False, "assistant": assistant_msg}, status_code=400)

    try:
        logger.info("ingest_url_start owner=%s url=%s source_title=%s", owner, raw_url, source_title.strip())
        title, text = await fetch_url_text(raw_url)
        source = source_title.strip() or title
        prefix = f"{owner}::{source}" if owner else source

        chunks = chunk_text(text, prefix=prefix)
        if not chunks:
            raise RuntimeError("No chunkable text extracted from URL")

        texts = [c.text for c in chunks]
        ids = [c.chunk_id for c in chunks]
        metas = [
            {"source": source, "kind": "url", "url": raw_url, "owner": owner, "start": c.start, "end": c.end}
            for c in chunks
        ]

        embs = await anyio.to_thread.run_sync(lambda: embed_texts(texts))
        await anyio.to_thread.run_sync(lambda: db.add(ids=ids, documents=texts, metadatas=metas, embeddings=embs))

        assistant_msg = {
            "role": "assistant",
            "ts": _now(),
            "text": f"✅ Ingested {len(chunks)} chunks from URL:\n{source}\n{raw_url}",
            "sources": [],
        }
        _history_append(owner, assistant_msg)
        logger.info(
            "ingest_url_success owner=%s source=%s chunks=%d duration_ms=%.2f",
            owner,
            source,
            len(chunks),
            (time.perf_counter() - t0) * 1000.0,
        )
        return JSONResponse({"ok": True, "assistant": assistant_msg})

    except Exception as e:
        logger.exception(
            "ingest_url_failed owner=%s url=%s duration_ms=%.2f",
            owner,
            raw_url,
            (time.perf_counter() - t0) * 1000.0,
        )
        assistant_msg = {
            "role": "assistant",
            "ts": _now(),
            "text": (
                f"⚠️ URL ingest failed: {e}\n\n"
                "This page may block scraping or require JavaScript rendering. "
                "Try another URL, upload a PDF, or paste text into a .txt/.md file."
            ),
            "sources": [],
        }
        _history_append(owner, assistant_msg)
        return JSONResponse({"ok": False, "assistant": assistant_msg}, status_code=400)


@app.post("/chat/ingest/file")
async def chat_ingest_file(owner: str = Form("default"), file: UploadFile = File(...)):
    owner = _normalize_owner(owner)
    t0 = time.perf_counter()

    try:
        data = await _read_upload_limited(file, MAX_UPLOAD_BYTES)
    except ValueError:
        assistant_msg = {
            "role": "assistant",
            "ts": _now(),
            "text": f"⚠️ File too large. Max allowed is {MAX_UPLOAD_MB} MB.",
            "sources": [],
        }
        _history_append(owner, assistant_msg)
        return JSONResponse({"ok": False, "assistant": assistant_msg}, status_code=413)
    filename = file.filename or "upload"
    lower = filename.lower()

    if lower.endswith(".pdf"):
        kind = "pdf"
        text = await anyio.to_thread.run_sync(lambda: extract_text_from_pdf(data))
    elif lower.endswith((".txt", ".md")):
        kind = "text"
        text = await anyio.to_thread.run_sync(lambda: extract_text_from_txt(data))
    else:
        assistant_msg = {
            "role": "assistant",
            "ts": _now(),
            "text": "⚠️ Unsupported file type. Please upload: .pdf, .txt, or .md",
            "sources": [],
        }
        _history_append(owner, assistant_msg)
        return JSONResponse({"ok": False, "assistant": assistant_msg})

    if not text.strip():
        assistant_msg = {
            "role": "assistant",
            "ts": _now(),
            "text": f"⚠️ No text extracted from `{filename}`. (If it’s scanned, OCR isn’t enabled yet.)",
            "sources": [],
        }
        _history_append(owner, assistant_msg)
        return JSONResponse({"ok": False, "assistant": assistant_msg})

    prefix = f"{owner}::{filename}" if owner else filename
    chunks = chunk_text(text, prefix=prefix)

    texts = [c.text for c in chunks]
    ids = [c.chunk_id for c in chunks]
    metas = [{"source": filename, "kind": kind, "owner": owner, "start": c.start, "end": c.end} for c in chunks]

    embs = await anyio.to_thread.run_sync(lambda: embed_texts(texts))
    await anyio.to_thread.run_sync(lambda: db.add(ids=ids, documents=texts, metadatas=metas, embeddings=embs))
    logger.info(
        "ingest_file_success owner=%s filename=%s kind=%s chunks=%d duration_ms=%.2f",
        owner,
        filename,
        kind,
        len(chunks),
        (time.perf_counter() - t0) * 1000.0,
    )

    assistant_msg = {
        "role": "assistant",
        "ts": _now(),
        "text": f"✅ Ingested {len(chunks)} chunks from file:\n{filename}",
        "sources": [],
    }
    _history_append(owner, assistant_msg)

    return JSONResponse({"ok": True, "assistant": assistant_msg})


# -----------------------------------------------------------------------------
# Streaming chat endpoint (NDJSON)
# -----------------------------------------------------------------------------
@app.post("/chat/stream")
async def chat_stream(
    request: Request,
    owner: str = Form("default"),
    message: str = Form(...),
    mode: str = Form("study_guide"),
    provider: str = Form("claude_code"),  # openai | claude | claude_code | codex
    model: str = Form(""),
    k: int = Form(8),
    history_turns: int = Form(DEFAULT_HISTORY_TURNS),
    history_enabled: bool = Form(True),
    thinking_enabled: bool = Form(False),
    thinking_level: str = Form("medium"),
    show_thinking: bool = Form(True),
    true_streaming: bool = Form(True),
    selected_library_ids: str = Form(""),
):
    owner = _normalize_owner(owner)
    msg = (message or "").strip()
    if not msg:
        return JSONResponse({"ok": False, "error": "empty_message"}, status_code=400)
    if len(msg) > MAX_CHAT_MESSAGE_CHARS:
        return JSONResponse(
            {"ok": False, "error": "message_too_large", "message": f"Max message length is {MAX_CHAT_MESSAGE_CHARS} characters."},
            status_code=413,
        )
    try:
        mode, provider, k, history_turns = _sanitize_post_params(mode, provider, k, history_turns)
    except ValueError as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)

    requested_provider = provider
    requested_model = (model or "").strip()
    _ensure_owner_loaded(owner)
    thinking_level = normalize_thinking_level(thinking_level)
    resolved_provider, resolved_model, provider_note = _resolve_provider_model_pair(provider, model)
    if resolved_provider != provider:
        logger.info(
            "chat_stream_provider_model_adjusted owner=%s provider=%s->%s model=%s",
            owner,
            provider,
            resolved_provider,
            resolved_model,
        )
    provider = resolved_provider
    model = resolved_model
    logger.info(
        "chat_stream_start owner=%s mode=%s provider=%s model=%s k=%d history_enabled=%s thinking_enabled=%s thinking_level=%s show_thinking=%s true_streaming=%s selected_library_ids=%s",
        owner,
        mode,
        provider,
        model,
        k,
        history_enabled,
        thinking_enabled,
        thinking_level,
        show_thinking,
        true_streaming,
        bool((selected_library_ids or "").strip()),
    )

    async def gen() -> AsyncGenerator[bytes, None]:
        t0 = time.perf_counter()
        user_msg = {"role": "user", "text": msg, "ts": _now()}
        _history_append(owner, user_msg)

        yield _ndjson({"type": "start", "ts": _now()})
        if show_thinking:
            active_model = (model or "").strip() or "auto"
            if requested_provider != provider or ((requested_model or "") != (model or "")):
                yield _ndjson(
                    {
                        "type": "status",
                        "text": (
                            f"Requested provider/model: {requested_provider or 'auto'} / {requested_model or 'auto'} "
                            f"-> using: {provider or 'auto'} / {active_model}"
                        ),
                    }
                )
            else:
                yield _ndjson({"type": "status", "text": f"Using provider/model: {provider or 'auto'} / {active_model}"})
            yield _ndjson({"type": "status", "text": "Retrieving relevant passages…"})
            if using_local_embeddings() and not local_embedding_model_loaded():
                yield _ndjson(
                    {
                        "type": "status",
                        "text": "First run: pulling embeddings model from Hugging Face. This can take a few minutes…",
                    }
                )
            if provider_note:
                yield _ndjson({"type": "status", "text": provider_note})
            if (selected_library_ids or "").strip():
                yield _ndjson({"type": "status", "text": "Adding selected library items as pinned context…"})

        # Retrieve
        try:
            with anyio.fail_after(RETRIEVAL_TIMEOUT_SECONDS):
                q_emb = await anyio.to_thread.run_sync(
                    lambda: embed_texts([msg])[0],
                    abandon_on_cancel=True,
                )
                owner_where = {"owner": owner} if owner else None
                res = await anyio.to_thread.run_sync(
                    lambda: db.query(q_emb, k=k, where=owner_where),
                    abandon_on_cancel=True,
                )
        except TimeoutError:
            err = (
                f"Retrieval timed out after {int(RETRIEVAL_TIMEOUT_SECONDS)}s. "
                "Try a shorter question, re-run once embeddings are warm, or ingest smaller sources."
            )
            logger.exception("chat_stream_retrieval_timeout owner=%s timeout_s=%.1f", owner, RETRIEVAL_TIMEOUT_SECONDS)
            yield _ndjson({"type": "error", "message": err})
            assistant_msg = {"role": "assistant", "text": f"⚠️ {err}", "ts": _now(), "sources": []}
            _history_append(owner, assistant_msg)
            yield _ndjson({"type": "done", "ts": _now()})
            return
        except Exception as e:
            err = f"Retrieval failed: {e}"
            logger.exception("chat_stream_retrieval_failed owner=%s", owner)
            yield _ndjson({"type": "error", "message": err})
            assistant_msg = {"role": "assistant", "text": f"⚠️ {err}", "ts": _now(), "sources": []}
            _history_append(owner, assistant_msg)
            yield _ndjson({"type": "done", "ts": _now()})
            return

        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        ids = res.get("ids", [[]])[0]

        results: List[Dict[str, Any]] = []
        passages: List[Dict[str, Any]] = []
        for i in range(len(docs)):
            if owner and metas[i].get("owner") != owner:
                continue
            results.append({"id": ids[i], "text": docs[i], "meta": metas[i], "distance": float(dists[i])})
            passages.append({"text": docs[i], "meta": metas[i]})

        # Defensive fallback: if owner-filter query returned nothing, retry a global query
        # and apply owner filtering locally. This avoids false negatives on some DB filter paths.
        if owner and not results:
            try:
                with anyio.fail_after(RETRIEVAL_TIMEOUT_SECONDS):
                    res2 = await anyio.to_thread.run_sync(
                        lambda: db.query(q_emb, k=max(k * 3, 24), where=None),
                        abandon_on_cancel=True,
                    )
                docs2 = res2.get("documents", [[]])[0]
                metas2 = res2.get("metadatas", [[]])[0]
                dists2 = res2.get("distances", [[]])[0]
                ids2 = res2.get("ids", [[]])[0]
                for i in range(len(docs2)):
                    if (metas2[i] or {}).get("owner") != owner:
                        continue
                    results.append({"id": ids2[i], "text": docs2[i], "meta": metas2[i], "distance": float(dists2[i])})
                    passages.append({"text": docs2[i], "meta": metas2[i]})
                    if len(results) >= k:
                        break
            except Exception:
                pass

        selected_ids = [
            x.strip()
            for x in (selected_library_ids or "").split(",")
            if x.strip()
        ][:8]
        pinned_passages: List[Dict[str, Any]] = []
        pinned_sources: List[Dict[str, Any]] = []
        if selected_ids:
            for sid in selected_ids:
                item = get_source_item(owner, sid, SOURCE_DIR)
                if not item:
                    continue
                text = read_source_text_blob(str(item.get("text_path") or ""))
                text = (text or "").strip()
                if not text:
                    continue
                pinned_passages.append(
                    {
                        "text": text[:4000],
                        "meta": {
                            "source": item.get("title") or item.get("original_name") or sid,
                            "kind": "library_selected",
                            "owner": owner,
                            "source_id": sid,
                            "url": item.get("url") or "",
                        },
                    }
                )
                pinned_sources.append(
                    {
                        "source": item.get("title") or item.get("original_name") or sid,
                        "kind": "library_selected",
                        "url": item.get("url") or "",
                        "distance": None,
                        "preview": text[:240],
                    }
                )

        if not results and not pinned_passages:
            logger.info(
                "chat_stream_no_results owner=%s k=%d duration_ms=%.2f",
                owner,
                k,
                (time.perf_counter() - t0) * 1000.0,
            )
            assistant_text = "I couldn’t find anything in your ingested materials for that yet. Try ingesting a PDF/URL first."
            history_for_breaks = _history_snapshot(owner)
            add_break = should_add_break_reminder(history_for_breaks)
            if add_break:
                reminder = build_break_reminder(history_for_breaks)
                assistant_text = f"{assistant_text}\n\n{reminder}"
            assistant_msg = {
                "role": "assistant",
                "text": assistant_text,
                "ts": _now(),
                "sources": [],
                "break_reminder": add_break,
            }
            _history_append(owner, assistant_msg)

            yield _ndjson({"type": "delta", "text": assistant_text})
            yield _ndjson({"type": "done", "ts": _now()})
            return

        sources_payload = [
            {
                "source": (r["meta"] or {}).get("source"),
                "kind": (r["meta"] or {}).get("kind"),
                "url": (r["meta"] or {}).get("url"),
                "distance": r["distance"],
                "preview": (r["text"] or "")[:240],
            }
            for r in results[:5]
        ]
        sources_payload.extend(pinned_sources[:5])
        yield _ndjson({"type": "sources", "sources": sources_payload})

        # History-aware prompt
        history_before_response = _history_snapshot(owner)
        convo = _build_history_block(history_before_response[:-1], history_turns) if history_enabled else ""
        query_for_prompt = f"Conversation so far:\n{convo}\n\nNew question:\n{msg}" if convo else msg
        try:
            study_style = get_global_style()
        except Exception:
            logger.exception("chat_stream_style_load_failed owner=%s", owner)
            study_style = ""
        logger.info("chat_stream_style owner=%s style_chars=%d", owner, len(study_style))
        all_passages = pinned_passages + passages
        prompt = build_prompt(mode=mode, query=query_for_prompt, passages=all_passages, study_style=study_style)

        if show_thinking:
            thinking_label = "Thinking…"
            if thinking_enabled and provider == "claude":
                thinking_label = f"Thinking… (Claude level: {thinking_level.replace('_', ' ')})"
            yield _ndjson({"type": "status", "text": thinking_label})

        assistant_text = ""

        # True streaming where possible, fallback otherwise
        try:
            p = (provider or "").strip().lower()

            if p == "openai":
                m = (model or DEFAULT_OPENAI_MODEL).strip()
                if true_streaming:
                    try:
                        async for delta in _stream_openai(prompt, m):
                            assistant_text += delta
                            yield _ndjson({"type": "delta", "text": delta})
                    except Exception:
                        text = await _run_generate_resilient(
                            request,
                            p,
                            prompt,
                            m,
                            thinking_enabled=thinking_enabled,
                            thinking_level=thinking_level,
                        )
                        assistant_text = text
                        yield _ndjson({"type": "delta", "text": assistant_text})
                else:
                    text = await _run_generate_resilient(
                        request,
                        p,
                        prompt,
                        m,
                        thinking_enabled=thinking_enabled,
                        thinking_level=thinking_level,
                    )
                    assistant_text = text
                    for i in range(0, len(assistant_text), 80):
                        yield _ndjson({"type": "delta", "text": assistant_text[i : i + 80]})
                        await asyncio.sleep(0.01)

            elif p == "claude":
                m = (model or DEFAULT_CLAUDE_MODEL).strip()
                if true_streaming:
                    try:
                        async for delta in _stream_anthropic(
                            prompt,
                            m,
                            thinking_enabled=thinking_enabled,
                            thinking_level=thinking_level,
                        ):
                            assistant_text += delta
                            yield _ndjson({"type": "delta", "text": delta})
                    except Exception:
                        text = await _run_generate_resilient(
                            request,
                            p,
                            prompt,
                            m,
                            thinking_enabled=thinking_enabled,
                            thinking_level=thinking_level,
                        )
                        assistant_text = text
                        yield _ndjson({"type": "delta", "text": assistant_text})
                else:
                    text = await _run_generate_resilient(
                        request,
                        p,
                        prompt,
                        m,
                        thinking_enabled=thinking_enabled,
                        thinking_level=thinking_level,
                    )
                    assistant_text = text
                    for i in range(0, len(assistant_text), 80):
                        yield _ndjson({"type": "delta", "text": assistant_text[i : i + 80]})
                        await asyncio.sleep(0.01)

            elif p == "codex":
                m = (model or DEFAULT_CODEX_MODEL).strip()
                if not codex_cli_available() and os.getenv("OPENAI_API_KEY", "").strip():
                    if show_thinking:
                        yield _ndjson({"type": "status", "text": "Codex CLI unavailable; using OpenAI API fallback…"})
                    p = "openai"
                    if true_streaming:
                        try:
                            async for delta in _stream_openai(prompt, m):
                                assistant_text += delta
                                yield _ndjson({"type": "delta", "text": delta})
                        except Exception:
                            text = await _run_generate_resilient(
                                request,
                                p,
                                prompt,
                                m,
                                thinking_enabled=thinking_enabled,
                                thinking_level=thinking_level,
                            )
                            assistant_text = text
                            yield _ndjson({"type": "delta", "text": assistant_text})
                    else:
                        text = await _run_generate_resilient(
                            request,
                            p,
                            prompt,
                            m,
                            thinking_enabled=thinking_enabled,
                            thinking_level=thinking_level,
                        )
                        assistant_text = text
                        for i in range(0, len(assistant_text), 80):
                            yield _ndjson({"type": "delta", "text": assistant_text[i : i + 80]})
                            await asyncio.sleep(0.01)
                else:
                    if true_streaming:
                        try:
                            async for delta in _stream_codex(prompt, m):
                                assistant_text += delta
                                yield _ndjson({"type": "delta", "text": delta})
                        except Exception:
                            text = await _run_generate_resilient(
                                request,
                                p,
                                prompt,
                                m,
                                thinking_enabled=thinking_enabled,
                                thinking_level=thinking_level,
                            )
                            assistant_text = text
                            yield _ndjson({"type": "delta", "text": assistant_text})
                    else:
                        text = await _run_generate_resilient(
                            request,
                            p,
                            prompt,
                            m,
                            thinking_enabled=thinking_enabled,
                            thinking_level=thinking_level,
                        )
                        assistant_text = text
                        for i in range(0, len(assistant_text), 80):
                            yield _ndjson({"type": "delta", "text": assistant_text[i : i + 80]})
                            await asyncio.sleep(0.01)

            elif p == "claude_code":
                m = (model or DEFAULT_CLAUDE_MODEL).strip()
                if not claude_code_cli_available() and os.getenv("ANTHROPIC_API_KEY", "").strip():
                    if show_thinking:
                        yield _ndjson({"type": "status", "text": "Claude Code unavailable; using Claude API fallback…"})
                    if true_streaming:
                        try:
                            async for delta in _stream_anthropic(
                                prompt,
                                m,
                                thinking_enabled=thinking_enabled,
                                thinking_level=thinking_level,
                            ):
                                assistant_text += delta
                                yield _ndjson({"type": "delta", "text": delta})
                        except Exception:
                            text = await _run_generate_resilient(
                                request,
                                p,
                                prompt,
                                m,
                                thinking_enabled=thinking_enabled,
                                thinking_level=thinking_level,
                            )
                            assistant_text = text
                            yield _ndjson({"type": "delta", "text": assistant_text})
                    else:
                        text = await _run_generate_resilient(
                            request,
                            p,
                            prompt,
                            m,
                            thinking_enabled=thinking_enabled,
                            thinking_level=thinking_level,
                        )
                        assistant_text = text
                        for i in range(0, len(assistant_text), 80):
                            yield _ndjson({"type": "delta", "text": assistant_text[i : i + 80]})
                            await asyncio.sleep(0.01)
                else:
                    text = await _run_generate_resilient(
                        request,
                        p,
                        prompt,
                        m,
                        thinking_enabled=thinking_enabled,
                        thinking_level=thinking_level,
                    )
                    assistant_text = text
                    for i in range(0, len(assistant_text), 80):
                        yield _ndjson({"type": "delta", "text": assistant_text[i : i + 80]})
                        await asyncio.sleep(0.01)

            else:
                # Others: fallback (UI will show chunked streaming)
                m = (model or DEFAULT_CLAUDE_MODEL).strip()
                text = await _run_generate_resilient(
                    request,
                    p,
                    prompt,
                    m,
                    thinking_enabled=thinking_enabled,
                    thinking_level=thinking_level,
                )
                assistant_text = text
                for i in range(0, len(assistant_text), 80):
                    yield _ndjson({"type": "delta", "text": assistant_text[i : i + 80]})
                    await asyncio.sleep(0.01)

        except asyncio.CancelledError:
            logger.info("chat_stream_client_disconnected owner=%s provider=%s", owner, provider)
            return
        except Exception as e:
            logger.exception("chat_stream_generation_failed owner=%s provider=%s", owner, provider)
            context_answer = _context_only_answer(msg, results)
            active_model = (model or "").strip() or "auto"
            assistant_text = (
                f"⚠️ Generation failed ({e}).\n\n"
                f"Requested provider/model: {requested_provider or 'auto'} / {requested_model or 'auto'}\n"
                f"Active provider/model: {provider or 'auto'} / {active_model}\n\n"
                f"{context_answer}"
            )
            yield _ndjson({"type": "delta", "text": assistant_text})

        if not (assistant_text or "").strip():
            logger.warning("chat_stream_generation_empty owner=%s provider=%s", owner, provider)
            active_model = (model or "").strip() or "auto"
            assistant_text = (
                "⚠️ Generation returned empty output from the selected provider.\n\n"
                f"Requested provider/model: {requested_provider or 'auto'} / {requested_model or 'auto'}\n"
                f"Active provider/model: {provider or 'auto'} / {active_model}\n\n"
                f"{_context_only_answer(msg, results)}"
            )
            yield _ndjson({"type": "delta", "text": assistant_text})

        assistant_msg = {
            "role": "assistant",
            "text": assistant_text,
            "ts": _now(),
            "sources": sources_payload,
            "mode": mode,
            "provider": provider,
            "model": model,
        }
        history_for_breaks = _history_snapshot(owner)
        add_break = should_add_break_reminder(history_for_breaks)
        if add_break:
            reminder = build_break_reminder(history_for_breaks)
            assistant_msg["text"] = f"{assistant_text}\n\n{reminder}"
            assistant_msg["break_reminder"] = True
            yield _ndjson({"type": "delta", "text": f"\n\n{reminder}"})
        _history_append(owner, assistant_msg)
        logger.info(
            "chat_stream_done owner=%s passages=%d response_chars=%d duration_ms=%.2f",
            owner,
            len(passages),
            len(assistant_text or ""),
            (time.perf_counter() - t0) * 1000.0,
        )

        yield _ndjson({"type": "done", "ts": _now()})

    return StreamingResponse(gen(), media_type="application/x-ndjson")
