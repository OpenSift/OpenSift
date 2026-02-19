from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import os
import re
import secrets
import threading
import time
from contextlib import asynccontextmanager
from urllib.parse import urlparse
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Deque, DefaultDict, Dict, List, Optional, Tuple
from uuid import uuid4

import anyio
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.status import HTTP_303_SEE_OTHER

from app.chunking import chunk_text
from app.ingest import extract_text_from_pdf, extract_text_from_txt, fetch_url_text
from app.llm import embed_texts
from app.atomic_io import atomic_write_json, path_lock
from app.logging_utils import configure_logging
from app.providers import (
    codex_auth_detected,
    DEFAULT_CLAUDE_MODEL,
    DEFAULT_OPENAI_MODEL,
    SUPPORTED_CLAUDE_MODELS,
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

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------
os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)

AUTH_STATE_PATH = os.path.join(os.getcwd(), ".opensift_auth.json")
OPENSIFT_VERSION = "1.2.0-alpha"

logger = configure_logging("opensift.ui")


@asynccontextmanager
async def _app_lifespan(_app: FastAPI):
    await _print_startup_token()
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
ALLOWED_HOSTS = {"127.0.0.1", "localhost", "::1", "[::1]"}


class LocalhostOnlyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        client_host = request.client.host if request.client else ""
        host_header = request.headers.get("host", "")
        host_only = host_header.split(":")[0].strip().lower()

        is_loopback_ip = client_host in ("127.0.0.1", "::1")
        is_allowed_host = host_only in ALLOWED_HOSTS

        if not (is_loopback_ip and is_allowed_host):
            return PlainTextResponse(
                "OpenSift is configured for localhost-only access.",
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


async def _stream_anthropic(prompt: str, model: str, thinking_enabled: bool = False) -> AsyncGenerator[str, None]:
    import anthropic  # type: ignore

    client = anthropic.Anthropic()
    params: Dict[str, Any] = {
        "model": model,
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": prompt}],
    }
    if thinking_enabled:
        params["thinking"] = {"type": "enabled", "budget_tokens": 2048}

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


def _run_generate(provider: str, prompt: str, model: str, thinking_enabled: bool = False) -> str:
    provider = (provider or "").strip().lower()
    model = (model or "").strip()

    if provider == "openai":
        m = model or DEFAULT_OPENAI_MODEL
        return generate_with_openai(prompt, model=m)

    if provider == "claude":
        m = model or DEFAULT_CLAUDE_MODEL
        return generate_with_claude(prompt, model=m, thinking_enabled=thinking_enabled)

    if provider == "claude_code":
        # Prefer explicit model; otherwise default to the configured Claude default.
        m = model or DEFAULT_CLAUDE_MODEL
        return generate_with_claude_code(prompt, model=m)

    if provider == "codex":
        m = model or DEFAULT_OPENAI_MODEL
        return generate_with_codex(prompt, model=m)

    raise RuntimeError(f"Unknown provider: {provider}")


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
    response = templates.TemplateResponse(
        "chat.html",
        {
            "request": request,
            "owner": owner,
            "history": history,
            "csrf_token": csrf_token,
            "default_claude_model": DEFAULT_CLAUDE_MODEL,
            "supported_claude_models": SUPPORTED_CLAUDE_MODELS,
        },
    )
    _set_csrf_cookie(response, request, csrf_token)
    return response


@app.get("/chat", response_class=HTMLResponse)
async def chat_page(request: Request, owner: str = "default"):
    owner = _normalize_owner(owner)
    history = _history_snapshot(owner)
    csrf_token = _csrf_token_for_request(request)
    response = templates.TemplateResponse(
        "chat.html",
        {
            "request": request,
            "owner": owner,
            "history": history,
            "csrf_token": csrf_token,
            "default_claude_model": DEFAULT_CLAUDE_MODEL,
            "supported_claude_models": SUPPORTED_CLAUDE_MODELS,
        },
    )
    _set_csrf_cookie(response, request, csrf_token)
    return response


@app.post("/chat/clear")
async def chat_clear(owner: str = Form("default")):
    owner = _normalize_owner(owner)
    _history_clear(owner)
    return JSONResponse({"ok": True})


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
async def session_delete(owner: str = Form(...)):
    owner = _normalize_owner(owner)
    if not owner:
        return JSONResponse({"ok": False, "error": "owner_required"}, status_code=400)
    _history_delete_owner(owner)
    ok = delete_session(owner, SESSION_DIR)
    if not ok:
        return JSONResponse({"ok": False, "error": "not_found"}, status_code=404)
    return JSONResponse({"ok": True, "owner": owner})


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
    owner: str = Form("default"),
    message: str = Form(...),
    mode: str = Form("study_guide"),
    provider: str = Form("claude_code"),  # openai | claude | claude_code | codex
    model: str = Form(""),
    k: int = Form(8),
    history_turns: int = Form(DEFAULT_HISTORY_TURNS),
    history_enabled: bool = Form(True),
    thinking_enabled: bool = Form(False),
    show_thinking: bool = Form(True),
    true_streaming: bool = Form(True),
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

    _ensure_owner_loaded(owner)
    logger.info(
        "chat_stream_start owner=%s mode=%s provider=%s model=%s k=%d history_enabled=%s thinking_enabled=%s show_thinking=%s true_streaming=%s",
        owner,
        mode,
        provider,
        model,
        k,
        history_enabled,
        thinking_enabled,
        show_thinking,
        true_streaming,
    )

    async def gen() -> AsyncGenerator[bytes, None]:
        t0 = time.perf_counter()
        user_msg = {"role": "user", "text": msg, "ts": _now()}
        _history_append(owner, user_msg)

        yield _ndjson({"type": "start", "ts": _now()})
        if show_thinking:
            yield _ndjson({"type": "status", "text": "Retrieving relevant passages…"})

        # Retrieve
        try:
            q_emb = await anyio.to_thread.run_sync(lambda: embed_texts([msg])[0])
            owner_where = {"owner": owner} if owner else None
            res = await anyio.to_thread.run_sync(lambda: db.query(q_emb, k=k, where=owner_where))
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
                res2 = await anyio.to_thread.run_sync(lambda: db.query(q_emb, k=max(k * 3, 24), where=None))
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

        if not results:
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
        yield _ndjson({"type": "sources", "sources": sources_payload})

        # History-aware prompt
        history_before_response = _history_snapshot(owner)
        convo = _build_history_block(history_before_response[:-1], history_turns) if history_enabled else ""
        query_for_prompt = f"Conversation so far:\n{convo}\n\nNew question:\n{msg}" if convo else msg
        study_style = get_global_style()
        logger.info("chat_stream_style owner=%s style_chars=%d", owner, len(study_style))
        prompt = build_prompt(mode=mode, query=query_for_prompt, passages=passages, study_style=study_style)

        if show_thinking:
            thinking_label = "Thinking…"
            if thinking_enabled and provider == "claude":
                thinking_label = "Thinking… (Claude extended thinking enabled)"
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
                        text = await anyio.to_thread.run_sync(lambda: _run_generate(p, prompt, m))
                        assistant_text = text
                        yield _ndjson({"type": "delta", "text": assistant_text})
                else:
                    text = await anyio.to_thread.run_sync(lambda: _run_generate(p, prompt, m))
                    assistant_text = text
                    for i in range(0, len(assistant_text), 80):
                        yield _ndjson({"type": "delta", "text": assistant_text[i : i + 80]})
                        await asyncio.sleep(0.01)

            elif p == "claude":
                m = (model or DEFAULT_CLAUDE_MODEL).strip()
                if true_streaming:
                    try:
                        async for delta in _stream_anthropic(prompt, m, thinking_enabled=thinking_enabled):
                            assistant_text += delta
                            yield _ndjson({"type": "delta", "text": delta})
                    except Exception:
                        text = await anyio.to_thread.run_sync(
                            lambda: _run_generate(p, prompt, m, thinking_enabled=thinking_enabled)
                        )
                        assistant_text = text
                        yield _ndjson({"type": "delta", "text": assistant_text})
                else:
                    text = await anyio.to_thread.run_sync(
                        lambda: _run_generate(p, prompt, m, thinking_enabled=thinking_enabled)
                    )
                    assistant_text = text
                    for i in range(0, len(assistant_text), 80):
                        yield _ndjson({"type": "delta", "text": assistant_text[i : i + 80]})
                        await asyncio.sleep(0.01)

            elif p == "codex":
                m = (model or DEFAULT_OPENAI_MODEL).strip()
                if true_streaming:
                    try:
                        async for delta in _stream_codex(prompt, m):
                            assistant_text += delta
                            yield _ndjson({"type": "delta", "text": delta})
                    except Exception:
                        text = await anyio.to_thread.run_sync(lambda: _run_generate(p, prompt, m))
                        assistant_text = text
                        yield _ndjson({"type": "delta", "text": assistant_text})
                else:
                    text = await anyio.to_thread.run_sync(lambda: _run_generate(p, prompt, m))
                    assistant_text = text
                    for i in range(0, len(assistant_text), 80):
                        yield _ndjson({"type": "delta", "text": assistant_text[i : i + 80]})
                        await asyncio.sleep(0.01)

            else:
                # claude_code or others: fallback (UI will show chunked streaming)
                m = (model or DEFAULT_CLAUDE_MODEL).strip()
                text = await anyio.to_thread.run_sync(lambda: _run_generate(p, prompt, m))
                assistant_text = text
                for i in range(0, len(assistant_text), 80):
                    yield _ndjson({"type": "delta", "text": assistant_text[i : i + 80]})
                    await asyncio.sleep(0.01)

        except Exception as e:
            logger.exception("chat_stream_generation_failed owner=%s provider=%s", owner, provider)
            bullets = "\n".join([f"- {(r['text'] or '')[:240].strip()}…" for r in results[:3]])
            assistant_text = (
                f"⚠️ Generation failed ({e}).\n\n"
                "Here are the most relevant passages I found:\n\n"
                f"{bullets}\n\n"
                "If you set a provider (OpenAI/Claude/Claude Code/Codex), I can generate study guides/quizzes."
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
