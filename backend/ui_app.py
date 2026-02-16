from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from collections import defaultdict, deque
from datetime import datetime
from typing import Any, AsyncGenerator, Deque, DefaultDict, Dict, List, Optional, Tuple

import anyio
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import StreamingResponse
from starlette.status import HTTP_303_SEE_OTHER

from app.chunking import chunk_text
from app.ingest import extract_text_from_pdf, extract_text_from_txt, fetch_url_text
from app.llm import embed_texts
from app.providers import (
    build_prompt,
    generate_with_claude,
    generate_with_claude_code,
    generate_with_openai,
)
from app.vectordb import VectorDB

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------
os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)

AUTH_STATE_PATH = os.path.join(os.getcwd(), ".opensift_auth.json")

app = FastAPI(title="OpenSift UI", version="0.5.2")

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

db = VectorDB()

CHAT_HISTORY: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)


def _now() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _ndjson(obj: Dict[str, Any]) -> bytes:
    return (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")


# -----------------------------------------------------------------------------
# Localhost-only middleware (defense-in-depth)
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

# -----------------------------------------------------------------------------
# Rate limiting (in-memory sliding window)
# -----------------------------------------------------------------------------
# Tweak these freely:
RL_WINDOW_SECONDS = 60

RL_CHAT_STREAM_PER_WINDOW = 30   # /chat/stream
RL_CHAT_INGEST_PER_WINDOW = 12   # /chat/ingest/*
RL_CHAT_OTHER_PER_WINDOW = 30    # other /chat/* POSTs
RL_LOGIN_PER_WINDOW = 12         # /login POST and /set-password POST

# Store timestamps for (client, bucket)
_RL: Dict[Tuple[str, str], Deque[float]] = {}
_RL_MAX_KEYS = 2048  # safety cap


def _rl_bucket(path: str, method: str) -> Optional[Tuple[str, int]]:
    """
    Returns (bucket_name, limit) or None if not rate-limited.
    """
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


def _rl_prune_global(now: float) -> None:
    # Prevent unbounded growth in case of pathological usage
    if len(_RL) <= _RL_MAX_KEYS:
        return
    # Drop oldest buckets quickly (simple heuristic)
    keys = list(_RL.keys())[: len(_RL) - _RL_MAX_KEYS]
    for k in keys:
        _RL.pop(k, None)


def _rl_check(client: str, bucket: str, limit: int, now: float) -> Tuple[bool, int, int]:
    """
    Returns (allowed, remaining, retry_after_seconds)
    """
    key = (client, bucket)
    q = _RL.get(key)
    if q is None:
        q = deque()
        _RL[key] = q

    # Evict old timestamps
    cutoff = now - RL_WINDOW_SECONDS
    while q and q[0] < cutoff:
        q.popleft()

    if len(q) >= limit:
        # retry-after until the oldest request falls out of window
        retry_after = max(1, int((q[0] + RL_WINDOW_SECONDS) - now))
        return (False, 0, retry_after)

    q.append(now)
    remaining = max(0, limit - len(q))
    return (True, remaining, 0)


def _is_api_path(path: str) -> bool:
    # Used for JSON error responses
    return path.startswith("/chat/")


class RateLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        method = request.method.upper()
        b = _rl_bucket(path, method)
        if not b:
            return await call_next(request)

        bucket_name, limit = b

        # Localhost-only means client will be loopback, but keep keying consistent anyway.
        client_ip = request.client.host if request.client else "unknown"
        now = time.time()

        _rl_prune_global(now)
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
                return JSONResponse({"ok": False, "error": "rate_limited", "message": msg}, status_code=429, headers=headers)
            return PlainTextResponse(msg, status_code=429, headers=headers)

        # Add some helpful headers for debugging
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
    with open(AUTH_STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


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


EXEMPT_PATHS = {
    "/login",
    "/set-password",
    "/health",
}
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


@app.on_event("startup")
async def _print_startup_token():
    print("\n" + "=" * 72)
    print("OpenSift Local Auth")
    print(f"Generated login token (valid until restart): {GEN_TOKEN}")
    print(f"Password set: {'YES' if _has_password() else 'NO'}")
    print("=" * 72 + "\n")


# -----------------------------------------------------------------------------
# Auth pages
# -----------------------------------------------------------------------------
@app.get("/health")
async def health():
    return JSONResponse({"ok": True, "time": _now()})


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse(
        "login.html",
        {
            "request": request,
            "mode": "login",
            "has_password": _has_password(),
            "token": GEN_TOKEN,
            "error": None,
        },
    )


@app.post("/login")
async def login_submit(
    request: Request,
    password: str = Form(""),
    token: str = Form(""),
):
    token = (token or "").strip()
    password = (password or "").strip()

    ok = False
    if token and secrets.compare_digest(token, GEN_TOKEN):
        ok = True
    elif password and _verify_password(password):
        ok = True

    if not ok:
        return templates.TemplateResponse(
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

    resp = RedirectResponse(url="/chat", status_code=HTTP_303_SEE_OTHER)
    resp.set_cookie(
        AUTH_COOKIE_NAME,
        _make_session_cookie(),
        httponly=True,
        samesite="lax",
        secure=False,
        max_age=AUTH_TTL_SECONDS,
    )
    return resp


@app.get("/set-password", response_class=HTMLResponse)
async def set_password_page(request: Request):
    return templates.TemplateResponse(
        "login.html",
        {
            "request": request,
            "mode": "set_password",
            "has_password": _has_password(),
            "token": GEN_TOKEN,
            "error": None,
        },
    )


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
        return templates.TemplateResponse(
            "login.html",
            {
                "request": request,
                "mode": "set_password",
                "has_password": _has_password(),
                "token": GEN_TOKEN,
                "error": "Invalid token. Copy/paste the generated token exactly.",
            },
            status_code=401,
        )

    if len(new_password) < 8:
        return templates.TemplateResponse(
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

    if new_password != confirm_password:
        return templates.TemplateResponse(
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

    _set_password(new_password)

    resp = RedirectResponse(url="/chat", status_code=HTTP_303_SEE_OTHER)
    resp.set_cookie(
        AUTH_COOKIE_NAME,
        _make_session_cookie(),
        httponly=True,
        samesite="lax",
        secure=False,
        max_age=AUTH_TTL_SECONDS,
    )
    return resp


@app.post("/logout")
async def logout():
    resp = RedirectResponse(url="/login", status_code=HTTP_303_SEE_OTHER)
    resp.delete_cookie(AUTH_COOKIE_NAME)
    return resp


# -----------------------------------------------------------------------------
# UI pages
# -----------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def root(request: Request, owner: str = "default"):
    return templates.TemplateResponse(
        "chat.html",
        {"request": request, "owner": owner, "history": CHAT_HISTORY[owner]},
    )


@app.get("/chat", response_class=HTMLResponse)
async def chat_page(request: Request, owner: str = "default"):
    return templates.TemplateResponse(
        "chat.html",
        {"request": request, "owner": owner, "history": CHAT_HISTORY[owner]},
    )


@app.post("/chat/clear")
async def chat_clear(owner: str = Form("default")):
    CHAT_HISTORY[owner].clear()
    return JSONResponse({"ok": True})


# -----------------------------------------------------------------------------
# Ingest endpoints (chat-first)
# -----------------------------------------------------------------------------
@app.post("/chat/ingest/url")
async def chat_ingest_url(
    owner: str = Form("default"),
    url: str = Form(...),
    source_title: str = Form(""),
):
    title, text = await fetch_url_text(url)
    source = source_title.strip() or title
    prefix = f"{owner}::{source}" if owner else source

    chunks = chunk_text(text, prefix=prefix)
    texts = [c.text for c in chunks]
    ids = [c.chunk_id for c in chunks]
    metas = [
        {"source": source, "kind": "url", "url": url, "owner": owner, "start": c.start, "end": c.end}
        for c in chunks
    ]

    embs = embed_texts(texts)
    db.add(ids=ids, documents=texts, metadatas=metas, embeddings=embs)

    assistant_msg = {
        "role": "assistant",
        "ts": _now(),
        "text": f"✅ Ingested {len(chunks)} chunks from URL:\n{source}\n{url}",
        "sources": [],
    }
    CHAT_HISTORY[owner].append(assistant_msg)

    return JSONResponse({"ok": True, "assistant": assistant_msg})


@app.post("/chat/ingest/file")
async def chat_ingest_file(
    owner: str = Form("default"),
    file: UploadFile = File(...),
):
    data = await file.read()
    filename = file.filename or "upload"

    lower = filename.lower()
    if lower.endswith(".pdf"):
        kind = "pdf"
        text = extract_text_from_pdf(data)
    elif lower.endswith((".txt", ".md")):
        kind = "text"
        text = extract_text_from_txt(data)
    else:
        assistant_msg = {
            "role": "assistant",
            "ts": _now(),
            "text": "⚠️ Unsupported file type. Please upload: .pdf, .txt, or .md",
            "sources": [],
        }
        CHAT_HISTORY[owner].append(assistant_msg)
        return JSONResponse({"ok": False, "assistant": assistant_msg})

    if not text.strip():
        assistant_msg = {
            "role": "assistant",
            "ts": _now(),
            "text": f"⚠️ No text extracted from `{filename}`. (If it’s scanned, OCR isn’t enabled yet.)",
            "sources": [],
        }
        CHAT_HISTORY[owner].append(assistant_msg)
        return JSONResponse({"ok": False, "assistant": assistant_msg})

    prefix = f"{owner}::{filename}" if owner else filename
    chunks = chunk_text(text, prefix=prefix)

    texts = [c.text for c in chunks]
    ids = [c.chunk_id for c in chunks]
    metas = [
        {"source": filename, "kind": kind, "owner": owner, "start": c.start, "end": c.end}
        for c in chunks
    ]

    embs = embed_texts(texts)
    db.add(ids=ids, documents=texts, metadatas=metas, embeddings=embs)

    assistant_msg = {
        "role": "assistant",
        "ts": _now(),
        "text": f"✅ Ingested {len(chunks)} chunks from file:\n{filename}",
        "sources": [],
    }
    CHAT_HISTORY[owner].append(assistant_msg)

    return JSONResponse({"ok": True, "assistant": assistant_msg})


# -----------------------------------------------------------------------------
# Streaming chat endpoint (NDJSON)
# -----------------------------------------------------------------------------
@app.post("/chat/stream")
async def chat_stream(
    owner: str = Form("default"),
    message: str = Form(...),
    mode: str = Form("study_guide"),
    provider: str = Form("claude_code"),  # openai | claude | claude_code
    model: str = Form(""),
    k: int = Form(8),
):
    async def gen() -> AsyncGenerator[bytes, None]:
        user_msg = {"role": "user", "text": message, "ts": _now()}
        CHAT_HISTORY[owner].append(user_msg)

        yield _ndjson({"type": "start", "ts": _now()})
        yield _ndjson({"type": "status", "text": "Retrieving relevant passages…"})

        try:
            q_emb = await anyio.to_thread.run_sync(lambda: embed_texts([message])[0])
            res = await anyio.to_thread.run_sync(lambda: db.query(q_emb, k=int(k)))
        except Exception as e:
            yield _ndjson({"type": "error", "message": f"Retrieval failed: {e}"})
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
            results.append(
                {"id": ids[i], "text": docs[i], "meta": metas[i], "distance": float(dists[i])}
            )
            passages.append({"text": docs[i], "meta": metas[i]})

        if not results:
            assistant_text = (
                "I couldn’t find anything in your ingested materials for that yet. "
                "Try ingesting a PDF/URL first, or ask with different keywords."
            )
            assistant_msg = {"role": "assistant", "text": assistant_text, "ts": _now(), "sources": []}
            CHAT_HISTORY[owner].append(assistant_msg)

            yield _ndjson({"type": "delta", "text": assistant_text})
            yield _ndjson({"type": "done", "ts": _now()})
            return

        sources_payload = [
            {
                "source": r["meta"].get("source"),
                "kind": r["meta"].get("kind"),
                "url": r["meta"].get("url"),
                "distance": r["distance"],
                "preview": r["text"][:240],
            }
            for r in results[:5]
        ]
        yield _ndjson({"type": "sources", "sources": sources_payload})

        prompt = build_prompt(mode=mode, query=message, passages=passages)
        yield _ndjson({"type": "status", "text": "Thinking…"})

        assistant_text = ""
        gen_error: Optional[str] = None

        def _run_generate() -> str:
            if provider == "openai":
                return generate_with_openai(prompt, model=model or "gpt-4.1-mini")
            if provider == "claude":
                return generate_with_claude(prompt, model=model or "claude-3-5-sonnet-latest")
            if provider == "claude_code":
                return generate_with_claude_code(prompt, model=model or None)
            raise RuntimeError(f"Unknown provider: {provider}")

        try:
            assistant_text = await anyio.to_thread.run_sync(_run_generate)
        except Exception as e:
            gen_error = str(e)

        if gen_error:
            top = results[:3]
            bullets = "\n".join([f"- {r['text'][:240].strip()}…" for r in top])
            assistant_text = (
                "I can’t generate right now (provider/auth issue), but here are the most relevant notes I found:\n\n"
                f"{bullets}\n\n"
                "If you set a provider (OpenAI/Claude/Claude Code), I can turn this into a study guide/quiz."
            )

        chunk_size = 60
        for i in range(0, len(assistant_text), chunk_size):
            yield _ndjson({"type": "delta", "text": assistant_text[i : i + chunk_size]})
            await asyncio.sleep(0.01)

        assistant_msg = {
            "role": "assistant",
            "text": assistant_text,
            "ts": _now(),
            "sources": sources_payload,
        }
        CHAT_HISTORY[owner].append(assistant_msg)

        yield _ndjson({"type": "done", "ts": _now()})

    return StreamingResponse(gen(), media_type="application/x-ndjson")