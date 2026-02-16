from __future__ import annotations

import asyncio
import json
import os
from collections import defaultdict
from datetime import datetime
from typing import Any, AsyncGenerator, DefaultDict, Dict, List, Optional

import anyio
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import StreamingResponse

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

# Ensure UI can mount even if empty
os.makedirs("static", exist_ok=True)

app = FastAPI(title="OpenSift UI", version="0.4.1")

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

db = VectorDB()

# In-memory chat history per owner namespace
CHAT_HISTORY: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)


def _now() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _ndjson(obj: Dict[str, Any]) -> bytes:
    return (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")


# -------------------------
# Localhost-only security
# -------------------------

ALLOWED_HOSTS = {"127.0.0.1", "localhost", "[::1]", "::1"}


class LocalhostOnlyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Client IP as seen by uvicorn/starlette
        client_host = request.client.host if request.client else ""
        # Host header (may include port)
        host_header = request.headers.get("host", "")
        host_only = host_header.split(":")[0].strip().lower()

        # Allow only loopback IPs AND localhost-ish Host headers.
        # This prevents accidental exposure if someone runs with --host 0.0.0.0.
        is_loopback_ip = client_host in ("127.0.0.1", "::1")
        is_allowed_host = host_only in ALLOWED_HOSTS

        if not (is_loopback_ip and is_allowed_host):
            return PlainTextResponse(
                "OpenSift UI is configured for localhost-only access.",
                status_code=403,
            )

        return await call_next(request)


app.add_middleware(LocalhostOnlyMiddleware)


# -------------------------
# Chatbot UI (chat.html)
# -------------------------

@app.get("/", response_class=HTMLResponse)
async def root_chat(request: Request, owner: str = "default"):
    # Chat-first UI
    return templates.TemplateResponse(
        "chat.html",
        {
            "request": request,
            "owner": owner,
            "history": CHAT_HISTORY[owner],
        },
    )


@app.get("/chat", response_class=HTMLResponse)
async def chat_page(request: Request, owner: str = "default"):
    return templates.TemplateResponse(
        "chat.html",
        {
            "request": request,
            "owner": owner,
            "history": CHAT_HISTORY[owner],
        },
    )


@app.post("/chat/clear")
async def chat_clear(owner: str = Form("default")):
    CHAT_HISTORY[owner].clear()
    return JSONResponse({"ok": True})


# -------------------------
# Chat-first ingestion endpoints
# -------------------------

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


# -------------------------
# Streaming chat endpoint (NDJSON)
# -------------------------

@app.post("/chat/stream")
async def chat_stream(
    owner: str = Form("default"),
    message: str = Form(...),
    mode: str = Form("study_guide"),
    provider: str = Form("claude_code"),  # openai | claude | claude_code
    model: str = Form(""),
    k: int = Form(8),
):
    """
    Streams assistant response as NDJSON lines.
    Each line is a JSON object: {type: "...", ...}

    Types emitted:
      - start: {type, ts}
      - status: {type, text}
      - sources: {type, sources: [...]}
      - delta: {type, text}      (incremental assistant text)
      - done: {type, ts}
      - error: {type, message}
    """

    async def gen() -> AsyncGenerator[bytes, None]:
        user_ts = _now()
        user_msg = {"role": "user", "text": message, "ts": user_ts}
        CHAT_HISTORY[owner].append(user_msg)

        # Start
        yield _ndjson({"type": "start", "ts": _now()})
        yield _ndjson({"type": "status", "text": "Retrieving relevant passages…"})

        # Retrieve passages for grounding
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
            item = {"id": ids[i], "text": docs[i], "meta": metas[i], "distance": float(dists[i])}
            results.append(item)
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

        # Pseudo-streaming (chunked)
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