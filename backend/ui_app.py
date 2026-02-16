from __future__ import annotations

import os
from collections import defaultdict
from datetime import datetime
from typing import Any, DefaultDict, Dict, List, Optional

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
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

# Ensure UI can mount even if empty
os.makedirs("static", exist_ok=True)

app = FastAPI(title="OpenSift UI", version="0.2.0")

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

db = VectorDB()

# In-memory chat history per owner namespace
CHAT_HISTORY: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)


def _now() -> str:
    return datetime.utcnow().isoformat() + "Z"


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """
    Simple landing page for ingestion + search (classic UI).
    If you prefer, you can redirect directly to chat.
    """
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "message": request.query_params.get("msg"),
            "search_results": None,
            "generated": None,
        },
    )


@app.post("/ingest/url")
async def ingest_url(
    url: str = Form(...),
    source_title: str = Form(""),
    owner: str = Form("default"),
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

    return RedirectResponse(
        url=f"/?msg=Ingested+{len(chunks)}+chunks+from+URL",
        status_code=HTTP_303_SEE_OTHER,
    )


@app.post("/ingest/file")
async def ingest_file(
    file: UploadFile = File(...),
    owner: str = Form("default"),
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
        return RedirectResponse(
            url="/?msg=Unsupported+file+type+(pdf,txt,md)",
            status_code=HTTP_303_SEE_OTHER,
        )

    if not text.strip():
        return RedirectResponse(
            url="/?msg=No+text+extracted+from+file",
            status_code=HTTP_303_SEE_OTHER,
        )

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

    return RedirectResponse(
        url=f"/?msg=Ingested+{len(chunks)}+chunks+from+file",
        status_code=HTTP_303_SEE_OTHER,
    )


@app.get("/search", response_class=HTMLResponse)
async def search(
    request: Request,
    q: str = "",
    k: int = 5,
    owner: str = "default",
):
    """
    Classic search endpoint for index.html.
    """
    message = request.query_params.get("msg")
    search_results = None

    if q.strip():
        q_emb = embed_texts([q])[0]
        res = db.query(q_emb, k=k)

        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        ids = res.get("ids", [[]])[0]

        items: List[Dict[str, Any]] = []
        for i in range(len(docs)):
            if owner and metas[i].get("owner") != owner:
                continue
            items.append({"id": ids[i], "text": docs[i], "meta": metas[i], "distance": float(dists[i])})

        search_results = items

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "message": message,
            "search_results": search_results,
            "generated": None,
        },
    )


@app.post("/generate", response_class=HTMLResponse)
async def generate(
    request: Request,
    q: str = Form(...),
    mode: str = Form("study_guide"),
    provider: str = Form("claude_code"),  # openai | claude | claude_code
    model: str = Form(""),
    k: int = Form(8),
    owner: str = Form("default"),
):
    """
    Classic generate endpoint for index.html (retrieve + generate).
    """
    q_emb = embed_texts([q])[0]
    res = db.query(q_emb, k=k)

    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    ids = res.get("ids", [[]])[0]

    items: List[Dict[str, Any]] = []
    passages: List[Dict[str, Any]] = []
    for i in range(len(docs)):
        if owner and metas[i].get("owner") != owner:
            continue
        items.append({"id": ids[i], "text": docs[i], "meta": metas[i], "distance": float(dists[i])})
        passages.append({"text": docs[i], "meta": metas[i]})

    prompt = build_prompt(mode=mode, query=q, passages=passages)

    try:
        if provider == "openai":
            out = generate_with_openai(prompt, model=model or "gpt-4.1-mini")
        elif provider == "claude":
            out = generate_with_claude(prompt, model=model or "claude-3-5-sonnet-latest")
        else:
            out = generate_with_claude_code(prompt, model=model or None)
    except Exception as e:
        out = f"Generation failed: {e}"

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "message": None,
            "search_results": items,
            "generated": out,
        },
    )


# -------------------------
# Chatbot UI (chat.html)
# -------------------------

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


@app.post("/chat/message")
async def chat_message(
    owner: str = Form("default"),
    message: str = Form(...),
    mode: str = Form("study_guide"),
    provider: str = Form("claude_code"),  # openai | claude | claude_code
    model: str = Form(""),
    k: int = Form(8),
):
    user_msg = {"role": "user", "text": message, "ts": _now()}
    CHAT_HISTORY[owner].append(user_msg)

    # Retrieve passages for grounding
    q_emb = embed_texts([message])[0]
    res = db.query(q_emb, k=k)

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
        return JSONResponse({"ok": True, "assistant": assistant_msg})

    prompt = build_prompt(mode=mode, query=message, passages=passages)

    assistant_text = ""
    gen_error: Optional[str] = None

    try:
        if provider == "openai":
            assistant_text = generate_with_openai(prompt, model=model or "gpt-4.1-mini")
        elif provider == "claude":
            assistant_text = generate_with_claude(prompt, model=model or "claude-3-5-sonnet-latest")
        elif provider == "claude_code":
            assistant_text = generate_with_claude_code(prompt, model=model or None)
        else:
            gen_error = f"Unknown provider: {provider}"
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

    assistant_msg = {
        "role": "assistant",
        "text": assistant_text,
        "ts": _now(),
        "sources": [
            {
                "source": r["meta"].get("source"),
                "kind": r["meta"].get("kind"),
                "url": r["meta"].get("url"),
                "distance": r["distance"],
                "preview": r["text"][:240],
            }
            for r in results[:5]
        ],
    }
    CHAT_HISTORY[owner].append(assistant_msg)

    return JSONResponse({"ok": True, "assistant": assistant_msg})


@app.post("/chat/clear")
async def chat_clear(owner: str = Form("default")):
    CHAT_HISTORY[owner].clear()
    return JSONResponse({"ok": True})