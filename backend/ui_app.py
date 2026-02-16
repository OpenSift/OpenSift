from __future__ import annotations

from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.status import HTTP_303_SEE_OTHER

from app.vectordb import VectorDB
from app.chunking import chunk_text
from app.ingest import extract_text_from_pdf, extract_text_from_txt, fetch_url_text
from app.llm import embed_texts
from app.providers import build_prompt, generate_with_openai, generate_with_claude, generate_with_claude_code

app = FastAPI(title="OpenSift UI", version="0.1.0")

templates = Jinja2Templates(directory="templates")

# Optional static folder (icons/css later)
app.mount("/static", StaticFiles(directory="static"), name="static")

db = VectorDB()


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "message": None,
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

    return RedirectResponse(url=f"/?msg=Ingested+{len(chunks)}+chunks+from+URL", status_code=HTTP_303_SEE_OTHER)


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
        return RedirectResponse(url="/?msg=Unsupported+file+type+(pdf,txt,md)", status_code=HTTP_303_SEE_OTHER)

    if not text.strip():
        return RedirectResponse(url="/?msg=No+text+extracted+from+file", status_code=HTTP_303_SEE_OTHER)

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

    return RedirectResponse(url=f"/?msg=Ingested+{len(chunks)}+chunks+from+file", status_code=HTTP_303_SEE_OTHER)


@app.get("/search", response_class=HTMLResponse)
async def search(
    request: Request,
    q: str = "",
    k: int = 5,
    owner: str = "default",
):
    message = request.query_params.get("msg")
    search_results = None

    if q.strip():
        q_emb = embed_texts([q])[0]
        res = db.query(q_emb, k=k)

        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        ids = res.get("ids", [[]])[0]

        items = []
        for i in range(len(docs)):
            if owner and metas[i].get("owner") != owner:
                continue
            items.append(
                {
                    "id": ids[i],
                    "text": docs[i],
                    "meta": metas[i],
                    "distance": dists[i],
                }
            )

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
    # Retrieve passages
    q_emb = embed_texts([q])[0]
    res = db.query(q_emb, k=k)

    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    ids = res.get("ids", [[]])[0]

    items = []
    passages = []
    for i in range(len(docs)):
        if owner and metas[i].get("owner") != owner:
            continue
        items.append({"id": ids[i], "text": docs[i], "meta": metas[i], "distance": dists[i]})
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