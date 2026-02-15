from __future__ import annotations
from fastapi import FastAPI, UploadFile, File, HTTPException
from typing import List
from .schemas import IngestUrlRequest, SiftRequest, SiftResponse
from .vectordb import VectorDB
from .chunking import chunk_text
from .ingest import extract_text_from_pdf, extract_text_from_txt, fetch_url_text
from .llm import embed_texts, generate_study_output

app = FastAPI(title="OpenSift API", version="0.1.0")
db = VectorDB()

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/ingest/file")
async def ingest_file(file: UploadFile = File(...)):
    data = await file.read()
    name = file.filename or "upload"

    if name.lower().endswith(".pdf"):
        text = extract_text_from_pdf(data)
        kind = "pdf"
    elif name.lower().endswith((".txt", ".md")):
        text = extract_text_from_txt(data)
        kind = "text"
    else:
        raise HTTPException(status_code=400, detail="Supported: .pdf, .txt, .md")

    if not text.strip():
        raise HTTPException(status_code=400, detail="No text extracted.")

    chunks = chunk_text(text, prefix=name)
    texts = [c.text for c in chunks]
    ids = [c.chunk_id for c in chunks]
    metas = [{"source": name, "kind": kind, "start": c.start, "end": c.end} for c in chunks]

    embs = embed_texts(texts)
    db.add(ids=ids, documents=texts, metadatas=metas, embeddings=embs)

    return {"ingested": len(chunks), "source": name}

@app.post("/ingest/url")
async def ingest_url(req: IngestUrlRequest):
    title, text = await fetch_url_text(str(req.url))
    source = req.source_title or title

    chunks = chunk_text(text, prefix=source)
    texts = [c.text for c in chunks]
    ids = [c.chunk_id for c in chunks]
    metas = [{"source": source, "kind": "url", "url": str(req.url), "start": c.start, "end": c.end} for c in chunks]

    embs = embed_texts(texts)
    db.add(ids=ids, documents=texts, metadatas=metas, embeddings=embs)

    return {"ingested": len(chunks), "source": source, "url": str(req.url)}

@app.post("/sift", response_model=SiftResponse)
async def sift(req: SiftRequest):
    q_emb = embed_texts([req.query])[0]
    result = db.query(q_emb, k=req.k)

    docs = result.get("documents", [[]])[0]
    metas = result.get("metadatas", [[]])[0]
    dists = result.get("distances", [[]])[0]
    ids = result.get("ids", [[]])[0]

    passages: List[dict] = []
    sources: List[dict] = []

    for i in range(len(docs)):
        passages.append({"text": docs[i], "meta": metas[i]})
        sources.append({
            "id": ids[i],
            "source": metas[i].get("source"),
            "kind": metas[i].get("kind"),
            "url": metas[i].get("url"),
            "distance": dists[i],
        })

    answer = generate_study_output(req.mode, req.query, passages)

    return SiftResponse(answer=answer, sources=sources)