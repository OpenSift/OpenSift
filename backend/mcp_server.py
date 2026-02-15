from __future__ import annotations

import base64
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

from app.vectordb import VectorDB
from app.chunking import chunk_text
from app.ingest import extract_text_from_pdf, extract_text_from_txt, fetch_url_text
from app.llm import embed_texts  # embeddings for retrieval

# Optional generation providers (only used if you added sift_generate)
from app.providers import (
    build_prompt,
    generate_with_openai,
    generate_with_claude,
    generate_with_claude_code,
)

# ✅ MUST be defined before any @mcp.tool decorators
mcp = FastMCP("OpenSift")
db = VectorDB()


def _detect_and_extract_text(filename: str, data: bytes) -> tuple[str, str]:
    lower = filename.lower()
    if lower.endswith(".pdf"):
        return "pdf", extract_text_from_pdf(data)
    if lower.endswith((".txt", ".md")):
        return "text", extract_text_from_txt(data)
    raise ValueError("Supported: .pdf, .txt, .md")


@mcp.tool()
async def ingest_file(
    filename: str,
    content_base64: str,
    owner: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Ingest a file into the OpenSift vector store.
    """
    data = base64.b64decode(content_base64)
    kind, text = _detect_and_extract_text(filename, data)
    if not text.strip():
        return {"ok": False, "error": "No text extracted."}

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

    return {"ok": True, "ingested": len(chunks), "source": filename, "owner": owner}


@mcp.tool()
async def ingest_url(
    url: str,
    source_title: Optional[str] = None,
    owner: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Fetch & ingest a URL into the vector store.
    """
    title, text = await fetch_url_text(url)
    source = source_title or title
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

    return {"ok": True, "ingested": len(chunks), "source": source, "url": url, "owner": owner}


@mcp.tool()
async def search(
    query: str,
    k: int = 8,
    owner: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Retrieve top-k relevant passages for a query.
    """
    q_emb = embed_texts([query])[0]
    res = db.query(q_emb, k=k)

    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    ids = res.get("ids", [[]])[0]

    items: List[Dict[str, Any]] = []
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

    return {"ok": True, "query": query, "k": k, "owner": owner, "results": items}


@mcp.tool()
async def sift_generate(
    query: str,
    mode: str = "study_guide",
    k: int = 8,
    owner: Optional[str] = None,
    provider: str = "openai",  # "openai" | "claude" | "claude_code"
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Retrieve passages then generate an answer using a configured provider.
    """
    res = await search(query=query, k=k, owner=owner)
    passages = [{"text": r["text"], "meta": r["meta"]} for r in res["results"]]

    prompt = build_prompt(mode=mode, query=query, passages=passages)

    try:
        if provider == "claude_code":
            out = generate_with_claude_code(prompt, model=model)
        elif provider == "claude":
            out = generate_with_claude(prompt, model=model or "claude-3-5-sonnet-latest")
        else:
            out = generate_with_openai(prompt, model=model or "gpt-4.1-mini")
    except Exception as e:
        return {
            "ok": False,
            "error": str(e),
            "hint": "If you're using Codex, call `search` and let Codex generate the final response.",
            "sources": res["results"],
        }

    return {"ok": True, "answer": out, "sources": res["results"]}


if __name__ == "__main__":
    # Codex MCP expects stdio transport by default.
    mcp.run()