from __future__ import annotations

import base64
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

from app.vectordb import VectorDB
from app.chunking import chunk_text
from app.ingest import extract_text_from_pdf, extract_text_from_txt, fetch_url_text
from app.llm import embed_texts  # embeddings for retrieval
from app.logging_utils import configure_logging

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
logger = configure_logging("opensift.mcp")


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
    logger.info("mcp_ingest_file_start owner=%s filename=%s", owner, filename)
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
    logger.info("mcp_ingest_file_success owner=%s filename=%s chunks=%d", owner, filename, len(chunks))

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
    logger.info("mcp_ingest_url_start owner=%s url=%s", owner, url)
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
    logger.info("mcp_ingest_url_success owner=%s source=%s chunks=%d", owner, source, len(chunks))

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
    logger.info("mcp_search_start owner=%s k=%d", owner, k)
    q_emb = embed_texts([query])[0]
    owner_where = {"owner": owner} if owner else None
    res = db.query(q_emb, k=k, where=owner_where)

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

    if owner and not items:
        res2 = db.query(q_emb, k=max(int(k) * 3, 24), where=None)
        docs2 = res2.get("documents", [[]])[0]
        metas2 = res2.get("metadatas", [[]])[0]
        dists2 = res2.get("distances", [[]])[0]
        ids2 = res2.get("ids", [[]])[0]
        for i in range(len(docs2)):
            if (metas2[i] or {}).get("owner") != owner:
                continue
            items.append(
                {
                    "id": ids2[i],
                    "text": docs2[i],
                    "meta": metas2[i],
                    "distance": dists2[i],
                }
            )
            if len(items) >= k:
                break

    logger.info("mcp_search_done owner=%s results=%d", owner, len(items))
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
    logger.info("mcp_sift_generate_start owner=%s mode=%s provider=%s", owner, mode, provider)
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
        logger.exception("mcp_sift_generate_failed owner=%s provider=%s", owner, provider)
        return {
            "ok": False,
            "error": str(e),
            "hint": "If you're using Codex, call `search` and let Codex generate the final response.",
            "sources": res["results"],
        }

    logger.info("mcp_sift_generate_done owner=%s sources=%d", owner, len(res["results"]))
    return {"ok": True, "answer": out, "sources": res["results"]}


if __name__ == "__main__":
    # Codex MCP expects stdio transport by default.
    mcp.run()
