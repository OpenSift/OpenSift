from __future__ import annotations

import argparse
import asyncio
import os
import sys
import textwrap
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import anyio

from app.chunking import chunk_text
from app.ingest import extract_text_from_pdf, extract_text_from_txt, fetch_url_text
from app.llm import embed_texts
from app.providers import build_prompt, generate_with_claude, generate_with_claude_code, generate_with_openai
from app.vectordb import VectorDB


@dataclass
class ChatConfig:
    owner: str = "default"
    mode: str = "study_guide"  # study_guide | quiz | explain | etc (depends on build_prompt)
    provider: str = "claude_code"  # openai | claude | claude_code
    model: str = ""  # optional override
    k: int = 8
    wrap: int = 100
    stream_chunk_chars: int = 80
    stream_delay: float = 0.01


HELP_TEXT = """
Commands:
  /help
  /quit
  /owner <name>               Set owner/namespace (e.g., bio101)
  /mode <mode>                Set mode (e.g., study_guide, quiz, explain)
  /provider <p>               Set provider: openai | claude | claude_code
  /model <name>               Set model override (empty = default)
  /k <num>                    Set retrieval top-k (default 8)
  /clear                      Clear terminal (does not delete vector DB)
  /sources on|off              Toggle showing sources each response (default on)

Ingest:
  /ingest url <url> [title]   Ingest a URL (optional title)
  /ingest file <path>         Ingest a file (.pdf, .txt, .md)

Tips:
  - Use owners to separate subjects:
      /owner chem-midterm
  - Ask: "Make me a 10-question quiz with answers"
"""


def _wrap(text: str, width: int) -> str:
    return "\n".join(textwrap.fill(line, width=width) if line.strip() else "" for line in text.splitlines())


def _print_banner(cfg: ChatConfig) -> None:
    print("\n" + "=" * 72)
    print("OpenSift — Terminal Chat")
    print(f"Owner: {cfg.owner} | Mode: {cfg.mode} | Provider: {cfg.provider} | k={cfg.k}")
    print("Type /help for commands. Type /quit to exit.")
    print("=" * 72 + "\n")


def _parse_command(line: str) -> List[str]:
    # Minimal split that keeps quoted strings simple-ish
    # (You can make this shlex.split later if you want strict parsing.)
    return [p for p in line.strip().split(" ") if p]


def _read_file_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def _infer_and_extract_text(path: str, data: bytes) -> tuple[str, str]:
    lower = path.lower()
    if lower.endswith(".pdf"):
        return "pdf", extract_text_from_pdf(data)
    if lower.endswith(".txt") or lower.endswith(".md"):
        return "text", extract_text_from_txt(data)
    raise ValueError("Unsupported file type. Please use .pdf, .txt, or .md")


async def ingest_url(db: VectorDB, owner: str, url: str, source_title: str = "") -> Dict[str, Any]:
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

    return {"ok": True, "ingested": len(chunks), "source": source, "url": url, "owner": owner}


async def ingest_file(db: VectorDB, owner: str, path: str) -> Dict[str, Any]:
    data = await anyio.to_thread.run_sync(lambda: _read_file_bytes(path))
    kind, text = await anyio.to_thread.run_sync(lambda: _infer_and_extract_text(path, data))

    if not text.strip():
        return {"ok": False, "error": "no_text_extracted", "path": path, "owner": owner}

    filename = os.path.basename(path)
    prefix = f"{owner}::{filename}" if owner else filename

    chunks = chunk_text(text, prefix=prefix)
    texts = [c.text for c in chunks]
    ids = [c.chunk_id for c in chunks]
    metas = [{"source": filename, "kind": kind, "owner": owner, "start": c.start, "end": c.end} for c in chunks]

    embs = embed_texts(texts)
    db.add(ids=ids, documents=texts, metadatas=metas, embeddings=embs)

    return {"ok": True, "ingested": len(chunks), "source": filename, "path": path, "owner": owner}


def _run_generate(cfg: ChatConfig, prompt: str) -> str:
    if cfg.provider == "openai":
        return generate_with_openai(prompt, model=cfg.model or "gpt-4.1-mini")
    if cfg.provider == "claude":
        return generate_with_claude(prompt, model=cfg.model or "claude-3-5-sonnet-latest")
    if cfg.provider == "claude_code":
        return generate_with_claude_code(prompt, model=cfg.model or None)
    raise RuntimeError(f"Unknown provider: {cfg.provider}")


async def answer(db: VectorDB, cfg: ChatConfig, question: str, show_sources: bool = True) -> None:
    print("\n" + "-" * 72)
    print(f"YOU: {question}\n")

    # Retrieve
    try:
        q_emb = await anyio.to_thread.run_sync(lambda: embed_texts([question])[0])
        res = await anyio.to_thread.run_sync(lambda: db.query(q_emb, k=int(cfg.k)))
    except Exception as e:
        print(_wrap(f"⚠️ Retrieval failed: {e}", cfg.wrap))
        return

    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    ids = res.get("ids", [[]])[0]

    results: List[Dict[str, Any]] = []
    passages: List[Dict[str, Any]] = []
    for i in range(len(docs)):
        if cfg.owner and metas[i].get("owner") != cfg.owner:
            continue
        results.append({"id": ids[i], "text": docs[i], "meta": metas[i], "distance": float(dists[i])})
        passages.append({"text": docs[i], "meta": metas[i]})

    if not results:
        print(_wrap("🤷 No matches yet. Ingest a URL/PDF first, or try different keywords.", cfg.wrap))
        return

    if show_sources:
        print("SOURCES:")
        for r in results[:5]:
            meta = r["meta"] or {}
            src = meta.get("source") or "(unknown)"
            kind = meta.get("kind") or "doc"
            url = meta.get("url")
            dist = r["distance"]
            line = f"  - {src} [{kind}] (distance={dist:.4f})"
            if url:
                line += f" — {url}"
            print(_wrap(line, cfg.wrap))
        print()

    prompt = build_prompt(mode=cfg.mode, query=question, passages=passages)

    # Generate
    print("OPENSIFT:")
    try:
        text = await anyio.to_thread.run_sync(lambda: _run_generate(cfg, prompt))
    except Exception as e:
        # Fallback: show top relevant passages if provider/auth fails
        bullets = "\n".join([f"- {r['text'][:240].strip()}…" for r in results[:3]])
        fallback = (
            f"⚠️ Generation failed ({e}).\n\n"
            "Here are the most relevant passages I found:\n\n"
            f"{bullets}\n\n"
            "If you configure a provider (OpenAI/Claude/Claude Code), I can turn this into a study guide/quiz."
        )
        print(_wrap(fallback, cfg.wrap))
        return

    # Stream to terminal (fake-stream by chunking)
    for i in range(0, len(text), cfg.stream_chunk_chars):
        chunk = text[i : i + cfg.stream_chunk_chars]
        sys.stdout.write(chunk)
        sys.stdout.flush()
        await asyncio.sleep(cfg.stream_delay)

    print("\n" + "-" * 72 + "\n")


async def repl(cfg: ChatConfig) -> None:
    db = VectorDB()
    show_sources = True

    _print_banner(cfg)

    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye 👋")
            return

        if not line:
            continue

        if line.startswith("/"):
            parts = _parse_command(line)
            cmd = parts[0].lower()

            if cmd in ("/quit", "/exit"):
                print("Bye 👋")
                return

            if cmd == "/help":
                print(_wrap(HELP_TEXT.strip(), cfg.wrap))
                continue

            if cmd == "/owner" and len(parts) >= 2:
                cfg.owner = parts[1].strip()
                print(f"✅ Owner set to: {cfg.owner}")
                continue

            if cmd == "/mode" and len(parts) >= 2:
                cfg.mode = parts[1].strip()
                print(f"✅ Mode set to: {cfg.mode}")
                continue

            if cmd == "/provider" and len(parts) >= 2:
                p = parts[1].strip().lower()
                if p not in ("openai", "claude", "claude_code"):
                    print("⚠️ provider must be: openai | claude | claude_code")
                    continue
                cfg.provider = p
                print(f"✅ Provider set to: {cfg.provider}")
                continue

            if cmd == "/model":
                cfg.model = " ".join(parts[1:]).strip()
                print(f"✅ Model set to: {cfg.model or '(default)'}")
                continue

            if cmd == "/k" and len(parts) >= 2:
                try:
                    cfg.k = int(parts[1])
                    print(f"✅ k set to: {cfg.k}")
                except ValueError:
                    print("⚠️ k must be an integer")
                continue

            if cmd == "/sources" and len(parts) >= 2:
                v = parts[1].strip().lower()
                if v not in ("on", "off"):
                    print("⚠️ /sources on|off")
                    continue
                show_sources = (v == "on")
                print(f"✅ sources: {'on' if show_sources else 'off'}")
                continue

            if cmd == "/clear":
                os.system("clear" if os.name != "nt" else "cls")
                _print_banner(cfg)
                continue

            # Ingest commands
            if cmd == "/ingest" and len(parts) >= 3:
                kind = parts[1].strip().lower()

                if kind == "url":
                    url = parts[2].strip()
                    title = " ".join(parts[3:]).strip()
                    print(f"⏳ Ingesting URL: {url}")
                    t0 = time.time()
                    try:
                        res = await ingest_url(db, cfg.owner, url, title)
                        dt = time.time() - t0
                        print(f"✅ Ingested {res['ingested']} chunks from {res['source']} ({dt:.2f}s)")
                    except Exception as e:
                        print(_wrap(f"⚠️ URL ingest failed: {e}", cfg.wrap))
                    continue

                if kind == "file":
                    path = " ".join(parts[2:]).strip()
                    if not os.path.exists(path):
                        print("⚠️ File not found.")
                        continue
                    print(f"⏳ Ingesting file: {path}")
                    t0 = time.time()
                    try:
                        res = await ingest_file(db, cfg.owner, path)
                        dt = time.time() - t0
                        if res.get("ok"):
                            print(f"✅ Ingested {res['ingested']} chunks from {res['source']} ({dt:.2f}s)")
                        else:
                            print(_wrap(f"⚠️ {res.get('error','ingest_failed')}", cfg.wrap))
                    except Exception as e:
                        print(_wrap(f"⚠️ File ingest failed: {e}", cfg.wrap))
                    continue

                print("⚠️ Usage: /ingest url <url> [title]  OR  /ingest file <path>")
                continue

            print("⚠️ Unknown command. Type /help")
            continue

        # Normal question
        await answer(db, cfg, line, show_sources=show_sources)


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenSift Terminal Chat")
    parser.add_argument("--owner", default="default", help="Owner/namespace (default: default)")
    parser.add_argument("--mode", default="study_guide", help="Mode (default: study_guide)")
    parser.add_argument("--provider", default="claude_code", choices=["openai", "claude", "claude_code"])
    parser.add_argument("--model", default="", help="Model override (optional)")
    parser.add_argument("--k", type=int, default=8, help="Top-k retrieval (default: 8)")
    parser.add_argument("--wrap", type=int, default=100, help="Wrap width (default: 100)")
    args = parser.parse_args()

    cfg = ChatConfig(
        owner=args.owner,
        mode=args.mode,
        provider=args.provider,
        model=args.model,
        k=args.k,
        wrap=args.wrap,
    )

    asyncio.run(repl(cfg))


if __name__ == "__main__":
    main()