from __future__ import annotations

import argparse
import asyncio
import os
import sys
import textwrap
import time
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Dict, List

import anyio

from app.chunking import chunk_text
from app.ingest import extract_text_from_pdf, extract_text_from_txt, fetch_url_text
from app.llm import embed_texts
from app.providers import (
    DEFAULT_CLAUDE_MODEL,
    DEFAULT_OPENAI_MODEL,
    build_prompt,
    generate_with_claude,
    generate_with_claude_code,
    generate_with_openai,
)
from app.vectordb import VectorDB

from session_store import DEFAULT_DIR, list_sessions, load_session, save_session


@dataclass
class ChatConfig:
    owner: str = "default"
    mode: str = "study_guide"
    provider: str = "claude_code"  # openai | claude | claude_code
    model: str = ""
    k: int = 8
    wrap: int = 100
    history_turns: int = 10
    stream: bool = True
    show_sources: bool = True


HELP_TEXT = """
Commands:
  /help
  /quit
  /owner <name>
  /mode <mode>
  /provider <p>                 openai | claude | claude_code
  /model <name>                 Model override (empty = default)
  /k <num>
  /history <turns>
  /history on|off
  /sources on|off
  /stream on|off
  /clear

Sessions:
  /save
  /load [owner]
  /sessions

Ingest:
  /ingest url <url> [title]
  /ingest file <path>           (.pdf, .txt, .md)
"""


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _wrap(text: str, width: int) -> str:
    return "\n".join(
        textwrap.fill(line, width=width) if line.strip() else ""
        for line in text.splitlines()
    )


def _print_banner(cfg: ChatConfig) -> None:
    print("\n" + "=" * 72)
    print("OpenSift — Terminal Chat")
    print(
        f"Owner: {cfg.owner} | Mode: {cfg.mode} | Provider: {cfg.provider} | k={cfg.k} | "
        f"history={cfg.history_turns} | stream={'on' if cfg.stream else 'off'}"
    )
    print(f"Defaults: OpenAI={DEFAULT_OPENAI_MODEL} | Claude={DEFAULT_CLAUDE_MODEL}")
    print("Type /help for commands. Type /quit to exit.")
    print("=" * 72 + "\n")


def _parse_command(line: str) -> List[str]:
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


def _run_generate(cfg: ChatConfig, prompt: str) -> str:
    if cfg.provider == "openai":
        return generate_with_openai(prompt, model=cfg.model or DEFAULT_OPENAI_MODEL)
    if cfg.provider == "claude":
        return generate_with_claude(prompt, model=cfg.model or DEFAULT_CLAUDE_MODEL)
    if cfg.provider == "claude_code":
        return generate_with_claude_code(prompt, model=cfg.model or DEFAULT_CLAUDE_MODEL)
    raise RuntimeError(f"Unknown provider: {cfg.provider}")


async def _stream_openai(prompt: str, model: str) -> AsyncGenerator[str, None]:
    try:
        from openai import OpenAI  # type: ignore
    except Exception:
        raise RuntimeError("openai package not installed for streaming")

    client = OpenAI()
    stream = client.responses.stream(model=model, input=prompt)
    with stream as s:
        for event in s:
            if getattr(event, "type", "") == "response.output_text.delta":
                delta = getattr(event, "delta", "")
                if delta:
                    yield delta
        _ = s.get_final_response()


async def _stream_anthropic(prompt: str, model: str) -> AsyncGenerator[str, None]:
    try:
        import anthropic  # type: ignore
    except Exception:
        raise RuntimeError("anthropic package not installed for streaming")

    client = anthropic.Anthropic()
    with client.messages.stream(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        for text in stream.text_stream:
            if text:
                yield text


async def answer(
    db: VectorDB,
    cfg: ChatConfig,
    history: List[Dict[str, Any]],
    question: str,
    history_enabled: bool,
) -> None:
    print("\n" + "-" * 72)
    print(f"YOU: {question}\n")

    history.append({"role": "user", "text": question, "ts": _now()})
    save_session(cfg.owner, history, DEFAULT_DIR)

    # Retrieve
    try:
        q_emb = await anyio.to_thread.run_sync(lambda: embed_texts([question])[0])
        res = await anyio.to_thread.run_sync(lambda: db.query(q_emb, k=int(cfg.k)))
    except Exception as e:
        msg = f"⚠️ Retrieval failed: {e}"
        print(_wrap(msg, cfg.wrap))
        history.append({"role": "assistant", "text": msg, "ts": _now(), "sources": []})
        save_session(cfg.owner, history, DEFAULT_DIR)
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
        msg = "🤷 No matches yet. Ingest a URL/PDF first, or try different keywords."
        print(_wrap(msg, cfg.wrap))
        history.append({"role": "assistant", "text": msg, "ts": _now(), "sources": []})
        save_session(cfg.owner, history, DEFAULT_DIR)
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

    if cfg.show_sources:
        print("SOURCES:")
        for r in sources_payload:
            line = f"  - {r.get('source')} [{r.get('kind')}] (distance={float(r.get('distance') or 0):.4f})"
            if r.get("url"):
                line += f" — {r['url']}"
            print(_wrap(line, cfg.wrap))
        print()

    convo = _build_history_block(history[:-1], cfg.history_turns) if history_enabled else ""
    query_for_prompt = f"Conversation so far:\n{convo}\n\nNew question:\n{question}" if convo else question
    prompt = build_prompt(mode=cfg.mode, query=query_for_prompt, passages=passages)

    print("OPENSIFT:")
    assistant_text = ""

    if cfg.stream and cfg.provider in ("openai", "claude"):
        try:
            if cfg.provider == "openai":
                model = cfg.model or DEFAULT_OPENAI_MODEL
                async for delta in _stream_openai(prompt, model):
                    sys.stdout.write(delta)
                    sys.stdout.flush()
                    assistant_text += delta
            else:
                model = cfg.model or DEFAULT_CLAUDE_MODEL
                async for delta in _stream_anthropic(prompt, model):
                    sys.stdout.write(delta)
                    sys.stdout.flush()
                    assistant_text += delta
            print()
        except Exception:
            assistant_text = await anyio.to_thread.run_sync(lambda: _run_generate(cfg, prompt))
            print(_wrap(assistant_text, cfg.wrap))
    else:
        assistant_text = await anyio.to_thread.run_sync(lambda: _run_generate(cfg, prompt))
        print(_wrap(assistant_text, cfg.wrap))

    history.append({"role": "assistant", "text": assistant_text, "ts": _now(), "sources": sources_payload})
    save_session(cfg.owner, history, DEFAULT_DIR)

    print("\n" + "-" * 72 + "\n")


async def repl(cfg: ChatConfig) -> None:
    db = VectorDB()
    history: List[Dict[str, Any]] = load_session(cfg.owner, DEFAULT_DIR)
    history_enabled = True

    _print_banner(cfg)
    if history:
        print(_wrap(f"✅ Loaded session for '{cfg.owner}' with {len(history)} messages.", cfg.wrap))
        print()

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
                history = load_session(cfg.owner, DEFAULT_DIR)
                print(f"✅ Owner set to: {cfg.owner} (loaded {len(history)} msgs)")
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

            if cmd == "/history" and len(parts) >= 2:
                v = parts[1].strip().lower()
                if v in ("on", "off"):
                    history_enabled = (v == "on")
                    print(f"✅ history: {'on' if history_enabled else 'off'}")
                    continue
                try:
                    cfg.history_turns = int(parts[1])
                    print(f"✅ history turns: {cfg.history_turns}")
                except ValueError:
                    print("⚠️ /history <turns> OR /history on|off")
                continue

            if cmd == "/sources" and len(parts) >= 2:
                v = parts[1].strip().lower()
                if v not in ("on", "off"):
                    print("⚠️ /sources on|off")
                    continue
                cfg.show_sources = (v == "on")
                print(f"✅ sources: {'on' if cfg.show_sources else 'off'}")
                continue

            if cmd == "/stream" and len(parts) >= 2:
                v = parts[1].strip().lower()
                if v not in ("on", "off"):
                    print("⚠️ /stream on|off")
                    continue
                cfg.stream = (v == "on")
                print(f"✅ stream: {'on' if cfg.stream else 'off'}")
                continue

            if cmd == "/save":
                save_session(cfg.owner, history, DEFAULT_DIR)
                print(f"✅ Saved session for {cfg.owner}")
                continue

            if cmd == "/load":
                which = parts[1].strip() if len(parts) >= 2 else cfg.owner
                cfg.owner = which
                history = load_session(cfg.owner, DEFAULT_DIR)
                print(f"✅ Loaded session for {cfg.owner} ({len(history)} msgs)")
                continue

            if cmd == "/sessions":
                sess = list_sessions(DEFAULT_DIR)
                if not sess:
                    print("No saved sessions yet.")
                else:
                    print("Saved sessions:")
                    for s in sess:
                        print(f"  - {s}")
                continue

            if cmd == "/clear":
                os.system("clear" if os.name != "nt" else "cls")
                _print_banner(cfg)
                continue

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
                        msg = f"✅ Ingested {res['ingested']} chunks from {res['source']} ({dt:.2f}s)"
                        print(msg)
                        history.append({"role": "assistant", "text": msg, "ts": _now(), "sources": []})
                        save_session(cfg.owner, history, DEFAULT_DIR)
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
                            msg = f"✅ Ingested {res['ingested']} chunks from {res['source']} ({dt:.2f}s)"
                        else:
                            msg = f"⚠️ {res.get('error','ingest_failed')}"
                        print(msg)
                        history.append({"role": "assistant", "text": msg, "ts": _now(), "sources": []})
                        save_session(cfg.owner, history, DEFAULT_DIR)
                    except Exception as e:
                        print(_wrap(f"⚠️ File ingest failed: {e}", cfg.wrap))
                    continue

            print("⚠️ Unknown command. Type /help")
            continue

        await answer(db, cfg, history, line, history_enabled=history_enabled)


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenSift Terminal Chat")
    parser.add_argument("--owner", default="default", help="Owner/namespace (default: default)")
    parser.add_argument("--mode", default="study_guide", help="Mode (default: study_guide)")
    parser.add_argument("--provider", default="claude_code", choices=["openai", "claude", "claude_code"])
    parser.add_argument("--model", default="", help="Model override (optional)")
    parser.add_argument("--k", type=int, default=8, help="Top-k retrieval (default: 8)")
    parser.add_argument("--wrap", type=int, default=100, help="Wrap width (default: 100)")
    parser.add_argument("--history-turns", type=int, default=10, help="How many messages to include as history")
    parser.add_argument("--no-stream", action="store_true", help="Disable streaming output")
    parser.add_argument("--no-sources", action="store_true", help="Disable sources printing")
    args = parser.parse_args()

    cfg = ChatConfig(
        owner=args.owner,
        mode=args.mode,
        provider=args.provider,
        model=args.model,
        k=args.k,
        wrap=args.wrap,
        history_turns=args.history_turns,
        stream=not args.no_stream,
        show_sources=not args.no_sources,
    )

    asyncio.run(repl(cfg))


if __name__ == "__main__":
    main()