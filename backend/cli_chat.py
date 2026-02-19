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
from app.logging_utils import configure_logging
from app.providers import (
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
from app.wellness import build_break_reminder, should_add_break_reminder

from session_store import DEFAULT_DIR, list_sessions, load_session, save_session

logger = configure_logging("opensift.cli")


@dataclass
class ChatConfig:
    owner: str = "default"
    mode: str = "study_guide"
    provider: str = "claude_code"  # openai | claude | claude_code | codex
    model: str = ""
    k: int = 8
    wrap: int = 100
    history_turns: int = 10
    stream: bool = True
    true_streaming: bool = True
    show_thinking: bool = True
    thinking_enabled: bool = False
    show_sources: bool = True


HELP_TEXT = """
Commands:
  /help
  /quit
  /owner <name>
  /mode <mode>
  /provider <p>                 openai | claude | claude_code | codex
  /model <name>                 Model override (empty = default)
  /k <num>
  /history <turns>
  /history on|off
  /sources on|off
  /stream on|off
  /true-stream on|off
  /show-thinking on|off
  /thinking on|off
  /style                         Show global study style
  /style set <text>              Set global study style
  /style clear                   Clear global study style
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
    style = get_global_style()
    print("\n" + "=" * 72)
    print("OpenSift — Terminal Chat")
    print(
        f"Owner: {cfg.owner} | Mode: {cfg.mode} | Provider: {cfg.provider} | k={cfg.k} | "
        f"history={cfg.history_turns} | stream={'on' if cfg.stream else 'off'} | "
        f"true_stream={'on' if cfg.true_streaming else 'off'} | thinking={'on' if cfg.thinking_enabled else 'off'}"
    )
    print(f"Study style: {'configured' if style else 'default/none'}")
    print(f"Defaults: OpenAI={DEFAULT_OPENAI_MODEL} | Claude={DEFAULT_CLAUDE_MODEL}")
    print(f"Claude models: {', '.join(SUPPORTED_CLAUDE_MODELS)}")
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
    t0 = time.perf_counter()
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
    logger.info(
        "cli_ingest_url_success owner=%s source=%s chunks=%d duration_ms=%.2f",
        owner,
        source,
        len(chunks),
        (time.perf_counter() - t0) * 1000.0,
    )

    return {"ok": True, "ingested": len(chunks), "source": source, "url": url, "owner": owner}


async def ingest_file(db: VectorDB, owner: str, path: str) -> Dict[str, Any]:
    t0 = time.perf_counter()
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
    logger.info(
        "cli_ingest_file_success owner=%s source=%s chunks=%d duration_ms=%.2f",
        owner,
        filename,
        len(chunks),
        (time.perf_counter() - t0) * 1000.0,
    )

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
        return generate_with_claude(
            prompt,
            model=cfg.model or DEFAULT_CLAUDE_MODEL,
            thinking_enabled=cfg.thinking_enabled,
        )
    if cfg.provider == "claude_code":
        return generate_with_claude_code(prompt, model=cfg.model or DEFAULT_CLAUDE_MODEL)
    if cfg.provider == "codex":
        return generate_with_codex(prompt, model=cfg.model or DEFAULT_OPENAI_MODEL)
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


async def _stream_anthropic(prompt: str, model: str, thinking_enabled: bool = False) -> AsyncGenerator[str, None]:
    try:
        import anthropic  # type: ignore
    except Exception:
        raise RuntimeError("anthropic package not installed for streaming")

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


async def answer(
    db: VectorDB,
    cfg: ChatConfig,
    history: List[Dict[str, Any]],
    question: str,
    history_enabled: bool,
) -> None:
    t0 = time.perf_counter()
    print("\n" + "-" * 72)
    print(f"YOU: {question}\n")

    history.append({"role": "user", "text": question, "ts": _now()})
    save_session(cfg.owner, history, DEFAULT_DIR)

    if cfg.show_thinking:
        print("STATUS: Retrieving relevant passages...")

    # Retrieve
    try:
        q_emb = await anyio.to_thread.run_sync(lambda: embed_texts([question])[0])
        owner_where = {"owner": cfg.owner} if cfg.owner else None
        res = await anyio.to_thread.run_sync(lambda: db.query(q_emb, k=int(cfg.k), where=owner_where))
    except Exception as e:
        logger.exception("cli_retrieval_failed owner=%s", cfg.owner)
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

    if cfg.owner and not results:
        try:
            res2 = await anyio.to_thread.run_sync(lambda: db.query(q_emb, k=max(int(cfg.k) * 3, 24), where=None))
            docs2 = res2.get("documents", [[]])[0]
            metas2 = res2.get("metadatas", [[]])[0]
            dists2 = res2.get("distances", [[]])[0]
            ids2 = res2.get("ids", [[]])[0]
            for i in range(len(docs2)):
                if (metas2[i] or {}).get("owner") != cfg.owner:
                    continue
                results.append({"id": ids2[i], "text": docs2[i], "meta": metas2[i], "distance": float(dists2[i])})
                passages.append({"text": docs2[i], "meta": metas2[i]})
                if len(results) >= int(cfg.k):
                    break
        except Exception:
            pass

    if not results:
        logger.info("cli_no_results owner=%s k=%d", cfg.owner, int(cfg.k))
        msg = "🤷 No matches yet. Ingest a URL/PDF first, or try different keywords."
        add_break = should_add_break_reminder(history)
        if add_break:
            msg = f"{msg}\n\n{build_break_reminder(history)}"
        print(_wrap(msg, cfg.wrap))
        history.append({"role": "assistant", "text": msg, "ts": _now(), "sources": [], "break_reminder": add_break})
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
    study_style = get_global_style()
    prompt = build_prompt(mode=cfg.mode, query=query_for_prompt, passages=passages, study_style=study_style)

    if cfg.show_thinking:
        if cfg.provider == "claude" and cfg.thinking_enabled:
            print("STATUS: Thinking... (Claude extended thinking enabled)")
        else:
            print("STATUS: Thinking...")
    print("OPENSIFT:")
    assistant_text = ""

    if cfg.stream and cfg.provider in ("openai", "claude", "codex") and cfg.true_streaming:
        try:
            if cfg.provider == "openai":
                model = cfg.model or DEFAULT_OPENAI_MODEL
                async for delta in _stream_openai(prompt, model):
                    sys.stdout.write(delta)
                    sys.stdout.flush()
                    assistant_text += delta
            elif cfg.provider == "codex":
                model = cfg.model or DEFAULT_OPENAI_MODEL
                async for delta in _stream_codex(prompt, model):
                    sys.stdout.write(delta)
                    sys.stdout.flush()
                    assistant_text += delta
            else:
                model = cfg.model or DEFAULT_CLAUDE_MODEL
                async for delta in _stream_anthropic(prompt, model, thinking_enabled=cfg.thinking_enabled):
                    sys.stdout.write(delta)
                    sys.stdout.flush()
                    assistant_text += delta
            print()
        except Exception:
            assistant_text = await anyio.to_thread.run_sync(lambda: _run_generate(cfg, prompt))
            if cfg.stream:
                for i in range(0, len(assistant_text), 80):
                    chunk = assistant_text[i : i + 80]
                    sys.stdout.write(chunk)
                    sys.stdout.flush()
                    await asyncio.sleep(0.01)
                print()
            else:
                print(_wrap(assistant_text, cfg.wrap))
    else:
        assistant_text = await anyio.to_thread.run_sync(lambda: _run_generate(cfg, prompt))
        if cfg.stream:
            for i in range(0, len(assistant_text), 80):
                chunk = assistant_text[i : i + 80]
                sys.stdout.write(chunk)
                sys.stdout.flush()
                await asyncio.sleep(0.01)
            print()
        else:
            print(_wrap(assistant_text, cfg.wrap))

    add_break = should_add_break_reminder(history)
    if add_break:
        reminder = build_break_reminder(history)
        assistant_text = f"{assistant_text}\n\n{reminder}"
        print("\n" + _wrap(reminder, cfg.wrap))

    history.append(
        {
            "role": "assistant",
            "text": assistant_text,
            "ts": _now(),
            "sources": sources_payload,
            "break_reminder": add_break,
        }
    )
    save_session(cfg.owner, history, DEFAULT_DIR)
    logger.info(
        "cli_answer_done owner=%s mode=%s provider=%s response_chars=%d duration_ms=%.2f",
        cfg.owner,
        cfg.mode,
        cfg.provider,
        len(assistant_text or ""),
        (time.perf_counter() - t0) * 1000.0,
    )

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
                style = get_global_style()
                print(f"🎨 Global study style: {'configured' if style else 'default/none'}")
                continue

            if cmd == "/mode" and len(parts) >= 2:
                cfg.mode = parts[1].strip()
                print(f"✅ Mode set to: {cfg.mode}")
                continue

            if cmd == "/provider" and len(parts) >= 2:
                p = parts[1].strip().lower()
                if p not in ("openai", "claude", "claude_code", "codex"):
                    print("⚠️ provider must be: openai | claude | claude_code | codex")
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

            if cmd == "/true-stream" and len(parts) >= 2:
                v = parts[1].strip().lower()
                if v not in ("on", "off"):
                    print("⚠️ /true-stream on|off")
                    continue
                cfg.true_streaming = (v == "on")
                print(f"✅ true-stream: {'on' if cfg.true_streaming else 'off'}")
                continue

            if cmd == "/show-thinking" and len(parts) >= 2:
                v = parts[1].strip().lower()
                if v not in ("on", "off"):
                    print("⚠️ /show-thinking on|off")
                    continue
                cfg.show_thinking = (v == "on")
                print(f"✅ show-thinking: {'on' if cfg.show_thinking else 'off'}")
                continue

            if cmd == "/thinking" and len(parts) >= 2:
                v = parts[1].strip().lower()
                if v not in ("on", "off"):
                    print("⚠️ /thinking on|off")
                    continue
                cfg.thinking_enabled = (v == "on")
                print(f"✅ thinking: {'on' if cfg.thinking_enabled else 'off'}")
                continue

            if cmd == "/style":
                if len(parts) == 1:
                    style = get_global_style()
                    if not style:
                        print("No global study style set.")
                    else:
                        print("\nCurrent global study style:\n")
                        print(_wrap(style, cfg.wrap))
                    continue
                sub = parts[1].strip().lower()
                if sub == "clear":
                    set_global_style("")
                    print("✅ Cleared global study style.")
                    continue
                if sub == "set":
                    style_text = line.split(" ", 2)[2].strip() if len(parts) >= 3 else ""
                    if not style_text:
                        print("⚠️ Usage: /style set <text>")
                        continue
                    set_global_style(style_text)
                    print("✅ Updated global study style.")
                    continue
                print("⚠️ Usage: /style | /style set <text> | /style clear")
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
    parser.add_argument("--provider", default="claude_code", choices=["openai", "claude", "claude_code", "codex"])
    parser.add_argument("--model", default="", help="Model override (optional)")
    parser.add_argument("--k", type=int, default=8, help="Top-k retrieval (default: 8)")
    parser.add_argument("--wrap", type=int, default=100, help="Wrap width (default: 100)")
    parser.add_argument("--history-turns", type=int, default=10, help="How many messages to include as history")
    parser.add_argument("--no-stream", action="store_true", help="Disable streaming output")
    parser.add_argument("--no-true-stream", action="store_true", help="Disable provider-native true streaming")
    parser.add_argument("--no-show-thinking", action="store_true", help="Hide retrieval/thinking status lines")
    parser.add_argument("--thinking", action="store_true", help="Enable Claude extended thinking where supported")
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
        true_streaming=not args.no_true_stream,
        show_thinking=not args.no_show_thinking,
        thinking_enabled=bool(args.thinking),
        show_sources=not args.no_sources,
    )

    asyncio.run(repl(cfg))


if __name__ == "__main__":
    main()
