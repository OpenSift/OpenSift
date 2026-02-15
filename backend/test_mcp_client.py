from __future__ import annotations

import asyncio
import base64
import json
import os
import sys
from pathlib import Path
from typing import Iterable, Tuple

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


def _pretty(resp) -> str:
    """
    MCP responses usually include .content blocks.
    Print the first text block when available; otherwise JSON-ish fallback.
    """
    if hasattr(resp, "content") and resp.content:
        blk = resp.content[0]
        if hasattr(blk, "text"):
            return blk.text
    try:
        return json.dumps(resp, indent=2, default=str, ensure_ascii=False)
    except Exception:
        return str(resp)


def file_to_b64(path: str) -> str:
    data = Path(path).read_bytes()
    return base64.b64encode(data).decode("utf-8")


async def ingest_urls(
    session: ClientSession,
    owner: str,
    urls: Iterable[Tuple[str, str]],
) -> None:
    print("\n=== BULK INGEST URLS ===")
    for title, url in urls:
        res = await session.call_tool(
            "ingest_url",
            {"url": url, "owner": owner, "source_title": title},
        )
        print(f"\n-- {title} --")
        print(_pretty(res))


async def ingest_files(
    session: ClientSession,
    owner: str,
    file_paths: Iterable[str],
) -> None:
    print("\n=== BULK INGEST FILES ===")
    for p in file_paths:
        path = Path(p).expanduser().resolve()
        if not path.exists():
            print(f"\n-- SKIP (missing): {path}")
            continue

        res = await session.call_tool(
            "ingest_file",
            {
                "filename": path.name,
                "content_base64": file_to_b64(str(path)),
                "owner": owner,
            },
        )
        print(f"\n-- {path.name} --")
        print(_pretty(res))


async def run_search_checks(session: ClientSession, owner: str, queries: Iterable[str], k: int = 5) -> None:
    print("\n=== SEARCH CHECKS ===")
    for q in queries:
        res = await session.call_tool("search", {"query": q, "k": k, "owner": owner})
        text = _pretty(res)
        # Truncate so your terminal doesn't explode
        if len(text) > 2000:
            text = text[:2000] + "\n... (truncated) ..."
        print(f"\n## {q}\n{text}")


async def run_generate(
    session: ClientSession,
    owner: str,
    provider: str = "openai",  # "openai" | "claude" | "claude_code"
) -> None:
    print("\n=== SIFT GENERATE (optional) ===")
    gen_res = await session.call_tool(
        "sift_generate",
        {
            "query": "Make a short study guide for photosynthesis focused on exam questions.",
            "mode": "study_guide",
            "k": 8,
            "owner": owner,
            "provider": provider,
            # "model": "gpt-4.1-mini",
            # "model": "claude-3-5-sonnet-latest",
            # "model": "claude-3-5-haiku-20241022",
        },
    )
    print(_pretty(gen_res))


async def main() -> None:
    # Ensure we run from backend/ so mcp_server.py and app/ imports resolve
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    owner = "matt"

    # Edit these lists however you like
    urls = [
        ("Photosynthesis (Wiki)", "https://en.wikipedia.org/wiki/Photosynthesis"),
        ("Cellular respiration (Wiki)", "https://en.wikipedia.org/wiki/Cellular_respiration"),
        ("Citric acid cycle (Wiki)", "https://en.wikipedia.org/wiki/Citric_acid_cycle"),
    ]

    # Put your local file paths here (PDF/TXT/MD). Leave empty if you don't want file ingest yet.
    file_paths = [
        # "/absolute/path/to/your/notes.pdf",
        # "/absolute/path/to/your/study_guide.md",
        # "/absolute/path/to/your/lecture_notes.txt",
    ]

    search_queries = [
        "What are the stages of photosynthesis?",
        "Explain the light-dependent reactions and the Calvin cycle with inputs/outputs.",
        "Compare photosynthesis vs cellular respiration (high-level).",
    ]

    server = StdioServerParameters(
        command=sys.executable,
        args=["mcp_server.py"],
        env=os.environ.copy(),  # optional: pass env vars explicitly
    )

    async with stdio_client(server) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            print("\n=== TOOLS ===")
            tools = await session.list_tools()
            for t in tools.tools:
                print("-", t.name)

            # Feed data
            await ingest_urls(session, owner=owner, urls=urls)
            await ingest_files(session, owner=owner, file_paths=file_paths)

            # Confirm retrieval works
            await run_search_checks(session, owner=owner, queries=search_queries, k=5)

            # Optional generation (requires provider auth configured)
            # provider can be: "openai", "claude", "claude_code"
            await run_generate(session, owner=owner, provider="openai")


if __name__ == "__main__":
    asyncio.run(main())