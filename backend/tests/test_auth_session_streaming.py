from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

from fastapi.testclient import TestClient

import ui_app


def _authed_client(monkeypatch) -> TestClient:
    monkeypatch.setattr(ui_app, "_is_authenticated", lambda request: True)
    monkeypatch.setattr(ui_app, "_csrf_invalid", lambda request: False)
    monkeypatch.setattr(
        ui_app.LocalhostOnlyMiddleware,
        "dispatch",
        lambda self, request, call_next: call_next(request),
    )
    return TestClient(ui_app.app)


def test_startup_logs_do_not_expose_full_token(monkeypatch) -> None:
    lines = []

    def _capture(msg: str, *args):
        lines.append(msg % args if args else msg)

    monkeypatch.setattr(ui_app.logger, "info", _capture)
    asyncio.run(ui_app._print_startup_token())

    assert any("token_present=true" in line for line in lines)
    assert not any(ui_app.GEN_TOKEN in line for line in lines)


def test_session_import_replace_and_merge_persist(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(ui_app, "SESSION_DIR", str(tmp_path / "sessions"))
    ui_app.CHAT_HISTORY.clear()
    client = _authed_client(monkeypatch)

    base_payload = json.dumps(
        [
            {"role": "user", "text": "First"},
            {"role": "assistant", "text": "One"},
        ]
    )
    merge_payload = json.dumps([{"role": "user", "text": "Second"}])

    r1 = client.post("/chat/session/import", data={"owner": "alice", "payload": base_payload, "merge": "false"})
    assert r1.status_code == 200
    assert r1.json()["count"] == 2

    r2 = client.post("/chat/session/import", data={"owner": "alice", "payload": merge_payload, "merge": "true"})
    assert r2.status_code == 200
    assert r2.json()["count"] == 3

    r3 = client.get("/chat/session/export", params={"owner": "alice"})
    assert r3.status_code == 200
    history = r3.json()["history"]
    assert [m["text"] for m in history] == ["First", "One", "Second"]

    session_file = tmp_path / "sessions" / "alice.json"
    on_disk = json.loads(session_file.read_text(encoding="utf-8"))
    assert [m["text"] for m in on_disk] == ["First", "One", "Second"]


def test_chat_stream_returns_done_and_persists_messages(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(ui_app, "SESSION_DIR", str(tmp_path / "sessions"))
    ui_app.CHAT_HISTORY.clear()
    client = _authed_client(monkeypatch)

    monkeypatch.setattr(ui_app, "embed_texts", lambda texts: [[0.1] for _ in texts])

    class _DB:
        def query(self, q_emb, k=8, where=None):
            owner = (where or {}).get("owner", "default")
            return {
                "documents": [["A relevant passage."]],
                "metadatas": [[{"source": "doc.txt", "kind": "text", "owner": owner}]],
                "distances": [[0.02]],
                "ids": [["chunk-1"]],
            }

    monkeypatch.setattr(ui_app, "db", _DB())
    monkeypatch.setattr(
        ui_app,
        "_run_generate",
        lambda provider, prompt, model, thinking_enabled=False, thinking_level="medium": "Generated answer.",
    )

    resp = client.post(
        "/chat/stream",
        data={
            "owner": "bob",
            "message": "Explain this",
            "mode": "study_guide",
            "provider": "claude_code",
            "model": "",
            "k": "4",
            "history_turns": "5",
            "history_enabled": "true",
        },
    )
    assert resp.status_code == 200

    events = [json.loads(line) for line in resp.text.splitlines() if line.strip()]
    assert any(e.get("type") == "done" for e in events)
    assert any(e.get("type") == "delta" and "Generated answer." in e.get("text", "") for e in events)

    exported = client.get("/chat/session/export", params={"owner": "bob"}).json()["history"]
    assert len(exported) == 2
    assert exported[0]["role"] == "user"
    assert exported[1]["role"] == "assistant"


def test_chat_stream_hides_status_and_uses_buffered_mode(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(ui_app, "SESSION_DIR", str(tmp_path / "sessions"))
    ui_app.CHAT_HISTORY.clear()
    client = _authed_client(monkeypatch)

    monkeypatch.setattr(ui_app, "embed_texts", lambda texts: [[0.1] for _ in texts])

    class _DB:
        def query(self, q_emb, k=8, where=None):
            owner = (where or {}).get("owner", "default")
            return {
                "documents": [["A relevant passage."]],
                "metadatas": [[{"source": "doc.txt", "kind": "text", "owner": owner}]],
                "distances": [[0.02]],
                "ids": [["chunk-1"]],
            }

    monkeypatch.setattr(ui_app, "db", _DB())
    monkeypatch.setattr(
        ui_app,
        "_run_generate",
        lambda provider, prompt, model, thinking_enabled=False, thinking_level="medium": "X" * 240,
    )

    resp = client.post(
        "/chat/stream",
        data={
            "owner": "carol",
            "message": "Explain this",
            "mode": "study_guide",
            "provider": "claude_code",
            "model": "",
            "k": "4",
            "history_turns": "5",
            "history_enabled": "true",
            "show_thinking": "false",
            "true_streaming": "false",
        },
    )
    assert resp.status_code == 200

    events = [json.loads(line) for line in resp.text.splitlines() if line.strip()]
    status_events = [e for e in events if e.get("type") == "status"]
    delta_events = [e for e in events if e.get("type") == "delta"]

    assert not status_events
    assert len(delta_events) >= 2
    assert any(e.get("type") == "done" for e in events)


def test_run_generate_falls_back_from_claude_code_to_claude_api(monkeypatch) -> None:
    monkeypatch.setattr(ui_app, "claude_code_cli_available", lambda: False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(
        ui_app,
        "generate_with_claude",
        lambda prompt, model=None, thinking_enabled=False, thinking_level="medium": "fallback-ok",
    )

    out = ui_app._run_generate("claude_code", "hello", model="", thinking_enabled=True)
    assert out == "fallback-ok"


def test_chat_stream_retrieval_timeout_returns_error_and_done(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(ui_app, "SESSION_DIR", str(tmp_path / "sessions"))
    monkeypatch.setattr(ui_app, "RETRIEVAL_TIMEOUT_SECONDS", 5.0)
    ui_app.CHAT_HISTORY.clear()
    client = _authed_client(monkeypatch)

    def _slow_embed(_texts):
        time.sleep(6.0)
        return [[0.1]]

    monkeypatch.setattr(ui_app, "embed_texts", _slow_embed)

    resp = client.post(
        "/chat/stream",
        data={
            "owner": "timeout-owner",
            "message": "Explain this",
            "mode": "study_guide",
            "provider": "claude_code",
            "model": "",
            "k": "4",
            "history_turns": "5",
            "history_enabled": "true",
        },
    )
    assert resp.status_code == 200

    events = [json.loads(line) for line in resp.text.splitlines() if line.strip()]
    errors = [e for e in events if e.get("type") == "error"]
    assert errors
    assert "Retrieval timed out" in errors[0].get("message", "")
    assert any(e.get("type") == "done" for e in events)


def test_chat_stream_empty_generation_returns_fallback_text(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(ui_app, "SESSION_DIR", str(tmp_path / "sessions"))
    ui_app.CHAT_HISTORY.clear()
    client = _authed_client(monkeypatch)

    monkeypatch.setattr(ui_app, "embed_texts", lambda texts: [[0.1] for _ in texts])

    class _DB:
        def query(self, q_emb, k=8, where=None):
            owner = (where or {}).get("owner", "default")
            return {
                "documents": [["A relevant passage."]],
                "metadatas": [[{"source": "doc.txt", "kind": "text", "owner": owner}]],
                "distances": [[0.02]],
                "ids": [["chunk-1"]],
            }

    monkeypatch.setattr(ui_app, "db", _DB())
    monkeypatch.setattr(
        ui_app,
        "_run_generate",
        lambda provider, prompt, model, thinking_enabled=False, thinking_level="medium": "",
    )

    resp = client.post(
        "/chat/stream",
        data={
            "owner": "empty-out",
            "message": "Explain this",
            "mode": "study_guide",
            "provider": "claude_code",
            "model": "",
            "k": "4",
            "history_turns": "5",
            "history_enabled": "true",
            "true_streaming": "false",
        },
    )
    assert resp.status_code == 200

    events = [json.loads(line) for line in resp.text.splitlines() if line.strip()]
    deltas = [e.get("text", "") for e in events if e.get("type") == "delta"]
    assert any("Generation returned empty output" in text for text in deltas)
    assert any(e.get("type") == "done" for e in events)
