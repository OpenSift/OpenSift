from __future__ import annotations

import json
import os
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


def test_settings_page_has_expected_tabs_and_controls(monkeypatch) -> None:
    client = _authed_client(monkeypatch)
    resp = client.get("/settings", params={"owner": "bio101"})
    assert resp.status_code == 200

    html = resp.text
    assert "OpenSift Settings" in html
    assert 'data-tab="authTab"' in html
    assert 'data-tab="soulTab"' in html
    assert 'data-tab="wellnessTab"' in html
    assert 'id="savePasswordBtn"' in html
    assert 'id="rotateTokenBtn"' in html
    assert 'id="saveSoulBtn"' in html
    assert 'id="saveWellnessBtn"' in html
    assert 'id="backToChatBtn"' in html
    assert 'id="saveProviderSettingsBtn"' in html
    assert 'id="openaiApiKey"' in html
    assert 'id="anthropicApiKey"' in html
    assert 'id="claudeCodeToken"' in html
    assert 'id="codexOauthToken"' in html
    assert 'id="claudeCodeCmd"' in html
    assert 'id="codexCmd"' in html
    assert 'id="installClaudeBtn"' in html
    assert 'id="installClaudeCodeBtn"' in html
    assert 'id="installCodexBtn"' in html
    assert 'id="providerStatus"' in html
    assert 'data-csrf="' in html
    assert 'class="app"' in html
    assert 'class="sidebar"' in html
    assert 'class="main"' in html


def test_chat_page_has_ingest_controls(monkeypatch) -> None:
    client = _authed_client(monkeypatch)
    resp = client.get("/chat", params={"owner": "bio101"})
    assert resp.status_code == 200

    html = resp.text
    assert 'id="ingestUrlBtn"' in html
    assert 'id="ingestFileBtn"' in html
    assert 'id="ingestOut"' in html
    assert 'id="ingestTitle"' in html
    assert 'id="ingestUrl"' in html
    assert 'id="libraryBtn"' in html


def test_library_page_has_expected_controls(monkeypatch) -> None:
    client = _authed_client(monkeypatch)
    resp = client.get("/library", params={"owner": "bio101"})
    assert resp.status_code == 200

    html = resp.text
    assert "OpenSift Library" in html
    assert 'id="goChatBtn"' in html
    assert 'id="saveNoteBtn"' in html
    assert 'id="saveUrlBtn"' in html
    assert 'id="uploadBtn"' in html
    assert 'id="uploadProgress"' in html
    assert 'id="searchInput"' in html
    assert 'id="kindFilter"' in html
    assert 'id="listPane"' in html
    assert 'id="detailsPanel"' in html
    assert 'data-csrf="' in html


def test_chat_stream_emits_provider_model_discrepancy_status(monkeypatch, tmp_path: Path) -> None:
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

    async def _fake_stream_anthropic(prompt: str, model: str, thinking_enabled: bool = False, thinking_level: str = "medium"):
        _ = (prompt, model, thinking_enabled, thinking_level)
        yield "stream-ok"

    monkeypatch.setattr(ui_app, "db", _DB())
    monkeypatch.setattr(ui_app, "_stream_anthropic", _fake_stream_anthropic)

    resp = client.post(
        "/chat/stream",
        data={
            "owner": "bob",
            "message": "Explain this",
            "mode": "study_guide",
            "provider": "openai",
            "model": "claude-sonnet-4-6",
            "k": "4",
            "history_turns": "5",
            "history_enabled": "true",
            "show_thinking": "true",
            "true_streaming": "true",
        },
    )
    assert resp.status_code == 200

    events = [json.loads(line) for line in resp.text.splitlines() if line.strip()]
    statuses = [e.get("text", "") for e in events if e.get("type") == "status"]

    assert any("switching provider to Claude API" in s for s in statuses)
    assert any(e.get("type") == "delta" and "stream-ok" in e.get("text", "") for e in events)
    assert any(e.get("type") == "done" for e in events)


def test_chat_stream_falls_back_from_codex_to_openai_when_cli_missing(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(ui_app, "SESSION_DIR", str(tmp_path / "sessions"))
    ui_app.CHAT_HISTORY.clear()
    client = _authed_client(monkeypatch)

    monkeypatch.setenv("OPENAI_API_KEY", "x")
    monkeypatch.setattr(ui_app, "codex_cli_available", lambda: False)
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

    async def _fake_stream_openai(prompt: str, model: str):
        _ = (prompt, model)
        yield "openai-fallback-stream"

    monkeypatch.setattr(ui_app, "db", _DB())
    monkeypatch.setattr(ui_app, "_stream_openai", _fake_stream_openai)

    resp = client.post(
        "/chat/stream",
        data={
            "owner": "bob",
            "message": "Explain this",
            "mode": "study_guide",
            "provider": "codex",
            "model": "gpt-5.3-codex",
            "k": "4",
            "history_turns": "5",
            "history_enabled": "true",
            "show_thinking": "true",
            "true_streaming": "true",
        },
    )
    assert resp.status_code == 200

    events = [json.loads(line) for line in resp.text.splitlines() if line.strip()]
    statuses = [e.get("text", "") for e in events if e.get("type") == "status"]
    assert any("Codex CLI unavailable; using OpenAI API fallback" in s for s in statuses)
    assert any(e.get("type") == "delta" and "openai-fallback-stream" in e.get("text", "") for e in events)
    assert any(e.get("type") == "done" for e in events)


def test_chat_stream_reports_hf_warmup_status(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(ui_app, "SESSION_DIR", str(tmp_path / "sessions"))
    ui_app.CHAT_HISTORY.clear()
    client = _authed_client(monkeypatch)

    monkeypatch.setattr(ui_app, "using_local_embeddings", lambda: True)
    monkeypatch.setattr(ui_app, "local_embedding_model_loaded", lambda: False)
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
            "owner": "alice",
            "message": "Explain this",
            "mode": "study_guide",
            "provider": "claude_code",
            "k": "4",
        },
    )
    assert resp.status_code == 200
    events = [json.loads(line) for line in resp.text.splitlines() if line.strip()]
    statuses = [e.get("text", "") for e in events if e.get("type") == "status"]
    assert any("pulling embeddings model from Hugging Face" in s for s in statuses)


def test_settings_provider_endpoint_masks_and_updates(monkeypatch, tmp_path: Path) -> None:
    client = _authed_client(monkeypatch)
    env_path = tmp_path / ".env"
    monkeypatch.setattr(ui_app, "ENV_FILE_PATH", str(env_path))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-openai-abcdef")
    monkeypatch.delenv("OPENSIFT_CODEX_CMD", raising=False)

    get_resp = client.get("/chat/settings/providers")
    assert get_resp.status_code == 200
    payload = get_resp.json()
    assert payload["openai_api_key_set"] is True
    assert "sk-test-openai-abcdef" not in json.dumps(payload)
    assert payload["claude_code_cmd"]
    assert payload["codex_cmd"]

    post_resp = client.post(
        "/chat/settings/providers",
        data={
            "anthropic_api_key": "ant-xyz-12345",
            "codex_cmd": "/usr/local/bin/codex",
        },
    )
    assert post_resp.status_code == 200
    assert env_path.exists()
    text = env_path.read_text(encoding="utf-8")
    assert "ANTHROPIC_API_KEY=ant-xyz-12345" in text
    assert "OPENSIFT_CODEX_CMD=/usr/local/bin/codex" in text
    mode = os.stat(env_path).st_mode & 0o777
    assert mode == 0o600


def test_settings_provider_install_endpoint_updates_command(monkeypatch, tmp_path: Path) -> None:
    client = _authed_client(monkeypatch)
    env_path = tmp_path / ".env"
    monkeypatch.setattr(ui_app, "ENV_FILE_PATH", str(env_path))

    def _fake_install(target: str):
        assert target == "codex"
        return {
            "target": "codex",
            "package": "@openai/codex",
            "binary": "codex",
            "env_key": "OPENSIFT_CODEX_CMD",
            "cmd_path": "/app/.opensift_tools/bin/codex",
            "install_log_tail": "installed ok",
        }

    monkeypatch.setattr(ui_app, "_install_provider_cli", _fake_install)

    resp = client.post("/chat/settings/providers/install", data={"target": "codex"})
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["installed_target"] == "codex"
    assert payload["installed_cmd_path"] == "/app/.opensift_tools/bin/codex"

    text = env_path.read_text(encoding="utf-8")
    assert "OPENSIFT_CODEX_CMD=/app/.opensift_tools/bin/codex" in text


def test_settings_provider_install_stream_endpoint(monkeypatch) -> None:
    client = _authed_client(monkeypatch)

    async def _fake_events(target: str):
        assert target == "claude_code"
        yield {"type": "start", "target": target}
        yield {"type": "progress", "percent": 42, "eta_seconds": 12}
        yield {"type": "done", "target": target, "cmd_path": "/app/.opensift_tools/bin/claude"}

    monkeypatch.setattr(ui_app, "_install_provider_cli_stream_events", _fake_events)

    resp = client.post("/chat/settings/providers/install/stream", data={"target": "claude_code"})
    assert resp.status_code == 200
    lines = [line for line in resp.text.splitlines() if line.strip()]
    events = [json.loads(line) for line in lines]
    assert events[0]["type"] == "start"
    assert events[1]["type"] == "progress"
    assert events[-1]["type"] == "done"
