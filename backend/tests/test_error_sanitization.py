from __future__ import annotations

from fastapi.testclient import TestClient

import mcp_server
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


def test_session_import_invalid_payload_is_generic(monkeypatch) -> None:
    client = _authed_client(monkeypatch)
    resp = client.post("/chat/session/import", data={"owner": "alice", "payload": "{", "merge": "false"})
    assert resp.status_code == 400
    assert resp.json()["error"] == "invalid_payload"


def test_provider_settings_invalid_path_error_is_generic(monkeypatch) -> None:
    client = _authed_client(monkeypatch)
    resp = client.post("/chat/settings/providers", data={"codex_cmd": "codex;rm -rf /"})
    assert resp.status_code == 400
    assert resp.json()["error"] == "invalid_provider_settings"


def test_mcp_generate_error_is_sanitized(monkeypatch) -> None:
    async def _fake_search(*args, **kwargs):
        _ = (args, kwargs)
        return {"ok": True, "results": []}

    def _boom(*args, **kwargs):
        _ = (args, kwargs)
        raise RuntimeError("sensitive provider detail")

    monkeypatch.setattr(mcp_server, "search", _fake_search)
    monkeypatch.setattr(mcp_server, "generate_with_openai", _boom)

    out = mcp_server.anyio.run(lambda: mcp_server.sift_generate(query="hello"))
    assert out["ok"] is False
    assert out["error"] == "generation_failed"
    assert "sensitive provider detail" not in str(out)
