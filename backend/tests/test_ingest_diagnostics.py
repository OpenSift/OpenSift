from __future__ import annotations

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


def test_chat_ingest_url_returns_diagnostics_and_latest_snapshot(monkeypatch) -> None:
    client = _authed_client(monkeypatch)

    async def _fake_fetch_url_text_with_diagnostics(url: str):
        _ = url
        return (
            "Article",
            "This is a long enough article body for chunking. " * 20,
            {"kind": "url", "engine": "httpx+bs4", "request_url": "https://example.com/a"},
        )

    class _DB:
        def add(self, ids, documents, metadatas, embeddings):
            _ = (ids, documents, metadatas, embeddings)
            return None

    monkeypatch.setattr(ui_app, "fetch_url_text_with_diagnostics", _fake_fetch_url_text_with_diagnostics)
    monkeypatch.setattr(ui_app, "embed_texts", lambda texts: [[0.1] for _ in texts])
    monkeypatch.setattr(ui_app, "db", _DB())

    resp = client.post("/chat/ingest/url", data={"owner": "bio101", "url": "https://example.com/a", "source_title": "A"})
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["ok"] is True
    assert payload["diagnostics"]["status"] == "ok"
    assert payload["diagnostics"]["source"] == "chat_ingest_url"
    assert payload["diagnostics"]["parser"]["engine"] == "httpx+bs4"
    assert int(payload["diagnostics"]["chunk"]["count"]) >= 1
    assert int(payload["diagnostics"]["index"]["vector_count"]) >= 1

    latest = client.get("/chat/ingest/diagnostics/latest", params={"owner": "bio101"})
    assert latest.status_code == 200
    latest_payload = latest.json()
    assert latest_payload["ok"] is True
    assert latest_payload["diagnostics"]["source"] == "chat_ingest_url"
    assert latest_payload["diagnostics"]["status"] == "ok"


def test_chat_ingest_file_records_error_diagnostics(monkeypatch) -> None:
    client = _authed_client(monkeypatch)

    resp = client.post(
        "/chat/ingest/file",
        data={"owner": "bio101"},
        files={"file": ("bad.csv", b"a,b,c\n1,2,3", "text/csv")},
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["ok"] is False
    assert payload["diagnostics"]["status"] == "error"
    assert payload["diagnostics"]["error"]["code"] == "unsupported_file_type"

    latest = client.get("/chat/ingest/diagnostics/latest", params={"owner": "bio101"})
    assert latest.status_code == 200
    latest_payload = latest.json()
    assert latest_payload["diagnostics"]["status"] == "error"
    assert latest_payload["diagnostics"]["error"]["code"] == "unsupported_file_type"
