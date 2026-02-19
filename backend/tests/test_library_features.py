from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

import source_store
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


def test_library_list_pagination_sort_and_filters(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(ui_app, "SOURCE_DIR", str(tmp_path / "sources"))
    owner = "bio101"
    base = str(tmp_path / "sources")

    items = [
        {"id": "a1", "title": "Zeta Notes", "kind": "note", "folder": "week1", "tags": "pain,rat", "created_at": "2026-01-01T00:00:00Z"},
        {"id": "a2", "title": "Alpha PDF", "kind": "pdf", "folder": "week2", "tags": "exam", "created_at": "2026-01-03T00:00:00Z"},
        {"id": "a3", "title": "Beta URL", "kind": "url", "folder": "week1", "tags": "pain,review", "created_at": "2026-01-02T00:00:00Z"},
    ]
    for item in items:
        source_store.add_item(owner, item, base)

    client = _authed_client(monkeypatch)
    resp = client.get(
        "/chat/library/list",
        params={
            "owner": owner,
            "folder": "week1",
            "tags": "pain",
            "sort_by": "title",
            "sort_dir": "asc",
            "page": 1,
            "page_size": 1,
        },
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["pagination"]["total"] == 2
    assert payload["pagination"]["total_pages"] == 2
    assert payload["items"][0]["title"] == "Beta URL"
    assert "week1" in payload["folders"]


def test_library_update_metadata(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(ui_app, "SOURCE_DIR", str(tmp_path / "sources"))
    owner = "bio101"
    base = str(tmp_path / "sources")
    source_store.add_item(
        owner,
        {"id": "item1", "title": "Old", "kind": "note", "folder": "", "tags": "", "created_at": "2026-01-01T00:00:00Z"},
        base,
    )

    client = _authed_client(monkeypatch)
    resp = client.post(
        "/chat/library/update",
        data={"owner": owner, "item_id": "item1", "title": "New Title", "folder": "wk3", "tags": "tag1,tag2"},
    )
    assert resp.status_code == 200
    item = resp.json()["item"]
    assert item["title"] == "New Title"
    assert item["folder"] == "wk3"
    assert item["tags"] == "tag1,tag2"


def test_chat_stream_uses_selected_library_ids_as_pinned_context(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(ui_app, "SESSION_DIR", str(tmp_path / "sessions"))
    monkeypatch.setattr(ui_app, "SOURCE_DIR", str(tmp_path / "sources"))
    ui_app.CHAT_HISTORY.clear()
    owner = "bio777"

    source_id = "src1"
    text_path = source_store.write_text_blob(owner, source_id, "Pinned context body about asomaesthesia.", str(tmp_path / "sources"))
    source_store.add_item(
        owner,
        {
            "id": source_id,
            "title": "Pinned Source",
            "kind": "note",
            "text_path": text_path,
            "created_at": "2026-01-01T00:00:00Z",
            "folder": "wk1",
            "tags": "pain",
        },
        str(tmp_path / "sources"),
    )

    monkeypatch.setattr(ui_app, "embed_texts", lambda texts: [[0.1] for _ in texts])

    class _DB:
        def query(self, q_emb, k=8, where=None):
            return {"documents": [[]], "metadatas": [[]], "distances": [[]], "ids": [[]]}

    monkeypatch.setattr(ui_app, "db", _DB())
    monkeypatch.setattr(
        ui_app,
        "_run_generate",
        lambda provider, prompt, model, thinking_enabled=False, thinking_level="medium": "Pinned answer.",
    )

    client = _authed_client(monkeypatch)
    resp = client.post(
        "/chat/stream",
        data={
            "owner": owner,
            "message": "What is asomaesthesia?",
            "mode": "study_guide",
            "provider": "claude_code",
            "selected_library_ids": source_id,
            "true_streaming": "false",
        },
    )
    assert resp.status_code == 200
    events = [json.loads(line) for line in resp.text.splitlines() if line.strip()]
    assert any(e.get("type") == "delta" and "Pinned answer." in e.get("text", "") for e in events)
    src_events = [e for e in events if e.get("type") == "sources"]
    assert src_events
    assert any((s.get("kind") == "library_selected") for s in src_events[0].get("sources", []))
