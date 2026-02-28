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


def test_library_page_renders_pdf_preview_with_fallback_actions(monkeypatch) -> None:
    client = _authed_client(monkeypatch)

    resp = client.get(
        "/library",
        params={
            "owner": "bio101",
            "title": "Week 2 Slides",
            "pdf_url": "https://example.com/notes.pdf",
        },
    )
    assert resp.status_code == 200

    html = resp.text
    assert "OpenSift Library" in html
    assert 'id="goChatBtn"' in html
    assert 'id="saveNoteBtn"' in html
    assert 'id="saveUrlBtn"' in html
    assert 'id="uploadBtn"' in html
    assert 'id="listPane"' in html
    assert 'id="detailsPanel"' in html
    assert "PDF preview is not available in this browser." in html
    assert "Safari with Lockdown Mode enabled" in html


def test_library_page_rejects_non_http_pdf_url(monkeypatch) -> None:
    client = _authed_client(monkeypatch)

    resp = client.get(
        "/library",
        params={
            "owner": "bio101",
            "title": "Unsafe",
            "pdf_url": "javascript:alert(1)",
        },
    )
    assert resp.status_code == 200

    html = resp.text
    assert "OpenSift Library" in html
    assert 'href="https://example.com/notes.pdf"' not in html
    assert "javascript:alert(1)" not in html
    assert 'id="goChatBtn"' in html
