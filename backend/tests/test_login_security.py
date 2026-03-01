from __future__ import annotations

from fastapi.testclient import TestClient

import ui_app


def _client_with_localhost(monkeypatch) -> TestClient:
    monkeypatch.setattr(
        ui_app.LocalhostOnlyMiddleware,
        "dispatch",
        lambda self, request, call_next: call_next(request),
    )
    return TestClient(ui_app.app)


def test_login_page_does_not_render_full_generated_token(monkeypatch) -> None:
    client = _client_with_localhost(monkeypatch)
    resp = client.get("/login")
    assert resp.status_code == 200
    assert ui_app.GEN_TOKEN not in resp.text


def test_set_password_without_token_when_no_password(monkeypatch) -> None:
    client = _client_with_localhost(monkeypatch)
    monkeypatch.setattr(ui_app, "_has_password", lambda: False)

    resp = client.post(
        "/set-password",
        data={
            "token": "",
            "new_password": "securepass123",
            "confirm_password": "securepass123",
        },
        follow_redirects=False,
    )
    assert resp.status_code == 303
