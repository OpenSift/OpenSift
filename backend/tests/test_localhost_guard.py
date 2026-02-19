from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

import ui_app
import opensift


def _guarded_test_client() -> TestClient:
    app = FastAPI()
    app.add_middleware(ui_app.LocalhostOnlyMiddleware)

    @app.get("/health")
    async def _health():
        return {"ok": True}

    return TestClient(app)


def test_private_or_loopback_ip_helper() -> None:
    assert ui_app._is_private_or_loopback_ip("127.0.0.1")
    assert ui_app._is_private_or_loopback_ip("::1")
    assert ui_app._is_private_or_loopback_ip("172.18.0.2")
    assert not ui_app._is_private_or_loopback_ip("8.8.8.8")
    assert not ui_app._is_private_or_loopback_ip("not-an-ip")


def test_gateway_health_probe_host_for_bind_all_addresses() -> None:
    assert opensift._health_probe_host("0.0.0.0") == "127.0.0.1"
    assert opensift._health_probe_host("::") == "127.0.0.1"
    assert opensift._health_probe_host("[::]") == "127.0.0.1"
    assert opensift._health_probe_host("127.0.0.1") == "127.0.0.1"


def test_localhost_middleware_allows_0_0_0_0_host_in_private_mode(monkeypatch) -> None:
    monkeypatch.setenv("OPENSIFT_ALLOW_PRIVATE_CLIENTS", "true")
    monkeypatch.setattr(ui_app, "_is_private_or_loopback_ip", lambda _host: True)
    client = _guarded_test_client()
    resp = client.get("/health", headers={"host": "0.0.0.0:8001"})
    assert resp.status_code == 200


def test_localhost_middleware_block_message_mentions_private_relay(monkeypatch) -> None:
    monkeypatch.delenv("OPENSIFT_ALLOW_PRIVATE_CLIENTS", raising=False)
    monkeypatch.setattr(ui_app, "_is_private_or_loopback_ip", lambda _host: False)
    monkeypatch.setattr(ui_app, "_client_ip_is_trusted", lambda _host: False)
    client = _guarded_test_client()
    resp = client.get("/health", headers={"host": "localhost:8001"})
    assert resp.status_code == 403
    text = resp.text
    assert "localhost-only access" in text
    assert "Private Relay" in text
    assert "127.0.0.1:8001" in text


def test_localhost_middleware_allows_trusted_client_ip(monkeypatch) -> None:
    monkeypatch.delenv("OPENSIFT_ALLOW_PRIVATE_CLIENTS", raising=False)
    monkeypatch.setenv("OPENSIFT_TRUSTED_CLIENT_IPS", "203.0.113.10")
    monkeypatch.setattr(ui_app, "_is_private_or_loopback_ip", lambda _host: False)
    monkeypatch.setattr(ui_app, "_client_ip_is_trusted", lambda _host: True)
    client = _guarded_test_client()
    resp = client.get("/health", headers={"host": "localhost:8001"})
    assert resp.status_code == 200
