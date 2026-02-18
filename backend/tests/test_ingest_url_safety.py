from __future__ import annotations

import socket

import pytest

from app.ingest import _validate_remote_url_sync


def test_validate_remote_url_blocks_localhost() -> None:
    with pytest.raises(RuntimeError, match="blocked"):
        _validate_remote_url_sync("http://localhost:8000/test")


def test_validate_remote_url_blocks_private_dns_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_getaddrinfo(*args, **kwargs):
        return [
            (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("10.0.0.12", 0)),
        ]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)

    with pytest.raises(RuntimeError, match="private/local"):
        _validate_remote_url_sync("https://example.com")


def test_validate_remote_url_blocks_private_ip_literal() -> None:
    with pytest.raises(RuntimeError, match=r"(?i)private/local"):
        _validate_remote_url_sync("http://10.1.2.3")


def test_validate_remote_url_allows_public_dns_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_getaddrinfo(*args, **kwargs):
        return [
            (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("93.184.216.34", 0)),
        ]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)

    _validate_remote_url_sync("https://example.com/resource")
