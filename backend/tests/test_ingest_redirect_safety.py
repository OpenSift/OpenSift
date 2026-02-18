from __future__ import annotations

import pytest

from app.ingest import _resolve_redirect_url


def test_resolve_redirect_url_requires_location() -> None:
    with pytest.raises(RuntimeError, match="missing Location"):
        _resolve_redirect_url("https://example.com/a", "")


def test_resolve_redirect_url_resolves_relative_path() -> None:
    out = _resolve_redirect_url("https://example.com/a/b", "../c")
    assert out == "https://example.com/c"


def test_resolve_redirect_url_rejects_non_http_scheme() -> None:
    with pytest.raises(RuntimeError, match="unsupported URL scheme"):
        _resolve_redirect_url("https://example.com/a", "file:///etc/passwd")
