from __future__ import annotations

import pytest

from app.ingest import _validate_remote_url_sync


def test_validate_remote_url_rejects_embedded_credentials() -> None:
    with pytest.raises(RuntimeError):
        _validate_remote_url_sync("https://user:pass@example.com/article")


def test_validate_remote_url_rejects_non_standard_port() -> None:
    with pytest.raises(RuntimeError):
        _validate_remote_url_sync("https://example.com:8443/article")
