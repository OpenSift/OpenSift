from __future__ import annotations

from pathlib import Path

from app import soul


def test_soul_path_falls_back_when_home_is_nonexistent(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", "/nonexistent")
    monkeypatch.delenv("OPENSIFT_SOUL_PATH", raising=False)

    p = soul.soul_path()
    assert p == str(tmp_path / ".opensift" / "SOUL.md")


def test_ensure_soul_file_creates_default_in_fallback_path(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", "/nonexistent")
    monkeypatch.delenv("OPENSIFT_SOUL_PATH", raising=False)

    path = soul.ensure_soul_file()
    assert path == str(tmp_path / ".opensift" / "SOUL.md")
    assert (tmp_path / ".opensift" / "SOUL.md").exists()
