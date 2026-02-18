from __future__ import annotations

from pathlib import Path


def test_chat_template_has_no_user_data_innerhtml_interpolation() -> None:
    path = Path(__file__).resolve().parents[1] / "templates" / "chat.html"
    content = path.read_text(encoding="utf-8")

    banned_tokens = [
        "${s.owner}",
        "${s.last_ts",
        "${item.title",
        "${item.preview",
        "${a.title",
        "${a.notes",
        "${c.front",
        "${c.back",
        "${c.tags",
    ]

    for token in banned_tokens:
        assert token not in content, f"Found unsafe template interpolation token: {token}"
