from __future__ import annotations

import re
from pathlib import Path


def test_chat_template_has_no_user_data_innerhtml_interpolation() -> None:
    path = Path(__file__).resolve().parents[1] / "templates" / "chat.html"
    content = path.read_text(encoding="utf-8")

    # Guard against reintroducing unsafe HTML rendering with template interpolation.
    matches = re.findall(r"innerHTML\s*=\s*`[^`]*\$\{", content, flags=re.DOTALL)
    assert not matches
