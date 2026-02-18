from __future__ import annotations

import re
from pathlib import Path


def test_chat_template_posts_use_csrf_fetch() -> None:
    path = Path(__file__).resolve().parents[1] / "templates" / "chat.html"
    content = path.read_text(encoding="utf-8")

    assert 'function csrfFetch(url, options = {})' in content
    assert 'data-csrf="{{ csrf_token }}"' in content

    # Ensure there are no raw POST fetch calls bypassing csrfFetch.
    raw_post_fetch = re.findall(r"fetch\([^)]*\{[^}]*method:\s*\"POST\"", content, flags=re.DOTALL)
    assert not raw_post_fetch
