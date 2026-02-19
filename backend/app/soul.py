from __future__ import annotations

import os
import re
from typing import Any, Dict

SOUL_PATH_ENV = "OPENSIFT_SOUL_PATH"
GLOBAL_KEY = "__global__"

_OWNER_RE = re.compile(r"^[a-zA-Z0-9._-]{1,128}$")
_HEADER_RE = re.compile(r"^\s*##\s*owner\s*:\s*([a-zA-Z0-9._-]{1,128})\s*$", re.IGNORECASE)
_GLOBAL_HEADER_RE = re.compile(r"^\s*##\s*global\s*$", re.IGNORECASE)

DEFAULT_TEMPLATE = """# OpenSift SOUL

This file stores study-style personality settings for OpenSift.
Edit by hand or through OpenSift UI/CLI style controls.

## global
You are a calm, structured study coach.
- Explain ideas in plain language first, then add depth.
- Use concise section headings and bullet points.
- End with a short recap and 3 quick self-check questions.
"""


def _normalize_owner(owner: str) -> str:
    raw = (owner or "").strip()
    if not raw:
        return "default"
    safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", raw)[:128]
    if not safe:
        return "default"
    if not _OWNER_RE.match(safe):
        return "default"
    return safe


def soul_path() -> str:
    configured = (os.getenv(SOUL_PATH_ENV, "") or "").strip()
    if configured:
        return configured
    # In containers HOME may be /nonexistent; keep SOUL under app cwd in that case.
    home = os.path.expanduser("~")
    if home and home not in ("/", "/nonexistent"):
        return os.path.join(home, ".opensift", "SOUL.md")
    return os.path.join(os.getcwd(), ".opensift", "SOUL.md")


def ensure_soul_file(path: str | None = None) -> str:
    p = path or soul_path()
    candidates = [p]
    fallback = os.path.join(os.getcwd(), ".opensift", "SOUL.md")
    if fallback not in candidates:
        candidates.append(fallback)
    for candidate in candidates:
        try:
            parent = os.path.dirname(candidate)
            if parent:
                os.makedirs(parent, exist_ok=True)
            if not os.path.exists(candidate):
                with open(candidate, "w", encoding="utf-8") as f:
                    f.write(DEFAULT_TEMPLATE.rstrip() + "\n")
            return candidate
        except OSError:
            continue
    raise PermissionError(f"Unable to initialize SOUL file; attempted: {candidates}")


def load_soul_map(path: str | None = None) -> Dict[str, str]:
    p = ensure_soul_file(path)
    with open(p, "r", encoding="utf-8") as f:
        content = f.read()

    lines = content.splitlines()
    result: Dict[str, str] = {}
    current_owner: str | None = None
    buf: list[str] = []

    def flush() -> None:
        nonlocal current_owner, buf
        if current_owner is None:
            buf = []
            return
        text = "\n".join(buf).strip()
        if text:
            result[current_owner] = text
        buf = []

    for line in lines:
        if _GLOBAL_HEADER_RE.match(line):
            flush()
            current_owner = GLOBAL_KEY
            continue
        m = _HEADER_RE.match(line)
        if m:
            flush()
            current_owner = _normalize_owner(m.group(1))
            continue
        if current_owner is not None:
            buf.append(line)
    flush()
    return result


def get_study_style(owner: str, path: str | None = None) -> str:
    migrate_legacy_owner_styles(path)
    o = _normalize_owner(owner)
    data = load_soul_map(path)
    # Global style applies everywhere when configured.
    if data.get(GLOBAL_KEY, "").strip():
        return data[GLOBAL_KEY].strip()
    if o in data:
        return data[o]
    return data.get("default", "").strip()


def set_study_style(owner: str, style: str, path: str | None = None) -> Dict[str, str]:
    # Global by default for "saved everywhere" behavior.
    if not owner or owner.lower().strip() in ("global", "*", "all"):
        return set_global_style(style, path=path)
    o = _normalize_owner(owner)
    data = load_soul_map(path)
    clean = (style or "").strip()
    if clean:
        data[o] = clean
    else:
        data.pop(o, None)

    _write_soul_map(data, path)
    return data


def get_global_style(path: str | None = None) -> str:
    migrate_legacy_owner_styles(path)
    data = load_soul_map(path)
    return data.get(GLOBAL_KEY, "").strip()


def set_global_style(style: str, path: str | None = None) -> Dict[str, str]:
    data = load_soul_map(path)
    clean = (style or "").strip()
    if clean:
        data[GLOBAL_KEY] = clean
    else:
        data.pop(GLOBAL_KEY, None)
    _write_soul_map(data, path)
    return data


def _write_soul_map(data: Dict[str, str], path: str | None = None) -> None:
    p = ensure_soul_file(path)
    owners = sorted(data.keys())

    lines: list[str] = [
        "# OpenSift SOUL",
        "",
        "This file stores study-style personality settings for OpenSift.",
        "Edit by hand or through OpenSift UI/CLI style controls.",
        "",
    ]
    global_text = (data.get(GLOBAL_KEY) or "").strip()
    if global_text:
        lines.append("## global")
        lines.append(global_text)
        lines.append("")

    for o in owners:
        if o == GLOBAL_KEY:
            continue
        text = (data.get(o) or "").strip()
        if not text:
            continue
        lines.append(f"## owner:{o}")
        lines.append(text)
        lines.append("")

    with open(p, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")


def migrate_legacy_owner_styles(path: str | None = None) -> Dict[str, Any]:
    """
    Merge legacy per-owner entries into the global style block.
    Idempotent: runs safely multiple times.
    """
    data = load_soul_map(path)
    owners = [k for k in sorted(data.keys()) if k != GLOBAL_KEY]
    if not owners:
        return {"ok": True, "migrated": False, "owners": 0}

    owner_sections: list[str] = []
    for o in owners:
        text = (data.get(o) or "").strip()
        if not text:
            continue
        owner_sections.append(f"[owner:{o}]\n{text}")

    legacy_block = "\n\n".join(owner_sections).strip()
    current_global = (data.get(GLOBAL_KEY) or "").strip()

    if current_global:
        if legacy_block and legacy_block not in current_global:
            merged = f"{current_global}\n\nLegacy owner style notes migrated:\n\n{legacy_block}".strip()
        else:
            merged = current_global
    else:
        merged = f"Merged legacy owner styles:\n\n{legacy_block}".strip() if legacy_block else ""

    next_data: Dict[str, str] = {}
    if merged:
        next_data[GLOBAL_KEY] = merged
    _write_soul_map(next_data, path)
    return {"ok": True, "migrated": True, "owners": len(owners)}
