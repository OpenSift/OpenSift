from __future__ import annotations

import json
import os
import re
import secrets
from typing import Any, Dict, List, Optional

DEFAULT_DIR = os.path.join(os.getcwd(), ".opensift_library")


def _safe_owner(owner: str) -> str:
    owner = owner.strip() or "default"
    owner = re.sub(r"[^a-zA-Z0-9._-]+", "_", owner)
    return owner[:128]


def library_path(owner: str, base_dir: str = DEFAULT_DIR) -> str:
    os.makedirs(base_dir, exist_ok=True)
    return os.path.join(base_dir, f"{_safe_owner(owner)}.json")


def load_library(owner: str, base_dir: str = DEFAULT_DIR) -> List[Dict[str, Any]]:
    path = library_path(owner, base_dir)
    if not os.path.exists(path):
        return []

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return []

    if not isinstance(data, list):
        return []

    out: List[Dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        if not isinstance(item.get("id"), str):
            continue
        if not isinstance(item.get("text"), str):
            continue
        out.append(item)
    return out


def save_library(owner: str, items: List[Dict[str, Any]], base_dir: str = DEFAULT_DIR) -> None:
    path = library_path(owner, base_dir)
    items = items[-1000:]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2, ensure_ascii=False)


def add_item(
    owner: str,
    title: str,
    text: str,
    mode: str = "",
    sources: Optional[List[Dict[str, Any]]] = None,
    created_at: str = "",
    base_dir: str = DEFAULT_DIR,
) -> Dict[str, Any]:
    items = load_library(owner, base_dir)
    item = {
        "id": secrets.token_urlsafe(9),
        "owner": owner,
        "title": (title or "").strip() or "Saved Study Item",
        "mode": (mode or "").strip(),
        "text": text,
        "sources": sources or [],
        "created_at": created_at,
    }
    items.append(item)
    save_library(owner, items, base_dir)
    return item


def get_item(owner: str, item_id: str, base_dir: str = DEFAULT_DIR) -> Optional[Dict[str, Any]]:
    items = load_library(owner, base_dir)
    for item in items:
        if item.get("id") == item_id:
            return item
    return None


def delete_item(owner: str, item_id: str, base_dir: str = DEFAULT_DIR) -> bool:
    items = load_library(owner, base_dir)
    kept = [x for x in items if x.get("id") != item_id]
    if len(kept) == len(items):
        return False
    save_library(owner, kept, base_dir)
    return True
