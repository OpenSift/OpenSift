from __future__ import annotations

import os
import re
import secrets
import threading
from typing import Any, Dict, List, Optional

from app.atomic_io import atomic_write_json, path_lock

DEFAULT_DIR = os.path.join(os.getcwd(), ".opensift_sources")
_LOCK = threading.RLock()


def _safe_owner(owner: str) -> str:
    owner = (owner or "").strip() or "default"
    owner = re.sub(r"[^a-zA-Z0-9._-]+", "_", owner)
    return owner[:128]


def _owner_manifest_path(owner: str, base_dir: str = DEFAULT_DIR) -> str:
    os.makedirs(base_dir, exist_ok=True)
    return os.path.join(base_dir, f"{_safe_owner(owner)}.json")


def _owner_files_dir(owner: str, base_dir: str = DEFAULT_DIR) -> str:
    path = os.path.join(base_dir, "_files", _safe_owner(owner))
    os.makedirs(path, exist_ok=True)
    return path


def new_source_id() -> str:
    return secrets.token_urlsafe(9)


def load_items(owner: str, base_dir: str = DEFAULT_DIR) -> List[Dict[str, Any]]:
    path = _owner_manifest_path(owner, base_dir)
    with _LOCK, path_lock(path):
        if not os.path.exists(path):
            return []
        try:
            import json

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
        out.append(item)
    return out


def save_items(owner: str, items: List[Dict[str, Any]], base_dir: str = DEFAULT_DIR) -> None:
    path = _owner_manifest_path(owner, base_dir)
    with _LOCK:
        atomic_write_json(path, items[-5000:])


def add_item(owner: str, item: Dict[str, Any], base_dir: str = DEFAULT_DIR) -> Dict[str, Any]:
    with _LOCK:
        items = load_items(owner, base_dir)
        items.append(item)
        save_items(owner, items, base_dir)
    return item


def get_item(owner: str, item_id: str, base_dir: str = DEFAULT_DIR) -> Optional[Dict[str, Any]]:
    for item in load_items(owner, base_dir):
        if item.get("id") == item_id:
            return item
    return None


def delete_item(owner: str, item_id: str, base_dir: str = DEFAULT_DIR) -> Optional[Dict[str, Any]]:
    with _LOCK:
        items = load_items(owner, base_dir)
        removed: Optional[Dict[str, Any]] = None
        kept: List[Dict[str, Any]] = []
        for item in items:
            if item.get("id") == item_id and removed is None:
                removed = item
                continue
            kept.append(item)
        if removed is None:
            return None
        save_items(owner, kept, base_dir)
        return removed


def update_item(owner: str, item_id: str, patch: Dict[str, Any], base_dir: str = DEFAULT_DIR) -> Optional[Dict[str, Any]]:
    with _LOCK:
        items = load_items(owner, base_dir)
        updated: Optional[Dict[str, Any]] = None
        for idx, item in enumerate(items):
            if item.get("id") != item_id:
                continue
            merged = dict(item)
            merged.update(patch or {})
            items[idx] = merged
            updated = merged
            break
        if updated is None:
            return None
        save_items(owner, items, base_dir)
        return updated


def write_text_blob(owner: str, source_id: str, text: str, base_dir: str = DEFAULT_DIR) -> str:
    files_dir = _owner_files_dir(owner, base_dir)
    path = os.path.join(files_dir, f"{source_id}.txt")
    with _LOCK:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text or "")
    return path


def read_text_blob(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""


def write_binary_blob(owner: str, source_id: str, filename: str, data: bytes, base_dir: str = DEFAULT_DIR) -> str:
    files_dir = _owner_files_dir(owner, base_dir)
    safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", (filename or "upload").strip())[:180] or "upload"
    path = os.path.join(files_dir, f"{source_id}__{safe}")
    with _LOCK:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "wb") as f:
            f.write(data or b"")
    return path


def remove_file(path: str) -> None:
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except Exception:
        pass
