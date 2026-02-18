from __future__ import annotations

import json
import os
import re
import threading
from typing import Any, Dict, List

from app.atomic_io import atomic_write_json, path_lock

DEFAULT_DIR = os.path.join(os.getcwd(), ".opensift_sessions")
_LOCK = threading.RLock()


def _safe_owner(owner: str) -> str:
    owner = owner.strip() or "default"
    owner = re.sub(r"[^a-zA-Z0-9._-]+", "_", owner)
    return owner[:128]


def session_path(owner: str, base_dir: str = DEFAULT_DIR) -> str:
    os.makedirs(base_dir, exist_ok=True)
    return os.path.join(base_dir, f"{_safe_owner(owner)}.json")


def load_session(owner: str, base_dir: str = DEFAULT_DIR) -> List[Dict[str, Any]]:
    path = session_path(owner, base_dir)
    with _LOCK, path_lock(path):
        if not os.path.exists(path):
            return []
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                # minimal validation
                out: List[Dict[str, Any]] = []
                for item in data:
                    if not isinstance(item, dict):
                        continue
                    role = item.get("role")
                    text = item.get("text")
                    if role in ("user", "assistant") and isinstance(text, str):
                        out.append(item)
                return out
        except Exception:
            return []
    return []


def save_session(owner: str, history: List[Dict[str, Any]], base_dir: str = DEFAULT_DIR) -> None:
    path = session_path(owner, base_dir)
    with _LOCK:
        # Keep file reasonably sized: last 500 messages
        trimmed = history[-500:]
        atomic_write_json(path, trimmed)


def list_sessions(base_dir: str = DEFAULT_DIR) -> List[str]:
    if not os.path.isdir(base_dir):
        return []
    names = []
    for fn in os.listdir(base_dir):
        if fn.endswith(".json"):
            names.append(fn[:-5])
    return sorted(names)


def delete_session(owner: str, base_dir: str = DEFAULT_DIR) -> bool:
    path = session_path(owner, base_dir)
    with _LOCK, path_lock(path):
        if not os.path.exists(path):
            return False
        os.remove(path)
        return True
