from __future__ import annotations

import json
import os
import re
import secrets
from typing import Any, Dict, List

DEFAULT_DIR = os.path.join(os.getcwd(), ".opensift_quiz_attempts")


def _safe_owner(owner: str) -> str:
    owner = owner.strip() or "default"
    owner = re.sub(r"[^a-zA-Z0-9._-]+", "_", owner)
    return owner[:128]


def _path(owner: str, base_dir: str = DEFAULT_DIR) -> str:
    os.makedirs(base_dir, exist_ok=True)
    return os.path.join(base_dir, f"{_safe_owner(owner)}.json")


def load_attempts(owner: str, base_dir: str = DEFAULT_DIR) -> List[Dict[str, Any]]:
    path = _path(owner, base_dir)
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
        out.append(item)
    return out


def save_attempts(owner: str, attempts: List[Dict[str, Any]], base_dir: str = DEFAULT_DIR) -> None:
    path = _path(owner, base_dir)
    attempts = attempts[-2000:]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(attempts, f, indent=2, ensure_ascii=False)


def add_attempt(
    owner: str,
    title: str,
    score: int,
    total: int,
    notes: str,
    created_at: str,
    base_dir: str = DEFAULT_DIR,
) -> Dict[str, Any]:
    attempts = load_attempts(owner, base_dir)
    item = {
        "id": secrets.token_urlsafe(9),
        "owner": owner,
        "title": (title or "").strip() or "Quiz Attempt",
        "score": int(score),
        "total": int(total),
        "pct": round((float(score) / max(1.0, float(total))) * 100.0, 2),
        "notes": (notes or "").strip(),
        "created_at": created_at,
    }
    attempts.append(item)
    save_attempts(owner, attempts, base_dir)
    return item


def delete_attempt(owner: str, attempt_id: str, base_dir: str = DEFAULT_DIR) -> bool:
    attempts = load_attempts(owner, base_dir)
    kept = [x for x in attempts if x.get("id") != attempt_id]
    if len(kept) == len(attempts):
        return False
    save_attempts(owner, kept, base_dir)
    return True
