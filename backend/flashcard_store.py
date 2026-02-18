from __future__ import annotations

import json
import os
import re
import secrets
import threading
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from app.atomic_io import atomic_write_json, path_lock

DEFAULT_DIR = os.path.join(os.getcwd(), ".opensift_flashcards")
_LOCK = threading.RLock()


def _safe_owner(owner: str) -> str:
    owner = owner.strip() or "default"
    owner = re.sub(r"[^a-zA-Z0-9._-]+", "_", owner)
    return owner[:128]


def _path(owner: str, base_dir: str = DEFAULT_DIR) -> str:
    os.makedirs(base_dir, exist_ok=True)
    return os.path.join(base_dir, f"{_safe_owner(owner)}.json")


def _now_dt() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.isoformat().replace("+00:00", "Z")


def _parse_iso(value: str) -> Optional[datetime]:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None


def load_cards(owner: str, base_dir: str = DEFAULT_DIR) -> List[Dict[str, Any]]:
    path = _path(owner, base_dir)
    with _LOCK, path_lock(path):
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
        if not isinstance(item.get("front"), str):
            continue
        if not isinstance(item.get("back"), str):
            continue
        out.append(item)
    return out


def save_cards(owner: str, cards: List[Dict[str, Any]], base_dir: str = DEFAULT_DIR) -> None:
    path = _path(owner, base_dir)
    with _LOCK:
        trimmed = cards[-5000:]
        atomic_write_json(path, trimmed)


def add_card(owner: str, front: str, back: str, tags: str, created_at: str, base_dir: str = DEFAULT_DIR) -> Dict[str, Any]:
    with _LOCK:
        cards = load_cards(owner, base_dir)
        item = {
            "id": secrets.token_urlsafe(9),
            "owner": owner,
            "front": front.strip(),
            "back": back.strip(),
            "tags": tags.strip(),
            "created_at": created_at,
            "last_reviewed_at": "",
            "due_at": created_at,
            "interval_days": 0,
            "ease": 2.5,
            "streak": 0,
        }
        cards.append(item)
        save_cards(owner, cards, base_dir)
        return item


def delete_card(owner: str, card_id: str, base_dir: str = DEFAULT_DIR) -> bool:
    with _LOCK:
        cards = load_cards(owner, base_dir)
        kept = [x for x in cards if x.get("id") != card_id]
        if len(kept) == len(cards):
            return False
        save_cards(owner, kept, base_dir)
        return True


def due_cards(owner: str, base_dir: str = DEFAULT_DIR) -> List[Dict[str, Any]]:
    cards = load_cards(owner, base_dir)
    now = _now_dt()
    out: List[Dict[str, Any]] = []
    for c in cards:
        due = _parse_iso(c.get("due_at", "")) or now
        if due <= now:
            out.append(c)
    return out


def review_card(owner: str, card_id: str, rating: str, reviewed_at: str, base_dir: str = DEFAULT_DIR) -> Optional[Dict[str, Any]]:
    with _LOCK:
        cards = load_cards(owner, base_dir)
        idx = -1
        for i, c in enumerate(cards):
            if c.get("id") == card_id:
                idx = i
                break
        if idx < 0:
            return None

        card = cards[idx]
        interval = int(card.get("interval_days") or 0)
        ease = float(card.get("ease") or 2.5)
        streak = int(card.get("streak") or 0)

        r = (rating or "").strip().lower()
        if r == "again":
            interval = 1
            ease = max(1.3, ease - 0.2)
            streak = 0
        elif r == "hard":
            interval = max(1, int(round((interval or 1) * 1.2)))
            ease = max(1.3, ease - 0.15)
            streak += 1
        elif r == "easy":
            base = interval or 1
            interval = max(2, int(round(base * (ease + 0.25))))
            ease = min(3.2, ease + 0.15)
            streak += 1
        else:
            base = interval or 1
            interval = max(1, int(round(base * ease)))
            streak += 1

        reviewed_dt = _parse_iso(reviewed_at) or _now_dt()
        due_dt = reviewed_dt + timedelta(days=interval)

        card["last_reviewed_at"] = _iso(reviewed_dt)
        card["due_at"] = _iso(due_dt)
        card["interval_days"] = interval
        card["ease"] = round(ease, 3)
        card["streak"] = streak

        cards[idx] = card
        save_cards(owner, cards, base_dir)
        return card
