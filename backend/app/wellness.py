from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from app.atomic_io import atomic_write_json, path_lock

DEFAULT_ENABLED = True
DEFAULT_EVERY_USER_MSGS = 6
DEFAULT_MIN_MINUTES = 45
WELLNESS_SETTINGS_PATH = os.path.join(os.path.expanduser("~"), ".opensift", "wellness.json")
_LOCK = threading.RLock()


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name, str(default)).strip().lower()
    return raw not in ("0", "false", "no", "off")


def _env_int(name: str, default: int) -> int:
    try:
        val = int((os.getenv(name, str(default)) or str(default)).strip())
        return max(1, val)
    except Exception:
        return default


def _coerce_int(value: Any, default: int) -> int:
    try:
        return max(1, int(value))
    except Exception:
        return default


def _settings_path() -> str:
    return (os.getenv("OPENSIFT_WELLNESS_PATH", WELLNESS_SETTINGS_PATH) or WELLNESS_SETTINGS_PATH).strip()


def _load_settings_file() -> Dict[str, Any]:
    p = _settings_path()
    with _LOCK, path_lock(p):
        if not os.path.exists(p):
            return {}
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}


def _save_settings_file(data: Dict[str, Any]) -> None:
    p = _settings_path()
    with _LOCK:
        atomic_write_json(p, data)


def get_wellness_settings() -> Dict[str, Any]:
    saved = _load_settings_file()
    enabled = bool(saved.get("enabled")) if "enabled" in saved else _env_bool("OPENSIFT_BREAK_REMINDERS_ENABLED", DEFAULT_ENABLED)
    every_user_msgs = _coerce_int(saved.get("every_user_msgs"), _env_int("OPENSIFT_BREAK_REMINDER_EVERY_USER_MSGS", DEFAULT_EVERY_USER_MSGS))
    min_minutes = _coerce_int(saved.get("min_minutes"), _env_int("OPENSIFT_BREAK_REMINDER_MIN_MINUTES", DEFAULT_MIN_MINUTES))
    return {
        "enabled": enabled,
        "every_user_msgs": every_user_msgs,
        "min_minutes": min_minutes,
    }


def set_wellness_settings(
    *,
    enabled: Optional[bool] = None,
    every_user_msgs: Optional[int] = None,
    min_minutes: Optional[int] = None,
) -> Dict[str, Any]:
    current = get_wellness_settings()
    if enabled is not None:
        current["enabled"] = bool(enabled)
    if every_user_msgs is not None:
        current["every_user_msgs"] = max(1, int(every_user_msgs))
    if min_minutes is not None:
        current["min_minutes"] = max(1, int(min_minutes))
    _save_settings_file(current)
    return current


def reminders_enabled() -> bool:
    return bool(get_wellness_settings().get("enabled", DEFAULT_ENABLED))


def reminder_every_user_msgs() -> int:
    return int(get_wellness_settings().get("every_user_msgs", DEFAULT_EVERY_USER_MSGS))


def reminder_min_minutes() -> int:
    return int(get_wellness_settings().get("min_minutes", DEFAULT_MIN_MINUTES))


def _parse_ts(raw: str) -> Optional[datetime]:
    if not raw:
        return None
    s = raw.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _is_break_msg(m: Dict[str, Any]) -> bool:
    return bool(m.get("break_reminder") is True)


def should_add_break_reminder(history: List[Dict[str, Any]], settings: Optional[Dict[str, Any]] = None) -> bool:
    cfg = settings or get_wellness_settings()
    if not bool(cfg.get("enabled", DEFAULT_ENABLED)):
        return False

    user_count = sum(1 for m in history if m.get("role") == "user")
    if user_count <= 0:
        return False

    every_n = max(1, int(cfg.get("every_user_msgs", DEFAULT_EVERY_USER_MSGS)))
    due_by_count = (user_count % every_n) == 0

    last_reminder_ts: Optional[datetime] = None
    for m in reversed(history):
        if m.get("role") != "assistant":
            continue
        if not _is_break_msg(m):
            continue
        last_reminder_ts = _parse_ts(str(m.get("ts") or ""))
        break

    if last_reminder_ts is None:
        due_by_time = True
    else:
        elapsed_min = (_now_utc() - last_reminder_ts).total_seconds() / 60.0
        due_by_time = elapsed_min >= float(max(1, int(cfg.get("min_minutes", DEFAULT_MIN_MINUTES))))

    return due_by_count and due_by_time


def build_break_reminder(history: List[Dict[str, Any]]) -> str:
    user_count = sum(1 for m in history if m.get("role") == "user")
    reminders = [
        "Quick study break: drink some water and rest your eyes for 2-5 minutes.",
        "Pause for a short reset: stand up, stretch, and take a few deep breaths.",
        "Coffee or tea check: take a mindful break before the next question.",
        "Mental health check-in: step away briefly and come back with a clear head.",
        "If it is late, protect sleep: consider wrapping this session and reviewing tomorrow.",
    ]
    pick = reminders[user_count % len(reminders)]
    return f"Break reminder: {pick}"
