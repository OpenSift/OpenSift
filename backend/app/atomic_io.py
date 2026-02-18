from __future__ import annotations

import json
import os
import tempfile
import threading
from collections import defaultdict
from typing import Any, Dict

_LOCKS: Dict[str, threading.RLock] = defaultdict(threading.RLock)


def path_lock(path: str) -> threading.RLock:
    return _LOCKS[os.path.abspath(path)]


def atomic_write_json(path: str, data: Any, *, indent: int = 2) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    lock = path_lock(path)
    with lock:
        fd, tmp_path = tempfile.mkstemp(prefix=".tmp.", suffix=".json", dir=parent or ".")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=indent, ensure_ascii=False)
                f.write("\n")
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, path)
        finally:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except OSError:
                pass
