from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

_CONFIGURED = False


def _parse_level(raw: str) -> int:
    name = (raw or "INFO").strip().upper()
    return getattr(logging, name, logging.INFO)


def configure_logging(
    service: str = "opensift",
    level: Optional[str] = None,
    log_dir: Optional[str] = None,
) -> logging.Logger:
    global _CONFIGURED

    root = logging.getLogger()
    if _CONFIGURED:
        return logging.getLogger(service)

    log_level = _parse_level(level or os.getenv("OPENSIFT_LOG_LEVEL", "INFO"))
    logs_path = Path(log_dir or os.getenv("OPENSIFT_LOG_DIR", ".opensift_logs"))
    logs_path.mkdir(parents=True, exist_ok=True)

    max_bytes = int(os.getenv("OPENSIFT_LOG_MAX_BYTES", str(5 * 1024 * 1024)))
    backup_count = int(os.getenv("OPENSIFT_LOG_BACKUP_COUNT", "5"))
    log_file = logs_path / "opensift.log"

    fmt = logging.Formatter(
        fmt="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root.setLevel(log_level)
    root.handlers.clear()

    console = logging.StreamHandler()
    console.setLevel(log_level)
    console.setFormatter(fmt)
    root.addHandler(console)

    file_handler = RotatingFileHandler(
        filename=str(log_file),
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)

    _CONFIGURED = True
    logger = logging.getLogger(service)
    logger.info(
        "logging_configured level=%s log_file=%s max_bytes=%d backups=%d",
        logging.getLevelName(log_level),
        log_file,
        max_bytes,
        backup_count,
    )
    return logger

