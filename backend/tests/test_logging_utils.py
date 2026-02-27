from __future__ import annotations

import logging

from app import logging_utils


def _reset_logging_state() -> None:
    logging_utils._CONFIGURED = False  # type: ignore[attr-defined]
    root = logging.getLogger()
    root.handlers.clear()


def test_chroma_telemetry_logs_suppressed_by_default(monkeypatch, tmp_path) -> None:
    _reset_logging_state()
    monkeypatch.delenv("OPENSIFT_SUPPRESS_CHROMA_TELEMETRY_LOGS", raising=False)

    noisy_logger = logging.getLogger("chromadb.telemetry.product.posthog")
    noisy_logger.setLevel(logging.NOTSET)
    noisy_logger.propagate = True

    logging_utils.configure_logging(service="opensift.test", log_dir=str(tmp_path))

    assert noisy_logger.level == logging.CRITICAL
    assert noisy_logger.propagate is False


def test_chroma_telemetry_logs_can_be_reenabled(monkeypatch, tmp_path) -> None:
    _reset_logging_state()
    monkeypatch.setenv("OPENSIFT_SUPPRESS_CHROMA_TELEMETRY_LOGS", "false")

    noisy_logger = logging.getLogger("chromadb.telemetry.product.posthog")
    noisy_logger.setLevel(logging.NOTSET)
    noisy_logger.propagate = True

    logging_utils.configure_logging(service="opensift.test", log_dir=str(tmp_path))

    assert noisy_logger.level == logging.NOTSET
    assert noisy_logger.propagate is True
