from __future__ import annotations

"""
DEPRECATED (legacy): This module originally hosted the OpenSift API v0.1.0.

OpenSift is now driven by:
  - ui_app.py   : FastAPI app (localhost-only, auth, rate limiting, chat streaming, ingest)
  - opensift.py : launcher for UI + terminal modes

This file remains ONLY for backward compatibility with:
  uvicorn app.main:app

It re-exports the current FastAPI app so there is exactly ONE server code path.
"""

from ui_app import app  # noqa: F401