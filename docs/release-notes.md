# OpenSift Release Notes (Docs Copy)

## v1.1.3-alpha (Proposed)
Release date: 2026-02-18

This release focuses on comprehensive hardening of the web app and data pipeline: XSS prevention, URL ingest security controls, resource safeguards, concurrency-safe persistence, and deeper regression testing.

### Highlights
- Closed the primary persistent XSS risk in chat tool surfaces by removing unsafe untrusted HTML rendering patterns.
- Hardened URL ingest against SSRF vectors (private/local targets, redirect abuse, scheme abuse).
- Added upload/message/session payload limits and safer ingest read paths for better resilience under large inputs.
- Made persistence writes atomic and synchronized across sessions, study library, quiz attempts, flashcards, wellness settings, and auth state.
- Expanded automated regression tests for auth/session/streaming flows.

### Security
- Web UI rendering hardening in `backend/templates/chat.html`:
  - Removed risky content rendering patterns for study library, quiz history, and flashcards.
  - Rendered untrusted content with DOM node creation + `textContent`.
  - Kept `innerHTML` usage only for safe container clearing.
- URL ingest hardening in `backend/app/ingest.py`:
  - Blocks localhost/local domains and private/local/link-local/reserved IP targets.
  - Validates DNS-resolved addresses before fetch.
  - Enforces safe redirect behavior with:
    - per-hop URL re-validation
    - HTTP/HTTPS-only redirect targets
    - redirect loop detection
    - configurable redirect ceiling (`OPENSIFT_MAX_URL_REDIRECTS`)

### Reliability and Operations
- Ingest resource controls in `backend/ui_app.py`:
  - Chunked upload read with hard byte cap (`OPENSIFT_MAX_UPLOAD_MB`).
  - Max chat message size guard (`OPENSIFT_MAX_CHAT_MESSAGE_CHARS`).
  - Max session import payload guard (`OPENSIFT_MAX_SESSION_IMPORT_CHARS`).
  - Retrieval/history parameter clamping for safer request bounds.
- Heavy ingest/retrieval paths remain offloaded via `anyio.to_thread.run_sync(...)` for better event-loop responsiveness under load.

### Data Integrity and Concurrency
- Added shared atomic JSON utility in `backend/app/atomic_io.py`:
  - temp-file write + fsync + atomic replace
  - per-path locking helper
- Migrated store/auth persistence to atomic + lock-guarded behavior:
  - `backend/session_store.py`
  - `backend/study_store.py`
  - `backend/quiz_store.py`
  - `backend/flashcard_store.py`
  - `backend/app/wellness.py`
  - `backend/ui_app.py` auth state read/write paths
- Added lock-guarded chat history mutation helpers in `backend/ui_app.py` to prevent race-prone shared-state updates.

### Auth and Logging Hardening
- Removed full generated login token exposure from startup logs.
- Startup logs now report token presence and short hint suffix only.

### Framework and Runtime Cleanup
- Migrated FastAPI startup initialization from deprecated `@app.on_event("startup")` to lifespan handlers.
- Replaced deprecated `datetime.utcnow()` usage with timezone-aware UTC timestamps.

### Testing and CI
- Added regression tests in `backend/tests/test_auth_session_streaming.py` for:
  - startup log token non-exposure
  - session import replace/merge persistence behavior
  - NDJSON streaming completion + persisted history behavior
- Existing hardening tests remain active:
  - `backend/tests/test_chat_template_safety.py`
  - `backend/tests/test_chat_csrf_template.py`
  - `backend/tests/test_ingest_url_safety.py`
  - `backend/tests/test_ingest_redirect_safety.py`
- Current backend test suite status: 12 passing tests.

### Configuration Notes
- `OPENSIFT_MAX_URL_REDIRECTS`
- `OPENSIFT_ALLOW_PRIVATE_URLS` (trusted local development only)
- `OPENSIFT_MAX_UPLOAD_MB`
- `OPENSIFT_MAX_CHAT_MESSAGE_CHARS`
- `OPENSIFT_MAX_SESSION_IMPORT_CHARS`
- `OPENSIFT_MAX_HISTORY_TURNS`
- `OPENSIFT_MAX_RETRIEVAL_K`

### Upgrade Notes
- No user-facing migration required.
- Existing sessions and saved study assets continue to load normally.
- If private-network URL ingest is required, set `OPENSIFT_ALLOW_PRIVATE_URLS=true` only in trusted environments.

---

For full historical releases, see the repository root file: `RELEASE_NOTES.md`.

