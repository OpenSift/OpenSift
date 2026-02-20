# OpenSift Release Notes

## v1.5.0-alpha
Release date: 2026-02-20

This release focuses on a full UI polish pass and significant improvements to Library and chat interoperability, including global browsing, references visibility, and safer chat/library lifecycle behavior.

### Highlights
- Refreshed UI design system across chat, settings, library, and login:
  - consistent light dashboard style
  - unified card/input/button visual language
  - OpenSift light-blue accent theme
- Library is now globally browsable across owners from one place.
- Chat now visibly renders retrieved/pinned source references in responses.
- Chat deletion flow now supports keeping or deleting linked library items.
- Library "Ask From Chat" now reopens or creates the correct owner chat automatically.

### Added
- Global library browsing:
  - `all_owners` support in `GET /chat/library/list`
  - owner-aware operations for:
    - `GET /chat/library/get`
    - `GET /chat/library/download`
    - `GET /chat/library/preview`
    - `POST /chat/library/update`
    - `POST /chat/library/delete`
- New owner enumeration helper:
  - `list_owners()` in `backend/source_store.py`
- Explicit source references UI in chat:
  - rendered from stream `sources` events and persisted history payloads
- PDF preview resilience:
  - preview endpoint accepts legacy PDF entries by file extension/name even when kind metadata is older
  - browser fallback copy includes Safari Lockdown Mode guidance

### Changed
- Chat UX:
  - deleting a chat now prompts whether to also delete linked library items
  - backend `POST /chat/session/delete` accepts `delete_library_items` and returns `deleted_library_count`
- Library UX:
  - "Browse all owners" toggle in UI
  - item cards and details include owner context
  - "Ask From Chat" routes to the item owner and ensures a chat session exists
- UI identity text updated from "Gateway Dashboard" to "Gateway" across templates.

### Testing
- Added/expanded tests in:
  - `backend/tests/test_library_features.py`
    - all-owners list behavior
    - session-delete linked-library behavior
    - PDF preview compatibility path
  - `backend/tests/test_settings_and_stream_ui.py`
    - continued UI control coverage for updated templates
- Latest suite status during this release cycle: 63 passing tests (Docker).

### Versioning
- Bumped app version to `1.5.0-alpha` in:
  - `backend/opensift.py`
  - `backend/ui_app.py`

## v1.4.0-alpha
Release date: 2026-02-19

This release introduces a full persistent Library workflow for source management, with a dedicated UI, richer organization controls, and tighter chat integration.

### Highlights
- Added a dedicated **Library page** separate from chat, with browse/search and source detail views.
- Added persistent source storage for notes, URLs, PDFs, text files, and OCR'd image uploads.
- Added chat-side **pinned library context** selection so users can force specific sources into prompt context.
- Added library pagination, sorting, folder/tag filtering, and metadata editing for better large-collection usability.

### Added
- New source storage module:
  - `backend/source_store.py`
  - owner-scoped source manifests
  - persistent text/blob file handling
  - item update/delete helpers
- New Library endpoints in `backend/ui_app.py`:
  - `GET /library`
  - `GET /chat/library/list`
  - `GET /chat/library/get`
  - `POST /chat/library/note`
  - `POST /chat/library/url`
  - `POST /chat/library/upload`
  - `POST /chat/library/update`
  - `POST /chat/library/delete`
  - `GET /chat/library/download`
- New Library template:
  - `backend/templates/library.html`
  - includes upload progress with percentage + ETA
- New chat Library integration:
  - `backend/templates/chat.html`
  - “Library” navigation button
  - modal to select pinned library items for chat context

### Changed
- Retrieval/generation flow now supports optional `selected_library_ids` in `/chat/stream`:
  - selected library items are injected as pinned passages before normal semantic retrieval context.
- Library list API now supports:
  - pagination (`page`, `page_size`)
  - sorting (`sort_by`, `sort_dir`)
  - filtering (`kind`, `folder`, `tags`, `q`)
- Source metadata now supports:
  - `folder`
  - `tags`

### Testing
- Added tests:
  - `backend/tests/test_library_features.py`
  - library pagination/sort/filter
  - library metadata updates
  - pinned library context in chat stream
- Updated UI test coverage:
  - `backend/tests/test_settings_and_stream_ui.py`

### Versioning
- Bumped app version to `1.4.0-alpha` in:
  - `backend/opensift.py`
  - `backend/ui_app.py`

## v1.3.1-alpha
Release date: 2026-02-19

This release adds cross-surface thinking/streaming controls, a new security audit system, and a more complete Docker + setup onboarding path centered around the OpenSift gateway.

### Hotfix updates (2026-02-19)
- Docker provider install UX and diagnostics:
  - Added Settings actions to install `claude` / `claude-code` / `codex` CLIs inside Docker.
  - Added install progress stream with percent/ETA and persistent per-install logs.
- Docker auth/token path reliability:
  - Added Docker-first Codex auth discovery from `/app/.codex/auth.json` (with `~/.codex/auth.json` fallback).
  - Defaulted `OPENSIFT_CODEX_AUTH_PATH=/app/.codex/auth.json` in compose.
  - Added writable runtime home handling for non-root container user and mounted auth dirs (`backend/.codex`, `backend/.claude`).
- Codex CLI trust-directory compatibility:
  - Added `--skip-git-repo-check` support for Docker runs by default.
  - Added Codex CLI capability detection so flag placement matches installed CLI behavior.
- Streaming and generation resilience:
  - Hardened `claude` CLI generation path to treat exit-0/empty-stdout as failure and retry invocation variants.
  - Added provider fallback chain when a selected provider fails at generation time (Claude Code -> Claude API / Codex / OpenAI).
  - Added deterministic context-only fallback so chat always returns a usable answer from retrieved passages when external generation fails.
- Provider transparency and UI debugging:
  - Chat stream now reports requested vs active provider/model in status output and failure messages.
  - Added client-side provider/model normalization to avoid silent provider switches caused by incompatible model selections.
  - Added persistent status timeline in chat bubbles so transient runtime errors are visible after stream completion.
- Container/runtime stability:
  - Added fallback SOUL path handling for container environments with invalid home directories (e.g., `/nonexistent`) to prevent stream-time crashes.

### Upgrade checklist (from earlier alpha builds)
1. Rebuild and restart containers:
   - `docker compose down`
   - `docker compose up --build -d opensift-gateway`
2. Re-auth provider CLIs inside Docker (if using CLI providers):
   - Codex: `docker exec -it opensift-gateway sh -lc 'HOME=/app codex login --device-auth'`
   - Claude Code: `docker exec -it opensift-gateway claude setup-token`
3. Verify provider readiness in UI:
   - Open `http://127.0.0.1:8001/settings`
   - Confirm provider status and command paths on the Providers tab
4. Run a chat smoke test:
   - Send one message with `provider=codex` and `model=Auto`
   - Confirm status log shows requested/active provider and returns a non-empty response

### Highlights
- Added CLI parity for thinking and streaming controls to match the web UI.
- Added a built-in `security-audit` command with optional permission auto-fixes.
- Upgraded setup onboarding to run security checks and support Docker launch flows.
- Switched Docker runtime to a gateway-first architecture with optional terminal service.

### Added
- New security audit module:
  - `backend/app/security_audit.py`
  - Checks sensitive file/dir permissions and local risk signals
  - Optional permission repair (`--fix-perms`)
- New launcher command:
  - `python opensift.py security-audit [--fix-perms] [--fail-on-warn]`
- New setup behavior:
  - `python opensift.py setup --no-launch`
  - Setup now runs a security audit before launch decisions
  - Setup now checks for missing `claude` / `codex` CLIs and prompts to install them
  - New setup flag: `python opensift.py setup --skip-cli-install-prompts`
- New automated tests:
  - `backend/tests/test_security_audit.py`

### Changed
- Terminal chat now supports:
  - Claude thinking toggle
  - show/hide thinking status output
  - true native streaming toggle independent of normal streaming
- New terminal flags:
  - `--thinking`
  - `--no-show-thinking`
  - `--no-true-stream`
- New terminal slash commands:
  - `/thinking on|off`
  - `/show-thinking on|off`
  - `/true-stream on|off`
- Docker compose now provides:
  - `opensift-gateway` (UI + MCP via gateway)
  - `opensift-terminal` (interactive terminal profile)
  - dropped Linux capabilities, `no-new-privileges`, `tmpfs /tmp`
- `setup.sh` now:
  - enforces restrictive `.env` permissions
  - runs setup in no-launch mode + security audit
  - supports local or Docker launch targets
  - starts Docker gateway and optional Docker terminal

### Security
- Added local setup security posture checks for:
  - `.env`, `.opensift_auth.json`, `SOUL.md`
  - OpenSift state directories (`.opensift_*`, `.chroma`)
  - Codex auth file path (default `~/.codex/auth.json`)
  - debug logging and Docker socket exposure warnings
- Setup now auto-fixes restrictive permissions where possible (`chmod 600/700` policy).
- Post-audit hardening update:
  - Gateway now applies `.env` before spawning managed UI/MCP processes.
  - `opensift.py` setup writes `.env` with restrictive permissions (`0600`).
  - Docker compose no longer defaults runtime user to root when UID/GID env vars are unset.
  - Security audit now includes static template checks for:
    - XSS regression patterns (`innerHTML` non-empty assignments)
    - CSRF POST safety patterns (`csrfFetch` + CSRF token binding)
  - Added/expanded regression tests:
    - `backend/tests/test_security_audit.py`
    - `backend/tests/test_chat_csrf_template.py`
    - `backend/tests/test_settings_and_stream_ui.py`
  - Latest backend test status after this hardening pass: 38 passing tests.

### Versioning
- Bumped app version to `1.3.1-alpha` in:
  - `backend/opensift.py`
  - `backend/ui_app.py`

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
  - temp-file write + fsync + atomic replace.
  - per-path locking helper.
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
- Added UI integration tests in `backend/tests/test_settings_and_stream_ui.py` for:
  - `/settings` tabbed page render and control presence (Auth/SOUL/Wellness/Ingest)
  - `/chat/stream` provider-model discrepancy status events (e.g., OpenAI + Claude model auto-switch messaging)
- Existing hardening tests remain active:
  - `backend/tests/test_chat_template_safety.py`
  - `backend/tests/test_chat_csrf_template.py`
  - `backend/tests/test_ingest_url_safety.py`
  - `backend/tests/test_ingest_redirect_safety.py`
- Current backend test suite status: 36 passing tests.

### Configuration Notes
Security/reliability-sensitive controls used in this release:
- `OPENSIFT_MAX_URL_REDIRECTS`
- `OPENSIFT_ALLOW_PRIVATE_URLS` (use cautiously; intended for trusted local development only)
- `OPENSIFT_MAX_UPLOAD_MB`
- `OPENSIFT_MAX_CHAT_MESSAGE_CHARS`
- `OPENSIFT_MAX_SESSION_IMPORT_CHARS`
- `OPENSIFT_MAX_HISTORY_TURNS`
- `OPENSIFT_MAX_RETRIEVAL_K`

### Upgrade Notes
- No user-facing migration required.
- Existing sessions and saved study assets continue to load normally.
- If custom deployments rely on private-network URL ingest, explicitly set `OPENSIFT_ALLOW_PRIVATE_URLS=true` and apply strict network trust boundaries.

---

## v1.1.2-alpha (Proposed)
Release date: 2026-02-17

This patch release strengthens provider runtime reliability and adds deeper ChatGPT Codex integration, including auth auto-discovery, non-interactive execution, diagnostics, and UI controls.

### Highlights
- Added end-to-end ChatGPT Codex provider support across UI, terminal, MCP, and setup wizard.
- Added Codex auth auto-discovery from `~/.codex/auth.json`.
- Switched Codex execution to non-interactive `codex exec` to avoid TTY-only failures.
- Added UI wellness reminder toggles with persistent global settings.

### Added
- Setup wizard support for Codex credentials/config:
  - `CHATGPT_CODEX_OAUTH_TOKEN`
  - `OPENSIFT_CODEX_CMD`
  - `OPENSIFT_CODEX_ARGS`
- Runtime Codex provider support in:
  - `backend/ui_app.py`
  - `backend/cli_chat.py`
  - `backend/mcp_server.py`
- Native Codex streaming support via shared provider utility:
  - `stream_with_codex(...)`
- Health diagnostics flag:
  - `GET /health -> diagnostics.codex_auth_detected` (boolean, no secret exposure)
- Wellness settings API + UI toggles:
  - `GET /chat/wellness`
  - `POST /chat/wellness/set`
  - Sidebar controls for enable/disable + cadence tuning

### Changed
- Codex provider invocation is now non-interactive and consistent:
  - `codex exec ... -` instead of interactive TTY path
- Codex auth resolution order:
  1. `CHATGPT_CODEX_OAUTH_TOKEN` env var
  2. `~/.codex/auth.json` (override with `OPENSIFT_CODEX_AUTH_PATH`)
- Setup/default provider selection now recognizes Codex auth availability from both env and auth file.
- Gateway/provider summaries now report Codex auth readiness more accurately.

### Fixed
- Resolved common Codex runtime failure:
  - `Error: stdin is not a terminal`
- Added detection and clearer errors when `OPENSIFT_CODEX_CMD` points to the unrelated npm `codex` site-generator package.
- Improved provider failure messaging to make Codex misconfiguration easier to diagnose.

### Configuration
New or emphasized variables for this patch:
- `CHATGPT_CODEX_OAUTH_TOKEN`
- `OPENSIFT_CODEX_CMD`
- `OPENSIFT_CODEX_ARGS`
- `OPENSIFT_CODEX_AUTH_PATH` (default: `~/.codex/auth.json`)
- `OPENSIFT_BREAK_REMINDERS_ENABLED`
- `OPENSIFT_BREAK_REMINDER_EVERY_USER_MSGS`
- `OPENSIFT_BREAK_REMINDER_MIN_MINUTES`

### Notes
- Codex support now follows the same shared provider architecture used across OpenSift runtime surfaces.
- Diagnostics intentionally expose only auth presence (`true/false`) and never token values.

---

## v1.1.1-alpha (Proposed)
Release date: 2026-02-17

This patch release focuses on personality customization with a new global SOUL system that applies consistently across UI, terminal, and MCP workflows.

### Highlights
- Added global chatbot personality customization via `SOUL.md`.
- Unified personality behavior across all owners/sessions.
- Added automatic migration from legacy per-owner SOUL entries.

### Added
- New SOUL persistence module in `backend/app/soul.py` with:
  - global style read/write helpers
  - default SOUL file bootstrap and directory creation
  - compatibility support for legacy owner sections
- UI endpoints for global SOUL management:
  - `GET /chat/soul`
  - `POST /chat/soul/set`
- MCP tools for global SOUL access:
  - `soul_get`
  - `soul_set`

### Changed
- SOUL scope is now global by default and applied everywhere:
  - chat UI generation
  - terminal generation
  - MCP `sift_generate`
- Default SOUL path changed to:
  - `~/.opensift/SOUL.md`
- SOUL UI wording updated from owner-specific style to global style.

### Migration
- Legacy per-owner SOUL entries are merged into the global style block automatically.
- Migration runs idempotently on SOUL reads.
- Existing users do not need manual migration commands.

### Configuration
- `OPENSIFT_SOUL_PATH` remains supported for custom SOUL location.
- Recommended default:
  - `OPENSIFT_SOUL_PATH=~/.opensift/SOUL.md`

### Notes
- If no global style is set, OpenSift falls back to a neutral/default study style.

---

## v1.1.0-alpha (Proposed)
Release date: 2026-02-17

This release rounds out OpenSift from a proof-of-concept chatbot into a more complete student study app, with major improvements to launch flow, ingestion reliability, UI/UX, observability, and operational tooling.

### Highlights
- Added a unified OpenSift launcher with guided setup and gateway orchestration.
- Upgraded chat UI to a sidebar-first layout with persistent chat/session controls.
- Improved URL/PDF ingestion reliability for long and difficult documents.
- Added centralized, rotating logs across UI, terminal, launcher, and MCP server.
- Added CI pipeline and GitHub badges for release/build/license visibility.

### Added
- Guided setup workflow via `python opensift.py setup` for:
  - API/token configuration (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `CLAUDE_CODE_OAUTH_TOKEN`)
  - saved local environment in `backend/.env`
  - launch mode selection (`gateway`, `ui`, `terminal`, `both`)
- Gateway runner via `python opensift.py gateway [--with-mcp]` with:
  - supervised child process lifecycle
  - startup `/health` checks and timeout handling
  - coordinated graceful shutdown and failure propagation
- Central logging utility in `backend/app/logging_utils.py`:
  - console + rotating file logs
  - configurable level, directory, retention, and file size
- Access logging middleware in UI API with request IDs and timing metrics.
- MCP server logging for ingest/search/generate tool execution paths.
- CI workflow (`.github/workflows/ci.yml`) for dependency checks, bytecode compile checks, and launcher smoke tests.
- Project badges in README:
  - build/smoke test status
  - release version
  - release date
  - MIT license

### Changed
- Chat UI updated to a more modern assistant-style interface:
  - left-side session history panel
  - larger main chat workspace
  - improved visual system (typography, spacing, panel hierarchy, gradients)
  - reduced modal friction for new chat flow (direct in-UI session creation path)
- Provider defaults updated:
  - OpenAI default model: `gpt-5.2`
  - Claude default model alias: `claude-sonnet-4-5`
- Vector store writes now use upsert semantics for safer repeat ingestion behavior.

### Ingestion and Reliability Improvements
- URL ingestion hardening in `backend/app/ingest.py`:
  - retry logic with backoff for fetch failures
  - robust HTML cleanup/noise stripping
  - main-content candidate scoring
  - fallback body extraction when structural parsing is weak
  - content normalization + max-length truncation safeguards
- PDF ingestion improvements:
  - scanned/low-text page detection
  - OCR pass from embedded PDF images
  - OCR fallback via `pdf2image` + Tesseract for difficult documents
  - merged OCR/plain extraction strategy for better recall
- Retrieval robustness:
  - owner-scoped query with fallback broader scan to recover missed owner matches

### Fixed
- Multiple ingestion edge paths that previously returned too little/empty content now have stronger fallbacks.
- Operational blind spots during failures are reduced through structured error logging and request tracing.
- Re-ingesting previously seen sources is less error-prone due to vector upsert behavior.

### Developer Experience
- Launcher now exposes clear command surface:
  - `setup`, `ui`, `terminal`, `gateway`
- README updated with:
  - gateway and setup usage
  - logging configuration variables
  - supported provider flow

### Configuration
New logging-related environment variables:
- `OPENSIFT_LOG_LEVEL` (default: `INFO`)
- `OPENSIFT_LOG_DIR` (default: `.opensift_logs`)
- `OPENSIFT_LOG_MAX_BYTES` (default: `5242880`)
- `OPENSIFT_LOG_BACKUP_COUNT` (default: `5`)

OCR-related note:
- Best OCR results require system/runtime OCR dependencies (`pytesseract` and optionally `pdf2image` + Poppler + Tesseract binaries).

### Upgrade Notes
- Existing users can continue with current `.env`; run `python opensift.py setup` to migrate into the guided flow.
- For gateway-supervised local runs, prefer:
  - `python opensift.py gateway --with-mcp`
- For logs on disk, ensure the configured log directory is writable.

---

## v1.0.1-alpha
Release date: tagged in repository

### Summary
- Major bug-fix pass across ingestion and runtime flow.
- Added CI and repository badges.
- General quality improvements and cleanup.

---

## v1.0.0-alpha
Release date: tagged in repository

### Summary
- Introduced setup wizard foundations.
- Established early UI + terminal chat workflows.
- Baseline provider integrations and retrieval-augmented generation path.

---
