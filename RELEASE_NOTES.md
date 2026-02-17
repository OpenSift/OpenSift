# OpenSift Release Notes

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
