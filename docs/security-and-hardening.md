# Security and Hardening Status

Last updated: 2026-02-18

This page tracks the major security and reliability findings previously identified in OpenSift and their current implementation status.

## Finding Status

1. **Persistent XSS in chat tool rendering**
- Status: **Addressed**
- Notes:
  - Risky untrusted HTML interpolation paths were replaced with safe DOM rendering patterns (`textContent`) in tool UIs.
  - Remaining `innerHTML` usage is limited to safe container clearing.

2. **URL ingest SSRF surface**
- Status: **Addressed**
- Notes:
  - URL validation blocks localhost/local labels and private/local IP targets.
  - DNS-resolved targets are checked before fetch.
  - Redirect chain is revalidated with loop detection, scheme restrictions, and cap controls.

3. **Event-loop blocking in heavy ingest paths**
- Status: **Mitigated**
- Notes:
  - Embedding/query/db and heavy extraction paths are offloaded with worker-thread execution (`anyio.to_thread.run_sync`).

4. **No payload/upload size limits**
- Status: **Addressed**
- Notes:
  - Explicit limits were added for upload bytes, chat message length, and session import payload size.

5. **Non-atomic/unsynchronized persistence**
- Status: **Addressed**
- Notes:
  - Added shared atomic JSON write utility (temp write + fsync + replace).
  - Added lock-guarded persistence and in-memory history mutation helpers.

6. **Minimal auth/session hardening for production**
- Status: **Partially open by design**
- Notes:
  - Localhost-first deployment model is still assumed.
  - Runtime secrets are process-local and sessions are designed for local/dev use cases.

7. **Shallow CI/test coverage**
- Status: **Improved**
- Notes:
  - CI executes backend tests.
  - Added focused tests for ingest security controls, template safety, CSRF usage, and auth/session/streaming regressions.

## Controls Added
- `OPENSIFT_MAX_URL_REDIRECTS`
- `OPENSIFT_ALLOW_PRIVATE_URLS`
- `OPENSIFT_MAX_UPLOAD_MB`
- `OPENSIFT_MAX_CHAT_MESSAGE_CHARS`
- `OPENSIFT_MAX_SESSION_IMPORT_CHARS`
- `OPENSIFT_MAX_HISTORY_TURNS`
- `OPENSIFT_MAX_RETRIEVAL_K`

## Remaining Priority Work
- Add stricter production-mode auth/session controls for non-localhost deployments.
- Add higher-scale runtime/load testing for ingest throughput and queue behavior.
- Expand security test matrix (malicious payloads, fuzzing for endpoint input validation).

