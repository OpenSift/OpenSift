# OpenSift Security Advisories

Last updated: 2026-02-18

This document contains disclosure-ready advisory summaries for previously fixed vulnerabilities.

## GHSA-OSFT-2026-0001

### Title
Persistent XSS in chat tool rendering

### Severity
High

### CWE
- CWE-79: Improper Neutralization of Input During Web Page Generation (Cross-site Scripting)
- CWE-116: Improper Encoding or Escaping of Output

### CVSS
- CVSS:3.1/AV:N/AC:L/PR:L/UI:R/S:C/C:H/I:H/A:L
- Base score: 8.5 (High)

### Summary
OpenSift previously rendered untrusted user/model content in chat tool UI surfaces using unsafe HTML interpolation patterns. Stored content could execute JavaScript when later viewed in authenticated sessions.

### Impact
An attacker able to influence stored study/quiz/flashcard content could trigger script execution in a victim’s browser and perform actions in that local authenticated app session.

### Affected Versions
- Affected: versions before `v1.1.3-alpha`
- Patched: `v1.1.3-alpha`

### Fix
Replaced unsafe rendering paths with safe DOM construction and `textContent` output handling. Added regression tests to prevent reintroduction.

### References
- `backend/templates/chat.html`
- `backend/tests/test_chat_template_safety.py`
- `RELEASE_NOTES.md` (`v1.1.3-alpha`)

---

## GHSA-OSFT-2026-0002

### Title
SSRF risk in URL ingestion endpoint

### Severity
Medium

### CWE
- CWE-918: Server-Side Request Forgery (SSRF)
- CWE-20: Improper Input Validation

### CVSS
- CVSS:3.1/AV:N/AC:L/PR:L/UI:N/S:U/C:H/I:L/A:N
- Base score: 6.5 (Medium)

### Summary
OpenSift URL ingestion previously permitted overly broad server-side fetch behavior and could be coerced into requesting unsafe internal/local targets.

### Impact
Potential private-network probing/access from the OpenSift host process when ingesting attacker-controlled URLs.

### Affected Versions
- Affected: versions before `v1.1.3-alpha`
- Patched: `v1.1.3-alpha`

### Fix
Added layered validation: localhost/private target blocking, DNS resolution checks, per-hop redirect validation, loop detection, non-http(s) redirect rejection, and redirect hop limits.

### References
- `backend/app/ingest.py`
- `backend/tests/test_ingest_url_safety.py`
- `backend/tests/test_ingest_redirect_safety.py`
- `RELEASE_NOTES.md` (`v1.1.3-alpha`)

---

## GHSA-OSFT-2026-0003

### Title
Race-prone local persistence could cause state corruption/loss

### Severity
Medium

### CWE
- CWE-362: Concurrent Execution using Shared Resource with Improper Synchronization (Race Condition)
- CWE-367: Time-of-check Time-of-use (TOCTOU) Race Condition

### CVSS
- CVSS:3.1/AV:L/AC:L/PR:L/UI:N/S:U/C:L/I:H/A:L
- Base score: 6.0 (Medium)

### Summary
OpenSift previously used non-atomic and insufficiently synchronized local JSON persistence flows.

### Impact
Concurrent operations could cause lost updates or corrupted local state across sessions/study/quiz/flashcard/wellness/auth stores.

### Affected Versions
- Affected: versions before `v1.1.3-alpha`
- Patched: `v1.1.3-alpha`

### Fix
Implemented lock-guarded atomic writes (temp file + fsync + replace) and synchronized shared in-memory history mutation paths.

### References
- `backend/app/atomic_io.py`
- `backend/session_store.py`
- `backend/study_store.py`
- `backend/quiz_store.py`
- `backend/flashcard_store.py`
- `backend/app/wellness.py`
- `backend/ui_app.py`
- `backend/tests/test_auth_session_streaming.py`
- `RELEASE_NOTES.md` (`v1.1.3-alpha`)

