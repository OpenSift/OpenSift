# Security Policy

## Scope
OpenSift is currently a local-first, hobby proof-of-concept project. The primary supported security model is localhost-only usage.

## Supported Versions
Security fixes are applied to the latest development state on `main`.

| Version | Supported |
| --- | --- |
| v1.1.3-alpha and later | Yes |
| Earlier alpha versions | No |

## Reporting a Vulnerability
If you discover a security issue, please do **not** post exploit details in a public issue first.

Preferred workflow:
1. Open a private security advisory in GitHub (`Security` tab in the repository).
2. If that is not available, open an issue with minimal details and request a private disclosure channel.
3. Include reproduction steps, affected files/endpoints, impact, and suggested mitigation if available.

## Disclosure Expectations
- We aim to acknowledge reports quickly and validate severity before public disclosure.
- Please allow reasonable remediation time before publishing details.
- Coordinated disclosure is preferred.

## Security Model and Assumptions
- OpenSift UI/API are intended to run on loopback (`127.0.0.1` / `::1`).
- The app is not designed for direct internet exposure.
- Provider credentials (OpenAI/Anthropic/Claude Code/Codex) are user-managed secrets and should be protected locally.

## Current Hardening Controls
- Localhost-only access enforcement in web UI middleware.
- Auth + CSRF protection for state-changing routes.
- Rate limiting on login, chat, and ingest routes.
- Upload/payload size limits for ingest and session-import surfaces.
- URL ingest protections:
  - private/local target blocking
  - DNS target validation
  - redirect validation and loop protection
- Atomic, lock-guarded JSON persistence for local stores.
- Regression tests for XSS template safety, CSRF fetch usage, ingest safety, and auth/session/streaming flows.

## Fixed Advisories
Previously fixed issues are documented in:
- `docs/security-advisories.md`

## Known Limitations
- Deployment assumptions remain local/dev-first.
- Session signing secrets are process-ephemeral by default.
- This project may still contain security defects; threat modeling and penetration testing are not yet comprehensive.

## Operational Recommendations
- Keep OpenSift bound to localhost unless you add additional network-layer protections.
- Do not run with high-privilege credentials.
- Rotate provider tokens if you suspect exposure.
- Keep dependencies updated.
