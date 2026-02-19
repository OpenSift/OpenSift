# Contributing to OpenSift

Thanks for contributing to OpenSift.

This guide covers how to set up your environment, make changes safely, and submit pull requests that are easy to review.

## Project Scope
OpenSift is currently a local-first, alpha-stage study assistant. Contributions should prioritize:
- Security and reliability
- Clear student-facing UX
- Minimal regressions across UI, terminal, and provider flows

## Prerequisites
- Python 3.12+ (3.13 also works)
- `pip` and virtual environments
- Optional OCR dependencies for PDF OCR workflows (`tesseract`, `poppler`, etc.)

## Local Setup
From `backend/`:

```bash
./setup.sh
```

Or manual setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip setuptools wheel
pip install -r requirements.txt
pip install sentence-transformers pytest
```

Run setup wizard:

```bash
python opensift.py setup
```

## Running OpenSift
From `backend/`:

```bash
python opensift.py gateway --with-mcp
```

Useful alternatives:
- UI only: `python opensift.py ui --reload`
- Terminal only: `python opensift.py terminal --provider claude_code`

## Development Workflow
1. Create a branch from `main`.
2. Keep PRs focused (one feature/fix topic per PR).
3. Update tests and docs in the same PR.
4. Ensure local tests pass before opening PR.

## Coding Guidelines
- Follow existing code style and structure.
- Prefer simple, explicit logic over clever abstractions.
- Keep security in mind when handling:
  - untrusted input/rendering
  - URL/network fetches
  - local file persistence
  - auth/session flows
- For web UI:
  - avoid unsafe `innerHTML` for untrusted content
  - use safe DOM APIs and `textContent`

## Testing Requirements
Run backend tests:

```bash
cd backend
python -m pytest tests -q
```

For security-sensitive changes, include/adjust tests for:
- template safety / XSS regression
- ingest URL safety and redirects
- auth/session behavior
- streaming endpoint behavior

## Security Reporting
Do not open public exploit details first.

Please follow `SECURITY.md`:
- use GitHub Security Advisories when possible
- otherwise open a minimal issue and request private disclosure

## Documentation Expectations
If behavior changes, update:
- `README.md`
- `RELEASE_NOTES.md`
- relevant docs pages in this repo

If a release is being cut, keep runtime and image versioning aligned:
- App version constants (e.g., `OPENSIFT_VERSION`) should match the release notes version.
- Published Docker image tags should include both the release tag and `latest` (for example, `1.3.1-alpha` and `latest`).

If you maintain the docs site in a separate repository (`opensift.github.io`), keep that content in sync as part of your release process.

## Pull Request Checklist
- [ ] Change is scoped and explained clearly
- [ ] Tests added/updated where needed
- [ ] `pytest` passes locally
- [ ] Docs updated (`README.md` / release notes / docs)
- [ ] No secrets or sensitive local files committed

## Commit Messages
Use clear, descriptive messages. Conventional commit style is welcome but not required.

Examples:
- `fix(ingest): block private-network redirect targets`
- `test(ui): add regression coverage for chat stream persistence`
- `docs: update release notes for v1.1.3-alpha`

## Questions
If anything is unclear, open a discussion or issue with:
- expected behavior
- current behavior
- reproduction details
