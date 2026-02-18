# OpenSift Docs

Last updated: 2026-02-18  
Current release: **v1.1.3-alpha (Proposed)**

OpenSift is a RAG-powered study assistant for students. It ingests class materials (URLs, PDFs, notes), retrieves relevant passages, and generates study outputs such as guides, key points, quizzes, and explanations.

## What Is New
- Security and reliability hardening release (`v1.1.3-alpha`)
- Atomic/synchronized persistence across core stores
- URL ingest SSRF protections and redirect safeguards
- Expanded automated tests for auth/session/streaming
- FastAPI lifespan migration and UTC timestamp cleanup

See [Release Notes](release-notes.md).

## Core Features
- Chat-first study workflow (web UI + terminal)
- Ingestion from URL, PDF, TXT, and MD
- Session isolation by owner/namespace
- Study Library for saved generated outputs
- Quiz history tracking
- Flashcards with spaced-repetition review state
- Wellness reminders with UI toggles
- Global study personality via `SOUL.md`
- Multi-provider generation:
  - OpenAI
  - Anthropic Claude API
  - Claude Code
  - ChatGPT Codex CLI

## Quick Start
From `backend/`:

```bash
./setup.sh
python opensift.py setup
```

Then launch:

```bash
python opensift.py gateway --with-mcp
```

Or run only UI:

```bash
python opensift.py ui --reload
```

## Screenshots
![OpenSift Demo](assets/screenshot.png)
![Full Chat](assets/full.png)
![Study Guide](assets/study_guide.png)
![Key Points](assets/key_points.png)
![Quiz Me](assets/quiz_me.png)

## Key Docs
- [Release Notes](release-notes.md)
- [Security and Hardening](security-and-hardening.md)
- [Security Advisories](security-advisories.md)
- [Main README](../README.md)
