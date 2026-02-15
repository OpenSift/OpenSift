**OpenSift** is an AI-assisted study and research tool that helps students (and researchers) **sift through large amounts of information**—notes, PDFs, websites, and articles—to quickly surface what matters most.

It focuses on **ingestion, semantic retrieval, and synthesis**, making it easier to study for exams, quizzes, and deep reading tasks without getting lost in the noise.

---

## ✨ What OpenSift Does

- 📥 **Ingest content**
  - Websites (URLs)
  - PDFs
  - Text / Markdown files
- 🧠 **Chunk + embed** content into a searchable knowledge base
- 🔍 **Semantic search** across all ingested materials
- 📝 **AI-assisted synthesis** (study guides, summaries, quizzes)
- 🔐 **Flexible AI providers**
  - Local embeddings (no API keys required)
  - OpenAI (API key)
  - Claude (API key)
  - Claude Code (long-lived setup-token / subscription)

OpenSift is designed to work well with **Codex / MCP workflows**, so the AI agent can retrieve context and generate answers without direct API coupling.

---

## 🧱 Architecture Overview

- **MCP Server (stdio-based)**
  - Exposes tools like `ingest_url`, `ingest_file`, `search`, `sift_generate`
- **Vector Store**
  - ChromaDB (local, persistent)
- **Embeddings**
  - Default: local `sentence-transformers`
  - Optional: OpenAI embeddings if API key is set
- **Generation**
  - OpenAI
  - Claude (Anthropic)
  - Claude Code CLI (setup-token)

---

## 📁 Project Structure

```text
backend/
├── app/
│   ├── chunking.py        # Text chunking logic
│   ├── ingest.py          # URL + file ingestion
│   ├── llm.py             # Embeddings (local + OpenAI fallback)
│   ├── providers.py       # OpenAI / Claude / Claude Code generation
│   ├── settings.py        # Environment-based configuration
│   └── vectordb.py        # ChromaDB wrapper
├── mcp_server.py          # MCP server entrypoint
├── test_mcp_client.py     # Local MCP test + ingestion script
├── requirements.txt
└── .env                   # Optional secrets (ignored by git)