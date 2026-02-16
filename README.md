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
├── ui_app.py              # Web UI (FastAPI)
├── test_mcp_client.py     # MCP ingestion + search test client
├── templates/
│   └── index.html         # UI template
├── static/                # UI assets (icons/css)
├── requirements.txt
└── .env                   # Optional secrets (ignored by git)
```
---

## ⚡ Quick Start (5 Minutes)

This gets OpenSift running locally with **no API keys required**.

### 1) Clone the repository and enter the backend
```
bash
git clone https://github.com/your-org/opensift.git
cd opensift/backend
```
### 2) Create and activate a virtual environment
```
python3.13 -m venv .venv
source .venv/bin/activate
```
### 3) Install dependencies
```
pip install -U pip setuptools wheel
pip install -r requirements.txt
pip install sentence-transformers
```

### 4) Run the Web UI

```
uvicorn ui_app:app --reload --port 8001
```
Open your browser at:
```
http://127.0.0.1:8001
```

🎉 You now have a full UI where you can ingest content and search it interactively.

🖥️ Using the OpenSift Web UI

Ingest content

From the UI you can:
	•	Paste a URL (articles, documentation, Wikipedia, etc.)
	•	Upload PDF, TXT, or Markdown files
	•	Assign an Owner / Namespace (e.g. biology101, cs_midterm)

Namespaces let you isolate different courses or projects.

⸻

Search your material
	•	Enter a natural-language question
	•	OpenSift retrieves the most relevant passages
	•	Results are grounded in your ingested sources

Example queries:
	•	“What are the stages of photosynthesis?”
	•	“Compare cellular respiration and photosynthesis”
	•	“Summarize the Calvin cycle inputs and outputs”

⸻

Generate study content (optional)

If you configure an AI provider, OpenSift can:
	•	Generate study guides
	•	Produce key point summaries
	•	Create quizzes

Generation always uses retrieved passages from your material.

⸻

🧠 Architecture Overview
	•	Web UI
	•	FastAPI + Jinja2
	•	MCP Server (stdio-based)
	•	Tools: ingest_url, ingest_file, search, sift_generate
	•	Vector Store
	•	ChromaDB (local, persistent)
	•	Embeddings
	•	Default: local sentence-transformers
	•	Optional: OpenAI embeddings
	•	Generation
	•	OpenAI
	•	Claude
	•	Claude Code CLI



### 4a) Feed OpenSift information
Open test_mcp_client.py and add:
	•	URLs you want to study
	•	PDFs / TXT / MD files (lecture notes, articles, books)

Example URLs already included:
```
urls = [
    ("Photosynthesis (Wiki)", "https://en.wikipedia.org/wiki/Photosynthesis"),
    ("Cellular respiration (Wiki)", "https://en.wikipedia.org/wiki/Cellular_respiration"),
]
```
### 5) Run the test client
```
python test_mcp_client.py
```
You should see:
	•	MCP tools listed
	•	content ingested
	•	successful semantic searches

🎉 You are now searching your own study material.

⸻

### 6) Try your own searches

The test client runs example queries like:
	•	“What are the stages of photosynthesis?”
	•	“Compare photosynthesis vs cellular respiration”

Add your own:
```
search_queries = [
    "Explain the Calvin cycle step by step",
    "Which reactions produce ATP?",
]
```
### 🔍 Available MCP Tools

Tool | Description
ingest_url | Fetch and ingest a webpage
ingest_file | Ingest PDF / TXT / MD files
search | Semantic search over ingested content
sift_generate | Retrieve + generate study content

🔐 AI Provider Configuration (Optional)

OpenSift works without any API keys by default.

OpenAI
```
export OPENAI_API_KEY="sk-..."
```
Claude (Anthropic API)
```
export ANTHROPIC_API_KEY="sk-ant-..."
```
Claude Code (subscription / setup-token)
claude setup-token
export CLAUDE_CODE_OAUTH_TOKEN="sk-ant-oat01-..."
unset ANTHROPIC_API_KEY

Then call:
```
{
  "provider": "claude_code"
}
```
🚀 Why OpenSift?

Most study tools either:
	•	summarize without grounding, or
	•	require constant manual searching

OpenSift flips that model:
	•	You ingest everything once
	•	You retrieve exactly what matters
	•	AI works with your sources, not instead of them

It’s built for:
	•	exam preparation
	•	research synthesis
	•	large reading loads
	•	agent-based study workflows

⸻

🛣️ Roadmap (High-Level)
	•	✅ Local ingestion + semantic search
	•	✅ No-key local embeddings
	•	✅ MCP-based agent integration
	•	🔜 CLI ingestion (opensift ingest)
	•	🔜 Per-course / per-project collections
	•	🔜 Exam mode (quizzes + flashcards)
	•	🔜 Lightweight web UI

⸻

📜 License

MIT

⸻

🙌 Acknowledgements
	•	ChromaDB
	•	sentence-transformers
	•	MCP (Model Context Protocol)
	•	OpenAI & Anthropic ecosystems
