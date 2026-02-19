FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    NPM_CONFIG_UPDATE_NOTIFIER=false \
    HOME=/home/opensift

ARG INSTALL_CLAUDE_CODE_CLI=true
ARG INSTALL_CODEX_CLI=true
ARG CLAUDE_CODE_NPM_PACKAGE=@anthropic-ai/claude-code
ARG CODEX_NPM_PACKAGE=@openai/codex

WORKDIR /app

# Runtime packages for OCR/URL/PDF workflows.
RUN apt-get update && apt-get install -y --no-install-recommends \
    nodejs \
    npm \
    poppler-utils \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Optional CLI providers for containerized runs.
RUN set -eux; \
    if [ "${INSTALL_CLAUDE_CODE_CLI}" = "true" ]; then npm install -g "${CLAUDE_CODE_NPM_PACKAGE}"; fi; \
    if [ "${INSTALL_CODEX_CLI}" = "true" ]; then npm install -g "${CODEX_NPM_PACKAGE}"; fi

COPY backend/requirements.txt /app/requirements.txt

RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install -r /app/requirements.txt && \
    python -m pip install sentence-transformers pytest

COPY backend /app

# Non-root runtime by default with a writable/persistent home for CLI auth.
RUN addgroup --system opensift && \
    adduser --system --home /home/opensift --ingroup opensift opensift && \
    mkdir -p /home/opensift/.codex /home/opensift/.claude && \
    chown -R opensift:opensift /app /home/opensift
USER opensift

EXPOSE 8001

CMD ["python", "opensift.py", "gateway", "--host", "0.0.0.0", "--port", "8001", "--with-mcp"]
