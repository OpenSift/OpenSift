FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Runtime packages for OCR/URL/PDF workflows.
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt /app/requirements.txt

RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install -r /app/requirements.txt && \
    python -m pip install sentence-transformers pytest

COPY backend /app

# Non-root runtime by default.
RUN addgroup --system opensift && adduser --system --ingroup opensift opensift && \
    chown -R opensift:opensift /app
USER opensift

EXPOSE 8001

CMD ["python", "opensift.py", "gateway", "--host", "0.0.0.0", "--port", "8001", "--with-mcp"]
