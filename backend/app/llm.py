from __future__ import annotations

import os
from typing import List, Optional

from .settings import settings

# Optional OpenAI (only used if OPENAI_API_KEY is set)
try:
    from openai import OpenAI  # type: ignore
except Exception:
    OpenAI = None  # type: ignore

# Local embeddings
from sentence_transformers import SentenceTransformer


_local_model: Optional[SentenceTransformer] = None


def _get_local_model() -> SentenceTransformer:
    global _local_model
    if _local_model is None:
        # Small, fast, decent quality. Good MVP default.
        _local_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _local_model


def _openai_enabled() -> bool:
    # Allow either settings value OR environment variable.
    return bool(settings.openai_api_key) or bool(os.environ.get("OPENAI_API_KEY"))


def using_local_embeddings() -> bool:
    return not _openai_enabled()


def local_embedding_model_loaded() -> bool:
    return _local_model is not None


def warmup_local_embeddings() -> None:
    """
    Preload local embeddings model so first chat retrieval does not block on
    Hugging Face downloads/model initialization in the request path.
    """
    if not using_local_embeddings():
        return
    _ = _get_local_model()


def _openai_client():
    if not _openai_enabled():
        return None
    if OpenAI is None:
        raise RuntimeError("openai package not available, but OpenAI embeddings are enabled")
    # OpenAI() will read OPENAI_API_KEY from env automatically.
    return OpenAI()


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Embedding strategy:
    - If OpenAI embeddings are enabled: use OpenAI embeddings
    - Else: use local sentence-transformers embeddings (no keys required)
    """
    client = _openai_client()
    if client is not None:
        resp = client.embeddings.create(
            model=settings.embed_model,
            input=texts,
        )
        return [d.embedding for d in resp.data]

    # Local fallback
    model = _get_local_model()
    vecs = model.encode(texts, normalize_embeddings=True)
    return vecs.tolist()
