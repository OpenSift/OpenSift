from __future__ import annotations

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


def _openai_client():
    if not settings.openai_api_key:
        return None
    if OpenAI is None:
        raise RuntimeError("openai package not available, but OPENAI_API_KEY is set")
    return OpenAI(api_key=settings.openai_api_key)


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Embedding strategy:
    - If OPENAI_API_KEY is set: use OpenAI embeddings (higher quality + consistency)
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