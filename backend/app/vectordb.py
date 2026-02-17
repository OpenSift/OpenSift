from __future__ import annotations
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings
from .settings import settings

class VectorDB:
    def __init__(self) -> None:
        self.client = chromadb.PersistentClient(
            path=settings.chroma_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.col = self.client.get_or_create_collection(name=settings.collection_name)

    def add(
        self,
        ids: List[str],
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        embeddings: List[List[float]],
    ) -> None:
        # Upsert is safer than add for repeated ingests of the same source/chunk IDs.
        self.col.upsert(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)

    def query(
        self,
        embedding: List[float],
        k: int = 8,
        where: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return self.col.query(
            query_embeddings=[embedding],
            n_results=k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
