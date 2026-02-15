from __future__ import annotations
from dataclasses import dataclass
import re
from typing import List

@dataclass
class Chunk:
    text: str
    chunk_id: str
    start: int
    end: int

def normalize(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def chunk_text(text: str, chunk_size: int = 900, overlap: int = 120, prefix: str = "src") -> List[Chunk]:
    """
    Simple character-based chunking with overlap (good enough for MVP).
    """
    text = normalize(text)
    if not text:
        return []

    chunks: List[Chunk] = []
    i = 0
    n = len(text)
    idx = 0

    while i < n:
        j = min(n, i + chunk_size)
        # try not to cut mid-sentence too harshly
        window = text[i:j]
        last_break = max(window.rfind(". "), window.rfind("\n\n"), window.rfind("; "))
        if last_break > 200:
            j = i + last_break + 1

        chunk = text[i:j].strip()
        if chunk:
            chunks.append(Chunk(text=chunk, chunk_id=f"{prefix}:{idx}", start=i, end=j))
            idx += 1

        if j >= n:
            break
        i = max(0, j - overlap)

    return chunks