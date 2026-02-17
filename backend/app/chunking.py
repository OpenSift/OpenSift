from __future__ import annotations
from dataclasses import dataclass
import re
from typing import List, Optional

@dataclass
class Chunk:
    text: str
    chunk_id: str
    start: int
    end: int

def normalize(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def _choose_break(window: str, min_break: int) -> Optional[int]:
    # Prefer paragraph boundaries, then sentence boundaries.
    p = window.rfind("\n\n")
    if p >= min_break:
        return p + 2

    punct = [
        window.rfind(". "),
        window.rfind("! "),
        window.rfind("? "),
        window.rfind("; "),
    ]
    best = max(punct)
    if best >= min_break:
        return best + 1

    nl = window.rfind("\n")
    if nl >= min_break:
        return nl + 1

    return None


def chunk_text(
    text: str,
    chunk_size: int = 1100,
    overlap: int = 180,
    prefix: str = "src",
    max_chunks: Optional[int] = None,
) -> List[Chunk]:
    """
    Character chunking with overlap and break-aware boundaries.
    Tuned for long articles/notes while keeping chunks coherent.
    """
    text = normalize(text)
    if not text:
        return []

    chunks: List[Chunk] = []
    i = 0
    n = len(text)
    idx = 0

    while i < n:
        hard = min(n, i + chunk_size)
        lookahead = min(n, i + int(chunk_size * 1.25))
        window = text[i:lookahead]
        min_break = max(0, int(chunk_size * 0.60))
        chosen = _choose_break(window[: max(1, hard - i + int(chunk_size * 0.25))], min_break=min_break)
        if chosen is not None:
            j = i + chosen
        else:
            j = hard

        if j <= i:
            j = min(n, i + chunk_size)

        chunk = text[i:j].strip()
        if chunk:
            chunks.append(Chunk(text=chunk, chunk_id=f"{prefix}:{idx}:{i}:{j}", start=i, end=j))
            idx += 1
            if max_chunks is not None and len(chunks) >= max_chunks:
                break

        if j >= n:
            break
        ni = max(0, j - max(0, overlap))
        if ni <= i:
            ni = j
        i = ni

    return chunks
