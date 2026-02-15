from __future__ import annotations
from typing import Tuple
import httpx
from bs4 import BeautifulSoup
from pypdf import PdfReader
from io import BytesIO

def extract_text_from_pdf(data: bytes) -> str:
    reader = PdfReader(BytesIO(data))
    pages = []
    for p in reader.pages:
        pages.append(p.extract_text() or "")
    return "\n\n".join(pages).strip()

def extract_text_from_txt(data: bytes) -> str:
    # naive utf-8 with fallback
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("latin-1", errors="ignore")

async def fetch_url_text(url: str) -> Tuple[str, str]:
    """
    Returns (title, text) from a URL.
    """
    async with httpx.AsyncClient(follow_redirects=True, timeout=30) as client:
        r = await client.get(url, headers={"User-Agent": "OpenSift/0.1 (+study-tool)"})
        r.raise_for_status()

    html = r.text
    soup = BeautifulSoup(html, "html.parser")

    # Remove obvious non-content
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
        tag.decompose()

    title = (soup.title.string.strip() if soup.title and soup.title.string else url)

    # Prefer article text
    main = soup.find("article") or soup.body or soup
    text = main.get_text(separator="\n", strip=True)

    # Keep it readable
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return title, "\n".join(lines)