from __future__ import annotations

import asyncio
import io
import os
import re
from typing import List, Optional, Tuple

import httpx
from bs4 import BeautifulSoup
from pypdf import PdfReader

MAX_URL_RETRIES = 3
URL_TIMEOUT = httpx.Timeout(40.0, connect=15.0)
MAX_ARTICLE_CHARS = 350_000
MIN_EXTRACTED_CHARS = 500


def _normalize_text(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = [ln.strip() for ln in text.split("\n")]
    lines = [ln for ln in lines if ln]
    return "\n".join(lines).strip()


def _extract_title(soup: BeautifulSoup, fallback: str) -> str:
    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()
    if not title:
        h1 = soup.find("h1")
        if h1:
            title = h1.get_text(" ", strip=True)
    return title or fallback


def _strip_noise(soup: BeautifulSoup) -> None:
    for tag in soup(["script", "style", "noscript", "svg", "canvas", "iframe", "form", "header", "footer", "nav", "aside"]):
        tag.decompose()

    # Remove likely boilerplate/cookie/utility containers.
    noise_pat = re.compile(r"cookie|consent|banner|modal|popup|subscribe|newsletter|share|social|related|breadcrumbs?", re.I)
    for node in soup.find_all(True):
        cls = " ".join(node.get("class", [])).strip()
        nid = (node.get("id") or "").strip()
        if noise_pat.search(cls) or noise_pat.search(nid):
            node.decompose()


def _candidate_nodes(soup: BeautifulSoup) -> List:
    selectors = [
        "article",
        "main",
        "[role='main']",
        ".article",
        ".article-content",
        ".post-content",
        ".entry-content",
        "#article",
        "#main",
        "#content",
    ]

    out = []
    for sel in selectors:
        out.extend(soup.select(sel))
    out.extend(soup.find_all(["section", "div"]))
    return out


def _node_text_score(node) -> Tuple[int, str]:
    text = node.get_text("\n", strip=True)
    if not text:
        return (0, "")

    text = _normalize_text(text)
    if not text:
        return (0, "")

    p_count = len(node.find_all("p"))
    h_count = len(node.find_all(["h1", "h2", "h3", "h4"]))
    score = len(text) + p_count * 180 + h_count * 120
    return (score, text)


def _extract_main_text(soup: BeautifulSoup) -> str:
    best_text = ""
    best_score = 0

    for node in _candidate_nodes(soup):
        score, text = _node_text_score(node)
        if score > best_score:
            best_score = score
            best_text = text

    if len(best_text) >= MIN_EXTRACTED_CHARS:
        return best_text

    # Fallback: prioritize semantic reading tags.
    pieces: List[str] = []
    for tag in soup.find_all(["h1", "h2", "h3", "h4", "p", "li", "blockquote", "pre"]):
        t = tag.get_text(" ", strip=True)
        if not t:
            continue
        if len(t) < 2:
            continue
        pieces.append(t)

    return _normalize_text("\n".join(pieces))


def _truncate_for_storage(text: str, max_chars: int = MAX_ARTICLE_CHARS) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text

    cut = text[:max_chars]
    split = cut.rfind("\n")
    if split > max_chars * 0.7:
        cut = cut[:split]
    return cut.strip()


def _ocr_with_embedded_images(reader: PdfReader) -> str:
    try:
        from PIL import Image  # type: ignore
        import pytesseract  # type: ignore
    except Exception:
        return ""

    out: List[str] = []
    lang = os.getenv("OPENSIFT_OCR_LANG", "eng")

    for page in reader.pages:
        images = getattr(page, "images", None)
        if not images:
            continue
        for img in images:
            try:
                data = getattr(img, "data", None)
                if not data:
                    continue
                pil = Image.open(io.BytesIO(data))
                txt = pytesseract.image_to_string(pil, lang=lang) or ""
                txt = _normalize_text(txt)
                if txt:
                    out.append(txt)
            except Exception:
                continue

    return "\n\n".join(out).strip()


def _ocr_with_pdf2image(data: bytes) -> str:
    try:
        from pdf2image import convert_from_bytes  # type: ignore
        import pytesseract  # type: ignore
    except Exception:
        return ""

    try:
        images = convert_from_bytes(data, dpi=220, fmt="png", thread_count=2)
    except Exception:
        return ""

    out: List[str] = []
    lang = os.getenv("OPENSIFT_OCR_LANG", "eng")
    for img in images:
        try:
            txt = pytesseract.image_to_string(img, lang=lang) or ""
            txt = _normalize_text(txt)
            if txt:
                out.append(txt)
        except Exception:
            continue

    return "\n\n".join(out).strip()


def extract_text_from_pdf(data: bytes) -> str:
    reader = PdfReader(io.BytesIO(data))

    pages: List[str] = []
    blank_pages = 0
    for p in reader.pages:
        t = (p.extract_text() or "").strip()
        pages.append(t)
        if len(t) < 20:
            blank_pages += 1

    text = _normalize_text("\n\n".join(pages))

    # If mostly scanned/empty, try OCR fallbacks.
    is_likely_scanned = (blank_pages >= max(1, int(len(reader.pages) * 0.6))) or (len(text) < max(300, len(reader.pages) * 40))
    if is_likely_scanned:
        ocr_text = _ocr_with_embedded_images(reader)
        if len(ocr_text) < 300:
            ocr_text = _ocr_with_pdf2image(data)

        if len(ocr_text) > len(text):
            text = ocr_text
        elif ocr_text:
            text = _normalize_text(f"{text}\n\n{ocr_text}")

    return text.strip()


def extract_text_from_txt(data: bytes) -> str:
    # utf-8 first, then latin-1 fallback
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("latin-1", errors="ignore")


async def _download_html(url: str) -> str:
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; OpenSift/0.7; +study-tool)",
        "Accept": "text/html,application/xhtml+xml",
    }

    last_err: Optional[Exception] = None
    async with httpx.AsyncClient(follow_redirects=True, timeout=URL_TIMEOUT) as client:
        for attempt in range(1, MAX_URL_RETRIES + 1):
            try:
                r = await client.get(url, headers=headers)
                r.raise_for_status()
                return r.text
            except (httpx.TimeoutException, httpx.NetworkError, httpx.HTTPStatusError) as e:
                last_err = e
                if attempt >= MAX_URL_RETRIES:
                    break
                await asyncio.sleep(0.35 * attempt)

    if last_err:
        raise last_err
    raise RuntimeError("Unknown URL fetch error")


async def fetch_url_text(url: str) -> Tuple[str, str]:
    """
    Returns (title, text) from a URL with extraction/retry safeguards.
    """
    html = await _download_html(url)
    soup = BeautifulSoup(html, "html.parser")

    _strip_noise(soup)
    title = _extract_title(soup, fallback=url)

    text = _extract_main_text(soup)
    text = _truncate_for_storage(_normalize_text(text))

    if len(text) < 120:
        raise RuntimeError("Could not extract enough article text from URL")

    return title, text
