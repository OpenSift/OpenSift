from __future__ import annotations

import asyncio
import io
import ipaddress
import os
import re
import socket
from urllib.parse import urlparse
from typing import List, Optional, Tuple

import anyio
import httpx
from bs4 import BeautifulSoup
from pypdf import PdfReader

MAX_URL_RETRIES = 3
MAX_URL_REDIRECTS = max(0, int(os.getenv("OPENSIFT_MAX_URL_REDIRECTS", "5")))
URL_TIMEOUT = httpx.Timeout(40.0, connect=15.0)
MAX_ARTICLE_CHARS = 350_000
MIN_EXTRACTED_CHARS = 500
ALLOW_PRIVATE_URLS = os.getenv("OPENSIFT_ALLOW_PRIVATE_URLS", "").strip().lower() in ("1", "true", "yes", "on")


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


def _extract_page_text_fallback(soup: BeautifulSoup) -> str:
    root = soup.body if soup.body is not None else soup
    return _normalize_text(root.get_text("\n", strip=True))


def _truncate_for_storage(text: str, max_chars: int = MAX_ARTICLE_CHARS) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text

    cut = text[:max_chars]
    split = cut.rfind("\n")
    if split > max_chars * 0.7:
        cut = cut[:split]
    return cut.strip()


def _is_blocked_host_label(hostname: str) -> bool:
    h = (hostname or "").strip().lower().rstrip(".")
    if not h:
        return True
    if h in ("localhost",):
        return True
    if h.endswith(".localhost") or h.endswith(".local"):
        return True
    return False


def _is_blocked_ip(value: str) -> bool:
    try:
        ip = ipaddress.ip_address(value)
    except ValueError:
        return False

    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def _validate_remote_url_sync(url: str) -> None:
    parsed = urlparse((url or "").strip())
    if parsed.scheme not in ("http", "https"):
        raise RuntimeError("Only http/https URLs are allowed.")
    if not parsed.hostname:
        raise RuntimeError("URL must include a hostname.")
    if _is_blocked_host_label(parsed.hostname):
        raise RuntimeError("Local hostnames are blocked for URL ingest.")

    if ALLOW_PRIVATE_URLS:
        return

    # Block direct private IP usage and private IP DNS resolution.
    if _is_blocked_ip(parsed.hostname):
        raise RuntimeError("Private/local IP targets are blocked for URL ingest.")

    try:
        infos = socket.getaddrinfo(parsed.hostname, None, type=socket.SOCK_STREAM)
    except socket.gaierror as e:
        raise RuntimeError(f"DNS resolution failed for host: {parsed.hostname}") from e

    if not infos:
        raise RuntimeError(f"Could not resolve host: {parsed.hostname}")

    resolved_ips = []
    for info in infos:
        sockaddr = info[4]
        if not sockaddr:
            continue
        ip = str(sockaddr[0]).strip()
        if not ip:
            continue
        resolved_ips.append(ip)
        if _is_blocked_ip(ip):
            raise RuntimeError("URL resolves to a private/local address and is blocked.")

    if not resolved_ips:
        raise RuntimeError("Could not resolve a valid IP for URL host.")


def _is_redirect_status(status_code: int) -> bool:
    return status_code in (301, 302, 303, 307, 308)


def _resolve_redirect_url(current_url: str, location: str) -> str:
    loc = (location or "").strip()
    if not loc:
        raise RuntimeError("Redirect response missing Location header.")
    target = httpx.URL(current_url).join(loc)
    parsed = urlparse(str(target))
    if parsed.scheme not in ("http", "https"):
        raise RuntimeError("Redirected to unsupported URL scheme.")
    return str(target)


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
    current_url = (url or "").strip()
    seen_urls = set()
    redirects = 0
    last_err: Optional[Exception] = None

    async with httpx.AsyncClient(follow_redirects=False, timeout=URL_TIMEOUT) as client:
        while True:
            if current_url in seen_urls:
                raise RuntimeError("Redirect loop detected.")
            seen_urls.add(current_url)

            await anyio.to_thread.run_sync(lambda: _validate_remote_url_sync(current_url))

            response: Optional[httpx.Response] = None
            for attempt in range(1, MAX_URL_RETRIES + 1):
                try:
                    response = await client.get(current_url, headers=headers)
                    break
                except (httpx.TimeoutException, httpx.NetworkError) as e:
                    last_err = e
                    if attempt >= MAX_URL_RETRIES:
                        break
                    await asyncio.sleep(0.35 * attempt)

            if response is None:
                if last_err:
                    raise last_err
                raise RuntimeError("Unknown URL fetch error")

            if _is_redirect_status(response.status_code):
                if redirects >= MAX_URL_REDIRECTS:
                    raise RuntimeError(f"Too many redirects (>{MAX_URL_REDIRECTS}).")
                current_url = _resolve_redirect_url(str(response.url), response.headers.get("location", ""))
                redirects += 1
                continue

            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as e:
                last_err = e
                raise
            return response.text


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
        # Fallback for pages where structural extraction fails but readable body text exists.
        fallback_text = _truncate_for_storage(_extract_page_text_fallback(soup))
        if len(fallback_text) > len(text):
            text = fallback_text

    if len(text) < 80:
        raise RuntimeError("Could not extract enough article text from URL")

    return title, text
