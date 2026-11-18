import ipaddress
import socket
from urllib.parse import urlparse
import httpx
from bs4 import BeautifulSoup


def validate_public_url(url: str):
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Only public HTTP(S) URLs are allowed")
    host = parsed.hostname
    try:
        addresses = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        raise ValueError("URL hostname could not be resolved") from exc
    for address in addresses:
        ip = ipaddress.ip_address(address[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            raise ValueError("Private or local network URLs are not allowed")
    return url


async def extract_article(url: str):
    validate_public_url(url)
    async with httpx.AsyncClient(follow_redirects=True, timeout=12.0, headers={"User-Agent": "TruthLensAI/1.0"}) as client:
        response = await client.get(url)
        response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    title_tag = soup.find("title")
    title = title_tag.get_text(" ", strip=True) if title_tag else None
    og_title = soup.find("meta", property="og:title")
    if og_title and og_title.get("content"):
        title = og_title["content"]

    author = None
    author_meta = soup.find("meta", attrs={"name": "author"}) or soup.find("meta", property="article:author")
    if author_meta:
        author = author_meta.get("content")

    date = None
    date_meta = soup.find("meta", property="article:published_time") or soup.find("time")
    if date_meta:
        date = date_meta.get("content") or date_meta.get("datetime") or date_meta.get_text(" ", strip=True)

    for tag in soup(["script", "style", "nav", "header", "footer", "aside", "noscript"]):
        tag.decompose()
    paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
    text = "\n".join(p for p in paragraphs if len(p) > 40)
    if not text:
        raise ValueError("Readable article text could not be extracted")
    return {
        "headline": title,
        "text": text,
        "author": author,
        "publication_date": date,
        "domain": urlparse(str(url)).hostname,
    }
