import asyncio
from urllib.parse import quote_plus
import httpx
import feedparser


async def gdelt_search(query: str, limit: int = 10):
    url = "https://api.gdeltproject.org/api/v2/doc/doc"
    params = {"query": query, "mode": "artlist", "format": "json", "maxrecords": str(limit), "sort": "hybridrel"}
    async with httpx.AsyncClient(timeout=12.0) as client:
        response = await client.get(url, params=params)
        if response.status_code != 200:
            return []
        payload = response.json()
    return [{"title": item.get("title"), "publisher": item.get("domain"), "url": item.get("url"), "publication_date": item.get("seendate"), "snippet": item.get("title"), "relevance_score": 0.65, "evidence_stance": "mention"} for item in payload.get("articles", [])]


async def google_news_rss(query: str, limit: int = 10):
    url = f"https://news.google.com/rss/search?q={quote_plus(query)}&hl=en-US&gl=US&ceid=US:en"
    feed = await asyncio.to_thread(feedparser.parse, url)
    results = []
    for entry in feed.entries[:limit]:
        results.append({"title": entry.get("title"), "publisher": getattr(entry, "source", {}).get("title") if getattr(entry, "source", None) else None, "url": entry.get("link"), "publication_date": entry.get("published"), "snippet": entry.get("summary", ""), "relevance_score": 0.60, "evidence_stance": "mention"})
    return results


async def newsapi_search(query: str, api_key: str | None, limit: int = 10):
    if not api_key:
        return []
    async with httpx.AsyncClient(timeout=12.0) as client:
        response = await client.get("https://newsapi.org/v2/everything", params={"q": query, "pageSize": limit, "language": "en", "sortBy": "relevancy"}, headers={"X-Api-Key": api_key})
        if response.status_code != 200:
            return []
        payload = response.json()
    return [{"title": item.get("title"), "publisher": item.get("source", {}).get("name"), "url": item.get("url"), "publication_date": item.get("publishedAt"), "snippet": item.get("description") or item.get("title"), "relevance_score": 0.70, "evidence_stance": "mention"} for item in payload.get("articles", [])]


async def search_news(query: str, newsapi_key: str | None = None):
    calls = [gdelt_search(query), google_news_rss(query)]
    if newsapi_key:
        calls.append(newsapi_search(query, newsapi_key))
    results = await asyncio.gather(*calls, return_exceptions=True)
    combined = []
    seen = set()
    for provider_result in results:
        if isinstance(provider_result, Exception):
            continue
        for item in provider_result:
            url = item.get("url")
            if not url or url in seen:
                continue
            seen.add(url)
            combined.append(item)
    return sorted(combined, key=lambda item: item.get("relevance_score", 0), reverse=True)[:20]
