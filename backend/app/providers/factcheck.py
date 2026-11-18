import httpx


async def search_fact_checks(query: str, api_key: str | None = None):
    if not api_key:
        return []
    params = {"query": query, "languageCode": "en", "pageSize": 10, "key": api_key}
    async with httpx.AsyncClient(timeout=12.0) as client:
        response = await client.get("https://factchecktools.googleapis.com/v1alpha1/claims:search", params=params)
        if response.status_code != 200:
            return []
        payload = response.json()
    results = []
    for claim in payload.get("claims", []):
        text = claim.get("text", query)
        for review in claim.get("claimReview", []):
            results.append({"claim_searched": text, "publisher": review.get("publisher", {}).get("name"), "verdict": normalize_verdict(review.get("textualRating")), "url": review.get("url"), "review_date": review.get("reviewDate")})
    return results


def normalize_verdict(verdict: str | None):
    if not verdict:
        return "Unverified"
    lowered = verdict.lower()
    if "mostly true" in lowered:
        return "Mostly True"
    if "mostly false" in lowered or "misleading" in lowered:
        return "Misleading"
    if lowered in {"false", "fake", "pants on fire", "incorrect"}:
        return "False"
    if "partly" in lowered or "partially" in lowered:
        return "Partially False"
    if "true" in lowered:
        return "True"
    if "outdated" in lowered:
        return "Outdated"
    if "mixed" in lowered:
        return "Mixed"
    return "Unverified"
