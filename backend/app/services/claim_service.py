import re


def classify_claim(sentence: str):
    lowered = sentence.lower()
    if any(word in lowered for word in ("percent", "%", "million", "billion", "study", "data")):
        return "statistical/scientific claim"
    if any(word in lowered for word in ("minister", "government", "president", "senator", "bill")):
        return "government/political claim"
    if any(word in lowered for word in ("died", "killed", "attack", "fire", "earthquake", "election")):
        return "event/incident claim"
    if re.search(r"\b(cure|treat|prevents|causes|vaccine|virus|disease)\b", lowered):
        return "health/scientific claim"
    return "general factual claim"


def extract_claims(text: str, limit: int = 8):
    sentences = re.split(r"(?<=[.!?])\s+", text.replace("\n", " "))
    claims = []
    seen = set()
    for sentence in sentences:
        cleaned = " ".join(sentence.split())
        if len(cleaned) < 35:
            continue
        if not re.search(r"\b(is|are|was|were|has|have|said|announced|reported|will|causes|killed|vaccine|percent)\b", cleaned.lower(), re.I):
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        claims.append({"text": cleaned, "claim_type": classify_claim(cleaned)})
        if len(claims) >= limit:
            break
    return claims
