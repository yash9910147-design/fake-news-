import re


def evidence_stance(claim: str, source: dict):
    text = f"{source.get('title') or ''} {source.get('snippet') or ''}".lower()
    claim_words = {word for word in re.findall(r"[a-z]{4,}", claim.lower())}
    source_words = set(re.findall(r"[a-z]{4,}", text))
    overlap = len(claim_words & source_words) / max(1, len(claim_words))
    contradiction_words = {"denies", "false", "wrong", "debunked", "contradicts", "rejected", "not true"}
    support_words = {"confirmed", "reports", "announced", "verified", "according", "says"}
    stance = "unrelated"
    if overlap >= 0.25:
        if any(phrase in text for phrase in contradiction_words):
            stance = "contradicts"
        elif any(word in text for word in support_words):
            stance = "supports"
        else:
            stance = "mentions"
    source["relevance_score"] = round(min(1.0, overlap), 3)
    source["evidence_stance"] = stance
    return source


def assess(sources: list[dict], claims: list[dict], fact_checks: list[dict], ml: dict):
    support = sum(s.get("evidence_stance") == "supports" for s in sources)
    contradict = sum(s.get("evidence_stance") == "contradicts" for s in sources)
    relevant = sum(s.get("elevance_score", s.get("relevance_score", 0)) >= 0.25 for s in sources)
    verdicts = {v.get("verdict") for v in fact_checks}

    if verdicts & {"False", "Mostly False", "Misleading", "Partially False"}:
        final = "LIKELY MISLEADING"
    elif verdicts & {"True", "Mostly True"} and contradict == 0:
        final = "LIKELY CREDIBLE"
    elif contradict > support:
        final = "LIKELY MISLEADING"
    elif support >= 2 and contradict == 0:
        final = "LIKELY CREDIBLE"
    elif relevant == 0 and not fact_checks:
        final = "UNVERIFIED"
    else:
        final = "MIXED EVIDENCE"

    explanation = {
        "factors": [
            {"name": "ML model", "value": ml.get("label"), "detail": "Text-pattern prediction only; not a factual verdict."},
            {"name": "Fact-check evidence", "value": sorted(v for v in verdicts if v), "detail": "Normalized verdicts from configured fact-check sources."},
            {"name": "Source evidence", "value": {"supporting": support, "contradicting": contradict, "relevant": relevant}, "detail": "Keyword overlap and stance heuristics; not semantic fact confirmation."},
        ],
        "limitations": [
            "The saved Linear SVM uses historical labeled text and has not been calibrated on live news.",
            "Retrieved articles may mention or share an underlying claim without independently verifying it.",
            "Source and stance scores are transparent heuristics, not proof of factual truth.",
        ],
        "missing_evidence": [] if (fact_checks or relevant) else ["No matching fact-check or relevant retrieved article was found."],
    }
    return final, explanation
