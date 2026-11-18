from backend.app.services.claim_service import extract_claims
from backend.app.services.evidence_service import evidence_stance, assess
from backend.app.providers.factcheck import normalize_verdict


def test_claim_extraction_returns_sentences():
    text = ("The government announced that 25 percent of schools will receive funding. "
            "Officials said the program will start next year.")
    claims = extract_claims(text)
    assert claims
    assert "claim_type" in claims[0]


def test_evidence_stance_is_transparent():
    source = {"title": "Study contradicts claim", "snippet": "The study reports that the claim was false and denied", "url": "https://example.com"}
    result = evidence_stance("The study is true", source)
    assert result["evidence_stance"] in {"contradicts", "supports", "mentions", "unrelated"}
    assert 0 <= result["relevance_score"] <= 1


def test_assessment_does_not_force_fake_real():
    final, explanation = assess([], [], [], {"label": "Fake"})
    assert final == "UNVERIFIED"
    assert explanation["missing_evidence"]


def test_factcheck_verdict_normalization():
    assert normalize_verdict("Mostly true") == "Mostly True"
    assert normalize_verdict("pants on fire") == "False"
    assert normalize_verdict(None) == "Unverified"
