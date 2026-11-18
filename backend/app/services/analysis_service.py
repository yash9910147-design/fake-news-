from datetime import datetime
from sqlalchemy.orm import Session
from ..database import Analysis, Claim, FactCheck, MongoDatabase, Source
from ..core.config import settings
from .ml_service import predict
from .claim_service import extract_claims
from .evidence_service import evidence_stance, assess
from ..providers.news import search_news
from ..providers.factcheck import search_fact_checks


async def analyze_text(db: Session, text: str, input_type: str = "text"):
    ml = predict(text)
    claims = extract_claims(text)
    if not claims:
        claims = [{"text": text[:500], "claim_type": "general factual claim"}]

    search_terms = [claims[0]["text"]] if claims else [text[:240]]
    source_results = await search_news(search_terms[0], settings.newsapi_key)

    fact_results = []
    for claim in claims:
        checks = await search_fact_checks(claim["text"], settings.factcheck_api_key)
        claim["fact_checks"] = checks
        fact_results.extend(checks)

    for source in source_results:
        evidence_stance(search_terms[0], source)
    final, explanation = assess(source_results, claims, fact_results, ml)
    explanation["ml_model"] = {
        "name": ml["model_name"],
        "version": ml["model_version"],
        "input_mode": ml["input_mode"],
        "score_type": ml["score_type"],
    }

    if isinstance(db, MongoDatabase):
        created_at = datetime.utcnow()
        document = {
            "original_input": text,
            "input_type": input_type,
            "extracted_text": text,
            "headline": None,
            "ml": {
                key: value
                for key, value in ml.items()
                if key != "processed_text"
            },
            "final_assessment": final,
            "explanation": explanation,
            "created_at": created_at,
            "claims": claims,
            "sources": source_results,
            "fact_checks": fact_results,
        }
        saved = db.save_analysis(document)
        saved["created_at"] = created_at
        return saved

    row = Analysis(original_input=text, input_type=input_type, extracted_text=text, headline=None,
                   ml_prediction=ml["label"], ml_score=ml.get("score"), final_assessment=final,
                   explanation=explanation, created_at=datetime.utcnow())
    db.add(row)
    db.flush()
    for claim in claims:
        claim_row = Claim(analysis_id=row.id, claim_text=claim["text"], claim_type=claim["claim_type"],
                          assessment="Unverified", confidence=None,
                          evidence_summary={"fact_check_count": len(claim.get("fact_checks", []))})
        db.add(claim_row)
        db.flush()
        for check in claim.get("fact_checks", []):
            db.add(FactCheck(claim_id=claim_row.id, publisher=check.get("publisher"), verdict=check.get("verdict"),
                             url=check.get("url"), review_date=check.get("review_date")))
    for source in source_results:
        db.add(Source(analysis_id=row.id, title=source.get("title"), publisher=source.get("publisher"),
                      url=source["url"], publication_date=source.get("publication_date"),
                      relevance_score=source.get("relevance_score"), evidence_stance=source.get("evidence_stance"),
                      source_metadata={"snippet": source.get("snippet")}))
    db.commit()
    db.refresh(row)

    return {"id": row.id, "original_input": text, "input_type": input_type, "headline": None,
            "extracted_text": text, "ml": ml, "claims": claims, "sources": source_results,
            "fact_checks": fact_results, "final_assessment": final, "explanation": explanation,
            "created_at": row.created_at}
