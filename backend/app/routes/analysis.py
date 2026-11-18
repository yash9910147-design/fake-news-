from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session
from slowapi import Limiter
from slowapi.util import get_remote_address

from ..database import Analysis, MongoDatabase, get_db
from ..schemas import TextAnalysisRequest, UrlAnalysisRequest, AnalysisResponse
from ..services.analysis_service import analyze_text
from ..services.article_service import extract_article

limiter = Limiter(key_func=get_remote_address)
router = APIRouter()


@router.post("/analyze/text", response_model=AnalysisResponse)
@limiter.limit("20/minute")
async def analyze_text_endpoint(request: Request, payload: TextAnalysisRequest, db: Session = Depends(get_db)):
    try:
        return await analyze_text(db, payload.text, "text")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/analyze/url", response_model=AnalysisResponse)
@limiter.limit("10/minute")
async def analyze_url_endpoint(request: Request, payload: UrlAnalysisRequest, db: Session = Depends(get_db)):
    try:
        article = await extract_article(str(payload.url))
        result = await analyze_text(db, article["text"], "url")
        result["headline"] = article["headline"]
        return result
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"URL extraction failed: {exc}") from exc


@router.get("/history")
def history(limit: int = 50, db: Session = Depends(get_db)):
    limit = min(limit, 100)
    if isinstance(db, MongoDatabase):
        return db.list_history(limit)
    rows = db.query(Analysis).order_by(Analysis.created_at.desc()).limit(min(limit, 100)).all()
    return [{"id": row.id, "input_type": row.input_type, "headline": row.headline,
             "ml_prediction": row.ml_prediction, "final_assessment": row.final_assessment,
             "created_at": row.created_at.isoformat()} for row in rows]


@router.get("/analysis/{analysis_id}")
def get_analysis(analysis_id: str, db: Session = Depends(get_db)):
    if isinstance(db, MongoDatabase):
        row = db.get_analysis(str(analysis_id))
        if not row:
            raise HTTPException(status_code=404, detail="Analysis not found")
        return row
    try:
        numeric_id = int(analysis_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Analysis not found") from exc
    row = db.query(Analysis).filter(Analysis.id == numeric_id).first()
    if not row:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return {"id": row.id, "original_input": row.original_input, "input_type": row.input_type,
            "headline": row.headline, "extracted_text": row.extracted_text,
            "ml": {"label": row.ml_prediction, "score": row.ml_score,
                   "score_type": row.explanation.get("ml_model", {}).get(
                       "score_type", "decision score; not a probability"
                   ),
                   "model_name": row.explanation.get("ml_model", {}).get(
                       "name", "Linear SVM"
                   ),
                   "model_version": row.explanation.get("ml_model", {}).get(
                       "version", "fake_news_tfidf"
                   ),
                   "input_mode": row.explanation.get("ml_model", {}).get(
                       "input_mode", "article"
                   )},
            "claims": [{"text": claim.claim_text, "claim_type": claim.claim_type,
                        "fact_checks": [{"claim_searched": claim.claim_text, "publisher": check.publisher,
                                         "verdict": check.verdict, "url": check.url, "review_date": check.review_date}
                                        for check in claim.fact_checks]} for claim in row.claims],
            "sources": [{"title": source.title, "publisher": source.publisher, "url": source.url,
                         "publication_date": source.publication_date, "relevance_score": source.relevance_score,
                         "evidence_stance": source.evidence_stance} for source in row.sources],
            "fact_checks": [{"claim_searched": claim.claim_text, "publisher": check.publisher,
                             "verdict": check.verdict, "url": check.url, "review_date": check.review_date}
                            for claim in row.claims for check in claim.fact_checks],
            "final_assessment": row.final_assessment, "explanation": row.explanation,
            "created_at": row.created_at}
