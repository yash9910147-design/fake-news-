from fastapi import APIRouter, Depends, Query
from sqlalchemy import func
from sqlalchemy.orm import Session

from ..database import Analysis, MongoDatabase, get_db
from ..schemas import HealthResponse, ModelInfo
from ..services.ml_service import get_model, model_info
from ..providers.news import search_news
from ..providers.factcheck import search_fact_checks
from ..core.config import settings

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health(db=Depends(get_db)):
    loaded = True
    try:
        get_model()
    except FileNotFoundError:
        loaded = False
    if isinstance(db, MongoDatabase):
        db.ping()
        database = "mongodb"
    else:
        database = settings.database_url.split(":")[0]
    return {
        "status": "ok" if loaded else "degraded",
        "model_loaded": loaded,
        "database": database,
    }


@router.get("/model/info", response_model=ModelInfo)
def model_info_endpoint():
    return model_info()


@router.get("/dashboard/stats")
def dashboard_stats(db: Session = Depends(get_db)):
    if isinstance(db, MongoDatabase):
        return db.dashboard_stats()
    total = db.query(func.count(Analysis.id)).scalar() or 0
    assessments = dict(db.query(Analysis.final_assessment, func.count(Analysis.id)).group_by(Analysis.final_assessment).all())
    predictions = dict(db.query(Analysis.ml_prediction, func.count(Analysis.id)).group_by(Analysis.ml_prediction).all())
    recent = db.query(Analysis).order_by(Analysis.created_at.desc()).limit(5).all()
    return {"total_analyses": total, "assessments": assessments, "ml_predictions": predictions,
            "recent_analyses": [{"id": r.id, "final_assessment": r.final_assessment, "created_at": r.created_at.isoformat()} for r in recent]}


@router.get("/news/search")
async def news_search(q: str = Query(min_length=2, max_length=300)):
    return await search_news(q, settings.newsapi_key)


@router.get("/fact-check/search")
async def factcheck_search(q: str = Query(min_length=2, max_length=300)):
    return await search_fact_checks(q, settings.factcheck_api_key)
