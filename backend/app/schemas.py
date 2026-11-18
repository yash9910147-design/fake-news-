from datetime import datetime
from typing import Any, Literal
from pydantic import BaseModel, Field, HttpUrl


class TextAnalysisRequest(BaseModel):
    text: str = Field(min_length=20, max_length=50_000)


class UrlAnalysisRequest(BaseModel):
    url: HttpUrl


class SourceResult(BaseModel):
    title: str | None = None
    publisher: str | None = None
    url: str
    publication_date: str | None = None
    snippet: str | None = None
    relevance_score: float = 0.0
    evidence_stance: str = "unverified"


class FactCheckResult(BaseModel):
    claim_searched: str
    publisher: str | None = None
    verdict: str | None = None
    url: str | None = None
    review_date: str | None = None


class ClaimResult(BaseModel):
    text: str
    claim_type: str
    fact_checks: list[FactCheckResult] = []


class AnalysisResponse(BaseModel):
    id: int | str | None = None
    original_input: str
    input_type: Literal["text", "url"]
    headline: str | None = None
    extracted_text: str
    ml: dict[str, Any]
    claims: list[ClaimResult]
    sources: list[SourceResult]
    fact_checks: list[FactCheckResult]
    final_assessment: str
    explanation: dict[str, Any]
    created_at: datetime


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    database: str


class ModelInfo(BaseModel):
    model_name: str
    model_version: str
    score_type: str
    preprocessing: str
    metrics: dict[str, Any]
