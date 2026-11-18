from datetime import datetime
from functools import lru_cache
import os
from bson import ObjectId
from pymongo import MongoClient
from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, JSON, String, Text, create_engine
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from ..core.config import settings

engine = create_engine(settings.database_url, connect_args={"check_same_thread": False} if settings.database_url.startswith("sqlite") else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class MongoDatabase:
    def __init__(self):
        if not settings.mongodb_uri:
            raise RuntimeError("MONGODB_URI must be configured for this deployment")
        self.client = get_mongo_client()
        self.database = self.client[settings.mongodb_database]
        self.analyses = self.database["analyses"]
        self.analyses.create_index("created_at")
        self.analyses.create_index("final_assessment")
        self.analyses.create_index("ml.label")

    def ping(self):
        self.client.admin.command("ping")

    def save_analysis(self, document):
        result = self.analyses.insert_one(document)
        document["id"] = str(result.inserted_id)
        return document

    def list_history(self, limit):
        return [
            {
                "id": str(row["_id"]),
                "input_type": row["input_type"],
                "headline": row.get("headline"),
                "ml_prediction": row.get("ml", {}).get("label"),
                "final_assessment": row["final_assessment"],
                "created_at": row["created_at"].isoformat(),
            }
            for row in self.analyses.find(
                {}, {"input_type": 1, "headline": 1, "ml": 1, "final_assessment": 1, "created_at": 1}
            ).sort("created_at", -1).limit(limit)
        ]

    def get_analysis(self, analysis_id):
        if not ObjectId.is_valid(analysis_id):
            return None
        row = self.analyses.find_one({"_id": ObjectId(analysis_id)})
        if row is None:
            return None
        row["id"] = str(row.pop("_id"))
        row["created_at"] = row["created_at"]
        return row

    def dashboard_stats(self):
        total = self.analyses.count_documents({})
        assessments = {
            row["_id"]: row["count"]
            for row in self.analyses.aggregate(
                [{"$group": {"_id": "$final_assessment", "count": {"$sum": 1}}}]
            )
        }
        predictions = {
            row["_id"]: row["count"]
            for row in self.analyses.aggregate(
                [{"$group": {"_id": "$ml.label", "count": {"$sum": 1}}}]
            )
            if row["_id"] is not None
        }
        return {
            "total_analyses": total,
            "assessments": assessments,
            "ml_predictions": predictions,
            "recent_analyses": self.list_history(5),
        }


@lru_cache(maxsize=1)
def get_mongo_client():
    return MongoClient(settings.mongodb_uri, serverSelectionTimeoutMS=5000)


class Analysis(Base):
    __tablename__ = "analyses"
    id = Column(Integer, primary_key=True)
    original_input = Column(Text, nullable=False)
    input_type = Column(String(20), nullable=False)
    extracted_text = Column(Text, nullable=True)
    headline = Column(String(500), nullable=True)
    ml_prediction = Column(String(20), nullable=True)
    ml_score = Column(Float, nullable=True)
    final_assessment = Column(String(30), nullable=False)
    explanation = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    claims = relationship("Claim", back_populates="analysis", cascade="all, delete-orphan")
    sources = relationship("Source", back_populates="analysis", cascade="all, delete-orphan")

class Claim(Base):
    __tablename__ = "claims"
    id = Column(Integer, primary_key=True)
    analysis_id = Column(Integer, ForeignKey("analyses.id"), nullable=False, index=True)
    claim_text = Column(Text, nullable=False)
    claim_type = Column(String(50), nullable=True)
    assessment = Column(String(30), nullable=True)
    confidence = Column(Float, nullable=True)
    evidence_summary = Column(JSON, nullable=True)
    analysis = relationship("Analysis", back_populates="claims")
    fact_checks = relationship("FactCheck", back_populates="claim", cascade="all, delete-orphan")

class Source(Base):
    __tablename__ = "sources"
    id = Column(Integer, primary_key=True)
    analysis_id = Column(Integer, ForeignKey("analyses.id"), nullable=False, index=True)
    title = Column(String(500), nullable=True)
    publisher = Column(String(300), nullable=True)
    url = Column(Text, nullable=False)
    publication_date = Column(String(100), nullable=True)
    relevance_score = Column(Float, nullable=True)
    evidence_stance = Column(String(30), nullable=True)
    source_metadata = Column(JSON, nullable=True)
    analysis = relationship("Analysis", back_populates="sources")

class FactCheck(Base):
    __tablename__ = "fact_checks"
    id = Column(Integer, primary_key=True)
    claim_id = Column(Integer, ForeignKey("claims.id"), nullable=False, index=True)
    publisher = Column(String(300), nullable=True)
    verdict = Column(String(100), nullable=True)
    url = Column(Text, nullable=True)
    review_date = Column(String(100), nullable=True)
    claim = relationship("Claim", back_populates="fact_checks")

def get_db():
    if settings.mongodb_uri:
        yield MongoDatabase()
        return
    if os.getenv("VERCEL"):
        raise RuntimeError("Set MONGODB_URI in Vercel project environment variables")
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    Base.metadata.create_all(bind=engine)
