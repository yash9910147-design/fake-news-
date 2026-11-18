from datetime import datetime
from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, JSON, String, Text, create_engine
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from .core.config import settings


engine = create_engine(settings.database_url, connect_args={"check_same_thread": False} if settings.database_url.startswith("sqlite") else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


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
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    Base.metadata.create_all(bind=engine)
