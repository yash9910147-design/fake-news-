from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "TruthLens AI"
    database_url: str = "sqlite:///./truthlens.db"
    mongodb_uri: str | None = None
    mongodb_database: str = "truthlens"
    newsapi_key: str | None = None
    factcheck_api_key: str | None = None
    llm_api_key: str | None = None
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    model_path: Path = Path("models/fake_news_tfidf.joblib")


settings = Settings()
