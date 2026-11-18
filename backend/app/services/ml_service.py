from functools import lru_cache
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fake_news_detector_runtime import (
    HEADLINE_MODEL_PATH,
    clean_text,
    load_model,
    load_model_metadata,
    predict_news,
)
from ..core.config import settings


@lru_cache(maxsize=1)
def get_model():
    return load_model(settings.model_path)


def predict(text: str):
    cleaned = clean_text(text)
    if not cleaned:
        raise ValueError("Enter some meaningful news text")
    return predict_news(text, model_path=settings.model_path)


def model_info():
    metadata = load_model_metadata(settings.model_path)
    headline_metadata = load_model_metadata(HEADLINE_MODEL_PATH)
    return {
        "model_name": "Full-article and headline classifiers",
        "model_version": "fake_news_tfidf + fake_news_headline",
        "score_type": "decision score; not a probability",
        "preprocessing": (
            "Same project clean_text preprocessing for both models. Text up to "
            "280 characters uses the headline model; longer text uses the article model."
        ),
        "metrics": {
            "article_model": metadata,
            "headline_model": headline_metadata,
        },
    }
