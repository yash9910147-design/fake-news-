import gzip
import html
import json
import math
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "fake_news_tfidf.joblib"
HEADLINE_MODEL_PATH = ROOT / "models" / "fake_news_headline.joblib"
MODEL_INFERENCE_PATH = ROOT / "models" / "fake_news_tfidf.inference.json.gz"
HEADLINE_INFERENCE_PATH = ROOT / "models" / "fake_news_headline.inference.json.gz"
LABELS = {0: "Fake", 1: "Real"}
HEADLINE_MAX_LENGTH = 280


def clean_text(value):
    text = html.unescape(str(value)).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


@lru_cache(maxsize=3)
def load_inference_model(model_path):
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Inference model not found at {model_path}. Train and export the model first."
        )
    with gzip.open(model_path, "rt", encoding="utf-8") as model_file:
        model = json.load(model_file)
    for transformer in model["transformers"]:
        transformer["vocabulary"] = {
            term: index for index, term in enumerate(transformer.pop("terms"))
        }
    return model


@lru_cache(maxsize=4)
def load_model(model_path=MODEL_PATH):
    model_path = Path(model_path)
    if model_path.resolve() == MODEL_PATH.resolve():
        return load_inference_model(MODEL_INFERENCE_PATH)
    if model_path.suffixes[-2:] == [".json", ".gz"]:
        return load_inference_model(model_path)
    import joblib

    if not model_path.exists():
        raise FileNotFoundError(
            f"No trained model at {model_path}. Run the training script first."
        )
    return joblib.load(model_path)


@lru_cache(maxsize=1)
def load_headline_model():
    return load_inference_model(HEADLINE_INFERENCE_PATH)


def load_model_metadata(model_path=MODEL_PATH):
    metadata_path = Path(model_path).with_name(f"{Path(model_path).stem}.metadata.json")
    if not metadata_path.exists():
        return {}
    with metadata_path.open(encoding="utf-8") as metadata_file:
        return json.load(metadata_file)


def _vectorize_transformer(text, transformer):
    vocabulary = transformer["vocabulary"]
    if transformer["analyzer"] == "word":
        stop_words = set(transformer["stop_words"])
        tokens = [
            token
            for token in re.findall(r"(?u)\b\w\w+\b", text.lower())
            if token not in stop_words
        ]
        min_n, max_n = transformer["ngram_range"]
        counts = Counter(tokens if min_n <= 1 else ())
        if min_n <= 2 <= max_n:
            counts.update(
                f"{left} {right}" for left, right in zip(tokens, tokens[1:])
            )
    else:
        counts = Counter()
        min_n, max_n = transformer["ngram_range"]
        for token in text.split():
            padded = f" {token} "
            for size in range(min_n, max_n + 1):
                counts.update(
                    padded[index : index + size]
                    for index in range(len(padded) - size + 1)
                )

    weighted = {}
    squared_norm = 0.0
    for term, count in counts.items():
        index = vocabulary.get(term)
        if index is None:
            continue
        value = (1.0 + math.log(count)) * transformer["idf"][index]
        weighted[index] = value
        squared_norm += value * value
    if squared_norm:
        norm = math.sqrt(squared_norm)
        for index in weighted:
            weighted[index] /= norm
    return weighted


def _portable_scores(model, text):
    if model["format_version"] != 1:
        raise ValueError(f"Unsupported inference model format: {model['format_version']}")
    if model["classifier"] == "linear":
        score = model["intercepts"][0]
        terms = []
        for transformer in model["transformers"]:
            weighted = _vectorize_transformer(text, transformer)
            coefficients = transformer["coefficients"][0]
            terms.extend(coefficients[index] * value for index, value in weighted.items())
        return math.fsum([score, *terms])

    if model["classifier"] != "multinomial_nb":
        raise ValueError(f"Unsupported inference classifier: {model['classifier']}")
    class_scores = list(model["intercepts"])
    for transformer in model["transformers"]:
        weighted = _vectorize_transformer(text, transformer)
        for class_index, coefficients in enumerate(transformer["coefficients"]):
            class_scores[class_index] += math.fsum(
                coefficients[index] * value for index, value in weighted.items()
            )
    return class_scores


def predict_news(text, model_path=MODEL_PATH):
    cleaned = clean_text(text)
    if not cleaned:
        raise ValueError("Enter some news text to classify")

    is_default_model = Path(model_path).resolve() == MODEL_PATH.resolve()
    is_headline = is_default_model and len(cleaned) <= HEADLINE_MAX_LENGTH
    model = load_headline_model() if is_headline else load_model(model_path)
    decision_score = None
    decision_threshold = None
    probability_score = None
    if isinstance(model, dict):
        model_score = _portable_scores(model, cleaned)
        if model["classifier"] == "linear":
            decision_score = float(model_score)
        else:
            maximum = max(model_score)
            exponentials = [math.exp(score - maximum) for score in model_score]
            probabilities = [
                probability / sum(exponentials) for probability in exponentials
            ]
            class_index = probabilities.index(max(probabilities))
            prediction = int(model["classes"][class_index])
            probability_score = probabilities[class_index]
        if is_headline and decision_score is not None:
            decision_threshold = load_model_metadata(HEADLINE_MODEL_PATH).get(
                "decision_threshold", model["decision_threshold"]
            )
            prediction = int(decision_score >= decision_threshold)
        elif decision_score is not None:
            prediction = int(decision_score > 0)
    else:
        classifier = model.named_steps["classifier"]
        if hasattr(classifier, "decision_function"):
            decision_score = float(model.decision_function([cleaned])[0])
        if is_headline and decision_score is not None:
            decision_threshold = load_model_metadata(HEADLINE_MODEL_PATH).get(
                "decision_threshold", 0.0
            )
            prediction = int(decision_score >= decision_threshold)
        else:
            prediction = int(model.predict([cleaned])[0])

    result = {
        "label": LABELS[prediction],
        "model_name": (
            "Logistic Regression headline model"
            if is_headline
            else "Linear SVM article model"
        ),
        "model_version": (
            HEADLINE_MODEL_PATH.stem if is_headline else MODEL_PATH.stem
        ),
        "input_mode": "headline" if is_headline else "article",
        "confidence": None,
        "processed_text": cleaned,
    }
    if decision_threshold is not None:
        result["decision_threshold"] = decision_threshold
    if is_headline and decision_score is not None:
        result["score"] = decision_score
        result["score_type"] = "decision score; not a probability"
    elif not isinstance(model, dict) and hasattr(classifier, "predict_proba"):
        result["score"] = float(model.predict_proba([cleaned])[0][prediction])
        result["score_type"] = "model probability; not a fact-check"
    elif probability_score is not None:
        result["score"] = probability_score
        result["score_type"] = "model probability; not a fact-check"
    elif decision_score is not None:
        result["score"] = decision_score
        result["score_type"] = "decision score; not a probability"
    return result
