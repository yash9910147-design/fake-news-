import joblib
import pandas as pd
import pytest
from sklearn.naive_bayes import MultinomialNB

from fake_news_detector import (
    HEADLINE_INFERENCE_PATH,
    HEADLINE_MODEL_PATH,
    MODEL_INFERENCE_PATH,
    MODEL_PATH,
    _portable_scores,
    build_pipeline,
    clean_text,
    load_inference_model,
    load_model,
    predict_news,
    split_dataset,
    temporal_split_dataset,
)
from fake_news_detector_runtime import (
    _portable_scores as runtime_portable_scores,
    load_inference_model as load_runtime_inference_model,
    predict_news as runtime_predict_news,
)


def test_clean_text_removes_markup_urls_and_digits():
    text = "  FACTS <b>Proven</b> at https://example.com/story 2024! "

    assert clean_text(text) == "facts proven at"


def test_split_dataset_is_disjoint_and_stratified():
    data = pd.DataFrame(
        {
            "news": [f"article {index}" for index in range(100)],
            "label": [index % 2 for index in range(100)],
        }
    )

    splits = split_dataset(data)
    training_ids = set(splits["Training"].index)
    validation_ids = set(splits["Validation"].index)
    test_ids = set(splits["Test"].index)

    assert (len(training_ids), len(validation_ids), len(test_ids)) == (64, 16, 20)
    assert not training_ids & validation_ids
    assert not training_ids & test_ids
    assert not validation_ids & test_ids
    assert splits["Training"]["label"].value_counts(normalize=True).to_dict() == {
        0: pytest.approx(0.5),
        1: pytest.approx(0.5),
    }


def test_pipeline_combines_word_and_character_tfidf():
    texts = [
        "false rumor scandal",
        "false political rumor",
        "official government report",
        "official confirmed report",
    ]
    pipeline = build_pipeline(MultinomialNB())

    pipeline.fit(texts, [0, 0, 1, 1])
    feature_names = pipeline.named_steps["features"].get_feature_names_out()

    assert any(name.startswith("word_ngrams__") for name in feature_names)
    assert any(name.startswith("character_ngrams__") for name in feature_names)


def test_temporal_split_keeps_dates_ordered_and_groups_together():
    publication_dates = pd.date_range("2020-01-01", periods=100).repeat(2)
    data = pd.DataFrame(
        {
            "news": [f"article {index}" for index in range(200)],
            "label": [0, 1] * 100,
            "published_at": publication_dates,
        }
    )
    data.loc[0, "published_at"] = pd.NaT

    splits, boundaries = temporal_split_dataset(data)
    training_dates = set(splits["Training"]["published_at"])
    validation_dates = set(splits["Validation"]["published_at"])
    test_dates = set(splits["Test"]["published_at"])

    assert splits["Training"]["published_at"].max() < splits["Validation"]["published_at"].min()
    assert splits["Validation"]["published_at"].max() < splits["Test"]["published_at"].min()
    assert not training_dates & validation_dates
    assert not training_dates & test_dates
    assert not validation_dates & test_dates
    assert all(set(frame["label"]) == {0, 1} for frame in splits.values())
    assert boundaries["undated_rows"] == 1


def test_predict_news_loads_a_saved_pipeline(tmp_path):
    training_text = [
        "sensational fabricated rumor scandal hoax",
        "sensational fabricated political rumor hoax",
        "official confirmed government report policy",
        "official confirmed public government report",
    ]
    labels = [0, 0, 1, 1]
    pipeline = build_pipeline(MultinomialNB())
    pipeline.fit(training_text, labels)
    model_path = tmp_path / "fake_news_tfidf.joblib"
    joblib.dump(pipeline, model_path)
    load_model.cache_clear()

    result = predict_news("official government report", model_path=model_path)

    assert result["label"] in {"Fake", "Real"}
    assert result["score_type"] == "model probability; not a fact-check"


def test_short_headline_uses_headline_model():
    result = predict_news("LeT’s Lateef Bhat killed in Kulgam encounter")

    assert result["label"] == "Real"
    assert result["input_mode"] == "headline"
    assert result["model_name"] == "Logistic Regression headline model"
    assert result["score_type"] == "decision score; not a probability"
    assert result["score"] >= result["decision_threshold"]
    assert runtime_predict_news("LeT’s Lateef Bhat killed in Kulgam encounter") == result


@pytest.mark.parametrize(
    ("model_path", "inference_path", "samples"),
    [
        (
            MODEL_PATH,
            MODEL_INFERENCE_PATH,
            [
                "Government announces new economic plan",
                "A viral story claims the election was cancelled",
            ],
        ),
        (
            HEADLINE_MODEL_PATH,
            HEADLINE_INFERENCE_PATH,
            [
                "LeT’s Lateef Bhat killed in Kulgam encounter",
                "Officials confirm new public health measures",
            ],
        ),
    ],
)
def test_portable_model_scores_match_sklearn(model_path, inference_path, samples):
    pipeline = joblib.load(model_path)
    portable_model = load_inference_model(str(inference_path))
    runtime_model = load_runtime_inference_model(str(inference_path))

    for sample in samples:
        cleaned = clean_text(sample)
        expected = pipeline.decision_function([cleaned])[0]
        assert _portable_scores(portable_model, cleaned) == pytest.approx(
            expected, abs=1e-12
        )
        assert runtime_portable_scores(runtime_model, cleaned) == pytest.approx(
            expected, abs=1e-12
        )
